# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T
from torch.utils.data import Dataset
from util.misc import get_local_rank, get_local_size, nested_tensor_from_tensor_list
import random
from PIL import Image
import os.path
from typing import Any, Callable, Optional, Tuple, List
from pycocotools.coco import COCO

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cross_domain=False, domain_flag = -1):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.cross_domain = cross_domain
        self.domain_flag = domain_flag
        if self.cross_domain:
            self.strong_transforms = make_coco_transforms(image_set="train", cross_domain=cross_domain, strong_aug=True)
            self.weak_transforms   = make_coco_transforms(image_set="train", cross_domain=cross_domain, strong_aug=False)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            if self.cross_domain:
                weak_img, target = self.weak_transforms(img, target)
                strong_img, target = self.strong_transforms(weak_img, target)
                target["domain"] = torch.tensor([self.domain_flag])
                return weak_img, strong_img, target
            else:
                img, target = self._transforms(img, target)
        target["domain"] = torch.tensor([self.domain_flag]) # this dataset only be used in the inference. So this value does not matter
        return img, target




def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks



class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target


def make_coco_transforms(image_set, cross_domain=False, strong_aug=False):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        if not cross_domain:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ),
                normalize,
            ])
        if strong_aug:
            return  T.Compose([
                # T.RandomHorizontalFlip(),
                # T.RandomSelect(
                #     T.RandomResize(scales, max_size=1333),
                #     T.Compose([
                #         T.RandomResize([400, 500, 600]),
                #         T.RandomSizeCrop(384, 600),
                #         T.RandomResize(scales, max_size=1333),
                #     ])
                # ),
                T.StrongAug(),
                T.ToTensor(),
            ])
        else:
            return  T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ),
                normalize,
            ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    if image_set == 'test2':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "images" / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "images" / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "images" / "test2017", root / "annotations" / f'image_info_test-dev2017.json'),
    }

    paths = get_paths(args.coco_path)
    img_folder=paths["cityscapes"]['train_img']
    ann_file=paths['cityscapes']['train_anno']

    if image_set == "train":
        print("====load train dataset====")
        img_folder = paths["cityscapes"]['train_img'] 
        ann_file   = paths['cityscapes']['train_anno']
        if args.cdod:
            source_domain="cityscapes"
            target_domain="foggy_cityscapes"
            return DADataset(
                    source_img_folder=paths[source_domain]['train_img'],
                    source_ann_file=paths[source_domain]['train_anno'],
                    target_img_folder=paths[target_domain]['train_img'],
                    target_ann_file=paths[target_domain]['train_anno'],
                    transforms=make_coco_transforms(image_set),
                    return_masks=False,
                    cache_mode=False,
                    local_rank=args.rank,
                    local_size=args.world_size,
                    )

    if image_set == "test":
        print("====load test dataset====")
        img_folder=paths["foggy_cityscapes"]['val_img']
        ann_file=paths['foggy_cityscapes']['val_anno']
        domain_flag = 1     # 1 for target domain

    if image_set == "val":
        print("====load validation dataset====")
        img_folder=paths["cityscapes"]['val_img']
        ann_file=paths['cityscapes']['val_anno']
        domain_flag = -1    # -1 or 0 for source domain

    if image_set == "test2":
        print("====load test dataset====")
        img_folder=paths["foggy_cityscapes"]['val_img']
        ann_file=paths['foggy_cityscapes']['val_anno']
        domain_flag = -1     # 1 for target domain

    
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks, domain_flag=domain_flag)
    return dataset


def get_paths(root):
    root = Path(root)
    return {
        'cityscapes': {
            'train_img': root / 'cityscapes/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'cityscapes/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'cityscapes_caronly': {
            'train_img': root / 'cityscapes/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_caronly_train.json',
            'val_img': root / 'cityscapes/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_caronly_val.json',
        },
        'foggy_cityscapes': {
            'train_img': root / 'cityscapes/leftImg8bit_foggy/train',
            'train_anno': root / 'cityscapes/annotations/foggy_cityscapes_train.json',
            'val_img': root / 'cityscapes/leftImg8bit_foggy/val',
            'val_anno': root / 'cityscapes/annotations/foggy_cityscapes_val.json',
        },
        'sim10k': {
            'train_img': root / 'sim10k/VOC2012/JPEGImages',
            'train_anno': root / 'sim10k/annotations/sim10k_caronly.json',
        },
        'bdd_daytime': {
            'train_img': root / 'bdd_daytime/train',
            'train_anno': root / 'bdd_daytime/annotations/bdd_daytime_train.json',
            'val_img': root / 'bdd_daytime/val',
            'val_anno': root / 'bdd_daytime/annotations/bdd_daytime_val.json',
        }
    }


class DADataset(Dataset):
    def __init__(self, source_img_folder, source_ann_file, target_img_folder, target_ann_file,
                 transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        self.source = CocoDetection(
            img_folder=source_img_folder,
            ann_file=source_ann_file,
            transforms=transforms,
            return_masks=return_masks,
            cross_domain=True,
            domain_flag=-1,
            # cache_mode=cache_mode,
            # local_rank=local_rank,
            # local_size=local_size
        )

        self.target = CocoDetection(
            img_folder=target_img_folder,
            ann_file=target_ann_file,
            transforms=transforms,
            return_masks=return_masks,
            cross_domain=True,
            domain_flag=1,
            # cache_mode=cache_mode,
            # local_rank=local_rank,
            # local_size=local_size
        )


    def __len__(self):
        return max(len(self.source), len(self.target))
        # return len(self.combined_data)

    def __getitem__(self, idx):
        '''
        go to util/misc.py to modify the collate_strong_weaken_fn()
        '''
        wa_source_img, sa_source_img, source_target = self.source[idx % len(self.source)]
        wa_target_img, sa_target_img, _             = self.target[idx % len(self.target)]
        return wa_source_img, sa_source_img, wa_target_img,  sa_target_img,  source_target