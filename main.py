 # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json



import random
import time
from pathlib import Path
import os, sys
from typing import Optional


from util.logger import setup_logger

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
# from models import build_DABDETR
from models import build_dab_deformable_detr
from util.utils import clean_state_dict
import copy
# from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
from warmup_scheduler_pytorch import WarmUpScheduler

def get_args_parser():
    parser = argparse.ArgumentParser('DAB-DETR', add_help=False)

    # about dn args
    parser.add_argument('--use_dn', action="store_true",
                        help="use denoising training.")
    parser.add_argument('--scalar', default=5, type=int,
                        help="number of dn groups")
    parser.add_argument('--label_noise_scale', default=0.2, type=float,
                        help="label noise ratio to flip")
    parser.add_argument('--box_noise_scale', default=0.4, type=float,
                        help="box noise scale to shift and scale")

    # about lr
    parser.add_argument('--lr', default=1e-4, type=float, 
                        help='learning rate')
    parser.add_argument('--lr_backbone', default=1e-5, type=float, 
                        help='learning rate for backbone')

    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--drop_lr_now', action="store_true", help="load checkpoint and drop for 12epoch setting")
    parser.add_argument('--save_checkpoint_interval', default=10, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--modelname', '-m', type=str, required=True, choices=['dab_deformable_detr'])
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--pe_temperatureH', default=20, type=int, 
                        help="Temperature for height positional encoding.")
    parser.add_argument('--pe_temperatureW', default=20, type=int, 
                        help="Temperature for width positional encoding.")
    parser.add_argument('--batch_norm_type', default='FrozenBatchNorm2d', type=str, 
                        choices=['SyncBatchNorm', 'FrozenBatchNorm2d', 'BatchNorm2d'], help="batch norm type for backbone")

    # * Transformer
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone_freeze_keywords', nargs="+", type=str, 
                        help='freeze some layers in backbone. for catdet5.')
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true', 
                        help="Using pre-norm in the Transformer blocks.")    
    parser.add_argument('--num_select', default=300, type=int, 
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--transformer_activation', default='prelu', type=str)
    parser.add_argument('--num_patterns', default=0, type=int, 
                        help='number of pattern embeddings. See Anchor DETR for more details.')
    parser.add_argument('--random_refpoints_xy', action='store_true', 
                        help="Random init the x,y of anchor boxes and freeze them.")

    # for DAB-Deformable-DETR
    parser.add_argument('--two_stage', default=False, action='store_true', 
                        help="Using two stage variant for DAB-Deofrmable-DETR")
    parser.add_argument('--num_feature_levels', default=4, type=int, 
                        help='number of feature levels')
    parser.add_argument('--dec_n_points', default=4, type=int, 
                        help="number of deformable attention sampling points in decoder layers")
    parser.add_argument('--enc_n_points', default=4, type=int, 
                        help="number of deformable attention sampling points in encoder layers")


    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float, 
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=1, type=float, 
                        help="loss coefficient for cls")
    parser.add_argument('--mask_loss_coef', default=1, type=float, 
                        help="loss coefficient for mask")
    parser.add_argument('--dice_loss_coef', default=1, type=float, 
                        help="loss coefficient for dice")
    parser.add_argument('--bbox_loss_coef', default=5, type=float, 
                        help="loss coefficient for bbox L1 loss")
    parser.add_argument('--giou_loss_coef', default=2, type=float, 
                        help="loss coefficient for bbox GIOU loss")
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', type=float, default=0.25, 
                        help="alpha for focal loss")


    # dataset parameters
    parser.add_argument('--dataset_file', default='cityscapes')
    # parser.add_argument('--dataset_file2', default='cityscapes')
    parser.add_argument('--coco_path', type=str, required=True)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true', 
                        help="Using for debug only. It will fix the size of input images to the maximum.")


    # Traing utils
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--note', default='', help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+', 
                        help="A list of keywords to ignore when loading pretrained models.")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help="eval only. w/o Training.")
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--debug', action='store_true', 
                        help="For debug only. It will perform only a few steps during trainig and val.")
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true', 
                        help="For eval only. Save the outputs for all images.")
    parser.add_argument('--save_log', action='store_true', 
                        help="If save the training prints to the log file.")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")

    # for cross domain adaptation
    parser.add_argument('--cdod', action='store_false', help="Used for cross domain object detection, need to load two datasets (source&target)")
    parser.add_argument('--teacher_weights', default='', help='resume from checkpoint for teacher')
    parser.add_argument('--use_pseudo_labels', action="store_true", help="use teacher model to generate pseudo labels in training training.")
    parser.add_argument('--training_phase', default='wa_src_tgt', help='training phases in [wa_src_tgt, sa_src_tgt, sa_src_tgt_ema]')
    parser.add_argument('--not_with_box_refine', action='store_true',  help='training phases in [wa_src_tgt, sa_src_tgt, sa_src_tgt_ema]')
    parser.add_argument('--bkbs_da_loss_xy', default=0.10, type=float, help="loss coefficient for c2f")
    parser.add_argument('--bkbs_da_loss_zd', default=0.01, type=float, help="loss coefficient for c2f")
    parser.add_argument('--srcs_da_loss_xy', default=0.10, type=float, help="loss coefficient for c2f")
    parser.add_argument('--srcs_da_loss_zd', default=0.01, type=float, help="loss coefficient for c2f")
    parser.add_argument('--tgts_da_loss_xy', default=0.10, type=float, help="loss coefficient for c2f")
    parser.add_argument('--tgts_da_loss_zd', default=0.01, type=float, help="loss coefficient for c2f")
    parser.add_argument('--srcs_mi_loss',    default=6e-5, type=float, help="loss coefficient for c2f")
    parser.add_argument('--tgts_mi_loss',    default=6e-5, type=float, help="loss coefficient for c2f")
    parser.add_argument('--bkbs_mi_loss',    default=6e-5, type=float, help="loss coefficient for c2f")
    
    parser.add_argument('--src6_da_loss_xy', default=0.10, type=float, help="loss coefficient for c2f")
    parser.add_argument('--src6_da_loss_zd', default=0.01, type=float, help="loss coefficient for c2f")
    parser.add_argument('--tgt5_da_loss_xy', default=0.10, type=float, help="loss coefficient for c2f")
    parser.add_argument('--tgt5_da_loss_zd', default=0.01, type=float, help="loss coefficient for c2f")
    parser.add_argument('--tgt0_da_loss_xy', default=0.10, type=float, help="loss coefficient for c2f")
    parser.add_argument('--tgt0_da_loss_zd', default=0.01, type=float, help="loss coefficient for c2f")
    parser.add_argument('--src6_mi_loss',    default=6e-5, type=float, help="loss coefficient for c2f")
    parser.add_argument('--tgt5_mi_loss',    default=6e-5, type=float, help="loss coefficient for c2f")
    parser.add_argument('--tgt0_mi_loss',    default=6e-5, type=float, help="loss coefficient for c2f")
    
    parser.add_argument('--margin_src',    default=0.75, type=float, help="hinge loss margin on src side")
    parser.add_argument('--margin_tgt',    default=0.75, type=float, help="hinge loss margin on tgt side")
    parser.add_argument('--with_aqt',    default=0, type=int, help="hinge loss margin on tgt side")
    parser.add_argument('--space_q',    default=0, type=float, help="hinge loss margin on tgt side")
    parser.add_argument('--chann_q',    default=0, type=float, help="hinge loss margin on tgt side")
    parser.add_argument('--insta_q',    default=0, type=float, help="hinge loss margin on tgt side")

    return parser


def build_model_main(args):
    
    if args.modelname.lower() == 'dab_deformable_detr':
        model, criterion, postprocessors = build_dab_deformable_detr(args)
    
    else:
        raise NotImplementedError

    return model, criterion, postprocessors

def build_teacher_model_main(args):

    args2 = copy.deepcopy(args)
    # args2.hidden_dim = 256
    # args2.enc_layers = 6
    model = build_dab_deformable_detr(args2, model_only=True)
    # model = build_DABDETR(args2, model_only=True)
    return model

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def main(args):
    utils.init_distributed_mode(args)
    # torch.autograd.set_detect_anomaly(True)
    
    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['output_dir'] = args.output_dir
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="DAB-DETR")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config.json")
        # print("args:", vars(args))
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    set_random_seed(seed=seed, deterministic=False)
    logger.info("===seed: {}".format(seed) + '\n')
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)
    if args.use_pseudo_labels:
        teacher = build_teacher_model_main(args)
        teacher.to(device)
    else:
        teacher = None
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
        if args.use_pseudo_labels:
            teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
            teacher_without_ddp = teacher.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))


    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    dataset_test = build_dataset(image_set='test', args=args)
    dataset_test2 = build_dataset(image_set='test2', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
        sampler_test2 = DistributedSampler(dataset_test2, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        sampler_test2 = torch.utils.data.SequentialSampler(dataset_test2)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    if args.cdod:
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_strong_weaken_fn, num_workers=args.num_workers)
        
    else:
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    data_loader_test2 = DataLoader(dataset_test2, args.batch_size, sampler=sampler_test2,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    torch_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # add lr warmup, see https://pytorch.org/ignite/generated/ignite.handlers.param_scheduler.create_lr_scheduler_with_warmup.html
    lr_scheduler = WarmUpScheduler(optimizer, torch_lr_scheduler,
                                   len_loader=len(data_loader_train),
                                   warmup_steps=500,
                                   warmup_start_lr=2e-6,
                                   warmup_mode='linear')
    
    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)
        

    
    # load model weights for both teach and student
    print("Start training")
    if args.teacher_weights and args.use_pseudo_labels:
        checkpoint = torch.load(args.teacher_weights, map_location='cpu')
        teacher_without_ddp.load_state_dict(checkpoint['model'])

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

            if args.drop_lr_now:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1

    if not args.resume and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))
        # import ipdb; ipdb.set_trace()


    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'

        print("--------begin to evaluate on source test len(data_loader_val) = {}---------".format(len(data_loader_val)))
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors,
            data_loader_val, base_ds, device, args.output_dir, 
            wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        if args.dataset_file != "cityscapes":
            log_stats = {
                     **{f'test_{k}': v for k, v in test_stats.items()}
                     }
        else:
            print("--------begin to evaluate on target test1 len(data_loader_test) = {}---------".format(len(data_loader_test)))
            test_stats1, coco_evaluator1 = evaluate(
                model, criterion, postprocessors, 
                data_loader_test, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval1.pth")
            
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval3.pth")

            log_stats = {
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        **{f'foggy_test_{k}': v for k, v in test_stats1.items()},
                        # **{f'foggy_test2_{k}': v for k, v in test_stats2.items()}
                        }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return
    # copy student model to teacher
    # teacher = copy.deepcopy(model)
    # teacher.load_state_dict(model.state_dict())
    if args.teacher_weights and args.use_pseudo_labels:
        checkpoint = torch.load(args.teacher_weights, map_location='cpu')
        print("=======>load teacher model weights <==========")
        teacher.module.load_state_dict(checkpoint['model'])
        print("=======>load student model weights <==========")
        model.module.load_state_dict(checkpoint['model'])
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, teacher=teacher,
                wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, postprocessors=postprocessors, logger=(logger if args.save_log else None))
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1== 0:
            # if (epoch + 1) % args.lr_drop == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}_beforedrop.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        print("--------begin to evaluate on source test ---------")
        torch.cuda.empty_cache()
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        torch.cuda.empty_cache()
        if args.dataset_file != "cityscapes":
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        else:
            print("--------begin to evaluate on target test1 len(data_loader_test) = {}---------".format(len(data_loader_test)))
            test_stats1, coco_evaluator1 = evaluate(
                model, criterion, postprocessors, data_loader_test, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        **{f'foggy_test_{k}': v for k, v in test_stats1.items()},
                        # **{f'foggy_test2_{k}': v for k, v in test_stats2.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 1== 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
                if args.dataset_file == "cityscapes":
                    (output_dir / 'eval1').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator1.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 1== 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator1.coco_eval["bbox"].eval,
                                    output_dir / "eval1" / name)
                    (output_dir / 'eval2').mkdir(exist_ok=True)
                    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print("Now time: {}".format(str(datetime.datetime.now())))


@torch.no_grad()
def _copy_main_model(self):
    # initialize all parameters
    if comm.get_world_size() > 1:
        rename_model_dict = {
            key[7:]: value for key, value in self.model.state_dict().items()
        }
        self.model_teacher.load_state_dict(rename_model_dict)
    else:
        self.model_teacher.load_state_dict(self.model.state_dict())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
