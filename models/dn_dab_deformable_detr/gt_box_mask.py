from math import sqrt
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
# from .util import box_ops

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def gaussian2D(radius_x, radius_y, sigma=1, dtype=torch.float32, device='cpu'):
    """Generate 2D gaussian kernel.

    Args:
        radius (int): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
        dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
        device (str): Device of gaussian tensor. Default: 'cpu'.

    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    x = torch.arange(
        -radius_x, radius_x + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius_y, radius_y + 1, dtype=dtype, device=device).view(-1, 1)

    h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()

    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_box_2D(radius, box_wh, sigma=1, dtype=torch.float32, device='cpu'):
    """Generate 2D gaussian kernel.

    Args:
        radius (int): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
        dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
        device (str): Device of gaussian tensor. Default: 'cpu'.

    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    l = torch.arange( -radius, 0, dtype=dtype, device=device)
    m = torch.zeros(box_wh, device=device)
    r = torch.arange(1, radius+1, dtype=dtype, device=device)
    a = torch.cat((l, m, r), dim=0)
    x = a.view(1, -1)
    y = a.view(-1, 1)
    h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def gen_gaussian_target(heatmap, center, radius_x, radius_y, k=1):
    """Generate 2D gaussian heatmap.

    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius (int): Radius of gaussian kernel.
        k (int): Coefficient of gaussian kernel. Default: 1.

    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    # diameter = 2 * radius + 1
    diameter = radius_x + radius_y + 1
    gaussian_kernel = gaussian2D(
        radius_x, radius_y,  sigma=diameter / 6, dtype=heatmap.dtype, device=heatmap.device)

    x, y = center

    height, width = heatmap.shape[:2]

    left, right = min(x, radius_x), min(width - x, radius_x + 1)
    top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[radius_y - top:radius_y + bottom,
                                      radius_x - left:radius_x + right]
    out_heatmap = heatmap
    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right])

    return out_heatmap
    



def _do_paste_mask(boxes, base_box_hws, base_box_rad, img_h, img_w, skip_empty=True):
    """Paste instance masks acoording to boxes.

    This implementation is modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        masks (Tensor): N, 1, H, W
        boxes (Tensor): N, 4
        img_h (int): Height of the image to be pasted.
        img_w (int): Width of the image to be pasted.
        skip_empty (bool): Only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        tuple: (Tensor, tuple). The first item is mask tensor, the second one
            is the slice object.
        If skip_empty == False, the whole image will be pasted. It will
            return a mask of shape (N, img_h, img_w) and an empty tuple.
        If skip_empty == True, only area around the mask will be pasted.
            A mask of shape (N, h', w') and its start and end coordinates
            in the original image will be returned.
    """
    # On GPU, paste all masks together (up to chunk size)
    # by using the entire image to sample the masks
    # Compared to pasting them one by one,
    # this has more operations but is faster on COCO-scale dataset.
    N = len(boxes)
    # masks = torch.ones((N, 1, 14, 14), device=boxes.device)
    masks = gaussian_box_2D(base_box_rad, base_box_hws, device=boxes.device)
    masks = masks[None, None, ...].repeat(N, 1, 1, 1)
    device = masks.device
    if skip_empty:
        x0_int, y0_int = torch.clamp(
            boxes.min(dim=0).values.floor()[:2] - 1,
            min=0).to(dtype=torch.int32)
        x1_int = torch.clamp(
            boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
        y1_int = torch.clamp(
            boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
    else:
        x0_int, y0_int = 0, 0
        x1_int, y1_int = img_w, img_h
    x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

    N = masks.shape[0]

    img_y = torch.arange(
        y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
    img_x = torch.arange(
        x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1
    # img_x, img_y have shapes (N, w), (N, h)
    if torch.isinf(img_x).any():
        inds = torch.where(torch.isinf(img_x))
        img_x[inds] = 0
    if torch.isinf(img_y).any():
        inds = torch.where(torch.isinf(img_y))
        img_y[inds] = 0

    gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
    gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
    grid = torch.stack([gx, gy], dim=3)
    # print("grid.shape = ", grid.shape)
    # print("img_w, img_h = ", img_w, img_h)
    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), grid, align_corners=False)

    if skip_empty:
        return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
    else:
        return img_masks[:, 0], ()



def gt_box_mask_generate(heatmap, boxes, categories, indices, target_sizes):
    """
    generate mask for boxes;
    TODO: will use grid sample next;
    TODO: will use gaussian heat map next;
    heatmap (tensor): batch, h, w, num_cls
    boxes   (tensor): num_obj, 4
    categories(tensor): num_obj,
    indices (tensor): num_obj
    """
    box_resize_factor = 1.0
    base_box_hws = 40
    base_box_rad = int(base_box_hws * (box_resize_factor - 1)) //2
    batch, height, width, num_cls = heatmap.shape
    boxes[:, 2:] = boxes[:, 2:] * box_resize_factor
    boxes = box_cxcywh_to_xyxy(boxes)
    # scale_fct = torch.stack([width, height, width, height], dim=1)


    # scale_fct = torch.tensor([[width, height, width, height]], device=boxes.device)
    boxes = boxes * target_sizes
    masks = _do_paste_mask(boxes, base_box_hws, base_box_rad, img_h=height, img_w=width, skip_empty=False)[0]
    
    heatmap = heatmap.permute(0, 3, 1, 2).contiguous()
    for i, bid in enumerate(indices):
        category = categories[i]
        # x, y, w, h = boxes[i]
        # w, h = w * 1.1, h * 1.1

        # # cx = int(x*width)
        # # cy = int(y*height)
        # # radius_x = int(w*width/2)
        # # radius_y = int(h*height/2)
        # left  = math.floor(width * max(x - w/2, 0))
        # right = math.ceil(width * min(x + w/2, 1))    # need to +1?
        # top   = math.floor(height * max(y - h/2, 0))
        # bot   = math.ceil(height * min(y + h/2, 1))
        # # if i == 1:
        # #     print("left, right, top, bot = ", left, right, top, bot)
        # #     print("x, y, w, h = ", x, y, w, h)

        # # one_gaussian = gen_gaussian_target(heatmap[bid, category], (cx, cy), radius_x, radius_y)
        # heatmap[bid, category] = one_gaussian
        heatmap[bid, category] = heatmap[bid, category] + masks[i]

    heatmap = heatmap.permute(0, 2, 3, 1)
    heatmap[:, : , :, -1] = (1 - (heatmap[:, :, :, :80].sum(dim=-1, keepdim=False))).clamp(max=1.0, min=0.0)
    heatmap = heatmap.clamp(max=1.0, min=0)
    return heatmap



if __name__ == '__main__':
    # test
    boxes = torch.tensor([[0, 0, 2, 2.5], [0, 0, 2, 2.5]]).float()
    img_h, img_w =  8, 8
    feats = _do_paste_mask(boxes, img_h, img_w, skip_empty=False)
    # print(feats[0].shape)
    # print(feats[0][0])
    g5 = gaussian2D(3, 3, sigma=5)
    print(g5.shape)
    print(g5)
    pass