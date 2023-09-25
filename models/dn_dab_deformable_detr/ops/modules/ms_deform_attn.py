# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction
# from ..grl import GradientReversal

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, cross_domain=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.cross_domain = cross_domain
        if cross_domain:
            unit_dim = d_model//3
            self.unit_dim = unit_dim
            self.attention_weights = nn.Linear(unit_dim*2, n_heads * n_levels * n_points)
            self.sampling_offsets = nn.Linear(unit_dim*2, n_heads * n_levels * n_points * 2)
            self.value_proj = nn.Linear(unit_dim*2, unit_dim*3)
            self.output_proj = nn.Linear(unit_dim*3, unit_dim*2)

            # self.attention_weights_zd = nn.Linear(unit_dim*1, n_heads * n_levels * n_points)
            self.attention_weights_zd = nn.Linear(unit_dim*3, n_heads * n_levels * n_points)
            self.value_proj_zd = nn.Linear(d_model, d_model)
            self.output_proj_zd = nn.Linear(d_model, unit_dim)
            self.top_grl = GradientReversal()
            self.bot_grl = GradientReversal()
            self.top_grl2 = GradientReversal()
            self.bot_grl2 = GradientReversal()
        else:

            self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
            self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
            self.value_proj = nn.Linear(d_model, d_model)
            self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
        if self.cross_domain:
            constant_(self.attention_weights_zd.weight.data, 0.)
            constant_(self.attention_weights_zd.bias.data, 0.)
            xavier_uniform_(self.value_proj_zd.weight.data)
            constant_(self.value_proj_zd.bias.data, 0.)
            xavier_uniform_(self.output_proj_zd.weight.data)
            constant_(self.output_proj_zd.bias.data, 0.)


    def cd_forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        cross domain forward
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        
        #step 1: sample points
        # N, Len_q, n_heads, n_levels, n_points, 2
        sample_query = query[..., :self.unit_dim*2]
        sampling_offsets = self.sampling_offsets(sample_query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        # step 2: top branch 
        # step 2.1 top branch val
        # top_val = torch.cat((input_flatten[..., :self.unit_dim*2], self.top_grl(input_flatten[..., self.unit_dim*2:])), dim=-1)
        top_val = input_flatten[..., :self.unit_dim*2]
        # value = self.value_proj(input_flatten[..., :self.unit_dim*2])
        top_val = self.value_proj(top_val)
        if input_padding_mask is not None:
            top_val = top_val.masked_fill(input_padding_mask[..., None], float(0))
        top_val = top_val.view(N, Len_in, self.n_heads, self.unit_dim*3 // self.n_heads)

        # step 2.2 top branch query
        # top_query = torch.cat((query[..., :self.unit_dim*2], self.top_grl2(query[..., self.unit_dim*2:])), dim=-1)
        top_query = query[..., :self.unit_dim*2]
        attention_weights = self.attention_weights(top_query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights_softmaxed = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        # step 2.3 top branch results
        top_output = MSDeformAttnFunction.apply(
            top_val, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights_softmaxed, self.im2col_step)
        
        top_output = self.output_proj(top_output)

        # step 3 bot branch
        # step 3.1 bot branch val 
        bot_val = torch.cat((self.bot_grl(input_flatten[..., :self.unit_dim*2]), input_flatten[..., self.unit_dim*2:]), dim=-1)
        bot_val = self.value_proj_zd(bot_val)
        if input_padding_mask is not None:
            bot_val = bot_val.masked_fill(input_padding_mask[..., None], float(0))
        bot_val = bot_val.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        # step 3.2 bot branch query
        bot_query = torch.cat((self.bot_grl2(query[..., :self.unit_dim*2]), query[..., self.unit_dim*2:]), dim=-1)
        # bot_query = query_with_grl
        # bot_query = query[..., self.unit_dim*2:]
        attention_weights_zd = self.attention_weights_zd(bot_query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights_zd_softmaxed = F.softmax(attention_weights_zd, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        # step 3.3 bot branch results

        bot_output = MSDeformAttnFunction.apply(
            bot_val, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights_zd_softmaxed, self.im2col_step)
        bot_output = self.output_proj_zd(bot_output)

        # step 4 cat two branches
        output = torch.cat((top_output, bot_output), dim=-1)
        return output

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
