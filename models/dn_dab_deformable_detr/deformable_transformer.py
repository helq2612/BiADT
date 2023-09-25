# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DModified from eformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn
from .utils import DomainAttention, GradientReversal, remove_mask_and_warp
from .attention import MultiheadAttention
from .grl import GradientReversal


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 use_dab=False, high_dim_query_update=False, no_sine_embed=False, 
                 with_aqt=0,
                 space_q=0,
                 chann_q=0,
                 insta_q=0):
        super().__init__()
        self.with_aqt = with_aqt

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_dab = use_dab

        #========================================================================
        # for domain alignment
        if with_aqt==1:
            space_align =    True if space_q !=0 else False
            chann_align =    True if chann_q !=0 else False
            instance_align = True if insta_q !=0 else False
        else:
            space_align = False
            chann_align = False
            instance_align = False
        self.space_align = space_align
        self.chann_align = chann_align
        self.instance_align = instance_align
        unit_dim = d_model // 3     # 384//3, or 256//2
        if space_align:
            self.space_query = nn.Parameter(torch.empty(1, 1, unit_dim*2))
        if chann_align:
            # self.chann_query is actually an embedding layer for chann query
            # We keep the name for consistency
            self.chann_query = nn.Linear(unit_dim*2, 1)
            self.grl = GradientReversal()
        if instance_align:
            self.insta_query = nn.Parameter(torch.empty(1, 1, unit_dim*2))
        #========================================================================

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points, space_align=space_align, chann_align=chann_align)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points, instance_align=instance_align)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec, 
                                                            use_dab=use_dab, d_model=d_model, high_dim_query_update=high_dim_query_update, no_sine_embed=no_sine_embed)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            if not self.use_dab:
                self.reference_points = nn.Linear(d_model, 2)

        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            assert not self.use_dab, "use_dab must be True"

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage and not self.use_dab:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, attn_mask=None, domain_flag=None):
        """
        Input:
            - srcs: List([bs, c, h, w])
            - masks: List([bs, h, w])
        """
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)     # bs, \sum{hxw}, c 
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # add space/chann queries, from AQT
        space_query, chann_query, insta_query = None, None, None
        if self.training:
            if self.space_align:
                space_query = self.space_query.expand(src_flatten.shape[0], -1, -1)
            if self.chann_align:
                unit_dim = src_flatten.shape[-1]//3
                xy_src_flatten = src_flatten[..., :unit_dim*2]
                xy_lvl_pos_embed_flatten = lvl_pos_embed_flatten[..., :unit_dim*2]
                
                src_warped, pos_warped = remove_mask_and_warp(
                    xy_src_flatten, xy_lvl_pos_embed_flatten, mask_flatten, level_start_index, spatial_shapes
                )
                
                chann_query = self.chann_query(self.grl(src_warped+pos_warped)).flatten(0, 1).transpose(1, 2)
        # encoder
        memory, memorys, space_query, chann_query = self.encoder(src_flatten, space_query, chann_query, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        da_output = {}
        if self.training:
            if self.space_align:
                da_output['space_query'] = torch.cat(space_query, dim=1)
            if self.chann_align:
                da_output['chann_query'] = torch.cat(chann_query, dim=1)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.use_dab:
            reference_points = query_embed[..., self.d_model:].sigmoid() 
            tgt = query_embed[..., :self.d_model]
            # tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            init_reference_out = reference_points
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid() 
                # bs, num_quires, 2
            init_reference_out = reference_points

        if self.training and self.instance_align:
            insta_query = self.insta_query.expand(tgt.shape[0], -1, -1)

        # decoder
        hs, inter_references, insta_query = self.decoder(tgt, insta_query, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, 
                                            query_pos=query_embed if not self.use_dab else None, 
                                            src_padding_mask=mask_flatten, attn_mask=attn_mask,
                                            domain_flag=domain_flag)

        if self.training and self.instance_align:
            da_output['insta_query'] = insta_query

        inter_references_out = inter_references

        # get stacked memory
        lvl_srcs = []
        lvl_nums = [h_*w_ for h_, w_ in spatial_shapes]
        
        lvl_srcs = [src_flatten] + memorys
        return hs, init_reference_out, inter_references_out, None, None, lvl_srcs, da_output


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, space_align=False, chann_align=False):
        super().__init__()
        # space/chann align
        
        self.space_align = space_align
        self.chann_align = chann_align
        unit_dim = d_model//3
        if space_align:
            self.space_attn = DomainAttention(unit_dim*2, n_heads, dropout)
            
        if chann_align:
            self.chann_attn = DomainAttention(unit_dim*2, n_heads, dropout)

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, cross_domain=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout1z = nn.Dropout(dropout)
        self.norm1_xy = nn.LayerNorm(unit_dim*2)
        self.norm1_zd = nn.LayerNorm(unit_dim*1)

        self.ffn_xy = FFN(unit_dim*2, unit_dim*2,activation, dropout)
        self.ffn_zd = FFN(unit_dim*1, unit_dim*1,activation, dropout)

        self.grl = GradientReversal()
        self.grl0 = GradientReversal()

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def with_pos_domain_embed(tensor, pos):
        if pos is None:
            return tensor
        unit_dim = tensor.shape[-1]//3
        res = torch.cat((tensor[..., :unit_dim*2]+pos[..., :unit_dim*2], tensor[..., unit_dim*2:]*pos[..., unit_dim*2:]), dim=-1)
        return res

    @staticmethod
    def with_grl_pos_embed(tensor, pos, grl0):
        if pos is None:
            return tensor
        unit_dim = tensor.shape[-1]//3
        res = torch.cat((grl0(tensor[..., :unit_dim*2])+pos[..., :unit_dim*2], tensor[..., unit_dim*2:]+pos[..., unit_dim*2:]), dim=-1)
        return res

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, space_query, chann_query, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):

        # self attention
        unit_dim = src.shape[-1]//3
        src2 = self.self_attn.cd_forward(self.with_pos_embed(src, pos), 
                                        #  self.with_grl_pos_embed(src, pos, self.grl0),
                                         reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src_xy = src[..., :unit_dim*2] + self.dropout1(src2)[..., :unit_dim*2]
        src_zd = src[..., unit_dim*2:] + self.dropout1z(src2)[..., unit_dim*2:]
        src_xy = self.norm1_xy(src_xy)
        src_zd = self.norm1_zd(src_zd)
        
        src = torch.cat((src_xy, src_zd), dim=-1)
        
        
        if self.training:
            
            content_src = src[..., :2*unit_dim]
            xy_pos = pos[..., :2*unit_dim]
            if self.space_align:
                space_query = self.space_attn(space_query, content_src, xy_pos, padding_mask)

            if self.chann_align:
                src_warped, pos_warped = remove_mask_and_warp(content_src, xy_pos, padding_mask, level_start_index, spatial_shapes)
                chann_query = self.chann_attn(
                    chann_query, # bsz * num_feature_levels, 1, H*W
                    src_warped.flatten(0, 1).transpose(1, 2), # bsz * num_feature_levels, C, H*W
                    pos_warped.flatten(0, 1).transpose(1, 2)
                )

        # ffn
        src_xy = self.ffn_xy(src[..., :unit_dim*2])
        src_zd = self.ffn_zd(src[..., unit_dim*2:])
        src = torch.cat((src_xy, src_zd), dim=-1)
        # src = self.forward_ffn(src)

        return src, space_query, chann_query


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, space_query, chann_query, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - padding_mask: [bs, sum(hi*wi)]
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_lebel, 2]
        """
        output = src
        outputs = []
        space_querys = []
        chann_querys = []
        # bs, sum(hi*wi), 256
        # import ipdb; ipdb.set_trace()
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output, space_query, chann_query  = layer(output, 
                                                        space_query, 
                                                        chann_query, 
                                                        pos, 
                                                        reference_points, 
                                                        spatial_shapes, 
                                                        level_start_index, 
                                                        padding_mask)
            outputs.append(output)
            space_querys.append(space_query)
            chann_querys.append(chann_query)

        return output, outputs, space_querys, chann_querys


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 instance_align = False):
        super().__init__()
        
        self.instance_align = instance_align
        unit_dim = d_model//3
        if instance_align:
            self.instance_attn = DomainAttention(unit_dim*2, n_heads, dropout)

        # cross attention
        # =========================Cross-atten=====================================
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, cross_domain=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout1z = nn.Dropout(dropout)
        # self.norm1 = nn.LayerNorm(d_model)
        self.norm1_xy = nn.LayerNorm(unit_dim*2)
        self.norm1_zd = nn.LayerNorm(unit_dim*1)

        # self attention
        # =========================Self-atten=====================================
        self.sa_qcontent_xyd_proj   = nn.Linear(unit_dim*2, unit_dim*2)
        self.sa_qpos_xyd_proj       = nn.Linear(unit_dim*2, unit_dim*2)
        self.sa_kcontent_xyd_proj   = nn.Linear(unit_dim*2, unit_dim*2)
        self.sa_kpos_xyd_proj       = nn.Linear(unit_dim*2, unit_dim*2)
        
        self.sa_v_proj_xyd = nn.Linear(unit_dim*2, unit_dim*3)
        self.sa_v_proj_xyz = nn.Linear(d_model, d_model)

        self.self_attn = MultiheadAttention(unit_dim*2, n_heads, dropout=dropout, vdim=unit_dim*2, atten_mode="crossDomainSelfAtt")
        self.src_xy_linear = nn.Linear(unit_dim*3, unit_dim*2)
        self.src_zd_linear = nn.Linear(d_model, unit_dim)
        
        self.dropout2_xy = nn.Dropout(dropout)
        self.dropout2_zd = nn.Dropout(dropout)
        self.norm2_xy = nn.LayerNorm(unit_dim*2)
        self.norm2_zd = nn.LayerNorm(unit_dim*1)
        
        # ffn
        # =========================FFN=====================================
        self.ffn_xy = FFN(unit_dim*2, unit_dim*2,activation, dropout)
        self.ffn_zd = FFN(unit_dim*1, unit_dim*1,activation, dropout)
        self.grl11 = GradientReversal()
        self.grl12 = GradientReversal()
        self.grl13 = GradientReversal()
        
        self.grl21 = GradientReversal()
        self.grl22 = GradientReversal()
        self.grl23 = GradientReversal()
        # self.grl31 = GradientReversal()
        self.grl0 = GradientReversal()
        self._reset_parameters()
    
    def _reset_parameters(self):
        xavier_uniform_(self.sa_qcontent_xyd_proj.weight)
        constant_(self.sa_qcontent_xyd_proj.bias, 0.)

        xavier_uniform_(self.sa_qpos_xyd_proj.weight)
        constant_(self.sa_qpos_xyd_proj.bias, 0.)

        xavier_uniform_(self.sa_kcontent_xyd_proj.weight)
        constant_(self.sa_kcontent_xyd_proj.bias, 0.)

        xavier_uniform_(self.sa_kpos_xyd_proj.weight)
        constant_(self.sa_kpos_xyd_proj.bias, 0.)

        xavier_uniform_(self.sa_v_proj_xyd.weight)
        constant_(self.sa_v_proj_xyd.bias, 0.)

        xavier_uniform_(self.sa_v_proj_xyz.weight)
        constant_(self.sa_v_proj_xyz.bias, 0.)

        xavier_uniform_(self.src_xy_linear.weight)
        constant_(self.src_xy_linear.bias, 0.)

        xavier_uniform_(self.src_zd_linear.weight)
        constant_(self.src_zd_linear.bias, 0.)
        
        
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    @staticmethod
    def with_pos_domain_embed(tensor, pos):
        if pos is None:
            return tensor
        unit_dim = tensor.shape[-1]//3
        res = torch.cat((tensor[..., :unit_dim*2]+pos[..., :unit_dim*2], tensor[..., unit_dim*2:]*pos[..., unit_dim*2:]), dim=-1)
        return res

    @staticmethod
    def with_grl_pos_embed(tensor, pos, grl0):
        if pos is None:
            return tensor
        unit_dim = tensor.shape[-1]//3
        res = torch.cat((grl0(tensor[..., :unit_dim*2])+pos[..., :unit_dim*2], tensor[..., unit_dim*2:]+pos[..., unit_dim*2:]), dim=-1)
        return res

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, insta_query, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None, self_attn_mask=None):
        # self attention
        # ========== Begin of Self-Attention =============
        unit_dim = tgt.shape[-1]//3
        # query projection for xy
        
        tgt_v_top  = tgt[..., :unit_dim*2].transpose(0, 1)
        
        tgt_top  = tgt[..., :unit_dim*2].transpose(0, 1)
        qpos_top = query_pos[..., :unit_dim*2].transpose(0, 1)

        q_xyd_content = self.sa_qcontent_xyd_proj(tgt_top)
        q_xyd_pos     = self.sa_qpos_xyd_proj(qpos_top)

        # key projection for xy
        k_xyd_content = self.sa_kcontent_xyd_proj(tgt_top)
        k_xyd_pos     = self.sa_kpos_xyd_proj(qpos_top)

        # query projection for xyz
        tgt_v_bot  = torch.cat((self.grl21(tgt[..., :unit_dim*2]),       tgt[..., unit_dim*2:]),       dim=-1).transpose(0, 1)
        tgt_bot    = torch.cat((self.grl22(tgt[..., :unit_dim*2]),       tgt[..., unit_dim*2:]),       dim=-1).transpose(0, 1)
        qpos_bot   = torch.cat((self.grl23(query_pos[..., :unit_dim*2]), query_pos[..., unit_dim*2:]), dim=-1).transpose(0, 1)

        # value projection
        v_xyd = self.sa_v_proj_xyd(tgt_v_top)
        v_xyz = self.sa_v_proj_xyz(tgt_v_bot)

        num_queries, bs, _ = q_xyd_content.shape
        hw, _, _ = k_xyd_content.shape

        q = q_xyd_content + q_xyd_pos
        k = k_xyd_content + k_xyd_pos

        q_xyz = None
        k_xyz = None
        
        tgt2, tgt3 = self.self_attn(q, k, q_xyz=q_xyz, k_xyz=k_xyz, value=v_xyd, value_xyz=v_xyz, attn_mask=self_attn_mask,
                            key_padding_mask=None)[0:2]
        
        tgt2 = self.src_xy_linear(tgt2)
        tgt3 = self.src_zd_linear(tgt3)


        tgt_xy = tgt[..., :unit_dim*2] + self.dropout2_xy(tgt2.transpose(0, 1))
        tgt_zd = tgt[..., unit_dim*2:] + self.dropout2_zd(tgt3.transpose(0, 1))
        tgt_xy = self.norm2_xy(tgt_xy)
        tgt_zd = self.norm2_zd(tgt_zd)
        tgt = torch.cat((tgt_xy, tgt_zd), dim=-1)
        # ========== End of Self-Attention =============
        
        # cross attention
        # ========== Begin of Cross-Attention =============
        tgt2 = self.cross_attn.cd_forward(self.with_pos_embed(tgt, query_pos),
                                        #   self.with_grl_pos_embed(tgt, query_pos, self.grl0), 
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt_xy = tgt[..., :unit_dim*2] + self.dropout1(tgt2)[..., :unit_dim*2]
        tgt_zd = tgt[..., unit_dim*2:] + self.dropout1z(tgt2)[..., unit_dim*2:]

        tgt_xy = self.norm1_xy(tgt_xy)
        tgt_zd = self.norm1_zd(tgt_zd)
        tgt = torch.cat((tgt_xy, tgt_zd), dim=-1)
        # ========== End of Cross-Attention =============

        unit_dim = tgt.shape[-1] // 3
        if self.training and self.instance_align:
            unit_dim = tgt.shape[-1] // 3
            xy_tgt = tgt[..., :2*unit_dim]
            xy_query_pos = query_pos[..., :2*unit_dim]
            insta_query = self.instance_attn(insta_query, xy_tgt, xy_query_pos)

        # ffn
        tgt_xy = self.ffn_xy(tgt[..., :unit_dim*2])
        tgt_zd = self.ffn_zd(tgt[..., unit_dim*2:])
        tgt = torch.cat((tgt_xy, tgt_zd), dim=-1)
        # tgt = self.forward_ffn(tgt)

        return tgt, insta_query


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, use_dab=False, d_model=256, high_dim_query_update=False, no_sine_embed=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.use_dab = use_dab
        self.d_model = d_model
        self.no_sine_embed = no_sine_embed
        if use_dab:
            self.query_scale = MLP(d_model*2//3, d_model*2//3, d_model*2//3, 2)
            if self.no_sine_embed:
                self.ref_point_head = MLP(4, d_model, d_model, 3)
            else:
                self.ref_point_head = MLP(2 * d_model * 2 // 3, d_model*2//3, d_model*2//3, 2)
        self.high_dim_query_update = high_dim_query_update
        if high_dim_query_update:
            self.high_dim_query_proj = MLP(d_model, d_model, d_model, 2)


    def forward(self, tgt, insta_query, reference_points, src, src_spatial_shapes,       
                src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, attn_mask=None, domain_flag=None):
        output = tgt
        if self.use_dab:
            assert query_pos is None
        # bs = src.shape[0]
        hidden_dim = tgt.shape[-1]
        unit_dim = hidden_dim//3        

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            # import ipdb; ipdb.set_trace()
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None] # bs, nq, 4, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            if self.use_dab:
                # import ipdb; ipdb.set_trace()
                if self.no_sine_embed:
                    raw_query_pos = self.ref_point_head(reference_points_input)
                else:
                    # query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # bs, nq, 256*2 
                    query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :], domain_flag=domain_flag) # bs, nq, 256*2 
                    unit_d_model = query_sine_embed.shape[-1]*2//5
                    raw_query_pos = self.ref_point_head(query_sine_embed[..., :2*unit_d_model]) # bs, nq, 256
                pos_scale = self.query_scale(output[..., :2*unit_dim]) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
                query_pos = torch.cat((query_pos, query_sine_embed[..., 2*unit_d_model:]), dim=-1)
            if self.high_dim_query_update and lid != 0:
                query_pos = query_pos + self.high_dim_query_proj(output)                 


            output, insta_query = layer(output, insta_query, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index,
                           src_padding_mask, self_attn_mask=attn_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output[..., :unit_dim*2])
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), insta_query

        return output, reference_points, insta_query


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        with_aqt=args.with_aqt,
        space_q=args.space_q,
        chann_q=args.chann_q,
        insta_q=args.insta_q,
        use_dab=True)

class FFN(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, d_model, d_ffn, activation, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.linear1.weight)
        constant_(self.linear1.bias, 0.)

        xavier_uniform_(self.linear2.weight)
        constant_(self.linear2.bias, 0.)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def gen_sineembed_for_position(pos_tensor, d_model=128, domain_flag=None):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(d_model, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / d_model)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    

    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        if domain_flag is None:
            pos = torch.cat((pos_y, pos_x), dim=2)
        else:
            pos_z = torch.ones_like(pos_x) * domain_flag[None, :, None] #/ dim_t
            pos = torch.cat((pos_y, pos_x, pos_z), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        if domain_flag is None:
            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:

            pos_z = torch.ones_like(pos_x) * domain_flag[:, None, None] #/ dim_t
            pos = torch.cat((pos_y, pos_x, pos_w, pos_h, pos_z), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

