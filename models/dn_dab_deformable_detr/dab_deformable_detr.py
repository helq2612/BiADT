# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DModified from deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import os
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy

from .dn_components import prepare_for_dn, dn_post_process, compute_dn_loss
from .grl import GradientReversal
from torchmetrics.classification import BinaryHingeLoss


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DABDeformableDETR(nn.Module):
    """ This is the DAB-Deformable-DETR for object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=True, two_stage=False,
                 use_dab=True, 
                 num_patterns=0,
                 random_refpoints_xy=False,
                 with_aqt=0,
                 space_q=0,
                 chann_q=0,
                 insta_q=0
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            use_dab: using dynamic anchor boxes formulation
            num_patterns: number of pattern embeddings
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        unit_dim = hidden_dim//3        
        self.unit_dim = unit_dim   

        self.num_classes = num_classes
        self.class_embed = nn.Linear(unit_dim*2, num_classes)
        self.bbox_embed = MLP(unit_dim*2, unit_dim*2, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.use_dab = use_dab
        self.num_patterns = num_patterns
        self.random_refpoints_xy = random_refpoints_xy
        # dn label enc
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)  # # for indicator
        if not two_stage:
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
            else:
                self.tgt_embed = nn.Embedding(num_queries, hidden_dim-1)  # for indicator
                self.refpoint_embed = nn.Embedding(num_queries, 4)
                if random_refpoints_xy:
                    # import ipdb; ipdb.set_trace()
                    self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
                    self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
                    self.refpoint_embed.weight.data[:, :2].requires_grad = False
                

        if self.num_patterns > 0:
            self.patterns_embed = nn.Embedding(self.num_patterns, hidden_dim)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_dummy_layer = 0
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers + num_dummy_layer
        
        with_img_cls = False
        if with_img_cls:
            self.class_img_embed = _get_clones(self.class_embed, 1)[0]


        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        
        # obj domain head
        self.grl = GradientReversal()
            
        encoder_output_align = True
        self.encoder_output_align = encoder_output_align
        self.encoder_layer_diff_align = True        # for each layer, if the align are different
        self.decoder_layer_diff_align = True
        self.biadt_encoder_mi_diff_align = True
        self.biadt_decoder_mi_diff_align = True
        if encoder_output_align:
            self.biadt_srcs_xy_domain = MLP(2*unit_dim, 2*unit_dim, 1, 3)
            for layer in self.biadt_srcs_xy_domain.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
            # self.biadt_srcs_xy_domain = nn.Linear(2*unit_dim, 1*unit_dim)
            self.biadt_srcs_zd_domain = MLP(1*unit_dim, 1*unit_dim, 1, 3)
            for layer in self.biadt_srcs_zd_domain.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)

            self.biadt_srcs_xy_domain_bg = MLP(2*unit_dim, 2*unit_dim, 1, 3)
            for layer in self.biadt_srcs_xy_domain_bg.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
            self.biadt_biadt_srcs_zd_domain_bg = MLP(1*unit_dim, 1*unit_dim, 1, 3)
            for layer in self.biadt_biadt_srcs_zd_domain_bg.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
            
            if self.encoder_layer_diff_align:
                num_encoder_layers = 6
                self.biadt_srcs_xy_domain = _get_clones(self.biadt_srcs_xy_domain, num_encoder_layers)
                self.biadt_srcs_zd_domain = _get_clones(self.biadt_srcs_zd_domain, num_encoder_layers)
                
        decoder_output_align = True
        
        self.decoder_output_align = decoder_output_align
        if decoder_output_align:
            self.biadt_tgts_xy_domain = MLP(2*unit_dim, 1*unit_dim, 1, 3)
            for layer in self.biadt_tgts_xy_domain.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
            self.biadt_tgts_zd_domain = MLP(1*unit_dim, 1*unit_dim, 1, 3)
            for layer in self.biadt_tgts_zd_domain.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
            if self.decoder_layer_diff_align:
                num_decoder_layers = 6
                self.biadt_tgts_xy_domain = _get_clones(self.biadt_tgts_xy_domain, num_decoder_layers)
                self.biadt_tgts_zd_domain = _get_clones(self.biadt_tgts_zd_domain, num_decoder_layers)

        self.with_aqt = with_aqt
        if with_aqt == 1:
            space_align = True if space_q != 0 else False
            chann_align = True if chann_q != 0 else False
            insta_align = True if insta_q != 0 else False
        else:
            space_align = False
            chann_align = False
            insta_align = False
        self.space_align = space_align
        self.chann_align = chann_align
        self.insta_align = insta_align
        
        if space_align:
            self.space_D = MLP(2*unit_dim, 2*unit_dim, 1, 3)
            for layer in self.space_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
        
        
        if chann_align:
            self.chann_D = MLP(2*unit_dim, 2*unit_dim, 1, 3)
            for layer in self.chann_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)
        
        if insta_align:
            self.insta_D = MLP(2*unit_dim, 2*unit_dim, 1, 3)
            for layer in self.insta_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)

        use_mi = True
        self.use_mi = use_mi
        if use_mi:
            self.biadt_encoder_mi = Mutual_loss()
            self.biadt_decoder_mi = Mutual_loss()
            # self.bkbs_mi = Mutual_loss()
            # self.decoder0_mi = Mutual_loss()
            if self.biadt_encoder_mi_diff_align:
                num_encoder_layers = 6
                self.biadt_encoder_mi = _get_clones(self.biadt_encoder_mi, num_encoder_layers+1)
            if self.biadt_decoder_mi_diff_align:
                num_decoder_layers = 6
                self.biadt_decoder_mi = _get_clones(self.biadt_decoder_mi, num_decoder_layers)
        # self.domain_scale = MLP(1*unit_dim, 1*unit_dim, 1*unit_dim, 2)
        # for layer in self.domain_scale.layers:
        #     nn.init.xavier_uniform_(layer.weight, gain=1)
        #     nn.init.constant_(layer.bias, 0)
        # self.domain_scale_tgts = MLP(1*unit_dim, 1*unit_dim, 1*unit_dim, 2)
        # self.domain_scale_aqt = MLP(2*unit_dim, 2*unit_dim, 2*unit_dim, 2)
        # for layer in self.domain_scale_aqt.layers:
        #     nn.init.xavier_uniform_(layer.weight, gain=1)
        #     nn.init.constant_(layer.bias, 0)

    def forward(self, samples: NestedTensor, dn_args=None, domain_flag=0, use_pseudo_label=False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples, domain_flag=domain_flag)
        # print("----features.shape = {}, pos.shape = {}".format([ele.shape for ele in features], [ele.shape for ele in pos]))
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        # import ipdb; ipdb.set_trace()

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask), domain_flag=domain_flag).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.two_stage:
            assert NotImplementedError
        elif self.use_dab:
            if self.num_patterns == 0:
                tgt_all_embed = tgt_embed = self.tgt_embed.weight           # nq, 256
                refanchor = self.refpoint_embed.weight      # nq, 4
                # query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
            else:
                # multi patterns is not used in this version
                assert NotImplementedError
        else:
            assert NotImplementedError

        # prepare for dn: for BiADT, actually we only use dab-deformable-detr, and we leave dn version for future 
        input_query_label, input_query_bbox, attn_mask, mask_dict = \
            prepare_for_dn(dn_args, tgt_all_embed, refanchor, src.size(0), self.training, self.num_queries, self.num_classes,
                           self.hidden_dim, self.label_enc)
        query_embeds = torch.cat((input_query_label, input_query_bbox), dim=2)

        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, lvl_srcs, da_transformer_output = \
            self.transformer(srcs, masks, pos, query_embeds, attn_mask, domain_flag=domain_flag)

        outputs_classes = []
        outputs_coords = []
        outputs_imgCls = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl][..., :self.unit_dim*2])
            tmp = self.bbox_embed[lvl](hs[lvl][..., :self.unit_dim*2])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
           
        out = {}

        # ================================================================================================
        # domain output
        unit_dim = self.unit_dim
        da_output = {}
        if self.training:
            # only take the source image to get the det loss
            B = outputs_class.shape[1]
            assert B%2 == 0
            outputs_class = outputs_class[:, :B//2]
            outputs_coord = outputs_coord[:, :B//2]
            
            if self.encoder_output_align:
                srcs_xy = [ele[..., :2*unit_dim] for ele in lvl_srcs]
                srcs_zd = [ele[..., 2*unit_dim:] for ele in lvl_srcs]
                
                for lvl in range(hs.shape[0]+1):
                    one_src_zd = srcs_zd[lvl]
                    one_src_xy = srcs_xy[lvl]
                    
                    if lvl == 0:
                        da_output[f'srcs_xy_{lvl}'] = self.biadt_srcs_xy_domain_bg(self.grl(one_src_xy))
                        da_output[f'srcs_zd_{lvl}'] = self.biadt_biadt_srcs_zd_domain_bg(one_src_zd)
                    else:
                        if self.encoder_layer_diff_align:
                            da_output[f'srcs_xy_{lvl}'] = self.biadt_srcs_xy_domain[lvl-1](self.grl(one_src_xy))
                            da_output[f'srcs_zd_{lvl}'] = self.biadt_srcs_zd_domain[lvl-1](one_src_zd)
                        else:
                            da_output[f'srcs_xy_{lvl}'] = self.biadt_srcs_xy_domain(self.grl(one_src_xy))
                            da_output[f'srcs_zd_{lvl}'] = self.biadt_srcs_zd_domain(one_src_zd)
                        
        
                    if self.use_mi:
                        if self.biadt_encoder_mi_diff_align:
                            da_output[f'srcs_mi_{lvl}'] = self.biadt_encoder_mi[lvl](one_src_xy, one_src_zd)
                        else:
                            da_output[f'srcs_mi_{lvl}'] = self.biadt_encoder_mi(one_src_xy, one_src_zd)
                

            if self.decoder_output_align:
                tgts_xy = [ele[..., :2*unit_dim] for ele in hs]
                tgts_zd = [ele[..., 2*unit_dim:] for ele in hs]
                
                for lvl in range(hs.shape[0]):
                    one_tgt_xy = tgts_xy[lvl]
                    one_tgt_zd = tgts_zd[lvl]
                    
                    if self.decoder_layer_diff_align:
                        da_output[f'tgts_xy_{lvl}'] = self.biadt_tgts_xy_domain[lvl](self.grl(one_tgt_xy))
                        da_output[f'tgts_zd_{lvl}'] = self.biadt_tgts_zd_domain[lvl](one_tgt_zd)
                        
                    else:
                        da_output[f'tgts_xy_{lvl}'] = self.biadt_tgts_xy_domain(self.grl(one_tgt_xy))
                        da_output[f'tgts_zd_{lvl}'] = self.biadt_tgts_zd_domain(one_tgt_zd)
                    
                    if self.use_mi:
                        
                        if self.biadt_decoder_mi_diff_align:
                            da_output[f'tgts_mi_{lvl}'] = self.biadt_decoder_mi[lvl](one_tgt_xy, one_tgt_zd)
                        else:
                            da_output[f'tgts_mi_{lvl}'] = self.biadt_decoder_mi(one_tgt_xy, one_tgt_zd)
                        
            if self.space_align:
                da_output['space_query'] = self.space_D(da_transformer_output['space_query'])
            if self.chann_align:
                da_output['chann_query'] = self.chann_D(da_transformer_output['chann_query'])
            if self.insta_align:
                da_output['insta_query'] = self.insta_D(da_transformer_output['insta_query'])
            
        # ================================================================================================
        
        outputs_class, outputs_coord = dn_post_process(outputs_class, outputs_coord, mask_dict)
            
        out['pred_logits'] = outputs_class[-1]
        out['pred_boxes']  = outputs_coord[-1]
        out['da_output'] = da_output
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        
        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
            if os.environ.get('IPDB_SHILONG_DEBUG') == 'INFO':
                import ipdb; ipdb.set_trace()        
        out["src_key_padding_mask"] = masks
        out["domain_flag"] = domain_flag
        
        return out, mask_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class Mutual_loss(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, unit_dim=128):
        super().__init__()
        self.fc1_x = nn.Linear(unit_dim*2, unit_dim*1)  # project xy
        self.fc1_y = nn.Linear(unit_dim*1, unit_dim*1)  # project zd
        self.fc2   = nn.Linear(unit_dim*1, unit_dim*1)
        # self.fc2_t = nn.Linear(unit_dim*1, unit_dim)
        self.fc3 = nn.Linear(unit_dim*1, 1)
        self._reset_parameters()

    def mine(self, x, y):
        h1 = F.leaky_relu(self.fc1_x(x)+self.fc1_y(y))
        h2 = F.leaky_relu(self.fc2(h1))
        h4 = self.fc3(h2)
        return h4
    
    def forward(self, xy, zd):
        """
        xy: bs, n, 256; domain invariant
        zd: bs, n, 128; domain specific
        """
        # xy = xy0.detach()
        joint = self.mine(xy, zd)
        # shuffled_xy = torch.index_select(xy, 1, torch.randperm(xy.shape[1]).cuda())
        # shuffled_zd
        # marginal = self.mine(shuffled_xy, zd)
        shuffled_zd = torch.index_select(zd, 1, torch.randperm(zd.shape[1]).cuda())
        marginal2 = self.mine(xy, shuffled_zd)
        # loss = torch.mean(joint, dim=1, keepdim=True) * 1  - torch.log(torch.mean(torch.exp(marginal), dim=1, keepdim=True)) #- torch.log(torch.mean(torch.exp(marginal2), dim=1, keepdim=True))
        loss = torch.mean(joint, dim=1, keepdim=True) * 1  - torch.log(torch.mean(torch.exp(marginal2), dim=1, keepdim=True))
        return loss.abs().mean() #* 0.5
        # return loss.mean() * (-1)

    def _reset_parameters(self):
        # xavier_uniform_(self.linear1.weight.data)

        nn.init.xavier_uniform_(self.fc1_x.weight, gain=1)
        nn.init.xavier_uniform_(self.fc1_y.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight,   gain=1)
        nn.init.xavier_uniform_(self.fc3.weight,   gain=1)

        # nn.init.normal_(self.fc1_x.weight,std=0.02)
        # nn.init.normal_(self.fc1_y.weight,std=0.02)
        # nn.init.normal_(self.fc2.weight,std=0.02)
        # nn.init.normal_(self.fc3.weight,std=0.02)

        nn.init.constant_(self.fc1_x.bias.data, 0.)
        nn.init.constant_(self.fc1_y.bias.data, 0.)
        nn.init.constant_(self.fc2.bias.data, 0.)
        nn.init.constant_(self.fc3.bias.data, 0.)


# def biHingeLoss(input, target, margin=0.75):
#     """
#     Bidirectional Hinge loss

#     """






class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, da_gamma=2, margin_src=0.55, margin_tgt=0.55):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.da_gamma = da_gamma
        self.hingeloss_tgt = nn.HingeEmbeddingLoss(margin=0.75, reduction='none')
        self.hingeloss_src = nn.HingeEmbeddingLoss(margin=0.75, reduction='none')
        self.margin_src = margin_src
        self.margin_tgt = margin_tgt

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        # if len(indices) == 0:
        #     losses = {
        #         "loss_ce": (outputs['pred_logits'] * torch.zeros_like(outputs['pred_logits'])).sum()
        #     }
        #     if log:
        #         # TODO this should probably be a separate loss, not hacked in this one here
        #         losses['class_error'] = (outputs['pred_logits'] * torch.zeros_like(outputs['pred_logits'])).sum()
        #     return losses
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        # if len(indices) == 0:
        #     losses = {
        #         "cardinality_error": (outputs['pred_logits'] * torch.zeros_like(outputs['pred_logits'])).sum()
        #     }
        #     return losses
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        # if len(indices) == 0:
        #     losses = {
        #         "loss_bbox": (outputs['pred_boxes'] * torch.zeros_like(outputs['pred_boxes'])).sum(),
        #         "loss_giou": (outputs['pred_boxes'] * torch.zeros_like(outputs['pred_boxes'])).sum(),
        #     }
        #     return losses
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_da(self, outputs, da_name, da_all_outputs, domain_flag, use_focal=False,  use_hinge = False):
        B = outputs.shape[0]
        assert B % 2 == 0
        
        targets = torch.empty_like(outputs)         
        targets[:B//2] = 0                          # 0 for source
        targets[B//2:] = 1                          # 1 for target
        
        if not use_hinge:
            loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
            if use_focal:
                prob = outputs.sigmoid()
                p_t = prob * targets + (1 - prob) * (1 - targets)
                loss = loss * ((1 - p_t) ** self.da_gamma)
        else:
                
            zd_name = da_name.replace("xy", "zd")
            zd_output = da_all_outputs[zd_name].detach()
            zd_output_sigmoid = zd_output.sigmoid() # make sure it is in [0, 1]; 0 for source, 1 for target
            
            margin_tgt=zd_output_sigmoid.clamp(min=self.margin_tgt)

            outputs_pred =  outputs.sigmoid()
            pos_delta = outputs_pred                # loss will decrease outputs_pred to margin_src (e.g. 0.4), for source img

            neg_delta = margin_tgt - outputs_pred   # loss will increase outputs_pred to margin_tgt (e.g. 0.7), for target img
            
            pos = (pos_delta.ge(0).detach() * pos_delta) * (targets.eq(0))                  # for source img, max(0, pos_delta)
            neg = (neg_delta.ge(0).detach() * neg_delta) * (targets.eq(1))                  # for target img, max(0, neg_delta)
            loss = pos + neg
        return loss.mean()


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, mask_dict=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        self.src_key_padding_mask = outputs['src_key_padding_mask']
        del outputs['src_key_padding_mask']
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        
        domain_flag = outputs["domain_flag"]
        self.domain_flag = domain_flag
        del outputs['domain_flag']
        losses = {}
        
        indices = self.matcher(outputs_without_aux, targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            if os.environ.get('IPDB_SHILONG_DEBUG') == 'INFO':
                import ipdb; ipdb.set_trace()
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)
        # dn loss computation, used for future work that using dn models
        aux_num = 0
        if 'aux_outputs' in outputs:
            aux_num = len(outputs['aux_outputs'])
        dn_losses = compute_dn_loss(mask_dict, self.training, aux_num, self.focal_alpha)
        losses.update(dn_losses)

        if 'da_output' in outputs:
            for k, v in outputs['da_output'].items():
                use_focal = False
                

                use_hinge = True
                if "srcs_xy_0" in k:
                    use_hinge = False
                if "srcs_zd_0" in k:
                    use_hinge = False
                if "zd" in k:
                    use_hinge = False
                # AQT alignmetns do not use hinge loss
                if 'space_query' in k:
                    use_hinge = False
                if 'chann_query' in k:
                    use_hinge = False
                if 'insta_query' in k:
                    use_hinge = False

                if 're' in k:
                    continue
                if 'mi' in k:
                    losses[f'loss_{k}'] = v #* (-1)
                    continue
                    
                losses[f'loss_{k}'] =  self.loss_da(v, k, outputs['da_output'], domain_flag, use_focal=use_focal,  use_hinge = use_hinge)
        return losses



class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_dab_deformable_detr(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "voc2clipart":
        num_classes = 25
    if args.dataset_file == "cityscapes":
        num_classes = 20
    if args.dataset_file == "sim10k2cityscapes":
        num_classes = 5
    if args.dataset_file == "bdd_daytime":
        num_classes = 20
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DABDeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        two_stage=args.two_stage,
        use_dab=True,
        num_patterns=args.num_patterns,
        random_refpoints_xy=args.random_refpoints_xy,
        with_aqt=args.with_aqt,
        space_q=args.space_q,
        chann_q=args.chann_q,
        insta_q=args.insta_q,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    # dn loss
    if args.use_dn:
        weight_dict['tgt_loss_ce'] = args.cls_loss_coef
        weight_dict['tgt_loss_bbox'] = args.bbox_loss_coef
        weight_dict['tgt_loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef    

    if args.aux_loss:
        aux_weight_dict = {}
        num_dummy_layer = 0
        for i in range(args.dec_layers - 1 + num_dummy_layer):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)


    for i in range(args.dec_layers - 1 + 2):
        weight_dict[f'loss_srcs_mi_{i}'] = args.srcs_mi_loss
        weight_dict[f'loss_srcs_xy_{i}'] = args.srcs_da_loss_xy    
        weight_dict[f'loss_srcs_zd_{i}'] = args.srcs_da_loss_zd    
        
        weight_dict[f'loss_tgts_mi_{i}'] = args.tgts_mi_loss
        weight_dict[f'loss_tgts_xy_{i}'] = args.tgts_da_loss_xy     
        weight_dict[f'loss_tgts_zd_{i}'] = args.tgts_da_loss_zd      
        
        
    # for da bkb: backbone part
    weight_dict[f'loss_srcs_mi_0'] = args.bkbs_mi_loss
    weight_dict[f'loss_srcs_xy_0'] = args.bkbs_da_loss_xy   
    weight_dict[f'loss_srcs_zd_0'] = args.bkbs_da_loss_zd
    
    # for src_6: last encoder layer
    weight_dict[f'loss_srcs_mi_6'] = args.src6_mi_loss
    weight_dict[f'loss_srcs_xy_6'] = args.src6_da_loss_xy   
    weight_dict[f'loss_srcs_zd_6'] = args.src6_da_loss_zd
    
    # for tgt_5: last decoder layer
    weight_dict[f'loss_tgts_mi_5'] = args.tgt5_mi_loss
    weight_dict[f'loss_tgts_xy_5'] = args.tgt5_da_loss_xy   
    weight_dict[f'loss_tgts_zd_5'] = args.tgt5_da_loss_zd
    
    # for tgt_0: first decoder layer
    weight_dict[f'loss_tgts_xy_0'] = args.tgt0_da_loss_xy   
    weight_dict[f'loss_tgts_zd_0'] = args.tgt0_da_loss_zd
    weight_dict[f'loss_tgts_mi_0'] = args.tgt0_mi_loss
    

    weight_dict['loss_space_query'] = args.space_q
    weight_dict['loss_chann_query'] = args.chann_q
    weight_dict['loss_insta_query'] = args.insta_q
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
        
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha, margin_src=args.margin_src, margin_tgt=args.margin_tgt)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
