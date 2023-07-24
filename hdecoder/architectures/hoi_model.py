# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import Boxes, ImageList, Instances, BitMasks, BoxMode

from .registry import register_model
from ..utils import configurable, get_class_names
from ..backbone import build_backbone, Backbone
from ..body.decoder.modules import MLP
from ..body import build_hoi_head
from ..modules.criterion import SetCriterionHOI
from ..modules.matcher import HungarianMatcherHOI

class CDNHOI(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        backbone,
        hoi_head,
        num_obj_classes,
        num_verb_classes,
        criterion,
        losses,
        num_queries,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        num_dec_layers_hopd: int,
        num_dec_layers_interaction,
    ):
        super().__init__()
        hidden_dim = hoi_head.d_model
        self.backbone = backbone
        self.hoid_head = hoi_head
        self.criterion = criterion
        self.losses = losses
        # self.num_queries = num_queries
        self.dec_layers_hopd = num_dec_layers_hopd
        self.dec_layers_interaction = num_dec_layers_interaction

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1).cuda()
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes).cuda()
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3).cuda()
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3).cuda()

        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        dec_cfg = cfg["MODEL"]["DECODER"]

        # build model
        backbone = build_backbone(cfg)
        hoi_head = build_hoi_head(cfg, backbone.output_shape())

        # for matching
        matcher = HungarianMatcherHOI(
            cost_obj_class=dec_cfg["COST_OBJECT_CLASS"], 
            cost_verb_class=dec_cfg["COST_VERB_CLASS"],
            cost_bbox=dec_cfg["COST_BBOX"], 
            cost_giou=dec_cfg["COST_GIOU"], 
            cost_matching=dec_cfg["COST_MATCHING"]
        )

        # for matching
        weight_dict = {}
        weight_dict['loss_obj_ce'] = dec_cfg["OBJ_LOSS_COEF"]
        weight_dict['loss_verb_ce'] = dec_cfg["VERB_LOSS_COEF"]
        weight_dict['loss_sub_bbox'] = dec_cfg["BBOX_LOSS_COEF"]
        weight_dict['loss_obj_bbox'] = dec_cfg["BBOX_LOSS_COEF"]
        weight_dict['loss_sub_giou'] = dec_cfg["GIOU_LOSS_COEF"]
        weight_dict['loss_obj_giou'] = dec_cfg["MATCHING_LOSS_COEF"]

        losses = ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality']
        criterion = SetCriterionHOI(
            dec_cfg["NUM_OBJECT_CLASSES"], 
            dec_cfg["NUM_OBJECT_QUERIES"], 
            dec_cfg["NUM_VERB_CLASSES"], 
            matcher=matcher,
            weight_dict=weight_dict, 
            eos_coef=dec_cfg["EOS_COEF"], 
            losses=losses)

        return {
            "backbone": backbone,
            "hoi_head": hoi_head,
            "criterion": criterion,
            "losses": losses,
            "num_queries": dec_cfg["NUM_OBJECT_QUERIES"],
            "num_obj_classes": dec_cfg["NUM_OBJECT_CLASSES"],
            "num_verb_classes": dec_cfg["NUM_VERB_CLASSES"],
            "pixel_mean": cfg["INPUT"]["PIXEL_MEAN"],
            "pixel_std": cfg["INPUT"]["PIXEL_STD"],
            "pixel_std": cfg["INPUT"]["PIXEL_STD"],
            "pixel_std": cfg["INPUT"]["PIXEL_STD"],
            "num_dec_layers_hopd": dec_cfg["HOPD_DEC_LAYERS"],
            "num_dec_layers_interaction" : dec_cfg["INTERACTION_DEC_LAYERS"],
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        outputs = self.forward_hoi(batched_inputs)
        if self.training:
            targets = self._prepare_targets(batched_inputs)
            losses_hoi = self.criterion(outputs, targets)
            return losses_hoi
        else:
            return self.evaluate_hoi(outputs)
        
    def forward_hoi(self, batched_inputs):
        assert "instances" in batched_inputs[0]
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 32)

        bs, c, h, w = images.tensor.shape
        features = self.backbone(images.tensor)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        hopd_out, interaction_decoder_out = self.hoid_head(features, mask=None, query_embed=query_embed)[:2]

        outputs_sub_coord = self.sub_bbox_embed(hopd_out).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(hopd_out).sigmoid()
        outputs_obj_class = self.obj_class_embed(hopd_out)
        outputs_verb_class = self.verb_class_embed(interaction_decoder_out)

        out = {
            'pred_obj_logits': outputs_obj_class[-1], 
            'pred_verb_logits': outputs_verb_class[-1],
            'pred_sub_boxes': outputs_sub_coord[-1], 
            'pred_obj_boxes': outputs_obj_coord[-1]}        
                                        
        out['aux_outputs'] = self._set_aux_loss(
            outputs_obj_class, 
            outputs_verb_class,
            outputs_sub_coord,
            outputs_obj_coord)

        return out

    def _prepare_targets(self, batched_inputs):
        new_targets = []
        for idx, batch_per_image in enumerate(batched_inputs):
            targets = batch_per_image["instances"]
            new_targets.append({k: v.to(self.device) for k, v in targets.items() if k != 'filename'})
        return new_targets
            
    def evaluate_hoi(self, outputs):
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = postprocessors['hoi'](outputs, orig_target_sizes)
        return outputs
    
    def hoi_inference(self):
        pass

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, outputs_matching=None):
        min_dec_layers_num = min(self.dec_layers_hopd, self.dec_layers_interaction)
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[-min_dec_layers_num : -1], outputs_verb_class[-min_dec_layers_num : -1], \
                                        outputs_sub_coord[-min_dec_layers_num : -1], outputs_obj_coord[-min_dec_layers_num : -1])]



@register_model
def get_hoi_model(cfg, **kwargs):
    return CDNHOI(cfg)