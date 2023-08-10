# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import itertools
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import Boxes, ImageList, Instances, BitMasks, BoxMode
from detectron2.data import MetadataCatalog

from .registry import register_model
from ..utils import configurable, get_class_names
from ..backbone import build_backbone, Backbone
from ..body.decoder.modules import MLP
from ..body import build_hoi_head
from ..modules.criterion import SetCriterionHOI
from ..modules.matcher import HungarianMatcherHOI
from ..modules.postprocessing import PostProcessHOI
from datasets.utils.misc import all_gather
from copy import deepcopy

class CDNHOI(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        backbone,
        hoi_head,
        criterion,
        losses,
        postprocessors,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        metadata,
        size_divisibility

    ):
        super().__init__()
        self.backbone = backbone
        self.hoid_head = hoi_head
        self.criterion = criterion
        self.losses = losses
        self.postprocessors = postprocessors
        self.metadata = metadata
        self.size_divisibility = size_divisibility

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

        if cfg["AUX_LOSS"]:
            min_dec_layers_num = min(dec_cfg["HOPD_DEC_LAYERS"], dec_cfg["INTERACTION_DEC_LAYERS"])
            aux_weight_dict = {}
            for i in range(min_dec_layers_num - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality']
        criterion = SetCriterionHOI(
            dec_cfg["NUM_OBJECT_CLASSES"], 
            dec_cfg["NUM_OBJECT_QUERIES"], 
            dec_cfg["NUM_VERB_CLASSES"], 
            matcher=matcher,
            weight_dict=weight_dict, 
            eos_coef=dec_cfg["EOS_COEF"], 
            losses=losses)

        postprocessors = PostProcessHOI()

        pixel_mean = cfg["INPUT"]["PIXEL_MEAN"]
        pixel_mean = cfg["INPUT"]["PIXEL_STD"]
        
        return {
            "backbone": backbone,
            "hoi_head": hoi_head,
            "criterion": criterion,
            "losses": losses,
            "postprocessors": postprocessors,
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_mean,
            "metadata": MetadataCatalog.get(cfg['DATASETS']['TRAIN'][0]),
            "size_divisibility": dec_cfg["SIZE_DIVISIBILITY"],
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, mode="vcoco"):
        if self.training:
            losses_hoi = self.forward_hoi(batched_inputs["vcoco"])
            return losses_hoi
        else:
            if mode == "vcoco":
                return self.evaluate_hoi(batched_inputs)

    def forward_hoi(self, batched_inputs):
        assert "instances" in batched_inputs[0]
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        # if "instances" in batched_inputs[0]:
        targets = self.prepare_targets(batched_inputs, images)
        features = self.backbone(images.tensor)
        
        # TODO not mask None
        # src, mask = features[-1].decompose()
        out = self.hoid_head(features, mask=None)
        losses_hoi = self.criterion(out, targets)
        
        del out
        return losses_hoi
    
    def prepare_targets(self, batched_inputs, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for idx, batch_per_image in enumerate(batched_inputs):
            targets = batch_per_image["hoi_instances"]
            new_target = {}
            for key, value in targets.items():
                if key == "file_name":
                    continue
                if "boxes" in key:
                    gt_boxes = value.to(self.device)
                    ratio = torch.tensor([w_pad,h_pad,w_pad,h_pad]).to(gt_boxes.device)[None,:]
                    gt_boxes = gt_boxes / ratio
                    xc,yc,w,h = (gt_boxes[:,0] + gt_boxes[:,2])/2, (gt_boxes[:,1] + gt_boxes[:,3])/2, gt_boxes[:,2] - gt_boxes[:,0], gt_boxes[:,3] - gt_boxes[:,1]
                    gt_boxes = torch.stack([xc,yc,w,h]).permute(1,0)
                    new_target[key] = gt_boxes
                if "labels" in key:
                    gt_labels = value.to(self.device)
                    new_target[key] = gt_labels
            new_targets.append(new_target)
        return new_targets
    
    def evaluate_hoi(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]

        # TODO size_divisibility
        images = ImageList.from_tensors(images, 32)
        orig_target_sizes = torch.stack([t["orig_size"] for t in batched_inputs], dim=0)

        features = self.backbone(images.tensor)
        outputs = self.hoid_head(features, mask=None)

        # TODO
        results = self.postprocessors(outputs, orig_target_sizes)

        return results
    
    def hoi_inference(self):
        pass

@register_model
def get_hoi_model(cfg, **kwargs):
    return CDNHOI(cfg)