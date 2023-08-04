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
        metadata

    ):
        super().__init__()
        self.backbone = backbone
        self.hoid_head = hoi_head
        self.criterion = criterion
        self.losses = losses
        self.postprocessors = postprocessors
        self.metadata = metadata

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
            "metadata": MetadataCatalog.get(cfg['DATASETS']['TRAIN'][0])
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
        images = ImageList.from_tensors(images, 32)

        features = self.backbone(images.tensor)
        
        # TODO not mask None
        # src, mask = features[-1].decompose()
        out = self.hoid_head(features, mask=None)
        targets = self._prepare_targets(batched_inputs)
        losses_hoi = self.criterion(out, targets)

        del out
        return losses_hoi

    def _prepare_targets(self, batched_inputs):
        new_targets = []
        for idx, batch_per_image in enumerate(batched_inputs):
            targets = batch_per_image["instances"]
            new_targets.append({k: v.to(self.device) for k, v in targets.items() if k != 'filename'})
        return new_targets
    
    def _prepare_eval_targets(self, batched_inputs):
        new_targets = []
        for idx, batch_per_image in enumerate(batched_inputs):
            targets = batch_per_image["instances"]
            for k, v in targets.items():
                if k != 'filename' and k != 'id' and k != 'img_id':
                    new_targets.append({k: v})
        return new_targets
            

    def evaluate_hoi(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]

        # TODO size_divisibility
        images = ImageList.from_tensors(images, 32)
        orig_target_sizes = torch.stack([t["instances"]["orig_size"] for t in batched_inputs], dim=0)

        features = self.backbone(images.tensor)
        outputs = self.hoid_head(features, mask=None)

        # TODO
        results = self.postprocessors(outputs, orig_target_sizes)

        return results

        # processed_results = []
        # processed_results.append({})

        # processed_results[-1]["predicts"] = list(itertools.chain.from_iterable(all_gather(results)))

        # targets = self._prepare_eval_targets(batched_inputs)
        # processed_results[-1]["gts"] = list(itertools.chain.from_iterable(all_gather(deepcopy(targets))))
        # return processed_results
    
    def hoi_inference(self):
        pass

@register_model
def get_hoi_model(cfg, **kwargs):
    return CDNHOI(cfg)