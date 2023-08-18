# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
import numpy as np
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
from ..modules.postprocessing import PostProcessHOI, OfficialPostProcessHOI
from datasets.utils.misc import all_gather
from copy import deepcopy

from utils.box_ops import box_cxcywh_to_xyxy

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

        if cfg["POSTPROCESS"]["OFFICIAL"]["USE"]:
            postprocessors = OfficialPostProcessHOI(correct_mat_dir=MetadataCatalog.get(cfg["DATASETS"]["TEST"][0]).correct_mat_dir)
        else:
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
    

    @torch.no_grad()
    def hoi_inference(self, image_ori, orig_size, transform, thr=0.1, return_only_outputs=False):

        height, width = orig_size
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

        orig_target_sizes = torch.as_tensor([int(height), int(width)])
        batched_inputs = [{'image': images, 'orig_size':orig_target_sizes}]
        
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 32)

        features = self.backbone(images.tensor)
        outputs = self.hoid_head(features, mask=None)

        if return_only_outputs:
            return outputs
        # print(outputs)

        out_obj_logits = outputs['pred_obj_logits']
        out_verb_logits = outputs['pred_verb_logits']
        out_sub_boxes =  outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)
        verb_scores = out_verb_logits.sigmoid()

        img_h = torch.as_tensor([orig_target_sizes[0]]).cuda(self.device)
        img_w = torch.as_tensor([orig_target_sizes[1]]).cuda(self.device)

        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)

        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
            sl = torch.full_like(ol, 0)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(b.to('cpu').numpy(), l.to('cpu').numpy())]

            hoi_scores = vs * os.unsqueeze(1)

            verb_labels = torch.arange(hoi_scores.shape[1]).view(1, -1).expand(hoi_scores.shape[0], -1)

            ids = torch.arange(b.shape[0])
            hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                    subject_id, object_id, category_id, score in zip(ids[:ids.shape[0] // 2].to('cpu').numpy(),
                                                                    ids[ids.shape[0] // 2:].to('cpu').numpy(),
                                                                    verb_labels.to('cpu').numpy(), hoi_scores.to('cpu').numpy())]
            current_result = {'predictions': bboxes, 'hoi_prediction': hois}
            results.append(current_result)

        results_filtered = []
        for i in range(len(results)):
            result_filtered = []
            result = results[i]
            for h in result['hoi_prediction']:
                score = np.max(h['score'])
                if score > thr:
                    obj_id =  h['object_id']
                    sub_id =  h['subject_id']

                    index = np.argmax(h['score'])
                    filtered_dict = { 'category_id': h['category_id'][index],
                            'object_id': obj_id,
                            'score': score,
                            'object_bbox' : result['predictions'][obj_id],
                            'subject_id' : result['predictions'][sub_id],
                            }
                    result_filtered.append(filtered_dict)
            results_filtered.append(result_filtered)
        return results_filtered


@register_model
def get_hoi_model(cfg, **kwargs):
    return CDNHOI(cfg)