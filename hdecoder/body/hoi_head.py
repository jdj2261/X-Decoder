# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict

import torch
from torch import nn
from detectron2.layers import ShapeSpec

from .registry import register_body
from .encoder import build_encoder
from .decoder import build_decoder
from ..utils import configurable
from ..body.decoder.modules import MLP

class CDN(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape,
        *,
        transformer_encoder,
        transformer_decoder,
        num_obj_classes,
        num_verb_classes,
        num_queries,
        num_dec_layers_hopd: int,
        num_dec_layers_interaction,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.encoder = transformer_encoder
        self.hoi_decoder = transformer_decoder

        hidden_dim = transformer_decoder.d_model
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1).cuda()
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes).cuda()
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3).cuda()
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3).cuda()
        self.dec_layers_hopd = num_dec_layers_hopd
        self.dec_layers_interaction = num_dec_layers_interaction

    @classmethod
    def from_config(
        cls,
        cfg,
        input_shape: Dict[str, ShapeSpec],
    ):
        enc_cfg = cfg["MODEL"]["ENCODER"]
        dec_cfg = cfg["MODEL"]["DECODER"]
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in enc_cfg["IN_FEATURES"]
            },
            "transformer_encoder": build_encoder(cfg, input_shape),
            "transformer_decoder": build_decoder(cfg),
            "num_queries": dec_cfg["NUM_OBJECT_QUERIES"],
            "num_obj_classes": dec_cfg["NUM_OBJECT_CLASSES"],
            "num_verb_classes": dec_cfg["NUM_VERB_CLASSES"],
            "num_dec_layers_hopd": dec_cfg["HOPD_DEC_LAYERS"],
            "num_dec_layers_interaction" : dec_cfg["INTERACTION_DEC_LAYERS"],
        }

    def forward(
        self,
        features,
        mask,
    ):
        # Encoder
        encoder_features, pos = self.encoder(features)

        bs, _, _, _ = pos.shape
        pos_embed = pos.flatten(2).permute(2, 0, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        # Decoder
        hopd_out, interaction_decoder_out, _ = self.hoi_decoder(encoder_features, mask, query_embed, pos_embed)
    
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
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, outputs_matching=None):
        min_dec_layers_num = min(self.dec_layers_hopd, self.dec_layers_interaction)
        return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                for a, b, c, d in zip(outputs_obj_class[-min_dec_layers_num : -1], outputs_verb_class[-min_dec_layers_num : -1], \
                                        outputs_sub_coord[-min_dec_layers_num : -1], outputs_obj_coord[-min_dec_layers_num : -1])]



@register_body
def get_hoi_head(cfg, input_shape):
    return CDN(cfg, input_shape)