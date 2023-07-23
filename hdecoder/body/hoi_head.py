# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict

from torch import nn
from detectron2.layers import ShapeSpec

from .registry import register_body
from .encoder import build_encoder
from .decoder import build_decoder
from ..utils import configurable


class CDN(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape,
        *,
        transformer_encoder,
        transformer_decoder,

    ):
        super().__init__()
        self.input_shape = input_shape
        self.encoder = transformer_encoder
        self.hoi_decoder = transformer_decoder
        self.d_model = transformer_decoder.d_model

    @classmethod
    def from_config(
        cls,
        cfg,
        input_shape: Dict[str, ShapeSpec],
    ):
        enc_cfg = cfg["MODEL"]["ENCODER"]
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in enc_cfg["IN_FEATURES"]
            },
            "transformer_encoder": build_encoder(cfg, input_shape),
            "transformer_decoder": build_decoder(cfg)
        }

    def forward(
        self,
        features,
        mask,
        query_embed
    ):

        # Encoder
        encoder_features, pos = self.encoder(features)
        pos_embed = pos.flatten(2).permute(2, 0, 1)

        # Decoder
        hopd_out, interaction_decoder_out, memory = self.hoi_decoder(encoder_features, mask, query_embed, pos_embed)
       
        return hopd_out, interaction_decoder_out, memory


@register_body
def get_hoi_head(cfg, input_shape):
    return CDN(cfg, input_shape)