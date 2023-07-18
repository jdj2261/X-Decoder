# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from timm.models.layers import trunc_normal_
from detectron2.layers import Conv2d
import fvcore.nn.weight_init as weight_init

from .registry import register_decoder

from ...utils import configurable
from ...modules import PositionEmbeddingSine
from ..transformer_blocks import TransformerDecoder, TransformerDecoderLayer


class HDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_queries: int,
    ):
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    @classmethod
    def from_config(cls):
        pass

    def forward(self, features):
        return self.layers(features)


class HumanObjectPairDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_dec_layers_hopd=3,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.hopd_decoder = TransformerDecoder(
            decoder_layer,
            num_dec_layers_hopd,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

    def forward(self, memory, mask, query_embed, pos_embed):
        tgt = torch.zeros_like(query_embed)
        hopd_out = self.hopd_decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        hopd_out = hopd_out.transpose(1, 2)
        return hopd_out


class InteractionDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_dec_layers_hopd=3,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.hopd_decoder = TransformerDecoder(
            decoder_layer,
            num_dec_layers_hopd,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

    def forward(self, memory, mask, interaction_query_embed, pos_embed):
        interaction_tgt = torch.zeros_like(interaction_query_embed)
        interaction_query_embed = interaction_query_embed.permute(1, 0, 2)

        interaction_decoder_out = self.interaction_decoder(interaction_tgt, memory, memory_key_padding_mask=mask,
                                  pos=pos_embed, query_pos=interaction_query_embed)
        interaction_decoder_out = interaction_decoder_out.transpose(1, 2)

        return interaction_decoder_out
    

@register_decoder
def get_masked_transformer_decoder(
    cfg, in_channels, lang_encoder, mask_classification, extra
):
    return HDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
