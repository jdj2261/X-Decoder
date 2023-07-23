# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .registry import register_decoder

from ...utils import configurable
from ..transformer_blocks import TransformerDecoder, TransformerDecoderLayer

class HDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        hidden_dim: int,
        nheads: int,
        dropout,
        dim_feedforward: int,
        pre_norm,
        num_dec_layers_hopd: int,
        num_dec_layers_interaction,
        return_intermediate_dec
    ):
        super().__init__()
        self.d_model = hidden_dim
        self.nhaed = nheads

        hopd_decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=nheads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            normalize_before=pre_norm
        )
        hopd_decoder_norm = nn.LayerNorm(hidden_dim)

        self.hopd_decoder = TransformerDecoder(
            decoder_layer=hopd_decoder_layer,
            num_layers=num_dec_layers_hopd,
            norm=hopd_decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        interaction_decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=nheads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            normalize_before=pre_norm
        )
        interaction_decoder_norm = nn.LayerNorm(hidden_dim)

        self.interaction_decoder = TransformerDecoder(
            decoder_layer=interaction_decoder_layer,
            num_layers=num_dec_layers_interaction,
            norm=interaction_decoder_norm,
            return_intermediate=return_intermediate_dec,
        )


    @classmethod
    def from_config(cls, cfg):
        ret = {}
        dec_cfg = cfg["MODEL"]["DECODER"]
        ret["hidden_dim"] = dec_cfg["HIDDEN_DIM"]
        ret["nheads"] = dec_cfg["NHEADS"]
        ret["dropout"] = dec_cfg["DROPOUT"]
        ret["dim_feedforward"] = dec_cfg["DIM_FEEDFORWARD"]
        ret["pre_norm"] = dec_cfg["PRE_NORM"]
        ret["num_dec_layers_hopd"] = dec_cfg["HOPD_DEC_LAYERS"]
        ret["num_dec_layers_interaction"] = dec_cfg["INTERACTION_DEC_LAYERS"]
        ret["return_intermediate_dec"] = dec_cfg["RETURN_INTERMEDIATE_DEC"]
        return ret


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

        interaction_query_embed = hopd_out[-1]
        interaction_query_embed = interaction_query_embed.permute(1, 0, 2)
        interaction_tgt = torch.zeros_like(interaction_query_embed)
        interaction_decoder_out = self.interaction_decoder(interaction_tgt, memory, memory_key_padding_mask=mask,
                                  pos=pos_embed, query_pos=interaction_query_embed)
        interaction_decoder_out = interaction_decoder_out.transpose(1, 2)

        return hopd_out, interaction_decoder_out, memory


@register_decoder
def get_masked_transformer_decoder(cfg):
    return HDecoder(cfg)