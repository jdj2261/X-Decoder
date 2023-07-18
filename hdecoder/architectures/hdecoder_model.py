# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


from ..utils import configurable, get_class_names
from ..backbone import build_backbone, Backbone


class HoiDecoder(nn.Module):
    @configurable
    def __init__(self, *, d):
        pass
