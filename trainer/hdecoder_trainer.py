# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import logging
import os
import json
import random
import copy
import itertools
from typing import Any, Dict, List, Set, Union
from datetime import datetime
from mpi4py import MPI

import numpy as np
import torch
from torch.utils.data import DataLoader

from detectron2.projects.deeplab import build_lr_scheduler
from fvcore.common.config import CfgNode
from infinibatch import iterators

from utils.distributed import is_main_process, get_world_size
from .default_trainer import DefaultTrainer
from .utils.serialization import JSONEncoder, filter_jsonable

logger = logging.getLogger(__name__)


class XDecoder_Trainer(DefaultTrainer):
    def __init__(self, opt):
        super().__init__(opt)