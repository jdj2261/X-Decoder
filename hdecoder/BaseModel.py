import os
import logging

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from utils.model import align_and_update_state_dicts

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self, opt, module: nn.Module):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.model = module

        t = []
        t.append(transforms.Resize(800, interpolation=Image.BICUBIC))
        self.tranform_for_inference = transforms.Compose(t)
        
    def forward(self, *inputs, **kwargs):
        outputs = self.model(*inputs, **kwargs)
        return outputs

    def inference(self, image_path, transform=None):
        if transform is None:
            transform = self.tranform_for_inference
        outputs = self.model.hoi_inference(image_path, transform)
        return outputs
        
    def save_pretrained(self, save_dir):
        torch.save(self.model.state_dict(), save_dir)

    def from_pretrained(self, load_dir):
        state_dict = torch.load(load_dir, map_location=self.opt["device"])
        state_dict = align_and_update_state_dicts(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        return self
