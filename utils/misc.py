# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
import math
import torch
import torch.distributed as dist

# HACK for evalution
def hook_metadata(metadata, name):
    if name == "cityscapes_fine_sem_seg_val":
        metadata.__setattr__("keep_sem_bgd", False)
    return metadata


def hook_opt(model, name):
    if name in [
        "cityscapes_fine_panoptic_val",
        "ade20k_panoptic_val",
        "bdd10k_40_panoptic_val",
        "cityscapes_fine_panoptic_val",
        "scannet_21_panoptic_val",
    ]:
        model.model.object_mask_threshold = 0.4
    else:
        model.model.object_mask_threshold = 0.8


# HACK for evalution
def hook_switcher(model, name):
    mappings = {}
    if name in [
        "cityscapes_fine_sem_seg_val",
        "scannet_21_val_seg",
        "scannet_38_val_seg",
        "scannet_41_val_seg",
        "sunrgbd_37_val_seg",
        "bdd10k_val_sem_seg",
        "ade20k_full_sem_seg_val",
    ]:
        mappings = {"SEMANTIC_ON": True, "INSTANCE_ON": False, "PANOPTIC_ON": False}
    elif name in ["cityscapes_fine_instance_seg_val"] or "seginw" in name:
        mappings = {"SEMANTIC_ON": False, "INSTANCE_ON": True, "PANOPTIC_ON": False}
    elif name in [
        "cityscapes_fine_panoptic_val",
        "scannet_21_panoptic_val",
        "bdd10k_40_panoptic_val",
    ]:
        mappings = {"SEMANTIC_ON": True, "INSTANCE_ON": False, "PANOPTIC_ON": True}
    elif name in [
        "coco_2017_val_panoptic_with_sem_seg",
        "ade20k_panoptic_val",
        "coco_2017_test-dev",
    ]:
        mappings = {"SEMANTIC_ON": True, "INSTANCE_ON": True, "PANOPTIC_ON": True}
    else:
        if name not in [
            "vlp_val",
            "vlp_captioning_val",
            "vlp_val2017",
            "vlp_captioning_val2017",
            "imagenet_val",
            "refcocog_val_google",
            "phrasecut_val",
            "phrasecut_test",
            "refcocop_val_unc",
            "refcoco_val_unc",
            "refcocog_val_umd",
        ]:
            assert False, "dataset switcher is not defined"
    for key, value in mappings.items():
        if key == "SEMANTIC_ON":
            model.model.semantic_on = value
        if key == "INSTANCE_ON":
            model.model.instance_on = value
        if key == "PANOPTIC_ON":
            model.model.panoptic_on = value


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, decay=0):
        self.val = val
        if decay:
            alpha = math.exp(-n / decay)  # exponential decay over 100 updates
            self.sum = alpha * self.sum + (1 - alpha) * val * n
            self.count = alpha * self.count + (1 - alpha) * n
        else:
            self.sum += val * n
            self.count += n
        self.avg = self.sum / self.count
