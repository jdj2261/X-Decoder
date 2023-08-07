# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import torch
import numpy as np
from PIL import Image

from torchvision import transforms

from pycocotools import mask
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from xdecoder.utils import configurable

__all__ = ["VCOCODatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    cfg_input = cfg["INPUT"]
    image_size = cfg_input["IMAGE_SIZE"]
    min_scale = cfg_input["MIN_SCALE"]
    max_scale = cfg_input["MAX_SCALE"]

    augmentation = []

    if cfg_input["RANDOM_FLIP"] != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg_input["RANDOM_FLIP"] == "horizontal",
                vertical=cfg_input["RANDOM_FLIP"] == "vertical",
            )
        )

    augmentation.extend(
        [
            T.ResizeScale(
                min_scale=min_scale,
                max_scale=max_scale,
                target_height=image_size,
                target_width=image_size,
            ),
            T.FixedSizeCrop(crop_size=(image_size, image_size)),
        ]
    )

    return augmentation

class VCOCODatasetMapper:
    @configurable
    def __init__(
        self,
        is_train=True,
        tfm_gens=None,
        image_format=None,
        min_size_test=None,
        max_size_test=None,
        num_queries=100,
    ):
        self.is_train = is_train
        self.tfm_gens = tfm_gens
        self.img_format = image_format
        self.min_size_test = min_size_test
        self.max_size_test = max_size_test
        self.num_queries = num_queries

        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        
        self._valid_verb_ids = range(29)

        t = []
        t.append(transforms.Resize(self.min_size_test, interpolation=Image.BICUBIC))
        self.transform = transforms.Compose(t)

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        print(cfg)
        if is_train:
            tfm_gens = build_transform_gen(cfg, is_train)
        else:
            tfm_gens = None

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg["INPUT"].get("FORMAT", "RGB"),
            "min_size_test": cfg["INPUT"]["MIN_SIZE_TEST"],
            "max_size_test": cfg["INPUT"]["MAX_SIZE_TEST"],
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        if self.is_train and len(dataset_dict["annotations"]) > self.num_queries:
            dataset_dict["annotations"] = dataset_dict["annotations"][:self.num_queries]
        
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image_shape = image.shape[:2]  # h, w
        h, w = image_shape[0], image_shape[1]
        
        boxes = [obj["bbox"] for obj in dataset_dict["annotations"]]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        if self.is_train:
            # Add index for confirming which boxes are kept after image transformation
            classes = [
                (i, self._valid_obj_ids.index(obj["category_id"]))
                for i, obj in enumerate(dataset_dict["annotations"])
            ]
        else:
            classes = [
                self._valid_obj_ids.index(obj["category_id"])
                for obj in dataset_dict["annotations"]
            ]
            
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self.is_train:
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            target["boxes"] = boxes
            target["labels"] = classes
            target["iscrowd"] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            image, transforms = T.apply_transform_gens(self.tfm_gens, image)

            kept_box_indices = [label[0] for label in target["labels"]]

            target["labels"] = target["labels"][:, 1]

            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            sub_obj_pairs = []
            for hoi in dataset_dict["hoi_annotation"]:
                if hoi["subject_id"] not in kept_box_indices or (
                    hoi["object_id"] != -1 and hoi["object_id"] not in kept_box_indices
                ):
                    continue
                sub_obj_pair = (hoi["subject_id"], hoi["object_id"])
                if sub_obj_pair in sub_obj_pairs:
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][
                        self._valid_verb_ids.index(hoi["category_id"])
                    ] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    if hoi["object_id"] == -1:
                        obj_labels.append(torch.tensor(len(self._valid_obj_ids)))
                    else:
                        obj_labels.append(
                            target["labels"][kept_box_indices.index(hoi["object_id"])]
                        )
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    verb_label[self._valid_verb_ids.index(hoi["category_id"])] = 1
                    sub_box = target["boxes"][kept_box_indices.index(hoi["subject_id"])]
                    if hoi["object_id"] == -1:
                        obj_box = torch.zeros((4,), dtype=torch.float32)
                    else:
                        obj_box = target["boxes"][
                            kept_box_indices.index(hoi["object_id"])
                        ]
                    verb_labels.append(verb_label)
                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)

            target["filename"] = dataset_dict["file_name"]

            if len(sub_obj_pairs) == 0:
                target["obj_labels"] = torch.zeros((0,), dtype=torch.int64)
                target["verb_labels"] = torch.zeros(
                    (0, len(self._valid_verb_ids)), dtype=torch.float32
                )
                target["sub_boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["obj_boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["matching_labels"] = torch.zeros((0,), dtype=torch.int64)
            else:
                target["obj_labels"] = torch.stack(obj_labels)
                target["verb_labels"] = torch.as_tensor(
                    verb_labels, dtype=torch.float32
                )
                target["sub_boxes"] = torch.stack(sub_boxes)
                target["obj_boxes"] = torch.stack(obj_boxes)
                target["matching_labels"] = torch.ones_like(target["obj_labels"])
        else:
            target["filename"] = dataset_dict["file_name"]
            target["boxes"] = boxes
            target["labels"] = classes
            target["id"] = dataset_dict["id"]
            target["img_id"] = int(
                dataset_dict["file_name"].rstrip(".jpg").split("_")[2]
            )

            image = self.transform(image)
            hois = []
            for hoi in dataset_dict["hoi_annotation"]:
                hois.append(
                    (
                        hoi["subject_id"],
                        hoi["object_id"],
                        self._valid_verb_ids.index(hoi["category_id"]),
                    )
                )
            target["hois"] = torch.as_tensor(hois, dtype=torch.int64)
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        dataset_dict["instances"] = target
        return dataset_dict
