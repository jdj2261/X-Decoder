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
from detectron2.structures import (
    Boxes,
    BoxMode,
    Instances,
)

from xdecoder.utils import configurable

__all__ = ["VCOCODatasetMapperModified"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    if is_train:
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
    else:
        cfg_input = cfg["INPUT"]
        image_size = cfg_input["IMAGE_SIZE"]
        augmentation = []

        augmentation.extend(
            [
                T.Resize((image_size, image_size)),
            ]
        )

    return augmentation

class VCOCODatasetMapperModified:
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

        tfm_gens = build_transform_gen(cfg, is_train)

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
        dataset_dict = copy.deepcopy(dataset_dict)
        # image
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        
        if self.is_train and len(dataset_dict["annotations"]) > self.num_queries:
            dataset_dict["annotations"] = dataset_dict["annotations"][:self.num_queries]
        
        
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2] # h, w
        h, w = image_shape[0], image_shape[1]
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict['width'] = int(w)
        dataset_dict['height'] = int(h)
        if not self.is_train:
            valid_instances = self._get_valid_instances(dataset_dict)
            dataset_dict["instances"] = valid_instances
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            boxes = [obj["bbox"] for obj in dataset_dict["annotations"]]
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

            annos = [
                self._transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            
            instances = self._annotations_to_instances(annos, image_shape)            
            dataset_dict["instances"] = instances

            hoi_annos = [hoi_anno for hoi_anno in dataset_dict.pop("hoi_annotation")]
            file_names = dataset_dict["file_name"]
            dataset_dict["hoi_instances"] = self._annotations_to_hoi_instances(instances, hoi_annos, file_names)
        return dataset_dict


    @staticmethod
    def _transform_instance_annotations(annotation, transforms, image_size):
        if isinstance(transforms, (tuple, list)):
            transforms = T.TransformList(transforms)
        # bbox is 1d (per-instance bounding box)
        bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYWH_ABS)
        # clip transformed bbox to image size
        bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
        annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
        annotation["bbox_mode"] = BoxMode.XYXY_ABS
        return annotation
    
    def _annotations_to_instances(self, annos, image_size):
        if self.is_train:
            boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
            target = Instances(image_size)
            target.gt_boxes = Boxes(boxes)
            target.gt_areas = (target.gt_boxes.tensor[:, 2] - target.gt_boxes.tensor[:, 0]) * (target.gt_boxes.tensor[:, 3] - target.gt_boxes.tensor[:, 1])

            classes = [
                (i, self._valid_obj_ids.index(obj["category_id"]))
                for i, obj in enumerate(annos)
            ]
            classes = torch.tensor(classes, dtype=torch.int64)

            kept_box_indices = [label[0] for label in classes]

            classes = classes[:, 1]

            target.gt_classes = classes
            target.gt_kept_box_indices = kept_box_indices

        return target

    def _annotations_to_hoi_instances(self, instances, hoi_annos, file_names):
        obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
        sub_obj_pairs = []

        hoi_instacnes = {}
        for hoi in hoi_annos:
            if hoi["subject_id"] not in instances.gt_kept_box_indices or (
                hoi["object_id"] != -1 and hoi["object_id"] not in instances.gt_kept_box_indices
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
                        instances.gt_classes[instances.gt_kept_box_indices.index(hoi["object_id"])]
                    )
                verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                verb_label[self._valid_verb_ids.index(hoi["category_id"])] = 1
                sub_box = instances.gt_boxes.tensor[instances.gt_kept_box_indices.index(hoi["subject_id"])]
                if hoi["object_id"] == -1:
                    obj_box = torch.zeros((4,), dtype=torch.float32)
                else:
                    obj_box = instances.gt_boxes.tensor[
                        instances.gt_kept_box_indices.index(hoi["object_id"])
                    ]
                verb_labels.append(verb_label)
                sub_boxes.append(sub_box)
                obj_boxes.append(obj_box)

            hoi_instacnes["file_name"] = file_names
            if len(sub_obj_pairs) == 0:
                hoi_instacnes["obj_labels"] = torch.zeros((0,), dtype=torch.int64)
                hoi_instacnes["verb_labels"] = torch.zeros(
                    (0, len(self._valid_verb_ids)), dtype=torch.float32
                )
                hoi_instacnes["sub_boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                hoi_instacnes["obj_boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            else:
                hoi_instacnes["obj_labels"] = torch.stack(obj_labels)
                hoi_instacnes["verb_labels"] = torch.as_tensor(
                    verb_labels, dtype=torch.float32
                )
                hoi_instacnes["sub_boxes"] = torch.stack(sub_boxes)
                hoi_instacnes["obj_boxes"] = torch.stack(obj_boxes)
        return hoi_instacnes
    
    def _get_valid_instances(self, dataset_dict):
        valid_instances = {}
        boxes = [obj["bbox"] for obj in dataset_dict["annotations"]]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        classes = [self._valid_obj_ids.index(obj["category_id"]) for obj in dataset_dict["annotations"]]
        classes = torch.tensor(classes, dtype=torch.int64)
        valid_instances["boxes"] = boxes
        valid_instances["classes"] = classes
        valid_instances["id"] = dataset_dict["id"]
        valid_instances["img_id"] = int(
            dataset_dict["file_name"].rstrip(".jpg").split("_")[2]
        )
        valid_instances["filename"] = dataset_dict["file_name"]
        hois = []
        for hoi in dataset_dict["hoi_annotation"]:
            hois.append(
                (
                    hoi["subject_id"],
                    hoi["object_id"],
                    self._valid_verb_ids.index(hoi["category_id"]),
                )
            )
        valid_instances["hois"] = torch.as_tensor(hois, dtype=torch.int64)
        return valid_instances