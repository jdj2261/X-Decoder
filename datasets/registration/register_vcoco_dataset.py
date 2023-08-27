# Copyright (c) Facebook, Inc. and its affiliates.
import os
import json

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

_PREDEFINED_SPLITS_VCOCO_CAPTION = {
    "vcoco_train": (
        "v-coco/images/train2014",
        "v-coco/annotations/trainval_vcoco.json",
        None,
        None,
        None,
        None
    ),
    "vcoco_val": (
        "v-coco/images/val2014", 
        "v-coco/annotations/test_vcoco.json", 
        "v-coco/annotations/corre_vcoco.npy",
        "v-coco/data/vcoco/vcoco_test.json",
        "v-coco/data/instances_vcoco_all_2014.json",
        "v-coco/data/splits/vcoco_test.ids",
    )
}


COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
]

VCOCO_ACTIONS = [
    {"id": 1, "name": "hold"},
    {"id": 2, "name": "stand"},
    {"id": 3, "name": "sit"},
    {"id": 4, "name": "ride"},
    {"id": 5, "name": "walk"},
    {"id": 6, "name": "look"},
    {"id": 7, "name": "hit"},
    {"id": 8, "name": "eat"},
    {"id": 9, "name": "jump"},
    {"id": 10, "name": "lay"},
    {"id": 11, "name": "talk_on_phone"},
    {"id": 12, "name": "carry"},
    {"id": 13, "name": "throw"},
    {"id": 14, "name": "catch"},
    {"id": 15, "name": "cut"},
    {"id": 16, "name": "run"},
    {"id": 17, "name": "work_on_computer"},
    {"id": 18, "name": "ski"},
    {"id": 19, "name": "surf"},
    {"id": 20, "name": "skateboard"},
    {"id": 21, "name": "smile"},
    {"id": 22, "name": "drink"},
    {"id": 23, "name": "kick"},
    {"id": 24, "name": "point"},
    {"id": 25, "name": "read"},
    {"id": 26, "name": "snowboard"},
]


def load_vcoco_json(image_root, anno_file):
    json_file = os.path.join(anno_file)
    with open(json_file) as f:
        vcoco_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(vcoco_anns):
        record = {}
        record["id"] = idx
        for key, val in v.items():
            if key == "file_name":
                record[key] = os.path.join(image_root, v["file_name"])
            else:
                record[key] = val
                if key == "annotations":
                    for anno in v[key]:
                        anno["bbox_mode"] = BoxMode.XYWH_ABS
        dataset_dicts.append(record)
    return dataset_dicts


def get_metadata():
    meta = {}
    return meta


def register_vcoco(
        name, 
        metadata, 
        image_root, 
        annot_json, 
        correct_mat_dir=None,
        vsrl_annot_file=None,
        coco_file=None,
        split_file=None
    ):
    DatasetCatalog.register(
        name,
        lambda: load_vcoco_json(image_root, annot_json),
    )
    
    MetadataCatalog.get(name).set(
        image_root=image_root,
        json_file=annot_json,
        evaluator_type="hoi",
        correct_mat_dir=correct_mat_dir,
        vsrl_annot_file=vsrl_annot_file,
        coco_annot_file=coco_file,
        split_file=split_file,
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_all_vcoco(root):
    for (
        prefix,
        (image_root, annot_root, correct_mat_dir, vsrl_annot_file, coco_file, split_file),
    ) in _PREDEFINED_SPLITS_VCOCO_CAPTION.items():
        if correct_mat_dir:
            correct_mat_dir = os.path.join(root, correct_mat_dir)
        if vsrl_annot_file:
            vsrl_annot_file = os.path.join(root, vsrl_annot_file)
        if coco_file:
            coco_file = os.path.join(root, coco_file)
        if split_file:
            split_file = os.path.join(root, split_file)

        register_vcoco(
            prefix,
            get_metadata(),
            os.path.join(root, image_root),
            os.path.join(root, annot_root),
            correct_mat_dir,
            vsrl_annot_file,
            coco_file,
            split_file,
        )

_root = os.getenv("DATASET", "./datasets")
register_all_vcoco(_root)