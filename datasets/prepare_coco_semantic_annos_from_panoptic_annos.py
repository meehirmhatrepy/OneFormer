#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import functools
import json
import multiprocessing as mp
import numpy as np
import os
import time
from fvcore.common.download import download
from panopticapi.utils import rgb2id
from PIL import Image

# from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

COCO_CATEGORIES = [
    # {"id": 0, "name": "background", "supercategory": "none", "isthing": 0, "color": [0, 0, 0]},
    {"id": 1, "name": "Backrind", "supercategory": "defect", "isthing": 1, "color": [220, 20, 60]},
    {"id": 2, "name": "crack", "supercategory": "defect", "isthing": 1, "color": [119, 11, 32]},
    {"id": 3, "name": "dent", "supercategory": "defect", "isthing": 1, "color": [0, 0, 142]},
    {"id": 4, "name": "elastomer", "supercategory": "object", "isthing": 1, "color": [0, 0, 230]},
    {"id": 5, "name": "extrusion", "supercategory": "defect", "isthing": 1, "color": [106, 0, 228]},
    {"id": 6, "name": "flow", "supercategory": "defect", "isthing": 1, "color": [0, 60, 100]},
    {"id": 7, "name": "over grinding", "supercategory": "defect", "isthing": 1, "color": [0, 80, 100]},
    {"id": 8, "name": "overtrimming", "supercategory": "defect", "isthing": 1, "color": [0, 0, 70]},
    {"id": 9, "name": "rgd", "supercategory": "defect", "isthing": 1, "color": [0, 0, 192]},
    {"id": 10, "name": "trimming", "supercategory": "defect", "isthing": 1, "color": [250, 170, 30]},
]


def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
    panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
    panoptic = rgb2id(panoptic)
    output = np.zeros_like(panoptic, dtype=np.uint8) + 255
    for seg in segments:
        cat_id = seg["category_id"]
        new_cat_id = id_map[cat_id]
        output[panoptic == seg["id"]] = new_cat_id
    Image.fromarray(output).save(output_semantic)


def separate_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.
    It maps all thing categories to class 0, and maps all unlabeled pixels to class 255.
    It maps all stuff categories to contiguous ids starting from 1.
    Args:
        panoptic_json (str): path to the panoptic json file, in COCO's format.
        panoptic_root (str): a directory with panoptic annotation files, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
        categories (list[dict]): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(categories) <= 254
    for i, k in enumerate(categories):
        id_map[k["id"]] = i
    # what is id = 0?
    # id_map[0] = 255
    print(id_map)

    with open(panoptic_json) as f:
        obj = json.load(f)

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        for anno in obj["annotations"]:
            file_name = anno["file_name"]
            segments = anno["segments_info"]
            input = os.path.join(panoptic_root, file_name)
            output = os.path.join(sem_seg_root, file_name)
            yield input, output, segments

    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    pool.starmap(
        functools.partial(_process_panoptic_to_semantic, id_map=id_map),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    dataset_dir = os.path.join(os.getenv("DETECTRON2_DATASETS", "datasets"), "coco")
    for s in ["val2017", "train2017"]:
        separate_coco_semantic_from_panoptic(
            os.path.join(dataset_dir, "annotations/panoptic_{}.json".format(s)),
            os.path.join(dataset_dir, "panoptic_{}".format(s)),
            os.path.join(dataset_dir, "panoptic_semseg_{}".format(s)),
            COCO_CATEGORIES,
        )
