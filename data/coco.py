
__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import os
import cv2
import numpy as np
from torch.utils import data
from pycocotools import mask
from utils.tranform import CommonTransforms
from .dataset import DatasetBuilder
from utils.image import load_rgb_image


skip_cls_ids = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]
num_cls = 80


def convert_cls_id_to_index(cls_id):
    """
    convert the class id to cls id
    :param cls_id:
    :return:
    """
    skip_pos = 0
    while skip_pos < len(skip_cls_ids) and skip_cls_ids[skip_pos] < cls_id:
        skip_pos += 1
    return cls_id - skip_pos - 1


def parse_segmentation(ann):
    """
    parse segmentation
    :param ann: annotation
    :return: list[numpy]
    """
    segm = ann["segmentation"]
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        segment_poly = segm[0]
        segments = np.array(segment_poly, dtype=np.float32).reshape((-1, 2))
    else:
        if type(segm['counts']) == list:
            # uncompressed RLE
            rle = mask.frPyObjects(segm, segm["size"][0], segm["size"][1])
        else:
            # rle
            rle = ann['segmentation']
        binary_mask = mask.decode(rle)
        segments = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return segments


class COCODataset(data.Dataset):
    """
    the dataset for coco. it includes the augmentation of data.
    """
    def __init__(self, data_dir, phase, ann_file, input_size, transforms=None):
        # save the parameters
        self._data_dir = data_dir
        self._phase = phase
        self._ann_file = ann_file
        if transforms is not None:
            self._transforms = transforms  # ADDED THIS
        else:
            self._transforms = CommonTransforms(input_size, num_cls)
        # initialize the coco api
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # locating index
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        path = os.path.join(self._data_dir, coco.loadImgs(img_id)[0]['file_name'])
        input_img = load_rgb_image(path)

        height, width, _ = input_img.shape
        polygons = []
        cls_ids = []
        for ann in anns:
            # handle boundary, reverse point(w,h) to (h,w)
            polygon = np.vstack(parse_segmentation(ann)).astype(np.int32)
            polygon[:, 0] = np.clip(polygon[:, 0], a_min=0, a_max=width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], a_min=0, a_max=height - 1)
            polygons.append(polygon)
            # handle category id
            cls_ids.append(convert_cls_id_to_index(ann["category_id"]))

        label = (cls_ids, polygons)
        input_img, label, trans_info = self._transforms(input_img, label, path)
        return input_img, label, trans_info

    def __len__(self):
        return len(self.ids)


class COCODatasetBuilder(DatasetBuilder):
    def __init__(self, data_dir, phase="train", ann_file=None):
        super().__init__(data_dir, phase, ann_file)

    def default_ann_file(self):
        return os.path.join(self._data_dir, "annotations.json")

    def get_dataset(self, **kwargs):
        return COCODataset(self._data_dir, self._phase, self._ann_file, **kwargs)