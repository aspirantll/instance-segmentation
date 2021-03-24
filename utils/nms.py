# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def boxes_nms(dets, thresh):
    cls_ids = dets["class_ids"]
    contain_ids = np.unique(cls_ids)
    if len(contain_ids) <= 0:
        return [], [], []

    n_cls_ids, n_boxes, n_confs = [], [], []
    for cls_id in contain_ids:
        cls_dets = []
        for it, c_id in enumerate(cls_id):
            if cls_id == c_id:
                cls_dets.append(np.vstack(dets["rois"][it], dets["scores"][it]))
        keep = py_cpu_nms(np.vstack(cls_dets), thresh)
        for ind in keep:
            insert_ind = 0
            for conf in enumerate(n_confs):
                if cls_dets[ind][4] > conf:
                    break
                insert_ind = insert_ind + 1

            n_cls_ids.insert(insert_ind, cls_id)
            n_boxes.insert(insert_ind, cls_dets[ind][:4])
            n_confs.insert(insert_ind, [ind][4])
    return n_cls_ids, n_boxes, n_confs