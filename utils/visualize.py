import os

__copyright__ = \
    """
    This code is based on
        # CenterNet (https://github.com/xingyizhou/CenterNet)
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import numpy as np
import cv2


def visualize_instance(img, instances, mask=False, alpha=0.3, gamma=0):
    polygons = []
    for obj in instances:
        poly = np.vstack(obj)
        polygons.append(poly)
    return visualize_objs(img, polygons, mask, alpha, gamma)


def visualize_objs(img, objs, mask=False, alpha=0.3, gamma=0):
    for obj in objs:
        c = ((np.random.random((1, 3)) * 0.6 + 0.4) * 255).tolist()[0]
        obj = obj.astype(np.int32)
        if mask:
            poly_img = img.copy()
            poly_img = cv2.fillPoly(poly_img, [obj.reshape(-1, 2)], c)
            img = cv2.addWeighted(poly_img, alpha, img, 1 - alpha, gamma)
        else:
            img = cv2.polylines(img, [obj.reshape(-1, 1, 2)], isClosed=True, color=c, thickness=2)
    return img


def visualize_kp(img, kps, color=None):
    c = ((np.random.random((1, 3)) * 0.6 + 0.4) * 255).tolist()[0] if color is None else color
    kps = kps.astype(np.int32)
    return cv2.drawKeypoints(img, cv2.KeyPoint_convert(np.array(kps).reshape((-1, 1, 2))), None,
                              color=c)


def visualize_obj_points(kps, centers, path, colors=((255, 0, 0), (0, 0, 255))):
    img = cv2.imread(path)
    visualize_kp(img, kps, colors[0])
    visualize_kp(img, centers.reshape(-1, 2), colors[1])
    return img


def visualize_box(img, centers, box_sizes, center_color=(255, 0, 0), mask=False):
    for center, box_size in zip(centers, box_sizes):
        w, h = box_size//2
        c_w, c_h = center.astype(np.int32)
        img = cv2.circle(img, (c_w, c_h), 1, center_color, 2)

        box = np.array([[c_w - w, c_h - h],
                                          [c_w - w, c_h + h],
                                          [c_w + w, c_h + h],
                                          [c_w + w, c_h - h]]).reshape(-1, 2)
        img = visualize_objs(img, [box], mask)

    return img



