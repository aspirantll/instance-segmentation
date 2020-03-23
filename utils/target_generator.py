__copyright__ = \
    """
    # This code is base on 
    # CenterNet (https://github.com/xingyizhou/CenterNet)
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import numpy as np


def gaussian_radius(det_size, min_overlap=0.8):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(height - x, radius + 1)
    top, bottom = min(y, radius), min(width - y, radius + 1)

    masked_heatmap = heatmap[x - left:x + right, y - top:y + bottom]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def generate_cls_mask(target_size, cls_locations_list, cls_ids_list, box_sizes_list=None, strategy="one-hot"):
    """
    generate the cls mask, the pixel is (w,h)
    :param target_size: tuple(b, h, w, c)
    :param cls_locations_list: list n * 2
    :param cls_ids_list: list n * 1
    :param box_sizes_list: list n * 2
    :param strategy: smoothing, one-hot
    :return: b * c * h * w
    """
    b, c, h, w = target_size
    target_mask = np.zeros(target_size, dtype=np.float32)
    for b_i in range(b):
        cls_locations = cls_locations_list[b_i]
        cls_ids = cls_ids_list[b_i]
        box_sizes = box_sizes_list[b_i]
        for i in range(len(cls_locations)):
            cls_location = cls_locations[i]
            cls_id = cls_ids[i]
            if strategy == "one-hot":
                target_mask[b_i, cls_id, cls_location[0], cls_location[1]] = 1
            elif strategy == "smoothing":
                box_size = box_sizes[i]
                radius = int(max(0, gaussian_radius(box_size)))
                target_mask[b_i, cls_id] = draw_umich_gaussian(target_mask[b_i, cls_id], cls_location, radius)
            else:
                raise ValueError("invalid strategy:{}".format(strategy))
    return target_mask


def generate_kp_mask(target_size, polygons_list, strategy="one-hot"):
    """
    generate the kp mask
    :param target_size: tuple(b, c, h, w)
    :param polygons_list: list(list(ndarray(n*2)))
    :param strategy: smoothing, one-hot
    :return: b* 1 * h * w
    """
    b, c, h, w = target_size
    assert c == 1
    assert b == len(polygons_list)
    target_mask = np.zeros(target_size, dtype=np.float32)
    for b_i in range(b):
        polygons = polygons_list[b_i]
        for polygon in polygons:
            if strategy == "one-hot":
                for pixel in polygon:
                    target_mask[b_i, 0, pixel[0], pixel[1]] = 1
            else:
                raise ValueError("invalid strategy:{}".format(strategy))
    return target_mask

