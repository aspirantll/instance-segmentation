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
from typing import Iterable
from utils import image
from alphashape import alphashape
import cv2
import torch
import torch.nn as nn
import numpy as np
from utils.visualize import visualize_kp, visualize_box
from utils.nms import py_cpu_nms
from utils.kmeans import kmeans

base_dir = r"E:\checkpoints\test"

def to_numpy(tensor):
    return tensor.cpu().numpy()


def nms_hm(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).byte()
    return keep


def find_internal_point(kps, default):
    """
    find a internal point for poly
    :param kps:
    :return:
    """
    kps = np.array(kps)
    if cv2.pointPolygonTest(kps, tuple(default), False) > 0:
        return default
    mean = kps.mean(axis=0).reshape(-1)
    if cv2.pointPolygonTest(kps, tuple(mean), False) > 0:
        return mean
    for i in range(kps.shape[0]):
        for j in range(1, kps.shape[0]):
            point = (kps[i] + kps[j]) / 2
            if cv2.pointPolygonTest(kps, tuple(point), False) > 0:
                return point
    return default


def select_points(mat, k):
    """
    takes a mat, return max k elements
    :param mat: 2-dims
    :param k:
    :return: (mask)
    """
    h, w = mat.shape
    mask = torch.zeros((h, w), dtype=torch.float32).to(mat.device)

    _, inds = mat.reshape(-1).topk(k)
    for ind in inds:
        mask[ind // w, ind % w] = 1
    selected_mat = (mat * mask).unsqueeze(0)
    return nms_hm(selected_mat).squeeze(0) * mask.byte()


def cartesian2polar(kps, center_loc):
    """
    transform the cartesian to polar
    :param kps:
    :param center_loc:
    :return:
    """
    polar_kps = []
    for pixel in kps:
        d_x, d_y = tuple(pixel - center_loc)
        # compute seta
        if d_x == 0 and d_y > 0:
            seta = np.pi / 2
        elif d_x == 0 and d_y < 0:
            seta = 3 * np.pi / 2
        else:
            seta = np.arctan(d_y / d_x)
            if d_x < 0:
                seta = seta + np.pi
            elif d_x > 0 and d_y < 0:
                seta = seta + 2 * np.pi
        # compute distance
        d = np.sqrt(d_x ** 2 + d_y ** 2)
        # put to result
        polar_kps.append(np.array([[seta, d]], dtype=np.float32))
    return np.vstack(polar_kps)


def polar2cartesian(kps, center_loc):
    """
    transform the polar to cartesian
    :param kps:
    :param center_loc:
    :return:
    """
    n_polar_kp_s = kps[:, 0]
    n_polar_kp_d = kps[:, 1]
    d_x = (n_polar_kp_d * np.cos(n_polar_kp_s)).reshape(-1, 1)
    d_y = (n_polar_kp_d * np.sin(n_polar_kp_s)).reshape(-1, 1)
    delta = np.hstack((d_x, d_y))
    return delta + center_loc


def filter_ghost_polygons(polygons, center):
    if not isinstance(polygons, Iterable):
        polygons = [polygons]
    max_area = 0
    max_poly = None
    for poly in polygons:
        np_poly = np.array(poly.exterior.coords).astype(np.float32)
        if poly.area > max_area and cv2.pointPolygonTest(np_poly, tuple(center), False) > 0:
            max_area = poly.area
            max_poly = np_poly
    return max_poly


def aug_group(pts, center_loc, alpha_ratio=2):
    """
    aug the points
    :param pts: n * 2
    :param center_loc: center location
    :return:
    """
    # h, w = wh
    # length = 2 * 2 * (h + w)
    # n = pts.shape[0]
    # alpha = n / length

    center_loc = center_loc.reshape(-1)
    # convert to polar, then sort by seta
    internal_point = find_internal_point(pts, center_loc)
    polar_pts = cartesian2polar(pts, internal_point)
    sorted_inds = np.argsort(polar_pts[:, 0])
    sorted_kp = np.array([pts[ind] for ind in sorted_inds])

    area = image.poly_to_mask(sorted_kp).sum()
    if area == 0:
        return None

    # n = sorted_kp.shape[0]
    # r = np.max(np.vstack([np.sqrt(np.power(
    #     sorted_kp[(i+1) % n] - sorted_kp[i], 2).sum()) for i in range(n)])) * alpha_ratio
    # bound_polygons = alphashape(pts, 1/r)
    #
    # poly = filter_ghost_polygons(bound_polygons, center_loc)
    # if poly is None and cv2.pointPolygonTest(sorted_kp, tuple(center_loc), False) > 0:
    #     return sorted_kp
    # else:
    #     return poly
    if cv2.pointPolygonTest(sorted_kp, tuple(center_loc), False) > 0:
        return sorted_kp
    else:
        return None


def draw_kp_mask(kp_mask, transforms, kp_threshold, infos, keyword):
    cv2.imwrite(r'{}\mask_{}{}'.format(base_dir, keyword, os.path.basename(infos.img_path)), to_numpy(kp_mask)*255)
    kp_arr = to_numpy(kp_mask.nonzero())
    img = cv2.imread(infos.img_path)
    draw_kp(img, kp_arr, transforms, kp_threshold, infos, keyword)


def draw_kp(img, kps, transforms, kp_threshold, infos, keyword):
    for kp in kps:
        true_pixel = kp.astype(np.float32)
        # transform to origin image pixel
        true_pixel = transforms.detransform_pixel(true_pixel, infos)
        # put to groups
        img = visualize_kp(img, true_pixel)
    cv2.imwrite(
        r'{}\{}_{}{}.png'.format(base_dir, os.path.basename(infos.img_path), keyword, kp_threshold),
        img)
    return img


def draw_box(box_sizes, centers, trans_info, transforms):
    centers = [transforms.detransform_pixel(center, trans_info)[0] for center in centers]
    box_sizes = [box_size[::-1] for box_size in box_sizes]

    img = cv2.imread(trans_info.img_path)
    img = visualize_box(img, centers, box_sizes, mask=True)
    cv2.imwrite(
        r'{}\{}_{}.png'.format(base_dir, os.path.basename(trans_info.img_path), "box"),
        img)


def draw_objs(img, kp_index, kp_mask, transforms, infos):
    l, c = kp_mask.shape
    for i in range(c):
        c_vec = kp_mask[:, i]
        c_kps = to_numpy(kp_index[to_numpy(c_vec.nonzero()), :])
        img = draw_kp(img, c_kps, transforms, i, infos, "objs")

    return img


def draw_candid(kps, lt, rb, img, color):
    cv2.rectangle(img, lt, rb, color)
    return cv2.drawKeypoints(img, cv2.KeyPoint_convert(kps.reshape((-1, 1, 2))), None,
                              color=color)


def decode_ct_hm(conf_mat, cls_mat, wh, num_classes, cls_th=100):
    cat, height, width = wh.size()
    center_mask = select_points(conf_mat, cls_th)
    center_cls = to_numpy(cls_mat.masked_select(center_mask))
    center_indexes = to_numpy(center_mask.nonzero())
    center_confs = to_numpy(conf_mat.masked_select(center_mask)).astype(np.float32)
    center_whs = to_numpy(wh.masked_select(center_mask)).reshape(cat, -1)

    keep_center_cls = []
    keep_center_indexes = []
    keep_center_confs = []
    keep_center_whs = []
    for c_i in range(0, num_classes):
        select_indexes = center_cls == c_i
        if select_indexes.sum() == 0:
            continue

        cls = center_cls[select_indexes]
        confs = center_confs[select_indexes]
        whs = center_whs[:, select_indexes]
        centers = center_indexes[select_indexes, :]
        boxes = np.array([[*(centers[j] - whs[:, j]/2), *(centers[j] + whs[:, j]/2), confs[j]] for j in range(centers.shape[0])], dtype=np.float32)
        keep = py_cpu_nms(boxes, thresh=0.5)

        keep_center_cls.extend(cls[keep])
        keep_center_indexes.extend(centers[keep])
        keep_center_confs.extend(confs[keep])
        keep_center_whs.extend(whs[:, keep].T)

    return keep_center_cls, keep_center_indexes, keep_center_confs, keep_center_whs


def group_kp(hm_kp, hm_ae, transforms, center_whs, center_indexes, center_cls, center_confs, infos, decode_cfg, device):
    """
    group the bounds key points
    :param hm_kp: heat map for key point, 0-1 mask, 2-dims:h*w
    :param hm_ae: heat map for associate embedding, 2-dims:h*w
    :param transforms: transforms for task
    :param center_indexes: the object centers
    :return: the groups
    """
    objs_num = len(center_indexes)
    # handle key point
    kp_mask = select_points(hm_kp, decode_cfg.kp_th)
    if objs_num == 0 or kp_mask.sum() == 0:
        return [], [], [], []

    # generate the centers
    centers_vector = torch.from_numpy(np.vstack(center_indexes)).float()
    if decode_cfg.draw_flag:
        draw_kp_mask(kp_mask, transforms, decode_cfg.kp_th, infos, "bound")

    # clear the non-active part
    allow_distances = np.vstack(center_whs).max(axis=1) * (0.5 + decode_cfg.wh_delta)
    correspond_index = kp_mask.nonzero()
    active_ae = hm_ae.masked_select(kp_mask.byte()).reshape(hm_ae.shape[0], -1).t() + correspond_index.float()
    correspond_vec, corrected_centers = kmeans(active_ae, objs_num, cluster_centers=centers_vector, device=device, allow_distances=allow_distances)

    # center pixel locations
    n_centers = []
    kps = []
    n_clss = []
    n_confs = []
    img = cv2.imread(infos.img_path)
    color = [int(e) for e in np.random.random_integers(0, 256, 3)]
    for i in range(objs_num):
        # filter the boxes
        h, w = tuple(center_whs[i])
        center_loc = center_indexes[i]

        center_loc = transforms.detransform_pixel(center_loc, infos)[0]
        x, y = center_loc[0], center_loc[1]

        # get the points for center
        kp_pixels = correspond_index[to_numpy((correspond_vec == i).nonzero())[:, 0], :]
        true_pixels = to_numpy(kp_pixels.float())
        # transform to origin image pixel
        true_pixels = transforms.detransform_pixel(true_pixels, infos)
        # filter the ghost point
        x_mask = (x - (0.5 + decode_cfg.wh_delta) * w < true_pixels[:, 0]) * (true_pixels[:, 0] < x + (0.5 + decode_cfg.wh_delta) * w)
        y_mask = (y - (0.5 + decode_cfg.wh_delta) * h < true_pixels[:, 1]) * (true_pixels[:, 1] < y + (0.5 + decode_cfg.wh_delta) * h)
        filter_mask = x_mask * y_mask
        # filter_mask = np.ones(kp_pixels.shape[0], dtype=np.bool)
        if filter_mask.sum() < decode_cfg.obj_pixel_th:
            continue

        # augment the groups
        np_poly = aug_group(true_pixels[filter_mask], center_loc, decode_cfg.alpha_ratio)

        if np_poly is not None:
            if decode_cfg.draw_flag:
                img = draw_candid(np_poly, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                            img, (color[0] * (i+1) % 256, color[1] * (i+1) % 256, color[2] * (i+1) % 256))
            # put to groups
            kps.append(np_poly)
            n_centers.append(center_loc)
            n_clss.append(center_cls[i])
            n_confs.append(center_confs[i])
    if decode_cfg.draw_flag:
        cv2.imwrite(
            r'{}\{}_{}.png'.format(base_dir, os.path.basename(infos.img_path), "candid"),
            img)
    return n_clss, n_confs, n_centers, kps


def decode_output(outs, infos, transforms, decode_cfg, device):
    """
    decode the model output
    :param outs:
    :param infos:
    :param transforms:
    :param decode_cfg:
    :param device:
    :return:
    """
    # get output
    hm_cls = torch.sigmoid(outs["hm_cls"])
    hm_kp = torch.sigmoid(outs["hm_kp"])
    ae = outs["ae"]
    wh = outs["wh"]

    # for each to handle the out
    b, cls, h, w = hm_cls.shape
    dets = []
    for b_i in range(b):
        hm_cls_mat = hm_cls[b_i]
        hm_kp_mat = hm_kp[b_i, 0]
        ae_mat = ae[b_i]
        wh_mat = wh[b_i]
        info = infos[b_i]

        # handle the center point
        max_conf_mat, cls_mat = hm_cls_mat.max(0)
        center_cls, center_indexes, center_confs, center_whs = decode_ct_hm(max_conf_mat, cls_mat, wh_mat, hm_cls_mat.shape[0], decode_cfg.cls_th)
        if decode_cfg.draw_flag:
            img = cv2.imread(info.img_path)
            draw_kp(img, center_indexes, transforms, decode_cfg.cls_th, info, "center")
            draw_box(center_whs, center_indexes, info, transforms)

        # group the key points
        center_cls, center_confs, center_indexes, groups = group_kp(hm_kp_mat, ae_mat, transforms, center_whs
                                                                    , center_indexes, center_cls, center_confs, info,
                                                                    decode_cfg, device)

        # append the results
        dets.append([e for e in zip(center_cls, center_confs, center_indexes, groups)])

    return dets
