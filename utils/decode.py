from typing import Iterable

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
import torch
import torch.nn as nn
import numpy as np
from utils import image
import alphashape
from utils.nms import py_cpu_nms
from utils.kmeans import kmeans

device = None


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


def nms(center_cls, center_confs, center_indexes, groups, img_size, threshold=0.5):
    num = len(center_cls)
    # flag for suppression
    suppression = np.zeros(num)
    # descent order sorted
    orders = np.argsort(- np.array(center_confs))
    # generate mask
    masks = [image.poly_to_mask([list(group.reshape(-1))], img_size) for group in groups]
    for i in range(num - 1):
        cur_index = orders[i]
        if suppression[cur_index] == 1:
            continue
        cur_mask = masks[cur_index]
        for j in range(i + 1, num):
            com_index = orders[j]
            if suppression[com_index] == 1:
                continue
            com_mask = masks[com_index]
            if image.is_cover(cur_mask, com_mask) or image.compute_iou_for_mask(cur_mask, com_mask) > threshold:
                suppression[com_index] = 1
    # filter the suppression
    n_center_cls, n_center_confs, n_center_indexes, n_groups = [], [], [], []
    for k in range(num):
        if suppression[k] == 0:
            n_center_cls.append(center_cls[k])
            n_center_confs.append(center_confs[k])
            n_center_indexes.append(center_indexes[k])
            n_groups.append(groups[k])
    return n_center_cls, n_center_confs, n_center_indexes, n_groups


def aug_group(pts, center_loc, rp=80):
    """
    aug the points
    :param pts: n * 2
    :param center_loc: center location
    :return:
    """
    center_loc = center_loc.reshape(-1)
    # convert to polar, then sort by seta
    internal_point = find_internal_point(pts, center_loc)
    polar_pts = cartesian2polar(pts, internal_point)
    sorted_inds = np.argsort(polar_pts[:, 0])
    sorted_kp = np.array([pts[ind] for ind in sorted_inds])
    # compute distance to center point, then sort
    d_vec = np.power(sorted_kp - center_loc, 2).sum(1)
    filtered_kp = sorted_kp[d_vec > np.percentile(d_vec, rp)]
    return filtered_kp


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


def remove_ghost_boxes(det_tuple, min_pixels):
    """
    filter the low quality segments
    :param det_tuple:
    :param min_pixels:
    :return:
    """
    center_cls, center_confs, center_indexes, kps, preds = det_tuple
    n_center_cls, n_center_confs, n_center_indexes, n_kps, n_preds = [], [], [], [], []
    for i in range(len(center_cls)):
        # obtain the item
        kp = kps[i]
        center_loc = center_indexes[i]

        if len(kp) < min_pixels:
            continue
        # remove the center obj which far from the mean
        bound_polygons = alphashape.alphashape(np.array(kp))
        poly = filter_ghost_polygons(bound_polygons, center_loc)
        if poly is not None:
            n_center_cls.append(center_cls[i])
            n_center_confs.append(center_confs[i])
            n_center_indexes.append(center_indexes[i])
            n_kps.append(poly.reshape(-1, 2))
            n_preds.append(preds[i])
    return n_center_cls, n_center_confs, n_center_indexes, n_kps, n_preds


def draw_kp_mask(kp_mask, transforms, kp_threshold, infos, keyword):
    cv2.imwrite(r'C:\data\checkpoints\test\mask_{}{}'.format(keyword, os.path.basename(infos.img_path)), kp_mask.numpy()*255)
    kp_arr = kp_mask.nonzero().numpy()
    draw_kp(kp_arr, transforms, kp_threshold, infos, keyword)


def draw_kp(kps, transforms, kp_threshold, infos, keyword):
    img = cv2.imread(infos.img_path)
    groups = []
    for kp in kps:
        true_pixel = kp.astype(np.float32)
        # transform to origin image pixel
        true_pixel = transforms.transform_pixel(true_pixel, infos)
        # put to groups
        groups.append(true_pixel)
    img_c = cv2.drawKeypoints(img, cv2.KeyPoint_convert(np.array(groups).reshape((-1, 1, 2))), None,
                              color=(0, 255, 0))
    cv2.imwrite(
        r'C:\data\checkpoints\test\{}_{}{}.png'.format(os.path.basename(infos.img_path), keyword, kp_threshold),
        img_c)


def draw_box(box_sizes, centers, trans_info, transforms):
    save_dir = r"C:\data\checkpoints\test"
    import matplotlib.pyplot as plt
    from PIL import Image
    from utils.visualize import visualize_box
    import os.path as op
    centers = [transforms.transform_pixel(center, trans_info)[0] for center in centers]
    box_sizes = [box_size[::-1] for box_size in box_sizes]

    fig = plt.figure(trans_info.img_path)
    pil_img = Image.open(trans_info.img_path)
    plt.imshow(pil_img)
    visualize_box(centers, box_sizes)
    plt.savefig(op.join(save_dir, "box_{}".format(op.basename(trans_info.img_path))))
    plt.close(fig)


def draw_objs(kp_index, kp_mask, transforms, infos):
    l, c = kp_mask.shape
    for i in range(c):
        c_vec = kp_mask[:, i]
        c_kps = kp_index[c_vec.nonzero().numpy(), :].numpy()
        draw_kp(c_kps, transforms, i, infos, "objs")


def draw_candid(kps, lt, rb, infos, ind):
    img = cv2.imread(infos.img_path)
    cv2.rectangle(img, lt, rb, (255, 0, 0))
    img_c = cv2.drawKeypoints(img, cv2.KeyPoint_convert(kps.reshape((-1, 1, 2))), None,
                              color=(0, 255, 0))
    cv2.imwrite(
        r'C:\data\checkpoints\test\{}_{}{}.png'.format(os.path.basename(infos.img_path), "candid", ind),
        img_c)


def decode_ct_hm(conf_mat, cls_mat, wh, num_classes, cls_th=100):
    cat, height, width = wh.size()
    center_mask = select_points(conf_mat, cls_th)
    center_cls = cls_mat.masked_select(center_mask).numpy()
    center_indexes = center_mask.nonzero().numpy()
    center_confs = conf_mat.masked_select(center_mask).numpy().astype(np.float32)
    center_whs = wh.masked_select(center_mask).numpy().reshape(cat, -1)

    keep_center_cls = []
    keep_center_indexes = []
    keep_center_confs = []
    keep_center_whs = []
    for c_i in range(num_classes):
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


def group_kp(hm_kp, hm_ae, transforms, center_whs, center_indexes, center_cls, center_confs, infos
             , max_distance=10, min_pixels=4, k=1000):
    """
    group the bounds key points
    :param hm_kp: heat map for key point, 0-1 mask, 2-dims:h*w
    :param hm_ae: heat map for associate embedding, 2-dims:h*w
    :param transforms: transforms for task
    :param center_indexes: the object centers
    :return: the groups
    """
    if len(center_indexes) == 0 or hm_kp.sum() == 0:
        return [], [], [], []
    # generate the centers
    centers_vector = torch.from_numpy(np.vstack(center_indexes)).float()
    # handle key point
    kp_mask = select_points(hm_kp, k)
    draw_kp_mask(kp_mask, transforms, k, infos, "bound")

    # clear the non-active part
    correspond_index = kp_mask.nonzero()
    active_ae = hm_ae.masked_select(kp_mask.byte()).reshape(hm_ae.shape[0], -1).t() + correspond_index.float()
    d_matrix = (active_ae.unsqueeze(1) - centers_vector.unsqueeze(0)).pow(2).sum(-1).sqrt()
    correspond_mask = d_matrix < max_distance
    # draw_objs(correspond_index, correspond_mask, transforms, infos)

    # center pixel locations
    center_indexes = [transforms.transform_pixel(center, infos)[0] for center in center_indexes]
    # foreach the active point to group
    kps = []
    kp_preds = []
    for i in range(centers_vector.shape[0]):
        # get the points for center
        kp_pixels = correspond_index[correspond_mask[:, i].nonzero().numpy()[:, 0], :]
        true_pixels = kp_pixels.float()
        # transform to origin image pixel
        true_pixels = transforms.transform_pixel(true_pixels.numpy(), infos)
        # filter the boxes
        h, w = tuple(center_whs[i])
        x, y = tuple(center_indexes[i])
        x_inds = (x - w / 2 - w * 0.1 < true_pixels[:, 0]) * (true_pixels[:, 0] < x + w / 2 + w * 0.1)
        y_inds = (y - h / 2 - h * 0.1 < true_pixels[:, 1]) * (true_pixels[:, 1] < y + h / 2 + h * 0.1)

        draw_candid(true_pixels, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                    infos, i)
        # put to groups, kp_preds
        kps.append(true_pixels[x_inds * y_inds, :])
        kp_preds.append(hm_kp[kp_pixels.numpy()[:, 0], kp_pixels.numpy()[:, 1]])
    # filter some group
    center_cls, center_confs, center_indexes, kps, kp_preds = remove_ghost_boxes(
        (center_cls, center_confs, center_indexes, kps, kp_preds), min_pixels)
    return center_cls, center_confs, center_indexes, kps


def decode_output(outs, infos, transforms, kp_th=10000, cls_th=1000):
    """
    decode the model output
    :param outs:
    :param infos:
    :param kp_k:
    :param cls_th:
    :param min_pixels:
    :return:
    """
    # get output
    hm_cls = torch.sigmoid(outs["hm_cls"])
    hm_kp = torch.sigmoid(outs["hm_kp"])
    hm_ae = outs["hm_ae"]
    hm_wh = outs["hm_wh"]

    # for each to handle the out
    b, cls, h, w = hm_cls.shape
    dets = []
    for b_i in range(b):
        hm_cls_mat = hm_cls[b_i].detach().cpu()
        hm_kp_mat = hm_kp[b_i, 0].detach().cpu()
        hm_ae_mat = hm_ae[b_i].detach().cpu()
        hm_wh_mat = hm_wh[b_i].detach().cpu()
        info = infos[b_i]

        # handle the center point
        max_conf_mat, cls_mat = hm_cls_mat.max(0)
        center_cls, center_indexes, center_confs, center_whs = decode_ct_hm(max_conf_mat, cls_mat, hm_wh_mat, hm_cls_mat.shape[0], cls_th)
        draw_kp(center_indexes, transforms, cls_th, info, "center")
        draw_box(center_whs, center_indexes, info, transforms)

        # group the key points
        center_cls, center_confs, center_indexes, groups = group_kp(hm_kp_mat, hm_ae_mat, transforms, center_whs
                                                                    , center_indexes, center_cls, center_confs, info,
                                                                    k=kp_th)

        # nms
        center_cls, center_confs, center_indexes, groups = nms(center_cls, center_confs, center_indexes, groups,
                                                               info.img_size)
        # append the results
        dets.append([e for e in zip(center_cls, center_confs, center_indexes, groups)])

    return dets
