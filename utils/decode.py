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


def remove_ghost_boxes(det_tuple, min_pixels, max_delta):
    """
    filter the low quality segments
    :param det_tuple:
    :param min_pixels:
    :return:
    """
    center_cls, center_confs, center_indexes, kps, preds = det_tuple
    n_center_cls, n_center_confs, n_center_indexes, n_kps, n_preds = [], [], [], [], []
    for i in range(center_cls.shape[0]):
        # obtain the item
        kp = kps[i]
        center_loc = center_indexes[i]

        if len(kp) < min_pixels:
            continue
        # remove the center obj which far from the mean
        # kp = aug_group(np.array(kp), center_loc)
        kp = cv2.convexHull(np.array(kp))
        # delta = cv2.pointPolygonTest(kp, tuple(center_loc), True)
        # if delta < - max_delta:
        #     continue
        n_center_cls.append(center_cls[i])
        n_center_confs.append(center_confs[i])
        n_center_indexes.append(center_indexes[i])
        n_kps.append(kp.reshape(-1, 2))
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


def draw_objs(kp_index, kp_mask, transforms, infos):
    l, c = kp_mask.shape
    for i in range(c):
        c_vec = kp_mask[:, i]
        c_kps = kp_index[c_vec.nonzero().numpy(), :].numpy()
        draw_kp(c_kps, transforms, i, infos, "objs")


def group_kp(hm_kp, hm_ae, transforms, center_indexes, center_cls, center_confs, infos
             , max_distance=0.2, min_pixels=4, max_delta=10, k=1000):
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
    centers_vector = torch.tensor([hm_ae[center[0], center[1]] for center in center_indexes])
    # handle key point
    kp_mask = select_points(hm_kp, k)
    draw_kp_mask(kp_mask, transforms, k, infos, "bound")
    # from utils.visualize import visualize_hm
    # from matplotlib import pyplot as plt
    # fig = plt.figure("ae")
    # visualize_hm(kp_mask.float() * hm_ae)
    # plt.savefig(r'C:\data\checkpoints\test\ae_{}'.format(os.path.basename(infos.img_path)))
    # plt.close(fig)

    # clear the non-active part
    active_ae = hm_ae.masked_select(kp_mask.byte())
    correspond_index = kp_mask.nonzero()
    # compute the distance
    d_matrix = (active_ae.unsqueeze(1) - centers_vector.unsqueeze(0)).abs()
    correspond_mask = d_matrix < max_distance
    # draw_objs(correspond_index, correspond_mask, transforms, infos)
    # foreach the active point to group
    kps = []
    kp_preds = []
    for i in range(centers_vector.shape[0]):
        # get the points for center
        kp_pixels = correspond_index[correspond_mask[:, i].nonzero().numpy(), :]
        true_pixels = kp_pixels.float()
        # transform to origin image pixel
        true_pixels = transforms.transform_pixel(true_pixels.numpy(), infos)
        # put to groups, kp_preds
        kps.append(true_pixels)
        kp_preds.append(hm_kp[kp_pixels.numpy().T])
    # center pixel locations
    center_indexes = [transforms.transform_pixel(center, infos)[0] for center in center_indexes]
    # filter some group
    center_cls, center_confs, center_indexes, kps, kp_preds = remove_ghost_boxes(
        (center_cls, center_confs, center_indexes, kps, kp_preds), min_pixels, max_delta)
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

    # for each to handle the out
    b, cls, h, w = hm_cls.shape
    dets = []
    for b_i in range(b):
        hm_cls_mat = hm_cls[b_i].detach().cpu()
        hm_kp_mat = hm_kp[b_i, 0].detach().cpu()
        hm_ae_mat = hm_ae[b_i, 0].detach().cpu()
        info = infos[b_i]

        # handle the center point
        max_conf_mat, cls_mat = hm_cls_mat.max(0)

        center_mask = select_points(max_conf_mat, cls_th)
        draw_kp_mask(center_mask, transforms, cls_th, info, "center")
        center_cls = cls_mat.masked_select(center_mask).numpy()
        center_indexes = center_mask.nonzero().numpy()
        center_confs = max_conf_mat.masked_select(center_mask).numpy().astype(np.float32)

        # group the key points
        center_cls, center_confs, center_indexes, groups = group_kp(hm_kp_mat, hm_ae_mat, transforms
                                                                    , center_indexes, center_cls, center_confs, info,
                                                                    k=kp_th)
        for i in range(len(center_indexes)):
            from utils.visualize import visualize_obj_points
            from matplotlib import pyplot as plt
            fig = plt.figure(str(i) + info.img_path)
            visualize_obj_points(groups[i], center_indexes[i], info.img_path, i)
            plt.close(fig)
        # nms
        center_cls, center_confs, center_indexes, groups = nms(center_cls, center_confs, center_indexes, groups,
                                                               info.img_size)
        # append the results
        dets.append([e for e in zip(center_cls, center_confs, center_indexes, groups)])

    return dets
