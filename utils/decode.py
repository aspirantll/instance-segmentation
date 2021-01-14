__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.
    
    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import math
import os
from typing import Iterable

from utils.utils import BBoxTransform, ClipBoxes, generate_coordinates, generate_corner
from utils import image
import cv2
import torch
import torch.nn as nn
from torchvision.ops.boxes import batched_nms
import numpy as np
from utils.visualize import visualize_kp, visualize_box
from utils.nms import py_cpu_nms, boxes_nms
from utils import parell_util

base_dir = r""
target_size = 1

xym = generate_coordinates()


def compute_scale(info):
    return target_size


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


def smooth_polygon(polar_pts, sorted_inds, k=360):
    d_seta = 2*np.pi/12
    selected_inds = []
    cur_ind = -1
    cur_dist = -1
    cur_bin = 0
    for ind in sorted_inds:
        index = math.floor(polar_pts[ind][0]/d_seta)
        if index != cur_bin:
            if cur_ind >= 0:
                selected_inds.append(cur_ind)
            cur_ind = -1
            cur_dist = -1
            cur_bin = index
        elif polar_pts[ind][1]>cur_dist:
            cur_ind = ind
            cur_dist = polar_pts[ind][1]
    if cur_ind >= 0:
        selected_inds.append(cur_ind)
    return selected_inds


def draw_kp_mask(kp_mask, transforms, kp_threshold, infos, keyword):
    cv2.imwrite(r'{}/mask_{}{}'.format(base_dir, keyword, os.path.basename(infos.img_path)), to_numpy(kp_mask)*255)
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
        r'{}/{}_{}{}.png'.format(base_dir, os.path.basename(infos.img_path), keyword, kp_threshold),
        img)
    return img


def draw_instance_map(instance_map, trans_info):
    cv2.imwrite(
        r'{}/{}_{}.png'.format(base_dir, os.path.basename(trans_info.img_path), "instances"),
        instance_map.astype(np.uint16)*1000)


def draw_box(boxes_lt, boxes_rb, boxes_cls, boxes_confs, trans_info):
    img = cv2.imread(trans_info.img_path)
    img = visualize_box(img, ((boxes_rb+boxes_lt)/2).astype(np.int32), (boxes_rb-boxes_lt).astype(np.int32), mask=True, cls_label=boxes_cls, cls_conf=boxes_confs)
    cv2.imwrite(
        r'{}/{}_{}.png'.format(base_dir, os.path.basename(trans_info.img_path), "box"),
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


def decode_ct_hm(conf_mat, cls_mat, wh, num_classes, cls_th, transforms, info):
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
        transformed_centers = transforms.detransform_pixel(centers, info)[:, ::-1]
        scaled_whs = whs * compute_scale(info)
        boxes = np.array([[*(transformed_centers[j] - scaled_whs[:, j]/2), *(transformed_centers[j] + scaled_whs[:, j]/2), confs[j]] for j in range(transformed_centers.shape[0])], dtype=np.float32)
        keep = py_cpu_nms(boxes, thresh=0.5)

        keep_center_cls.extend(cls[keep])
        keep_center_indexes.extend(centers[keep])
        keep_center_confs.extend(confs[keep])
        keep_center_whs.extend(whs[:, keep].T)

    return keep_center_cls, keep_center_indexes, keep_center_confs, keep_center_whs


def group_instance_map(ae_mat, boxes_cls, boxes_confs, boxes_lt, boxes_rb, device):
    """
    group the bounds key points
    :param hm_kp: heat map for key point, 0-1 mask, 2-dims:h*w
    :param hm_ae: heat map for associate embedding, 2-dims:h*w
    :param transforms: transforms for task
    :param center_indexes: the object centers
    :return: the groups
    """
    objs_num = len(boxes_cls)
    h, w = ae_mat.shape[1:]
    xym_s = xym[:, 0:h, 0:w].contiguous().to(device)
    spatial_emb = torch.tanh(ae_mat[0:2, :, :]) + xym_s
    sigma = ae_mat[2:3, :, :]
    center_indexes = ((boxes_lt+boxes_rb)/2).astype(np.int32)
    boxes_wh = (boxes_rb-boxes_lt).astype(np.int32)

    n_boxes_cls, n_boxes_confs, instance_ids = [],[],[]
    instance_map = torch.zeros(h, w, dtype=torch.uint8, device=device)
    conf_map = torch.zeros(h, w, dtype=torch.float32, device=device)
    for i in range(objs_num):
        center_index = center_indexes[i]
        box_wh = boxes_wh[i]

        if box_wh[0] < 2 or box_wh[1] < 2:
            continue

        center = xym_s[:, center_index[0], center_index[1]].view(2, 1, 1)
        lt, rb = generate_corner(center_index, box_wh, h, w, 1.0)
        selected_spatial_emb = spatial_emb[:, lt[0]:rb[0], lt[1]:rb[1]]
        s = torch.exp(sigma[:, center_index[0], center_index[1]])
        dist = torch.exp(-1 * torch.sum(torch.pow(selected_spatial_emb -
                                                  center, 2) * s, 0, keepdim=True)).squeeze()

        proposal = (dist > 0.5)
        # resolve the conflicts
        box_h, box_w = proposal.shape
        area = box_h*box_w
        if proposal.sum().item() < 128 or proposal.sum().item()/area < 0.3:
            continue

        # nms
        instance_map_cut = instance_map[lt[0]:rb[0], lt[1]:rb[1]]
        conf_map_cut = conf_map[lt[0]:rb[0], lt[1]:rb[1]]
        occupied_ids = instance_map_cut.unique().cpu().numpy()
        skip = False
        for occupied_id in occupied_ids:
            if occupied_id == 0:
                continue
            other_proposal = instance_map_cut.eq(occupied_id)
            overlapped_area = other_proposal*proposal
            if overlapped_area.sum().item()/proposal.sum().item() >= 0.5:
                skip = True
                break
            if (conf_map_cut[overlapped_area] >= dist[overlapped_area]).sum() > overlapped_area.sum()/2:
                proposal[overlapped_area] = False

        if skip or proposal.sum().item() < 128 or proposal.sum().item()/area < 0.3:
            continue

        cls_id = boxes_cls[i]
        conf = boxes_confs[i]
        instance_id = i+1

        instance_map[lt[0]:rb[0], lt[1]:rb[1]][proposal] = instance_id
        conf_map[lt[0]:rb[0], lt[1]:rb[1]][proposal] = dist[proposal]

        n_boxes_cls.append(cls_id)
        n_boxes_confs.append(conf)
        instance_ids.append(instance_id)

    return n_boxes_cls, n_boxes_confs, instance_ids, instance_map.cpu().numpy()


def decode_boxes(x, anchors, regression, classification, threshold, iou_threshold):
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]

    dets = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            dets.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            dets.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            dets.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return dets


def decode_single(ae_mat, dets, info, decode_cfg, device):
    cls_ids, boxes, confs = boxes_nms(dets, 0.2)
    if len(cls_ids) == 0:
        return ([], [])
    boxes_lt = boxes[:, :2][:, ::-1]
    boxes_rb = boxes[:, 2:][:, ::-1]

    cls_ids, confs, instance_ids, instance_map = group_instance_map(ae_mat, cls_ids, confs, boxes_lt, boxes_rb, device)
    if decode_cfg.draw_flag:
        draw_box(boxes_lt[:, ::-1], boxes_rb[:, ::-1], cls_ids, confs, info)
        draw_instance_map(instance_map, info)
    return ([e for e in zip(cls_ids, confs, instance_ids)], instance_map)


def decode_output(inputs, outs, infos, decode_cfg, device):
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
    kp_out, regression, classification, anchors = outs
    det_boxes = decode_boxes(inputs, anchors, regression, classification, decode_cfg.cls_th, decode_cfg.iou_th)

    dets = parell_util.multi_apply(decode_single, kp_out[0], det_boxes, infos, decode_cfg=decode_cfg, device=device)

    return dets
