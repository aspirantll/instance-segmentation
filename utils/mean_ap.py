import numpy as np
from terminaltables import AsciiTable
import data
from utils.image import compute_iou_for_poly, poly_to_mask


def compute_iou_matrix(polygons1, polygons2):
    m = len(polygons1)
    n = len(polygons2)

    iou_mat = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            iou_mat[i, j] = compute_iou_for_poly(polygons1[i], polygons2[j])

    return iou_mat


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def tpfp_default(det_polygons, det_confs, gt_polygons, gt_ignore, iou_thr, area_ranges=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_polygons (list): the detected bbox
        gt_polygons (list): ground truth bboxes of this image
        gt_ignore (list): indicate if gts are ignored for evaluation or not
        iou_thr (float): the iou thresholds

    Returns:
        tuple: (tp, fp), two arrays whose elements are 0 and 1
    """
    # convert to ndarray
    gt_ignore = np.array(gt_ignore)
    det_confs = np.array(det_confs)

    # prepare step
    num_dets = len(det_polygons)
    num_gts = len(gt_polygons)
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if num_gts == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = np.array([poly_to_mask(poly).sum() for poly in det_polygons])
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp
    ious = compute_iou_matrix(det_polygons, gt_polygons)
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)
    sort_inds = np.argsort(-det_confs)
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore, dtype=bool)
        else:
            gt_areas = np.array([poly_to_mask(poly).sum() for poly in gt_polygons])
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore[matched_gt] or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                area = np.array([poly_to_mask(poly).sum() for poly in det_polygons])
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def filter_inds(l, inds):
    assert len(l) == len(inds)
    results = []
    for i in range(len(l)):
        if inds[i]:
            results.append(l[i])
    return results


def flatten_dets(dets):
    flatten_vec = []
    for det in dets:
        flatten_vec.extend(det)

    return flatten_vec


def get_cls_results(det_results, gt_polygons, gt_labels, gt_ignore, class_id):
    """Get det results and gt information of a certain class."""
    cls_dets = [[det[0] for det in dets] for dets in det_results]
    poly_dets = [[det[-1] for det in dets] for dets in det_results]
    conf_dets = [[det[1] for det in dets] for dets in det_results]

    cls_poly_dets = []
    cls_conf_dets = []
    cls_gts = []
    cls_gt_ignore = []
    for ind in range(len(cls_dets)):
        gt_inds = [cls_id == class_id for cls_id in gt_labels[ind]]

        gts_img = filter_inds(gt_polygons[ind], gt_inds)  # gt polys of this class
        cls_gts.append(gts_img)

        if gt_ignore is None:
            cls_gt_ignore.append([0] * len(gts_img))
        else:
            cls_gt_ignore.append(filter_inds(gt_ignore[ind], gt_inds))

        det_inds = [cls_id == class_id for cls_id in cls_dets[ind]]

        cls_poly_dets.append(filter_inds(poly_dets[ind], det_inds))
        cls_conf_dets.append(filter_inds(conf_dets[ind], det_inds))

    return cls_poly_dets, cls_conf_dets, cls_gts, cls_gt_ignore


def eval_map(det_results,
             gt_polygons,
             gt_labels,
             num_classes,
             meters,
             gt_ignore=None,
             scale_ranges=None,
             iou_thr=0.5,
             dataset=None,
             print_summary=True):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list): a list of list, [[cls1_det, cls2_det, ...], ...]
        gt_polygons (list): ground truth bboxes of each image, a list of K*4
            array.
        gt_labels (list): ground truth labels of each image, a list of K array
        num_classes (int): the number of classes, include unlabeled
        meters (list): the length is equal to num_classes
        gt_ignore (list): gt ignore indicators of each image, a list of K array
        scale_ranges (list, optional): [(min1, max1), (min2, max2), ...]
        iou_thr (float): IoU threshold
        dataset (None or str or list): dataset name or dataset classes, there
            are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc.
        print_summary (bool): whether to print the mAP summary

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(gt_polygons) == len(gt_labels)
    assert len(meters) == num_classes
    if gt_ignore is not None:
        assert len(gt_ignore) == len(gt_labels)
        for i in range(len(gt_ignore)):
            assert len(gt_labels[i]) == len(gt_ignore[i])
    area_ranges = ([(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges]
                   if scale_ranges is not None else None)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1

    eval_results = []

    for i in range(1, num_classes):
        # get gt and det bboxes of this class
        poly_dets, conf_dets, cls_gts, cls_gt_ignore = get_cls_results(
            det_results, gt_polygons, gt_labels, gt_ignore, i)
        # calculate tp and fp for each image
        tpfp = [
            tpfp_default(poly_dets[j], conf_dets[j], cls_gts[j], cls_gt_ignore[j], iou_thr,
                         area_ranges) for j in range(len(poly_dets))
        ]
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale, gts ignored or beyond scale
        # are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, polys in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += np.sum(np.logical_not(cls_gt_ignore[j]))
            else:
                gt_areas = np.array([poly_to_mask(poly).sum() for poly in polys])
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum(
                        np.logical_not(cls_gt_ignore[j])
                        & (gt_areas >= min_area) & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        poly_dets = flatten_dets(poly_dets)
        num_dets = len(poly_dets)

        meter = meters[i]
        meter.update(tp, fp, conf_dets, num_dets, num_gts)
        eval_results.append(meter.eval(scale_ranges=scale_ranges, dataset=dataset))
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0
    if print_summary:
        print_map_summary(mean_ap, eval_results, dataset)

    return mean_ap, eval_results


def print_map_summary(mean_ap, results, dataset=None):
    """Print mAP and results of each class.

    Args:
        mean_ap(float): calculated from `eval_map`
        results(list): calculated from `eval_map`
        dataset(None or str or list): dataset name or dataset classes.
    """
    num_scales = len(results[0]['ap']) if isinstance(results[0]['ap'],
                                                     np.ndarray) else 1
    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    precisions = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
            precisions[:, i] = np.array(
                cls_result['precision'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']

    if dataset is None:
        label_names = [str(i) for i in range(1, num_classes + 1)]
    elif isinstance(dataset, str):
        label_names = data.get_cls_names(dataset)
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]
    header = ['class', 'gts', 'dets', 'recall', 'precision', 'ap']
    for i in range(num_scales):
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'],
                '{:.3f}'.format(recalls[i, j]),
                '{:.3f}'.format(precisions[i, j]), '{:.3f}'.format(aps[i, j])
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', '', '{:.3f}'.format(mean_ap[i])])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print(table.table)
