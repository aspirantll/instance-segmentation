__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.
    
    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import cv2
import torch
import torch.nn as nn
import numpy as np

from utils.target_generator import generate_all_annotations, generate_kp_mask
from utils.utils import BBoxTransform, ClipBoxes, postprocess, display, generate_coordinates


def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU


class DetFocalLoss(nn.Module):
    def __init__(self):
        super(DetFocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations, **kwargs):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():

                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = alpha_factor.cuda()
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce

                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    classification_losses.append(cls_loss.sum())
                else:

                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce

                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(cls_loss.sum())

                continue

            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            targets = torch.ones_like(classification) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones_like(targets) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss)
            if torch.cuda.is_available():
                zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # efficientdet style
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t()

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))

        # debug
        imgs = kwargs.get('imgs', None)
        if imgs is not None:
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            obj_list = kwargs.get('obj_list', None)
            out = postprocess(imgs.detach(),
                              torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(),
                              classifications.detach(),
                              regressBoxes, clipBoxes,
                              0.5, 0.3)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
            display(out, imgs, obj_list, imshow=False, imwrite=True)

        return [torch.stack(classification_losses).mean(dim=0), \
                torch.stack(regression_losses).mean(
                    dim=0) * 50]  # https://github.com/google/automl/blob/6fdd1de778408625c1faf368a327fe36ecd41bf7/efficientdet/hparams_config.py#L233


def zero_tensor(device):
    return torch.tensor(0, dtype=torch.float32).to(device)


def sigmoid_(tensor):
    return torch.clamp(torch.sigmoid(tensor), min=1e-4, max=1 - 1e-4)


class KPFocalLoss:

    def __init__(self, device):
        self._device = device

    def __call__(self, hm_kp, targets):
        # prepare step
        kp_mask = torch.from_numpy(targets).to(self._device)
        return focal_loss(sigmoid_(hm_kp), kp_mask)


def focal_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    n_pos_loss = pos_loss.sum()
    n_neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - n_neg_loss
    else:
        loss = loss - (n_pos_loss + n_neg_loss) / num_pos

    if torch.isnan(loss):
        raise RuntimeError("loss nan")
    return loss


class AELoss(object):
    def __init__(self, device, weight=1):
        self._device = device
        self._weight = weight
        self._xym = generate_coordinates().to(device)

    def __call__(self, ae, targets):
        """
        :param ae:
        :param targets: (cls_ids, centers,polygons)
        :return:
        """
        # prepare step
        b, c, h, w = ae.shape
        centers_list, polygons_list, kp_mask_list = targets

        xym_s = self._xym[:, 0:h, 0:w].contiguous()  # 2 x h x w

        ae_loss = zero_tensor(self._device)
        for b_i in range(b):
            centers = centers_list[b_i]
            polygons = polygons_list[b_i]
            n = len(centers)

            if n <= 0:
                continue

            spatial_emb = torch.tanh(ae[b_i, 0:2]) + xym_s  # 2 x h x w

            var_loss = zero_tensor(self._device)
            instance_loss = zero_tensor(self._device)

            centers_np = np.vstack(centers)
            centers_tensor = xym_s[:, centers_np[:, 0], centers_np[:, 1]].unsqueeze(1)

            for n_i in range(n):
                kps = polygons[n_i]
                # calculate the delta distance
                selected_emb = spatial_emb[:, kps[:, 0], kps[:, 1]].unsqueeze(2)
                dists = torch.exp(-1 * torch.sum(
                    torch.pow(selected_emb - centers_tensor, 2), 0))  # m x n
                instance_loss += (1-dists[:, n_i]).sum()
                var_loss += nn.functional.l1_loss(dists[:, n_i], torch.max(dists, dim=1)[0], size_average=False)

            ae_loss += (var_loss+instance_loss) / max(n, 1)

        # compute mean loss
        return self._weight * ae_loss / b


class TangentLoss(object):
    def __init__(self, device, weight=1):
        self._device = device
        self._weight = weight

    def __call__(self, inputs, targets):
        # prepare step
        b, c, h, w = inputs.shape
        polygons_list, normal_vector_list = targets

        tan_losses = []
        # foreach every batch
        for b_i in range(b):
            # select the active point
            polygons, normal_vector = polygons_list[b_i], normal_vector_list[b_i]
            n = len(polygons)
            tan_loss = zero_tensor(self._device)
            if n > 0:
                tan_mat = inputs[b_i]
                t_polygons = np.vstack(polygons).transpose()
                t_normals = np.vstack(normal_vector).transpose()
                normal_tensor = torch.from_numpy(t_normals).to(self._device)

                tan_tensor = tan_mat[:, t_polygons[0, :], t_polygons[1, :]]
                tan_tensor = tan_tensor / torch.clamp((tan_tensor * tan_tensor).sum(dim=0).sqrt(), min=1e-4)
                tan_loss += (1 - (normal_tensor * tan_tensor).sum(dim=0)).mean()

            tan_losses.append(tan_loss)

        # compute mean loss
        tan_loss = torch.stack(tan_losses).mean()
        return self._weight * tan_loss


class ComposeLoss(nn.Module):
    def __init__(self, device):
        super(ComposeLoss, self).__init__()
        self._device = device
        self._loss_names = ["cls_loss", "wh_loss", "kp_loss", "ae_loss", "tan_loss", "total_loss"]
        self.det_focal_loss = DetFocalLoss()
        self.kp_loss = KPFocalLoss(device)
        self.ae_loss = AELoss(device)
        self.tan_loss = TangentLoss(device)

    def forward(self, outputs, targets):
        # unpack the output
        kp_out, regression, classification, anchors = outputs
        det_annotations, kp_annotations, ae_annotations, tan_annotations = generate_all_annotations(kp_out[0].shape,
                                                                                                    targets)

        losses = []
        losses.extend(self.det_focal_loss(classification, regression, anchors,
                                          torch.from_numpy(det_annotations).to(self._device)))
        losses.append(self.kp_loss(kp_out[0], kp_annotations))
        losses.append(self.ae_loss(kp_out[1], ae_annotations))
        losses.append(self.tan_loss(kp_out[2], tan_annotations))

        # compute total loss
        total_loss = torch.stack(losses).sum()
        losses.append(total_loss)

        return total_loss, {self._loss_names[i]: losses[i] for i in range(len(self._loss_names))}

    def get_loss_states(self):
        return self._loss_names
