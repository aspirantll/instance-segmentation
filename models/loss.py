__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.
    
    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import torch
import torch.nn as nn
import numpy as np

from models.lovasz_losses import lovasz_hinge
from utils.target_generator import generate_all_annotations
from utils.utils import generate_coordinates, convert_corner_to_corner


def zero_tensor(device):
    return torch.tensor(0, dtype=torch.float32).to(device)


def to_numpy(tensor):
    return tensor.cpu().numpy()


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


class InstanceLoss(nn.Module):
    def __init__(self, device):
        super(InstanceLoss, self).__init__()
        self._device = device
        self._xym = generate_coordinates().to(device)

    def forward(self, classifications, regressions, anchors, associates, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        sigma_losses = []
        ae_losses = []

        bbox_annotations, instance_ids_list, instance_map_list = annotations
        b, _, h, w = associates.shape
        xym_s = self._xym[:, 0:h, 0:w].contiguous()  # 2 x h x w

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = bbox_annotations[j]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            instance_map = instance_map_list[j]
            instance_ids = instance_ids_list[j]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                alpha_factor = torch.ones_like(classification, device=self._device) * alpha
                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                bce = -(torch.log(1.0 - classification))

                cls_loss = focal_weight * bce

                regression_losses.append(zero_tensor(self._device))
                classification_losses.append(cls_loss.sum())
                sigma_losses.append(zero_tensor(self._device))
                ae_losses.append(zero_tensor(self._device))

                continue

            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            targets = torch.ones_like(classification, device=self._device) * -1

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones_like(targets, device=self._device) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss, device=self._device)

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            regression_loss = zero_tensor(device=self._device)
            sigma_loss = zero_tensor(device=self._device)
            ae_loss = zero_tensor(device=self._device)
            positive_argmax = IoU_argmax[positive_indices]
            positive_regressions = regression[positive_indices]
            positive_ann_indices = positive_argmax.unique()
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

                regression_diff = torch.abs(targets - regression[positive_indices, :4])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )

                spatial_emb = torch.tanh(associates[0:2, :, :]) + xym_s

                # associate embedding loss
                for ann_ind in positive_ann_indices:
                    in_mask = instance_map.eq(instance_ids[ann_ind]).view(1, h, w)  # 1 x h x w

                    anchor_indices = positive_argmax==ann_ind

                    # calculate var loss before exp
                    ann = to_numpy(bbox_annotation[ann_ind])
                    o_lt = ann[0:2][::-1].astype(np.int32)
                    o_rb = ann[2:4][::-1].astype(np.int32)

                    target_sigma = positive_regressions[anchor_indices, 4].mean()

                    sigma_loss = sigma_loss + \
                               torch.mean(
                                   torch.pow(positive_regressions[anchor_indices, 4] - target_sigma.detach(), 2))

                    s = torch.exp(target_sigma)
                    lt, rb = convert_corner_to_corner(o_lt, o_rb, h, w, 1.5)
                    selected_spatial_emb = spatial_emb[0, :, lt[0]:rb[0], lt[1]:rb[1]]
                    label_mask = in_mask[:, lt[0]:rb[0], lt[1]:rb[1]].float()
                    center_index = ((o_lt + o_rb) / 2).astype(np.int32)
                    center = xym_s[:, center_index[0], center_index[1]].view(2, 1, 1)
                    # calculate gaussian
                    dist = torch.exp(-1 * torch.sum(
                        torch.pow(selected_spatial_emb - center, 2) * s, 0, keepdim=True))

                    # apply lovasz-hinge loss
                    ae_loss = ae_loss + \
                                    lovasz_hinge(dist * 2 - 1, label_mask)

            regression_losses.append(regression_loss.mean())
            sigma_losses.append(sigma_loss/max(1, len(positive_ann_indices)))
            ae_losses.append(ae_loss/max(1, len(positive_ann_indices)))

        return [torch.stack(classification_losses).mean(dim=0),
                torch.stack(regression_losses).mean(
                    dim=0) * 50, torch.stack(sigma_losses).mean(dim=0), torch.stack(ae_losses).mean(dim=0)]  # https://github.com/google/automl/blob/6fdd1de778408625c1faf368a327fe36ecd41bf7/efficientdet/hparams_config.py#L233


class ComposeLoss(nn.Module):
    def __init__(self, device):
        super(ComposeLoss, self).__init__()
        self._device = device
        self._loss_names = ["cls_loss", "wh_loss", "sigma_loss", "ae_loss", "total_loss"]
        self.instance_loss = InstanceLoss(device)

    def forward(self, outputs, targets):
        # unpack the output
        kp_out, regression, classification, anchors = outputs
        det_annotations, instance_ids_list, instance_map_list = generate_all_annotations(kp_out.shape, targets, self._device)

        losses = []
        losses.extend(self.instance_loss(classification, regression, anchors, kp_out,
                                         (torch.from_numpy(det_annotations).to(self._device), instance_ids_list, instance_map_list)))

        # compute total loss
        total_loss = torch.stack(losses).sum()
        losses.append(total_loss)

        return total_loss, {self._loss_names[i]: losses[i] for i in range(len(self._loss_names))}

    def get_loss_states(self):
        return self._loss_names
