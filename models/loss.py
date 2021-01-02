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
from utils.utils import generate_coordinates


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


class ClsLoss(object):
    def __init__(self, device):
        self._device = device

    def __call__(self, cls, cls_mask):
        cls_mask_tensor = torch.from_numpy(cls_mask).to(self._device)
        return focal_loss(sigmoid_(cls), cls_mask_tensor)


class WHLoss(object):
    def __init__(self, device, type='smooth_l1', weight=0.1):
        self._device = device
        self._weight = weight
        if type == 'l1':
            self.loss = torch.nn.functional.l1_loss
        elif type == 'smooth_l1':
            self.loss = torch.nn.functional.smooth_l1_loss

    def __call__(self, wh, targets):
        wh_target, wh_mask = targets
        wh_target, wh_mask = torch.from_numpy(wh_target).to(self._device), torch.from_numpy(wh_mask).to(self._device)
        loss = self.loss(wh * wh_mask, wh_target * wh_mask, reduction='sum')
        loss = loss / (wh_mask.sum() + 1e-4)
        return self._weight * loss


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
        centers_list, polygons_list = targets

        xym_s = self._xym[:, 0:h, 0:w].contiguous()  # 2 x h x w

        ae_loss = zero_tensor(self._device)
        for b_i in range(b):
            centers = centers_list[b_i]
            polygons = polygons_list[b_i]
            n = len(centers)

            if n <= 0:
                continue

            spatial_emb = torch.tanh(ae[b_i, 0:2]) + xym_s  # 2 x h x w
            sigma = torch.exp(ae[b_i, 2:4])  # n_sigma x h x w

            var_loss = zero_tensor(self._device)
            instance_loss = zero_tensor(self._device)

            centers_np = np.vstack(centers)
            centers_tensor = xym_s[:, centers_np[:, 0], centers_np[:, 1]].unsqueeze(1)

            for n_i in range(n):
                center, kps = centers[n_i].astype(np.int32), polygons[n_i]

                # calculate gaussian
                center_s = xym_s[:, center[0], center[1]].view(2, 1, 1)
                pred = torch.exp(-1 * torch.sum(
                    torch.pow(spatial_emb - center_s, 2) * sigma, 0, keepdim=True))

                mask = torch.from_numpy(generate_kp_mask(kps, (h, w))).view(1, h, w).to(self._device)
                instance_loss += focal_loss(pred, mask)
                del pred

                # calculate the delta distance
                selected_emb = spatial_emb[:, kps[:, 0], kps[:, 1]].unsqueeze(2)
                selected_sigma = sigma[:, kps[:, 0], kps[:, 1]].unsqueeze(2)
                dists = torch.exp(-1 * torch.sum(
                    torch.pow(selected_emb - centers_tensor, 2) * selected_sigma, 0))  # m x n
                var_loss += nn.functional.l1_loss(dists[:, n_i], torch.max(dists, dim=1)[0], size_average=False)
                del dists

            ae_loss += (var_loss + instance_loss) / max(n, 1)

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
        self.cls_loss = ClsLoss(device)
        self.wh_loss = WHLoss(device)
        self.kp_loss = KPFocalLoss(device)
        self.ae_loss = AELoss(device)
        self.tan_loss = TangentLoss(device)

    def forward(self, outputs, targets):
        # unpack the output
        cls_out, wh_out, kp_out, ae_out, tan_out = outputs
        cls_annotations, wh_annotations, kp_annotations, ae_annotations, tan_annotations = generate_all_annotations(cls_out.shape, targets)

        losses = []
        losses.append(self.cls_loss(cls_out, cls_annotations))
        losses.append(self.wh_loss(wh_out, wh_annotations))
        losses.append(self.kp_loss(kp_out, kp_annotations))
        losses.append(self.ae_loss(ae_out, ae_annotations))
        losses.append(self.tan_loss(tan_out, tan_annotations))

        # compute total loss
        total_loss = torch.stack(losses).sum()
        losses.append(total_loss)

        return total_loss, {self._loss_names[i]: losses[i] for i in range(len(self._loss_names))}

    def get_loss_states(self):
        return self._loss_names
