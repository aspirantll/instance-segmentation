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
import math
import torch.nn as nn
import numpy as np

from utils.target_generator import generate_cls_mask, generate_kp_mask, generate_wh_target


def zero_tensor(device):
    return torch.tensor(0, dtype=torch.float32).to(device)


def sigmoid_(tensor):
    return torch.clamp(torch.sigmoid(tensor), min=1e-4, max=1-1e-4)


class WHDLoss(object):

    def __init__(self, device, alpha=0.8, beta=2, th=0.5):
        self._device = device
        self._alpha = alpha
        self._beta = beta
        self._th = th

    @staticmethod
    def get_loss_names():
        return ["kp_pos", "kp_neg", "kp_energy"]

    def __call__(self, hm_kp, targets):
        # prepare step
        hm_kp = sigmoid_(hm_kp)
        kp_targets = targets[-1].to(self._device)
        b, c, h, w = hm_kp.shape
        d_max = math.sqrt(h ** 2 + w ** 2)

        terms_neg = []
        terms_pos = []
        terms_eng = []
        print("kp mean:{}, max:{}, min:{}".format(hm_kp.mean().item(), hm_kp.max().item(), hm_kp.min().item()))
        # foreach every matrix
        for b_i in range(b):
            hm_mat = hm_kp[b_i, 0, :, :]
            # compute probability sum
            p_sum = hm_mat.sum()
            # compute Euclidean distance matrix
            d_matrix = torch.sum(kp_targets[b_i].pow(2), 0).float().sqrt()
            # compute term_1
            terms_neg.append(torch.sum(hm_mat.pow(self._beta) * d_matrix / p_sum))
            # compute term_2
            # expand the prob vector to n*m matrix
            pos_mask = d_matrix == 0
            pos_vec = hm_mat.masked_select(pos_mask)
            d_min_target = (1 - pos_vec).pow(self._beta) * d_max
            terms_pos.append(d_min_target.sum()/torch.clamp(pos_mask.float().sum(), min=1))
            # compute energy
            pt_mask = (hm_mat > self._th) * (1 - pos_mask)
            pt_pred = hm_mat.masked_select(pt_mask)
            energy_sum = (pt_pred - self._th).pow(self._beta).sum() * d_max
            pt_sum = pt_mask.sum()
            if pt_sum == 0:
                terms_eng.append(zero_tensor(self._device))
            else:
                terms_eng.append(energy_sum / pt_sum)
        # compute WHD loss
        term_neg = torch.stack(terms_neg).mean()
        term_pos = torch.stack(terms_pos).mean()
        term_eng = torch.stack(terms_eng).mean()
        return self._alpha * term_pos, (1 - self._alpha) * term_neg, (1 - self._alpha) * term_eng


class WHLoss(object):
    def __init__(self, device, type='smooth_l1'):
        self._device = device
        if type == 'l1':
            self.loss = torch.nn.functional.l1_loss
        elif type == 'smooth_l1':
            self.loss = torch.nn.functional.smooth_l1_loss

    @staticmethod
    def get_loss_names():
        return ["wh"]

    def __call__(self, hm_wh, targets):
        centers_list, _, polygons_list, box_sizes_list, _ = targets
        wh_target, wh_mask = generate_wh_target(hm_wh.shape, centers_list, box_sizes_list)
        wh_target, wh_mask = torch.from_numpy(wh_target).to(self._device), torch.from_numpy(wh_mask).to(self._device)
        loss = self.loss(hm_wh * wh_mask, wh_target * wh_mask, reduction='sum')
        loss = loss / (wh_mask.sum() + 1e-4)
        return [loss]


def sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction="sum"):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    pred = sigmoid_(inputs)
    pt = pred * targets + (1 - pred) * (1 - targets)
    log_pt = torch.log(pt)
    loss_mat = - torch.pow(1 - pt, gamma) * log_pt

    if alpha >= 0:
        loss_mat = alpha * loss_mat * targets + (1 - alpha) * (1-targets) * loss_mat

    pos_loss = loss_mat * targets
    neg_loss = loss_mat * (1-targets)
    if reduction == "mean":
        pos_loss = pos_loss.mean()
        neg_loss = neg_loss.mean()
    elif reduction == "sum":
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

    return pos_loss, neg_loss


class FocalLoss(object):
    def __init__(self, device, alpha=2, beta=4, epsilon=-1, init_loss_normalizer=100):
        self._device = device
        self._alpha = alpha
        self._beta = beta
        self._epsilon = epsilon
        self.loss_normalizer = init_loss_normalizer  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

    def __call__(self, heatmap, targets):
        # prepare step
        pos_mask = targets.eq(1).float()
        pos_loss, neg_loss = sigmoid_focal_loss(heatmap, pos_mask, self._epsilon, self._alpha, reduction="None")
        # weight the negative loss
        neg_loss = torch.pow(1 - targets, self._beta) * neg_loss

        num_pos = pos_mask.sum((1, 2, 3))
        loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + \
                          (1 - self.loss_normalizer_momentum) * num_pos
        self.loss_normalizer = loss_normalizer.mean()

        pos_loss = pos_loss.sum((1, 2, 3)) / torch.clamp_min(loss_normalizer, 1)
        neg_loss = neg_loss.sum((1, 2, 3)) / torch.clamp_min(loss_normalizer, 1)

        return pos_loss.mean(), neg_loss.mean()


class KPFocalLoss(FocalLoss):

    def __init__(self, device, alpha=0.2, beta=2, epsilon=0.9, init_loss_normalizer=100):
        super(KPFocalLoss, self).__init__(device, alpha, beta, epsilon, init_loss_normalizer)

    @staticmethod
    def get_loss_names():
        return ["kp_pos", "kp_neg"]

    def __call__(self, hm_kp, targets):
        # prepare step
        hm_pred = sigmoid_(hm_kp)
        print("kp mean:{}, max:{}, min:{}".format(hm_pred.mean().item(), hm_pred.max().item(), hm_pred.min().item()))
        _, _, polygons_list, _, _ = targets
        kp_mask = torch.from_numpy(generate_kp_mask(hm_kp.shape, polygons_list, strategy="one-hot")).to(self._device)
        return super().__call__(hm_kp, kp_mask)


class ClsFocalLoss(FocalLoss):

    def __init__(self, device, alpha=2, beta=4, epsilon=0.99, init_loss_normalizer=100):
        super(ClsFocalLoss, self).__init__(device, alpha, beta, epsilon, init_loss_normalizer)

    @staticmethod
    def get_loss_names():
        return ["cls_pos", "cls_neg"]

    def __call__(self, hm_cls, targets):
        # prepare step
        hm_pred = sigmoid_(hm_cls)
        print("cls mean:{}, max:{}, min:{}".format(hm_pred.mean().item(), hm_pred.max().item(), hm_pred.min().item()))
        centers_list, cls_ids_list, polygons_list, _, _ = targets
        # handle box size
        box_sizes = [[tuple(polygon.max(0) - polygon.min(0)) for polygon in polygons] for polygons in polygons_list]

        cls_mask = generate_cls_mask(hm_cls.shape, centers_list, cls_ids_list, box_sizes, strategy="smoothing")
        cls_mask = torch.from_numpy(cls_mask).to(self._device)
        return super().__call__(hm_cls, cls_mask)


class AELoss(object):
    def __init__(self, device):
        self._device = device

    def get_loss_names(self):
        return ["ae_loss"]

    def __call__(self, hm_ae, targets):
        """
        :param hm_ae:
        :param targets: (cls_ids, centers,polygons)
        :return:
        """
        # prepare step
        b, c, h, w = hm_ae.shape
        centers_list, _, polygons_list, _, _ = targets
        ae_losses = []
        # foreach every batch
        for b_i in range(b):
            # select the active point
            centers, polygons = centers_list[b_i], polygons_list[b_i]
            n = len(centers)
            ae_loss = zero_tensor(self._device)
            ae_mat = hm_ae[b_i]

            for c_j in range(n):
                center = centers[c_j]
                polygon = polygons[c_j]

                center_tensor = torch.from_numpy(center.astype(np.float32)).to(self._device)
                polygon_tensor = torch.from_numpy(polygon.astype(np.float32)).to(self._device)

                ae_tensor = torch.stack([ae_mat[:, p[0], p[1]] for p in polygon])

                ae_loss += (ae_tensor + polygon_tensor - center_tensor).pow(2).sum(dim=1).sqrt().mean()

            ae_losses.append(ae_loss / max(n, 0))

        # compute mean loss
        ae_loss = torch.stack(ae_losses).mean()
        return [ae_loss]


class ComposeLoss(nn.Module):
    def __init__(self, cls_loss_fn, kp_loss_fn, ae_loss_fn, wh_loss_fn):
        super(ComposeLoss, self).__init__()
        self._cls_loss_fn = cls_loss_fn
        self._kp_loss_fn = kp_loss_fn
        self._ae_loss_fn = ae_loss_fn
        self._wh_loss_fn = wh_loss_fn

        self._loss_names = []
        self._loss_names.extend(cls_loss_fn.get_loss_names())
        self._loss_names.extend(kp_loss_fn.get_loss_names())
        self._loss_names.extend(ae_loss_fn.get_loss_names())
        self._loss_names.extend(wh_loss_fn.get_loss_names())

        self._loss_names.append("total_loss")

    def forward(self, outputs, targets):
        # unpack the output
        hm_cls = outputs["hm_cls"]
        hm_kp = outputs["hm_kp"]
        hm_ae = outputs["hm_ae"]
        hm_wh = outputs["hm_wh"]

        losses = []
        # compute losses
        losses.extend(self._cls_loss_fn(hm_cls, targets))
        losses.extend(self._kp_loss_fn(hm_kp, targets))
        losses.extend(self._ae_loss_fn(hm_ae, targets))
        losses.extend(self._wh_loss_fn(hm_wh, targets))

        # compute total loss
        total_loss = torch.stack(losses).sum()
        losses.append(total_loss)

        return total_loss, {self._loss_names[i]: losses[i] for i in range(len(self._loss_names))}

    def get_loss_states(self):
        return self._loss_names
