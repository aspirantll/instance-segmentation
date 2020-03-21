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
from sklearn.utils.extmath import cartesian
from torch.nn import functional as F
import torch.nn as nn

import numpy as np

from utils.target_generator import generate_cls_mask


def zero_tensor(device):
    return torch.tensor(0, dtype=torch.float32).to(device)


def generalize_mean(tensor, dim, p=-2, keepdim=False):
    # """
    # Computes the softmin along some axes.
    # Softmin is the same as -softmax(-x), i.e,
    # softmin(x) = -log(sum_i(exp(-x_i)))

    # The smoothness of the operator is controlled with k:
    # softmin(x) = -log(sum_i(exp(-k*x_i)))/k

    # :param input: Tensor of any dimension.
    # :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    # :param keepdim: (bool) Whether the output tensor has dim retained or not.
    # :param k: (float>0) How similar softmin is to min (the lower the more smooth).
    # """
    # return -torch.log(torch.sum(torch.exp(-k*input), dim, keepdim))/k
    """
    The generalized mean. It corresponds to the minimum when p = -inf.
    https://en.wikipedia.org/wiki/Generalized_mean
    :param tensor: Tensor of any dimension.
    :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
    :param keepdim: (bool) Whether the output tensor has dim retained or not.
    :param p: (float<0).
    """
    assert p < 0
    res = torch.mean((tensor + 1e-6) ** p, dim, keepdim=keepdim) ** (1. / p)
    return res


class WHDLoss(object):

    def __init__(self, device, alpha=0.8, beta=2, epsilon=1e-6):
        self._device = device
        self._alpha = alpha
        self._beta = beta
        self._epsilon = epsilon

    def get_loss_names(self):
        return ["kp_pos", "kp_neg"]

    def __call__(self, hm_kp, targets):
        # prepare step
        hm_kp = torch.sigmoid(hm_kp)
        b, c, h, w = hm_kp.shape
        _, _, polygons_list = targets
        d_max = math.sqrt(h ** 2 + w ** 2)
        all_pixel_locations = torch.from_numpy(
            cartesian([np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32)])).to(self._device)
        terms_1 = []
        terms_2 = []
        print("kp mean:{}, max:{}, min:{}".format(hm_kp.mean().item(), hm_kp.max().item(), hm_kp.min().item()))
        # foreach every matrix
        for b_i in range(b):
            hm_mat = hm_kp[b_i, 0, :, :]
            polygons = polygons_list[b_i]
            # compute probability sum
            p_sum = hm_mat.sum()
            # expand to a vector
            hm_vec = hm_mat.view(-1)

            if len(polygons) == 0:
                terms_1.append(torch.sum(hm_vec.pow(self._beta) * d_max / (p_sum + self._epsilon)))
                terms_2.append(zero_tensor(self._device))
                continue

            # compose key points for the one img
            target_pixel_locations = torch.from_numpy(np.vstack(polygons)).float().to(self._device)
            # compute distance matrix
            diff = all_pixel_locations.unsqueeze(1) - target_pixel_locations.unsqueeze(0)
            # Euclidean distance
            d_matrix = torch.sum(diff.pow(2), -1).float().sqrt()
            # compute term_1
            d_min_all, d_min_index = torch.min(d_matrix, 1)
            terms_1.append(torch.sum(hm_vec.pow(self._beta) * d_min_all / (p_sum + self._epsilon)))
            # compute term_2
            # expand the prob vector to n*m matrix
            w_matrix = hm_vec.view(-1, 1).repeat(1, d_matrix.shape[1])
            weighted_d_matrix = w_matrix.pow(self._beta) * d_matrix + (1 - w_matrix).pow(self._beta) * d_max
            d_min_target = generalize_mean(weighted_d_matrix, dim=0, keepdim=False)
            terms_2.append(torch.mean(d_min_target))
        # compute WHD loss
        term_1 = torch.stack(terms_1).mean()
        term_2 = torch.stack(terms_2).mean()
        return (1 - self._alpha) * term_2, self._alpha * term_1


class FocalLoss(object):

    def __init__(self, device, alpha=2, beta=4, epsilon=1e-6):
        self._device = device
        self._alpha = alpha
        self._beta = beta
        self._epsilon = epsilon

    def get_loss_names(self):
        return ["cls_pos", "cls_neg"]

    def __call__(self, hm_cls, targets):
        # prepare step
        hm_cls = torch.sigmoid(hm_cls)
        print("cls mean:{}, max:{}, min:{}".format(hm_cls.mean().item(), hm_cls.max().item(), hm_cls.min().item()))
        cls_ids_list, centers_list, polygons_list = targets
        # handle box size
        box_sizes = [[tuple(polygon.max(0) - polygon.min(0)) for polygon in polygons] for polygons in polygons_list]

        cls_mask = generate_cls_mask(hm_cls.shape, centers_list, cls_ids_list, box_sizes, strategy="smoothing")
        cls_mask = torch.from_numpy(cls_mask).to(self._device)

        pos_mask = cls_mask.eq(1)
        neg_mask = cls_mask.lt(1)

        num_pos = pos_mask.float().sum()

        pos_hm = hm_cls[pos_mask]
        neg_hm = hm_cls[neg_mask]

        pos_loss = torch.log(pos_hm) * torch.pow(1 - pos_hm, self._alpha)
        neg_loss = torch.log(1 - neg_hm) * torch.pow(neg_hm, self._alpha) * torch.pow(1 - cls_mask[neg_mask], self._beta)

        pos_loss = - pos_loss.sum() / max(1, num_pos)
        neg_loss = - neg_loss.sum() / max(1, num_pos)

        return pos_loss, neg_loss


class AELoss(object):
    def __init__(self, device, alpha=0.5, beta=1, delta=2):
        self._device = device
        self._delta = delta
        self._alpha = alpha
        self._beta = beta

    def get_loss_names(self):
        return ["ae_push", "ae_pull", "ae_center"]

    def __call__(self, hm_ae, targets):
        """
        :param hm_ae:
        :param targets: (cls_ids, centers,polygons)
        :return:
        """
        # prepare step
        b, c, h, w = hm_ae.shape
        _ , centers_list, polygons_list = targets
        # handle the loss
        l_pulls = []
        l_pushs = []
        l_centers = []
        # foreach every batch
        for b_i in range(b):
            # select the active point
            centers, polygons = centers_list[b_i], polygons_list[b_i]
            n = len(centers)
            if n == 0:
                l_pulls.append(zero_tensor(self._device))
                l_pushs.append(zero_tensor(self._device))
                l_centers.append(zero_tensor(self._device))
                continue
            hm_mat = hm_ae[b_i, 0, :, :]
            active_aes = [hm_mat[polygon.T] for polygon in polygons]
            center_aes = hm_mat[np.vstack(centers).T]
            # compute the means
            ae_means = torch.stack([polygon.mean() for polygon in active_aes])
            # compute pull, namely variance
            ae_variances = [(active_aes[i]-ae_means[i]).pow(2).mean() for i in range(n)]
            l_pulls.append(torch.stack(ae_variances).mean())
            # compute push
            if n > 1:
                d_means = (ae_means.unsqueeze(1) - ae_means.unsqueeze(0)).abs()
                d_means = self._delta * (1 - torch.eye(n).to(self._device)) - d_means
                torch.nn.ReLU(inplace=True)(d_means)
                l_pushs.append(d_means.sum() / (n * (n - 1)))
            else:
                l_pushs.append(zero_tensor(self._device))
            # compute center
            center_diff = (center_aes - ae_means).abs()
            l_centers.append(center_diff.mean())
        # compute mean loss
        l_pull = torch.stack(l_pulls).mean()
        l_push = torch.stack(l_pushs).mean()
        l_center = torch.stack(l_centers).mean()
        return self._alpha * l_pull, (1 - self._alpha) * l_push, self._beta * l_center


class ComposeLoss(nn.Module):
    def __init__(self, cls_loss_fn, kp_loss_fn, ae_loss_fn):
        super(ComposeLoss, self).__init__()
        self._cls_loss_fn = cls_loss_fn
        self._kp_loss_fn = kp_loss_fn
        self._ae_loss_fn = ae_loss_fn

        self._loss_names = []
        self._loss_names.extend(cls_loss_fn.get_loss_names())
        self._loss_names.extend(kp_loss_fn.get_loss_names())
        self._loss_names.extend(ae_loss_fn.get_loss_names())
        self._loss_names.append("total_loss")

    def forward(self, outputs, targets):
        # unpack the output
        hm_cls = outputs["hm_cls"]
        hm_kp = outputs["hm_kp"]
        hm_ae = outputs["hm_ae"]
        if hm_cls.requires_grad:
            hm_cls.register_hook(lambda g: print("cls:{}".format(g.mean().item())))
        if hm_kp.requires_grad:
            hm_kp.register_hook(lambda g: print("kp:{}".format(g.mean().item())))
        if hm_ae.requires_grad:
            hm_ae.register_hook(lambda g: print("ae:{}".format(g.mean().item())))

        losses = []
        # compute losses
        losses.extend(self._cls_loss_fn(hm_cls, targets))
        losses.extend(self._kp_loss_fn(hm_kp, targets))
        losses.extend(self._ae_loss_fn(hm_ae, targets))

        # compute total loss
        total_loss = torch.stack(losses).sum()
        losses.append(total_loss)

        return total_loss, {self._loss_names[i]:losses[i] for i in range(len(self._loss_names))}

    def get_loss_states(self):
        return self._loss_names
