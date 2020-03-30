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
import torch.nn.functional as F
import numpy as np

from utils.tensor_util import unitize_redirection
from utils.target_generator import generate_cls_mask, generate_kp_mask, generate_batch_sdf


def zero_tensor(device):
    return torch.tensor(0, dtype=torch.float32).to(device)


def sigmoid_(tensor):
    return torch.clamp(torch.sigmoid(tensor), min=1e-4, max=1-1e-4)


def grad_img(img, gx=True, gy=True):
    if gx:
        x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        grad_x = F.conv2d(img, x_kernel, padding=1)
    if gy:
        y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        grad_y = F.conv2d(img, y_kernel, padding=1)
    if gx and not gy:
        return grad_x
    elif not gx and gy:
        return grad_y
    else:
        return grad_x, grad_y


def regular_item(u):
    mask = (u==0).float()
    du_dx, du_dy = grad_img(u)
    mu = torch.sqrt(du_dx ** 2 + du_dy ** 2)
    item_mask = (mu < 1).float()
    item1 = 1/(2 * np.pi)**2 * (1-torch.cos(2 * np.pi * mu))
    item2 = 1/2 * (mu - 1)**2
    p_mat = item1 * item_mask + item2 * (1 - item_mask)
    # return (p_mat * (1-mask)).mean()
    return torch.tensor(0, dtype=torch.float32)


class GACFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, gt):
        ctx.save_for_backward(inputs, gt)
        contour_mask = (inputs == 0).float()
        return (contour_mask != gt).float().sum()

    @staticmethod
    def backward(ctx, grad_output):
        u, i = ctx.saved_tensors

        g = torch.exp(- i * 1000)
        dg_dx, dg_dy = grad_img(g)

        du_dx, du_dy = grad_img(u)
        mu = torch.sqrt(du_dx**2 + du_dy**2)
        unit_du_dx = du_dx / (mu + 1e-6)
        unit_du_dy = du_dy / (mu + 1e-6)
        k = grad_img(unit_du_dx, gy=False) + grad_img(unit_du_dy, gx=False)
        return (g * mu * k + dg_dx * du_dx + dg_dy * du_dy) * grad_output, grad_output


class KPGACLoss(object):

    def __init__(self, device):
        self._device = device

    @staticmethod
    def get_loss_names():
        return ["kp_energy", "kp_regular"]

    def __call__(self, hm_kp, targets):
        # prepare step
        b, c, h, w = hm_kp.shape
        print("kp mean:{}, max:{}, min:{}".format(hm_kp.mean().item(), hm_kp.max().item(), hm_kp.min().item()))
        _, _, polygons_list = targets
        # generate the kp mask
        kp_mask = generate_kp_mask((b, c, h, w), polygons_list, strategy="one-hot")
        kp_mask = torch.from_numpy(kp_mask).to(self._device)
        GACFunction.apply(hm_kp, kp_mask)
        from utils.visualize import visualize_hm
        from matplotlib import pyplot as plt
        fig = plt.figure()
        visualize_hm(hm_kp[0, 0].detach())
        plt.savefig(r"C:\data\temp\kp.png")
        plt.close(fig)
        return GACFunction.apply(hm_kp, kp_mask), regular_item(hm_kp)


class KPLSLoss(object):
    def __init__(self, device):
        self._device = device
        self._mse = nn.MSELoss(reduce=True, size_average=False, reduction="sum")

    @staticmethod
    def get_loss_names():
        return ["kp"]

    def __call__(self, hm_kp, targets):
        # prepare step
        print("kp mean:{}, max:{}, min:{}".format(hm_kp.mean().item(), hm_kp.max().item(), hm_kp.min().item()))
        kp_target = targets[3].to(self._device)
        # unitize the output
        unitized_kp = unitize_redirection(hm_kp)
        # compute cosine similarity
        loss = self._mse(unitized_kp, kp_target)

        # from utils.visualize import visualize_hm
        # from matplotlib import pyplot as plt
        # fig = plt.figure()
        # visualize_hm((kp_target[0]*kp_target[0]).sum(0).detach())
        # plt.savefig(r"C:\data\temp\kp_true.png")
        # plt.close(fig)
        # fig = plt.figure()
        # visualize_hm((unitized_kp[0] * unitized_kp[0]).sum(0).detach())
        # plt.savefig(r"C:\data\temp\kp.png")
        # plt.close(fig)
        # fig = plt.figure()
        # visualize_hm(loss[0].detach())
        # plt.savefig(r"C:\data\temp\loss.png")
        # plt.close(fig)
        # fig = plt.figure()
        # visualize_hm(1 - loss[0].detach())
        # plt.savefig(r"C:\data\temp\sim.png")
        # plt.close(fig)

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
        kp_mask = targets.to(self._device)
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
        centers_list, cls_ids_list, polygons_list, _ = targets
        # handle box size
        box_sizes = [[tuple(polygon.max(0) - polygon.min(0)) for polygon in polygons] for polygons in polygons_list]

        cls_mask = generate_cls_mask(hm_cls.shape, centers_list, cls_ids_list, box_sizes, strategy="smoothing")
        cls_mask = torch.from_numpy(cls_mask).to(self._device)
        return super().__call__(hm_cls, cls_mask)


class AELoss(object):
    def __init__(self, device, alpha=0.5, beta=1, delta=2):
        self._device = device
        self._delta = delta
        self._alpha = alpha
        self._beta = beta

    def get_loss_names(self):
        return ["ae_pull", "ae_push", "ae_center"]

    def __call__(self, hm_ae, targets):
        """
        :param hm_ae:
        :param targets: (cls_ids, centers,polygons)
        :return:
        """
        # prepare step
        b, c, h, w = hm_ae.shape
        centers_list , _, polygons_list, _ = targets
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
            ae_variances = [(active_aes[i]-ae_means[i]).abs().mean() for i in range(n)]
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
