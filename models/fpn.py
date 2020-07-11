import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict

import configs.fpn_cfg as cfg

# Lowest and highest pyramid levels in the backbone network. For FPN, we assume
# that all networks have 5 spatial reductions, each by a factor of 2. Level 1
# would correspond to the input image, hence it does not make sense to use it.
LOWEST_BACKBONE_LVL = 2   # E.g., "conv2"-like level
HIGHEST_BACKBONE_LVL = 5  # E.g., "conv5"-like level

import math
import operator
from functools import reduce

import torch.nn.init as init


def XavierFill(tensor):
    """Caffe2 XavierFill Implementation"""
    size = reduce(operator.mul, tensor.shape, 1)
    fan_in = size / tensor.shape[0]
    scale = math.sqrt(3 / fan_in)
    return init.uniform_(tensor, -scale, scale)


def MSRAFill(tensor):
    """Caffe2 MSRAFill Implementation"""
    size = reduce(operator.mul, tensor.shape, 1)
    fan_out = size / tensor.shape[1]
    scale = math.sqrt(2 / fan_out)
    return init.normal_(tensor, 0, scale)


class AffineChannel2d(nn.Module):
    """ A simple channel-wise affine transformation operation """
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x):
        return x * self.weight.view(1, self.num_features, 1, 1) + \
            self.bias.view(1, self.num_features, 1, 1)

# ---------------------------------------------------------------------------- #
# Bits for specific architectures (ResNet50, ResNet101, ...)
# ---------------------------------------------------------------------------- #

def ResNet50_conv4_body():
    return ResNet_convX_body((3, 4, 6))


def ResNet50_conv5_body():
    return ResNet_convX_body((3, 4, 6, 3))


def ResNet101_conv4_body():
    return ResNet_convX_body((3, 4, 23))


def ResNet101_conv5_body():
    return ResNet_convX_body((3, 4, 23, 3))


def ResNet152_conv5_body():
    return ResNet_convX_body((3, 8, 36, 3))


# ---------------------------------------------------------------------------- #
# Generic ResNet components
# ---------------------------------------------------------------------------- #


class ResNet_convX_body(nn.Module):
    def __init__(self, block_counts):
        super().__init__()
        self.block_counts = block_counts
        self.convX = len(block_counts) + 1
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2

        self.res1 = globals()[cfg.RESNETS.STEM_FUNC]()
        dim_in = 64
        dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
        self.res2, dim_in = add_stage(dim_in, 256, dim_bottleneck, block_counts[0],
                                      dilation=1, stride_init=1)
        self.res3, dim_in = add_stage(dim_in, 512, dim_bottleneck * 2, block_counts[1],
                                      dilation=1, stride_init=2)
        self.res4, dim_in = add_stage(dim_in, 1024, dim_bottleneck * 4, block_counts[2],
                                      dilation=1, stride_init=2)
        if len(block_counts) == 4:
            stride_init = 2 if cfg.RESNETS.RES5_DILATION == 1 else 1
            self.res5, dim_in = add_stage(dim_in, 2048, dim_bottleneck * 8, block_counts[3],
                                          cfg.RESNETS.RES5_DILATION, stride_init)
            self.spatial_scale = 1 / 32 * cfg.RESNETS.RES5_DILATION
        else:
            self.spatial_scale = 1 / 16  # final feature scale wrt. original image scale

        self.dim_out = dim_in

        self._init_modules()

    def _init_modules(self):
        assert cfg.RESNETS.FREEZE_AT in [0, 2, 3, 4, 5]
        assert cfg.RESNETS.FREEZE_AT <= self.convX
        for i in range(1, cfg.RESNETS.FREEZE_AT + 1):
            freeze_params(getattr(self, 'res%d' % i))

        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        if cfg.RESNETS.USE_GN:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.gn1.weight': 'conv1_gn_s',
                'res1.gn1.bias': 'conv1_gn_b',
            }
            orphan_in_detectron = ['pred_w', 'pred_b']
        else:
            mapping_to_detectron = {
                'res1.conv1.weight': 'conv1_w',
                'res1.bn1.weight': 'res_conv1_bn_s',
                'res1.bn1.bias': 'res_conv1_bn_b',
            }
            orphan_in_detectron = ['conv1_b', 'fc1000_w', 'fc1000_b']

        for res_id in range(2, self.convX + 1):
            stage_name = 'res%d' % res_id
            mapping, orphans = residual_stage_detectron_mapping(
                getattr(self, stage_name), stage_name,
                self.block_counts[res_id - 2], res_id)
            mapping_to_detectron.update(mapping)
            orphan_in_detectron.extend(orphans)

        return mapping_to_detectron, orphan_in_detectron

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.RESNETS.FREEZE_AT + 1, self.convX + 1):
            getattr(self, 'res%d' % i).train(mode)

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'res%d' % (i + 1))(x)
        return x


class ResNet_roi_conv5_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
        stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        self.res5, self.dim_out = add_stage(dim_in, 2048, dim_bottleneck * 8, 3,
                                            dilation=1, stride_init=stride_init)
        self.avgpool = nn.AvgPool2d(7)

        self._init_modules()

    def _init_modules(self):
        # Freeze all bn (affine) layers !!!
        self.apply(lambda m: freeze_params(m) if isinstance(m, AffineChannel2d) else None)

    def detectron_weight_mapping(self):
        mapping_to_detectron, orphan_in_detectron = \
          residual_stage_detectron_mapping(self.res5, 'res5', 3, 5)
        return mapping_to_detectron, orphan_in_detectron

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        res5_feat = self.res5(x)
        x = self.avgpool(res5_feat)
        if cfg.MODEL.SHARE_RES5 and self.training:
            return x, res5_feat
        else:
            return x


def add_stage(inplanes, outplanes, innerplanes, nblocks, dilation=1, stride_init=2):
    """Make a stage consist of `nblocks` residual blocks.
    Returns:
        - stage module: an nn.Sequentail module of residual blocks
        - final output dimension
    """
    res_blocks = []
    stride = stride_init
    for _ in range(nblocks):
        res_blocks.append(add_residual_block(
            inplanes, outplanes, innerplanes, dilation, stride
        ))
        inplanes = outplanes
        stride = 1

    return nn.Sequential(*res_blocks), outplanes


def add_residual_block(inplanes, outplanes, innerplanes, dilation, stride):
    """Return a residual block module, including residual connection, """
    if stride != 1 or inplanes != outplanes:
        shortcut_func = globals()[cfg.RESNETS.SHORTCUT_FUNC]
        downsample = shortcut_func(inplanes, outplanes, stride)
    else:
        downsample = None

    trans_func = globals()[cfg.RESNETS.TRANS_FUNC]
    res_block = trans_func(
        inplanes, outplanes, innerplanes, stride,
        dilation=dilation, group=cfg.RESNETS.NUM_GROUPS,
        downsample=downsample)

    return res_block


# ------------------------------------------------------------------------------
# various downsample shortcuts (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_shortcut(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes,
                  outplanes,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        AffineChannel2d(outplanes),
    )


def basic_gn_shortcut(inplanes, outplanes, stride):
    return nn.Sequential(
        nn.Conv2d(inplanes,
                  outplanes,
                  kernel_size=1,
                  stride=stride,
                  bias=False),
        nn.GroupNorm(get_group_gn(outplanes), outplanes,
                     eps=cfg.GROUP_NORM.EPSILON)
    )


# ------------------------------------------------------------------------------
# various stems (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

def basic_bn_stem():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
        ('bn1', AffineChannel2d(64)),
        ('relu', nn.ReLU(inplace=True)),
        # ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True))]))
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))


def basic_gn_stem():
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)),
        ('gn1', nn.GroupNorm(get_group_gn(64), 64,
                             eps=cfg.GROUP_NORM.EPSILON)),
        ('relu', nn.ReLU(inplace=True)),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------

class bottleneck_transformation(nn.Module):
    """ Bottleneck Residual Block """

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 downsample=None):
        super().__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
        self.stride = stride

        self.conv1 = nn.Conv2d(
            inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=False)
        self.bn1 = AffineChannel2d(innerplanes)

        self.conv2 = nn.Conv2d(
            innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False,
            padding=1 * dilation, dilation=dilation, groups=group)
        self.bn2 = AffineChannel2d(innerplanes)

        self.conv3 = nn.Conv2d(
            innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.bn3 = AffineChannel2d(outplanes)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class bottleneck_gn_transformation(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes, innerplanes, stride=1, dilation=1, group=1,
                 downsample=None):
        super().__init__()
        # In original resnet, stride=2 is on 1x1.
        # In fb.torch resnet, stride=2 is on 3x3.
        (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)
        self.stride = stride

        self.conv1 = nn.Conv2d(
            inplanes, innerplanes, kernel_size=1, stride=str1x1, bias=False)
        self.gn1 = nn.GroupNorm(get_group_gn(innerplanes), innerplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.conv2 = nn.Conv2d(
            innerplanes, innerplanes, kernel_size=3, stride=str3x3, bias=False,
            padding=1 * dilation, dilation=dilation, groups=group)
        self.gn2 = nn.GroupNorm(get_group_gn(innerplanes), innerplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.conv3 = nn.Conv2d(
            innerplanes, outplanes, kernel_size=1, stride=1, bias=False)
        self.gn3 = nn.GroupNorm(get_group_gn(outplanes), outplanes,
                                eps=cfg.GROUP_NORM.EPSILON)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ---------------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------------- #

def residual_stage_detectron_mapping(module_ref, module_name, num_blocks, res_id):
    """Construct weight mapping relation for a residual stage with `num_blocks` of
    residual blocks given the stage id: `res_id`
    """
    if cfg.RESNETS.USE_GN:
        norm_suffix = '_gn'
    else:
        norm_suffix = '_bn'
    mapping_to_detectron = {}
    orphan_in_detectron = []
    for blk_id in range(num_blocks):
        detectron_prefix = 'res%d_%d' % (res_id, blk_id)
        my_prefix = '%s.%d' % (module_name, blk_id)

        # residual branch (if downsample is not None)
        if getattr(module_ref[blk_id], 'downsample'):
            dtt_bp = detectron_prefix + '_branch1'  # short for "detectron_branch_prefix"
            mapping_to_detectron[my_prefix
                                 + '.downsample.0.weight'] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix
                                 + '.downsample.1.weight'] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix
                                 + '.downsample.1.bias'] = dtt_bp + norm_suffix + '_b'

        # conv branch
        for i, c in zip([1, 2, 3], ['a', 'b', 'c']):
            dtt_bp = detectron_prefix + '_branch2' + c
            mapping_to_detectron[my_prefix
                                 + '.conv%d.weight' % i] = dtt_bp + '_w'
            orphan_in_detectron.append(dtt_bp + '_b')
            mapping_to_detectron[my_prefix
                                 + '.' + norm_suffix[1:] + '%d.weight' % i] = dtt_bp + norm_suffix + '_s'
            mapping_to_detectron[my_prefix
                                 + '.' + norm_suffix[1:] + '%d.bias' % i] = dtt_bp + norm_suffix + '_b'

    return mapping_to_detectron, orphan_in_detectron


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False


# ---------------------------------------------------------------------------- #
# FPN with ResNet
# ---------------------------------------------------------------------------- #

def get_group_gn(dim):
    """
    get number of groups used by GroupNorm, based on number of channels
    """
    dim_per_gp = cfg.GROUP_NORM.DIM_PER_GP
    num_groups = cfg.GROUP_NORM.NUM_GROUPS

    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0
        group_gn = num_groups
    return group_gn


def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, beta=1.0):
    """
    SmoothL1(x) = 0.5 * x^2 / beta      if |x| < beta
                  |x| - 0.5 * beta      otherwise.
    1 / N * sum_i alpha_out[i] * SmoothL1(alpha_in[i] * (y_hat[i] - y[i])).
    N is the number of batch elements in the input predictions
    """
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < beta).detach().float()
    in_loss_box = smoothL1_sign * 0.5 * torch.pow(in_box_diff, 2) / beta + \
                  (1 - smoothL1_sign) * (abs_in_box_diff - (0.5 * beta))
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    N = loss_box.size(0)  # batch size
    loss_box = loss_box.view(-1).sum(0) / N
    return loss_box


def fpn_ResNet50_conv5_body():
    return fpn(
        ResNet50_conv5_body, fpn_level_info_ResNet50_conv5()
    )


def fpn_ResNet50_conv5_body_bup():
    return fpn(
        ResNet50_conv5_body, fpn_level_info_ResNet50_conv5(),
        panet_buttomup=True
    )


def fpn_ResNet50_conv5_P2only_body():
    return fpn(
        ResNet50_conv5_body,
        fpn_level_info_ResNet50_conv5(),
        P2only=True
    )


def fpn_ResNet101_conv5_body():
    return fpn(
        ResNet101_conv5_body, fpn_level_info_ResNet101_conv5()
    )


def fpn_ResNet101_conv5_P2only_body():
    return fpn(
        ResNet101_conv5_body,
        fpn_level_info_ResNet101_conv5(),
        P2only=True
    )


def fpn_ResNet152_conv5_body():
    return fpn(
        ResNet152_conv5_body, fpn_level_info_ResNet152_conv5()
    )


def fpn_ResNet152_conv5_P2only_body():
    return fpn(
        ResNet152_conv5_body,
        fpn_level_info_ResNet152_conv5(),
        P2only=True
    )


# ---------------------------------------------------------------------------- #
# Functions for bolting FPN onto a backbone architectures
# ---------------------------------------------------------------------------- #
class fpn(nn.Module):
    """Add FPN connections based on the model described in the FPN paper.

    fpn_output_blobs is in reversed order: e.g [fpn5, fpn4, fpn3, fpn2]
    similarly for fpn_level_info.dims: e.g [2048, 1024, 512, 256]
    similarly for spatial_scale: e.g [1/32, 1/16, 1/8, 1/4]
    """
    def __init__(self, conv_body_func, fpn_level_info, P2only=False, panet_buttomup=False):
        super().__init__()
        self.fpn_level_info = fpn_level_info
        self.P2only = P2only
        self.panet_buttomup = panet_buttomup

        self.dim_out = fpn_dim = cfg.FPN.DIM
        min_level, max_level = get_min_max_levels()
        self.num_backbone_stages = len(fpn_level_info.blobs) - (min_level - LOWEST_BACKBONE_LVL)
        fpn_dim_lateral = fpn_level_info.dims
        self.spatial_scale = []  # a list of scales for FPN outputs

        #
        # Step 1: recursively build down starting from the coarsest backbone level
        #
        # For the coarest backbone level: 1x1 conv only seeds recursion
        self.conv_top = nn.Conv2d(fpn_dim_lateral[0], fpn_dim, 1, 1, 0)
        if cfg.FPN.USE_GN:
            self.conv_top = nn.Sequential(
                nn.Conv2d(fpn_dim_lateral[0], fpn_dim, 1, 1, 0, bias=False),
                nn.GroupNorm(get_group_gn(fpn_dim), fpn_dim,
                             eps=cfg.GROUP_NORM.EPSILON)
            )
        else:
            self.conv_top = nn.Conv2d(fpn_dim_lateral[0], fpn_dim, 1, 1, 0)
        self.topdown_lateral_modules = nn.ModuleList()
        self.posthoc_modules = nn.ModuleList()

        # For other levels add top-down and lateral connections
        for i in range(self.num_backbone_stages - 1):
            self.topdown_lateral_modules.append(
                topdown_lateral_module(fpn_dim, fpn_dim_lateral[i+1])
            )

        # Post-hoc scale-specific 3x3 convs
        for i in range(self.num_backbone_stages):
            if cfg.FPN.USE_GN:
                self.posthoc_modules.append(nn.Sequential(
                    nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1, bias=False),
                    nn.GroupNorm(get_group_gn(fpn_dim), fpn_dim,
                                 eps=cfg.GROUP_NORM.EPSILON)
                ))
            else:
                self.posthoc_modules.append(
                    nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1)
                )

            self.spatial_scale.append(fpn_level_info.spatial_scales[i])

        # add for panet buttom-up path
        if self.panet_buttomup:
            self.panet_buttomup_conv1_modules = nn.ModuleList()
            self.panet_buttomup_conv2_modules = nn.ModuleList()
            for i in range(self.num_backbone_stages - 1):
                if cfg.FPN.USE_GN:
                    self.panet_buttomup_conv1_modules.append(nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, 3, 2, 1, bias=True),
                        nn.GroupNorm(get_group_gn(fpn_dim), fpn_dim,
                                    eps=cfg.GROUP_NORM.EPSILON),
                        nn.ReLU(inplace=True)
                    ))
                    self.panet_buttomup_conv2_modules.append(nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1, bias=True),
                        nn.GroupNorm(get_group_gn(fpn_dim), fpn_dim,
                                    eps=cfg.GROUP_NORM.EPSILON),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    self.panet_buttomup_conv1_modules.append(
                        nn.Conv2d(fpn_dim, fpn_dim, 3, 2, 1)
                    )
                    self.panet_buttomup_conv2_modules.append(
                        nn.Conv2d(fpn_dim, fpn_dim, 3, 1, 1)
                    )

                #self.spatial_scale.append(fpn_level_info.spatial_scales[i])


        #
        # Step 2: build up starting from the coarsest backbone level
        #
        # Check if we need the P6 feature map
        if not cfg.FPN.EXTRA_CONV_LEVELS and max_level == HIGHEST_BACKBONE_LVL + 1:
            # Original FPN P6 level implementation from our CVPR'17 FPN paper
            # Use max pooling to simulate stride 2 subsampling
            self.maxpool_p6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
            self.spatial_scale.insert(0, self.spatial_scale[0] * 0.5)

        # Coarser FPN levels introduced for RetinaNet
        if cfg.FPN.EXTRA_CONV_LEVELS and max_level > HIGHEST_BACKBONE_LVL:
            self.extra_pyramid_modules = nn.ModuleList()
            dim_in = fpn_level_info.dims[0]
            for i in range(HIGHEST_BACKBONE_LVL + 1, max_level + 1):
                self.extra_pyramid_modules(
                    nn.Conv2d(dim_in, fpn_dim, 3, 2, 1)
                )
                dim_in = fpn_dim
                self.spatial_scale.insert(0, self.spatial_scale[0] * 0.5)

        if self.P2only:
            # use only the finest level
            self.spatial_scale = self.spatial_scale[-1]

        self._init_weights()

        # Deliberately add conv_body after _init_weights.
        # conv_body has its own _init_weights function
        self.conv_body = conv_body_func()  # e.g resnet

    def _init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Conv2d):
                XavierFill(m.weight)
                #MSRAFill(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        for child_m in self.children():
            if (not isinstance(child_m, nn.ModuleList) or
                not isinstance(child_m[0], topdown_lateral_module)):
                # topdown_lateral_module has its own init method
                child_m.apply(init_func)

    def detectron_weight_mapping(self):
        conv_body_mapping, orphan_in_detectron = self.conv_body.detectron_weight_mapping()
        mapping_to_detectron = {}
        for key, value in conv_body_mapping.items():
            mapping_to_detectron['conv_body.'+key] = value

        d_prefix = 'fpn_inner_' + self.fpn_level_info.blobs[0]
        if cfg.FPN.USE_GN:
            mapping_to_detectron['conv_top.0.weight'] = d_prefix + '_w'
            mapping_to_detectron['conv_top.1.weight'] = d_prefix + '_gn_s'
            mapping_to_detectron['conv_top.1.bias'] = d_prefix + '_gn_b'
        else:
            mapping_to_detectron['conv_top.weight'] = d_prefix + '_w'
            mapping_to_detectron['conv_top.bias'] = d_prefix + '_b'
        for i in range(self.num_backbone_stages - 1):
            p_prefix = 'topdown_lateral_modules.%d.conv_lateral' % i
            d_prefix = 'fpn_inner_' + self.fpn_level_info.blobs[i+1] + '_lateral'
            if cfg.FPN.USE_GN:
                mapping_to_detectron.update({
                    p_prefix + '.0.weight' : d_prefix + '_w',
                    p_prefix + '.1.weight' : d_prefix + '_gn_s',
                    p_prefix + '.1.bias': d_prefix + '_gn_b'
                })
            else:
                mapping_to_detectron.update({
                    p_prefix + '.weight' : d_prefix + '_w',
                    p_prefix + '.bias': d_prefix + '_b'
                })

        for i in range(self.num_backbone_stages):
            p_prefix = 'posthoc_modules.%d' % i
            d_prefix = 'fpn_' + self.fpn_level_info.blobs[i]
            if cfg.FPN.USE_GN:
                mapping_to_detectron.update({
                    p_prefix + '.0.weight': d_prefix + '_w',
                    p_prefix + '.1.weight': d_prefix + '_gn_s',
                    p_prefix + '.1.bias': d_prefix + '_gn_b'
                })
            else:
                mapping_to_detectron.update({
                    p_prefix + '.weight': d_prefix + '_w',
                    p_prefix + '.bias': d_prefix + '_b'
                })

        if hasattr(self, 'extra_pyramid_modules'):
            for i in len(self.extra_pyramid_modules):
                p_prefix = 'extra_pyramid_modules.%d' % i
                d_prefix = 'fpn_%d' % (HIGHEST_BACKBONE_LVL + 1 + i)
                mapping_to_detectron.update({
                    p_prefix + '.weight': d_prefix + '_w',
                    p_prefix + '.bias': d_prefix + '_b'
                })

        return mapping_to_detectron, orphan_in_detectron

    def forward(self, x):
        conv_body_blobs = [self.conv_body.res1(x)]
        for i in range(1, self.conv_body.convX):
            conv_body_blobs.append(
                getattr(self.conv_body, 'res%d' % (i+1))(conv_body_blobs[-1])
            )
        fpn_inner_blobs = [self.conv_top(conv_body_blobs[-1])]
        for i in range(self.num_backbone_stages - 1):
            fpn_inner_blobs.append(
                self.topdown_lateral_modules[i](fpn_inner_blobs[-1], conv_body_blobs[-(i+2)])
            )
        fpn_output_blobs = []
        if self.panet_buttomup:
            fpn_middle_blobs = []
        for i in range(self.num_backbone_stages):
            if not self.panet_buttomup:
                fpn_output_blobs.append(
                    self.posthoc_modules[i](fpn_inner_blobs[i])
                )
            else:
                fpn_middle_blobs.append(
                    self.posthoc_modules[i](fpn_inner_blobs[i])
                )
        if self.panet_buttomup:
            fpn_output_blobs.append(fpn_middle_blobs[-1])
            for i in range(2, self.num_backbone_stages + 1):
                fpn_tmp = self.panet_buttomup_conv1_modules[i - 2](fpn_output_blobs[0])
                #print(fpn_middle_blobs[self.num_backbone_stages - i].size())
                fpn_tmp = fpn_tmp + fpn_middle_blobs[self.num_backbone_stages - i]
                fpn_tmp = self.panet_buttomup_conv2_modules[i - 2](fpn_tmp)
                fpn_output_blobs.insert(0, fpn_tmp)        

        if hasattr(self, 'maxpool_p6'):
            fpn_output_blobs.insert(0, self.maxpool_p6(fpn_output_blobs[0]))

        if hasattr(self, 'extra_pyramid_modules'):
            blob_in = conv_body_blobs[-1]
            fpn_output_blobs.insert(0, self.extra_pyramid_modules(blob_in))
            for module in self.extra_pyramid_modules[1:]:
                fpn_output_blobs.insert(0, module(F.relu(fpn_output_blobs[0], inplace=True)))

        if self.P2only:
            # use only the finest level
            return fpn_output_blobs[-1]
        else:
            # use all levels
            return fpn_output_blobs


class topdown_lateral_module(nn.Module):
    """Add a top-down lateral module."""
    def __init__(self, dim_in_top, dim_in_lateral):
        super().__init__()
        self.dim_in_top = dim_in_top
        self.dim_in_lateral = dim_in_lateral
        self.dim_out = dim_in_top
        if cfg.FPN.USE_GN:
            self.conv_lateral = nn.Sequential(
                nn.Conv2d(dim_in_lateral, self.dim_out, 1, 1, 0, bias=False),
                nn.GroupNorm(get_group_gn(self.dim_out), self.dim_out,
                             eps=cfg.GROUP_NORM.EPSILON)
            )
        else:
            self.conv_lateral = nn.Conv2d(dim_in_lateral, self.dim_out, 1, 1, 0)

        self._init_weights()

    def _init_weights(self):
        if cfg.FPN.USE_GN:
            conv = self.conv_lateral[0]
        else:
            conv = self.conv_lateral

        if cfg.FPN.ZERO_INIT_LATERAL:
            init.constant_(conv.weight, 0)
        else:
            XavierFill(conv.weight)
        if conv.bias is not None:
            init.constant_(conv.bias, 0)

    def forward(self, top_blob, lateral_blob):
        # Lateral 1x1 conv
        lat = self.conv_lateral(lateral_blob)
        # Top-down 2x upsampling
        # td = F.upsample(top_blob, size=lat.size()[2:], mode='bilinear')
        td = F.upsample(top_blob, scale_factor=2, mode='nearest')
        # Sum lateral and top-down
        return lat + td


def get_min_max_levels():
    """The min and max FPN levels required for supporting RPN and/or RoI
    transform operations on multiple FPN levels.
    """
    min_level = LOWEST_BACKBONE_LVL
    max_level = HIGHEST_BACKBONE_LVL
    if cfg.FPN.MULTILEVEL_RPN and not cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.RPN_MAX_LEVEL
        min_level = cfg.FPN.RPN_MIN_LEVEL
    if not cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.ROI_MAX_LEVEL
        min_level = cfg.FPN.ROI_MIN_LEVEL
    if cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = max(cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.ROI_MAX_LEVEL)
        min_level = min(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.ROI_MIN_LEVEL)
    return min_level, max_level

# ---------------------------------------------------------------------------- #
# FPN level info for stages 5, 4, 3, 2 for select models (more can be added)
# ---------------------------------------------------------------------------- #

FpnLevelInfo = collections.namedtuple(
    'FpnLevelInfo',
    ['blobs', 'dims', 'spatial_scales']
)


def fpn_level_info_ResNet50_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_5_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_ResNet101_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_22_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )


def fpn_level_info_ResNet152_conv5():
    return FpnLevelInfo(
        blobs=('res5_2_sum', 'res4_35_sum', 'res3_7_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.)
    )
