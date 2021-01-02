import math
import torch
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_
import numpy as np

from .efficientnet import EfficientNet as EffNet


class EfficientNet(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, compound_coef, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{compound_coef}', load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.
        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps


def variance_scaling_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""
    initializer for SeparableConv in Regressor/Classifier
    reference: https://keras.io/zh/initializers/  VarianceScaling
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return _no_grad_normal_(tensor, 0., std)


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )


class EfficientDecoder(nn.Module):
    def __init__(self, channels, headers, concat_input=True):
        super().__init__()
        self.headers = headers
        self.concat_input = concat_input

        self.up_conv1 = up_conv(channels[0], 256)
        self.double_conv1 = double_conv(channels[1]+256, 256)
        self.up_conv2 = up_conv(256, 128)
        self.double_conv2 = double_conv(channels[2]+128, 128)
        self.up_conv3 = up_conv(128, 64)
        self.double_conv3 = double_conv(channels[3]+64, 64)
        self.up_conv4 = up_conv(64, 32)
        self.double_conv4 = double_conv(channels[4]+32, 32)

        if self.concat_input:
            self.up_conv_input = up_conv(32, 16)
            self.double_conv_input = double_conv(3+16, 16)

        for header, channel in self.headers.items():
            head_conv = nn.Conv2d(16, channel, kernel_size=1)
            self.__setattr__(header, head_conv)

    def forward(self, input_, blocks):
        x = blocks[-1]

        x = self.up_conv1(x)
        x = torch.cat([x, blocks[-2]], dim=1)
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        x = torch.cat([x, blocks[-3]], dim=1)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        x = torch.cat([x, blocks[-4]], dim=1)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        x = torch.cat([x, blocks[-5]], dim=1)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)

        outs = []
        for header in self.headers.keys():
            header_conv = self.__getattr__(header)
            outs.append(header_conv(x))

        return tuple(outs)


class EfficientSeg(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientSeg, self).__init__()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))

        self.num_classes = num_classes
        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

        channels = {
            0: [320, 112, 40, 24, 16],
            1: [320, 112, 40, 24, 16],
            2: [352, 120, 48, 24, 16],
            3: [384, 136, 48, 32, 24],
            4: [448, 160, 56, 32, 24],
            5: [512, 176, 64],
            6: [576, 200, 72],
            7: [576, 200, 72],
            8: [640, 224, 80],
        }

        self.obj_header = EfficientDecoder(channels[compound_coef], {"cls": num_classes, "wh": 2})
        self.ae_header = EfficientDecoder(channels[compound_coef], {"kp": 1, "ae": 4, "tan": 2})

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        blocks = self.backbone_net(inputs)

        cls_out, wh_out = self.obj_header(inputs, blocks)
        kp_out, ae_out, tan_out = self.ae_header(inputs, blocks)
        return cls_out, wh_out, kp_out, ae_out, tan_out

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')

    def init_weight(self):
        for name, module in self.named_modules():
            is_conv_layer = isinstance(module, nn.Conv2d)

            if is_conv_layer:
                if "conv_list" or "header" in name:
                    variance_scaling_(module.weight.data)
                else:
                    nn.init.kaiming_uniform_(module.weight.data)

                if module.bias is not None:
                    if "classifier.header" in name:
                        bias_value = -np.log((1 - 0.01) / 0.01)
                        torch.nn.init.constant_(module.bias, bias_value)
                    else:
                        module.bias.data.zero_()