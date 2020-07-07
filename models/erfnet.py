"""
this code base on
# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput,
                              (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1*dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1*dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output+input)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder (nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.mid_out = None
        self.heads = heads

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        for head, dims in self.heads.items():
            output_conv = nn.ConvTranspose2d(
                16, dims, 2, stride=2, padding=0, output_padding=0, bias=True)
            self.__setattr__(head, output_conv)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)
        self.mid_out = output
        outs = {}
        for head, _ in self.heads.items():
            output_conv = self.__getattr__(head)
            outs[head] = output_conv(self.mid_out)

        return outs

# ERFNet


class ERFNet(nn.Module):
    def __init__(self, num_classes, fixed_parts=None):  # use encoder to pass pretrained encoder
        super().__init__()
        self.encoder = Encoder(num_classes)

        # Decoders for cls and box size
        heads_cb = {
            "hm_cls": num_classes,
            "wh": 2
        }
        self.decoder_cb = Decoder(heads_cb)

        # Decoders for boundary
        heads_ak = {
            "hm_kp": 1,
            "ae": 2
        }
        self.decoder_ak = Decoder(heads_ak)

        if fixed_parts is not None:
            for part_name in fixed_parts:
                part_m = self.__getattr__(part_name)
                for p in part_m.parameters():
                    p.requires_grad = False

    def forward(self, input):
        features = self.encoder(input)  # predict=False by default
        outs = {}
        outs.update(self.decoder_ak(features))
        outs.update(self.decoder_cb(features))
        return outs

    def init_weight(self):
        def init_model_weights(layers, method="xavier", std=0.01):
            """
            init the weights for model
            :param layers:
            :param method: kaiming, xavier, None
            :param std: for normalize
            :param f: use focal loss
            :param pi: focal loss, foreground/background
            :return:
            """
            for m in layers.modules():
                if isinstance(m, torch.nn.Conv2d):
                    if method == "kaiming":
                        torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                    elif method == "xavier":
                        torch.nn.init.xavier_normal_(m.weight.data)
                    elif method == "uniform":
                        torch.nn.init.uniform_(m.weight)
                    elif method == "normal":
                        torch.nn.init.normal_(m.weight, std=std)

                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)

        # decoder_ak
        init_model_weights(self.decoder_ak, method="normal", std=0.01)
        # decoder_cb
        init_model_weights(self.decoder_cb, method="normal", std=0.01)

    def load_pretrained_weight(self):
        pass
