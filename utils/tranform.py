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
from collections import namedtuple
import torch
import numpy as np
from PIL import Image
from utils import image
from utils import cv2_aug_transforms

TransInfo = namedtuple('TransInfo', ['img_path', 'img_size'])

class Normalize(object):
    """Normalize a ``torch.tensor``

    Args:
        inputs (torch.tensor): tensor to be normalized.
        mean: (list): the mean of RGB
        std: (list): the std of RGB

    Returns:
        Tensor: Normalized tensor.
    """
    def __init__(self, div_value, mean, std):
        self.div_value = div_value
        self.mean = mean
        self.std =std

    def __call__(self, inputs):
        inputs = inputs.div(self.div_value)
        for t, m, s in zip(inputs, self.mean, self.std):
            t.sub_(m).div_(s)

        return inputs


class DeNormalize(object):
    """DeNormalize a ``torch.tensor``

    Args:
        inputs (torch.tensor): tensor to be normalized.
        mean: (list): the mean of RGB
        std: (list): the std of RGB

    Returns:
        Tensor: Normalized tensor.
    """
    def __init__(self, div_value, mean, std):
        self.div_value = div_value
        self.mean = mean
        self.std =std

    def __call__(self, inputs):
        result = inputs.clone()
        for i in range(result.size(0)):
            result[i, :, :] = result[i, :, :] * self.std[i] + self.mean[i]

        return result.mul_(self.div_value)


class ToTensor(object):
    """Convert a ``numpy.ndarray or Image`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        inputs (numpy.ndarray or Image): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    def __call__(self, inputs):
        if isinstance(inputs, Image.Image):
            channels = len(inputs.mode)
            inputs = np.array(inputs)
            inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], channels)
            inputs = torch.from_numpy(inputs.transpose(2, 0, 1))
        else:
            inputs = torch.from_numpy(inputs.transpose(2, 0, 1))

        return inputs.float()


class CoordinateReverser(object):
    def __call__(self, label):
        cls_ids, polygons = label
        polygons = [poly[:, ::-1].astype(np.int32) for poly in polygons]
        return cls_ids, polygons


class ReLabel(object):
    """
      255 indicate the background, relabel 255 to some value.
    """
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, inputs):
        assert isinstance(inputs, torch.LongTensor), 'tensor needs to be LongTensor'

        inputs[inputs == self.olabel] = self.nlabel
        return inputs


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs):
        for t in self.transforms:
            inputs = t(inputs)

        return inputs


class CommonTransforms(object):
    def __init__(self, trans_cfg, split="train"):
        self.configer = trans_cfg
        self.aug_trans = cv2_aug_transforms.CV2AugCompose(trans_cfg, split)
        self.img_transform = Compose([
            ToTensor(),
            Normalize(div_value=self.configer.get('normalize', 'div_value'),
                            mean=self.configer.get('normalize', 'mean'),
                            std=self.configer.get('normalize', 'std')), ]
        )
        self.label_transform = Compose([
            CoordinateReverser()
        ])

    def __call__(self, img, label=None, img_path=None):
        """
        compose transform the all the transform
        :param img:  rgb and the shape is h*w*c
        :param label: cls_ids, polygons, the pixel of polygons format as (w,h)
        :param img_path: as the key
        :return:
        """
        img_size = img.shape[:2]
        img, label = self.aug_trans(img, label=label)

        input_tensor = self.img_transform(img)
        if label is not None:
            label = self.label_transform(label)

        return input_tensor, label, TransInfo(img_path, img_size)

    def detransform_pixel(self, pixels, info):
        pixels = pixels.reshape(-1, 2)
        reversed_pixels = pixels[:, ::-1]
        img_size = info.img_size

        if 'resize' in self.configer.get('val_trans', 'trans_seq'):
            if 'scale' in self.configer.get('val_trans', 'resize'):
                scale = self.configer.get('val_trans', 'resize')['scale']
                w_scale_ratio, h_scale_ratio = 1 / scale, 1 / scale
                height, width = img_size
                target_size = (int(round(width * w_scale_ratio)), int(round(height * h_scale_ratio)))
                transform_matrix = image.get_affine_transform(img_size[::-1], target_size, inv=True)
                reversed_pixels = image.apply_affine_transform(reversed_pixels, transform_matrix, img_size[::-1])

        return reversed_pixels

    def tensor_to_image(self, tensor):
        denormalized_tensor = DeNormalize(div_value=self.configer.get('normalize', 'div_value'),
                                          mean=self.configer.get('normalize', 'mean'),
                                          std=self.configer.get('normalize', 'std'))(tensor)
        rgb_img = denormalized_tensor.numpy().transpose(1, 2, 0)
        return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)