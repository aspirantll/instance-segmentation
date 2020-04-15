__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

from collections import namedtuple
import cv2
import torch
import numpy as np
from torchvision.transforms import transforms
import torchvision.transforms.functional as F
from utils import image
from utils.target_generator import generate_cls_mask, generate_kp_mask, generate_batch_sdf

TransInfo = namedtuple('TransInfo', ['img_path', 'img_size', 'flipped_flag'])


class CommonTransforms(object):
    def __init__(self, input_size, num_cls=None, kp=True):
        if isinstance(input_size, tuple) or isinstance(input_size, list):
            input_size = np.array(input_size)
        self._input_size = input_size
        self._num_cls = num_cls
        self._kp = kp

    def regular_size(self, img_size):
        size_flag = img_size[1] > img_size[0]
        origin_size = img_size[::-1] if size_flag else img_size
        input_size = self._input_size[::-1] if size_flag else self._input_size
        return size_flag, origin_size, input_size

    def __call__(self, img, label=None, img_path=None, from_file=False):
        """
        compose transform the all the transform
        :param img:  rgb and the shape is h*w*c
        :param label: cls_ids, polygons, the pixel of polygons format as (w,h)
        :param img_path: as the key
        :return:
        """
        if from_file:
            img_size = np.array(img.shape[1:3]) * 2
            return img, label, TransInfo(img_path, img_size, False)

        img_size = img.shape[:2] if len(img.shape) == 3 else img.shape[2:4]
        # find the greater axis as the x
        size_flag, origin_size, input_size = self.regular_size(img_size)
        transform_matrix = image.get_affine_transform(origin_size, input_size)
        img = cv2.warpAffine(img, transform_matrix, tuple(self._input_size[::-1]))
        # todo normalize()
        input_tensor = F.to_tensor(img)
        if label is not None:
            cls_ids, polygons = label
            # handle box size
            box_sizes = [(polygon.max(0) - polygon.min(0))[::-1] for polygon in polygons]
            # transform pixel
            if size_flag:
                polygons = [image.apply_affine_transform(polygon, transform_matrix, input_size)
                            .astype(np.int32)[:, ::-1] for polygon in polygons]
            else:
                polygons = [image.apply_affine_transform(polygon[:, ::-1], transform_matrix, input_size)
                            .astype(np.int32) for polygon in polygons]

            # handle center
            centers = [polygon.mean(0).astype(np.int32) for polygon in polygons]

            if self._kp and len(cls_ids) != 0:
                kp_mask = generate_kp_mask((1, 1, self._input_size[0], self._input_size[1]), [polygons], strategy="one-hot")
                kp_target = torch.from_numpy(generate_batch_sdf(kp_mask))
                kp_target = kp_target[0]
            else:
                kp_target = torch.tensor([])
            label = (centers, cls_ids, polygons, box_sizes, kp_target)

        return input_tensor, label, TransInfo(img_path, img_size, False)

    def transform_pixel(self, pixels, info):
        anti_pixels = pixels.reshape(-1, 2)
        img_size = info.img_size
        # anti resize
        size_flag, origin_size, input_size = self.regular_size(img_size)
        anti_transform = image.get_affine_transform(origin_size, input_size, inv=True)
        if size_flag:
            anti_pixels = image.apply_affine_transform(anti_pixels[:, ::-1], anti_transform, img_size[::-1])
        else:
            anti_pixels = image.apply_affine_transform(anti_pixels, anti_transform, img_size)[:, ::-1]
        return anti_pixels

    def transform_image(self, img, info):
        img_size = info.img_size
        size_flag, origin_size, input_size = self.regular_size(img_size)

        # anti resize
        anti_transform = image.get_affine_transform(origin_size, input_size, inv=True)
        anti_img = cv2.warpAffine(img, anti_transform, img_size[::-1])
        return anti_img


class TrainTransforms(CommonTransforms):
    def __init__(self, input_size, num_cls, with_flip=False, with_aug_color=False):
        super(TrainTransforms, self).__init__(input_size, num_cls)
        self._with_flip = with_flip
        self._with_aug_color = with_aug_color

    def __call__(self, img, label=None, img_path=None, from_file=False):
        """
        compose transform the all the transform
        :param img:  rgb and the shape is h*w*c
        :param label: cls_mask, kp_mask, ae_mask
        :param img_path: as the key
        :return:
        """
        img_size = img.shape[:2] if len(img.shape) == 3 else img.shape[2:4]
        if from_file:
            input_tensor, label = img, label
        else:
            input_tensor, label, _ = super().__call__(img, label, img_path)

        if self._with_aug_color and np.random.randint(0, 2) == 0:
            pil_img = F.to_pil_image(input_tensor)
            pil_img = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)(pil_img)
            input_tensor = F.to_tensor(pil_img)

        flipped_flag = self._with_flip and np.random.randint(0, 2) == 0

        if flipped_flag:
            input_tensor = torch.flip(input_tensor, [2])
            if label is not None:
                centers, cls_ids, polygons, box_sizes, kp_target = label
                for c_i in range(len(centers)):
                    centers[c_i][1] = self._input_size[1] - centers[c_i][1] - 1
                    polygons[c_i][:, 1] = self._input_size[1] - polygons[c_i][:, 1] - 1
                kp_target = torch.flip(kp_target, [2])
                label = (centers, cls_ids, polygons, box_sizes, kp_target)

        return input_tensor, label, TransInfo(img_path, img_size, flipped_flag)

    def transform_pixel(self, pixels, info):
        """
        :param pixels:
        :param info:
        :return: pixels' format (w,h)
        """
        anti_pixels = pixels.reshape(-1, 2)
        # anti flip
        if info.flipped_flag:
            anti_pixels[:, 1] = self._input_size[1] - anti_pixels[:, 1] - 1

        return super().transform_pixel(anti_pixels, info)

    def transform_image(self, img, info):
        # anti flip
        if info.flipped_flag:
            img = img[:, ::-1, :]

        return super().transform_image(img, info)

