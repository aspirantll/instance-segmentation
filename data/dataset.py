__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import abc
import cv2


def load_rgb_image(img_path):
    input_img = cv2.imread(img_path)
    if input_img is None:
        raise ValueError("the img load error:{}".format(img_path))
    else:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    return input_img


def is_train_phase(phase):
    return phase == "train"


def is_val_phase(phase):
    return phase == "val"


class DatasetBuilder(object):
    def __init__(self, data_dir, phase="train", ann_file=None):
        if data_dir is None:
            raise Exception("The data_dir must be not None.")
        self._data_dir = data_dir
        self._phase = phase
        if ann_file is None:
            self._ann_file = self.default_ann_file()
        else:
            self._ann_file = ann_file

    @abc.abstractmethod
    def default_ann_file(self):
        pass

    @abc.abstractmethod
    def get_dataset(self, **kwargs):
        pass