__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import os
from torch.utils import data
from data.dataset import DatasetBuilder, load_rgb_image
from utils.tranform import CommonTransforms



class DirDataset(data.Dataset):
    """
    the dataset for image dir.
    """
    def __init__(self, data_dir, input_size, transforms=None):
        # save the parameters
        self._data_dir = data_dir
        self._input_size=input_size
        self._scale = 4
        if transforms is not None:
            self._transforms = transforms  # ADDED THIS
        else:
            self._transforms = CommonTransforms(input_size)
        # scan the dir
        self.imgs = []
        for name in os.listdir(data_dir):
            if name.endswith(r".jpg"):
                self.imgs.append(name)

    def __getitem__(self, index):
        # locating index
        path = os.path.join(self._data_dir, self.imgs[index])
        input_img = load_rgb_image(path)

        input_img, _, trans_info = self._transforms(input_img)
        return input_img, trans_info

    def __len__(self):
        return len(self.imgs)


class DirDatasetBuilder(DatasetBuilder):
    def __init__(self, data_dir,  phase, ann_file):
        super().__init__(data_dir, phase, ann_file)

    def default_ann_file(self):
        return ""

    def get_dataset(self, **kwargs):
        return DirDataset(self._data_dir, **kwargs)