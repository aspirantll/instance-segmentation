__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

from utils.target_generator import generate_sdf
import numpy as np
import torch
from sklearn.utils.extmath import cartesian


def sdf(mat):
    h, w = mat.shape
    all_pixels = cartesian([np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32)])
    target_pixels = np.vstack(mat.nonzero()).transpose()

    all_pixels = np.expand_dims(all_pixels, axis=1)
    target_pixels = np.expand_dims(target_pixels, axis=0)
    diff = - (all_pixels - target_pixels)

    distance = np.power(diff, 2).sum(axis=2)
    min_index = distance.argmin(axis=1)
    return diff[np.arange(min_index.shape[0]), min_index].reshape(h, w, 2)


# if __name__ == "__main__":
#     mat = np.random.random_integers(0, 1, size=(5, 5))
#     kp_vectors = generate_sdf(mat).transpose((2, 0, 1))
#     kp_vectors = torch.from_numpy(kp_vectors)
#     kp_sdf = torch.sqrt(torch.pow(kp_vectors, 2).sum(0))
#     kp_directions = kp_vectors / torch.clamp(kp_sdf, min=1).unsqueeze(0)
#
#     print(kp_directions.shape)


# if __name__ == "__main__":
#     from configs import Config
#     cfg = Config("./configs/train_cfg.yaml")
#     print(cfg)

if __name__ == "__main__":
    from utils.tensor_util import save_labels, load_labels
    data_file_path = r"C:\Users\liulei\Desktop\data.npy"
    arr = np.ones((16, 16))
    arr1 = np.ones((3, 3))
    save_labels(torch.from_numpy(arr), (arr1, arr1, arr1, torch.from_numpy(arr1)), data_file_path)
    img, label = load_labels(data_file_path)
    print(img)
    print(label)