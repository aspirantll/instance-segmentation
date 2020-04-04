import os
from math import log

from models import ERFNet

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

# if __name__ == "__main__":
#     from utils.label_io import save_labels, load_labels
#     data_file_path = r"C:\Users\liulei\Desktop\data.npy"
#     arr = np.ones((16, 16))
#     arr1 = np.ones((3, 3))
#     save_labels(torch.from_numpy(arr), (arr1, arr1, arr1, torch.from_numpy(arr1)), data_file_path)
#     img, label = load_labels(data_file_path)
#     print(img)
#     print(label)

if __name__ == "__main__":
    model = ERFNet(14)
    save_dir = "C:\data\checkpoints\erf"
    file_list = os.listdir(save_dir)
    file_list.sort(reverse=True)
    for file in file_list:
        if file.startswith("model_weights_") and file.endswith(".pth"):
            weight_path = os.path.join(save_dir, file)
            checkpoint = torch.load(weight_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
            start_epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"] if "best_loss" in checkpoint else np.inf
            break
    for m in model.hm_kp.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
    epoch = 0
    checkpoint = {
        'state_dict': model.state_dict(),
        'epoch': epoch,
        'best_loss': np.inf
    }
    weight_path = os.path.join(save_dir, "model_weights_{:0>8}.pth".format(epoch))
    # torch.save(best_model_wts, weight_path)
    torch.save(checkpoint, weight_path)