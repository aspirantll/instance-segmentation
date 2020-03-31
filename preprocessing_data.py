__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import argparse
import data
import tqdm
import os
from configs import Config
from utils.label_io import save_labels

# load arguments
print("loading the arguments...")
parser = argparse.ArgumentParser(description="training")
# add arguments
parser.add_argument("--cfg_path", help="the file of cfg", dest="cfg_path", default="./configs/train_cfg.yaml", type=str)
parser.add_argument("--phase", help="which phase", dest="phase", default="train", type=str)

# parse args
args = parser.parse_args()

cfg = Config(args.cfg_path)
data_cfg = cfg.data

if isinstance(data_cfg.input_size, str):
    data_cfg.input_size = eval(data_cfg.input_size)


def preprocess():
    dataloader = data.get_dataloader(1, data_cfg.dataset, data_cfg.train_dir,
                                           input_size=data_cfg.input_size, num_workers=2,
                                           phase=args.phase, transforms=None)
    target_dir = os.path.join(data_cfg.train_dir, 'preprocessed/' + args.phase)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for input_tensor, label, trans_info in tqdm.tqdm(dataloader):
        info = trans_info[0]
        img = input_tensor[0]
        centers = label[0][0]
        cls_ids = label[1][0]
        polygons = label[2][0]
        kp_target = label[3][0]

        filename = os.path.basename(info.img_path)[:-4] + ".npy"
        save_labels(img, (centers, cls_ids, polygons, kp_target), os.path.join(target_dir, filename))


if __name__ == "__main__":
    preprocess()