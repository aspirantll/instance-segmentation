from __future__ import print_function

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
import os
os.system("rm /home/work/anaconda3/lib/libmkldnn.so")
os.system("rm /home/work/anaconda3/lib/libmkldnn.so.0")
import torch
import numpy as np
import data
from models import create_model

from configs import Config, Configer
from utils.logger import Logger
from utils import decode
from utils.tranform import CommonTransforms
from utils.eval_util import evaluate_model, evaluate_masks_from_json
import moxing as mox
mox.file.shift('os', 'mox')

# global torch configs for training
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# Tensor type to use, select CUDA or not
torch.set_default_dtype(torch.float32)
use_cuda = torch.cuda.is_available()
device_type ='cuda' if use_cuda else 'cpu'
device = torch.device(device_type)

decode.device = device
decode.draw_flag = False

# load arguments
print("loading the arguments...")
parser = argparse.ArgumentParser(description="test")
# add arguments
parser.add_argument("--cfg_path", help="the file of cfg", dest="cfg_path", default="./configs/eval_cfg.yaml", type=str)
parser.add_argument("--start_epoch", help="start epoch desc", dest="start_epoch", default="-1", type=int)

# for modelarts
parser.add_argument("--data_url", required=False, type=str)
parser.add_argument("--init_method", required=False, type=str)
parser.add_argument("--train_url", required=False, type=str)
# parse args
args = parser.parse_args()

cfg = Config(args.cfg_path)
data_cfg = cfg.data
decode_cfg = Config(cfg.decode_cfg_path)
trans_cfg = Configer(configs=cfg.trans_cfg_path)

if data_cfg.num_classes == -1:
    data_cfg.num_classes = data.get_cls_num(data_cfg.dataset)
if isinstance(data_cfg.input_size, str):
    data_cfg.input_size = eval(data_cfg.input_size)
# validate the arguments
print("eval dir:", data_cfg.eval_dir)
if data_cfg.eval_dir is not None and not os.path.exists(data_cfg.eval_dir):
    raise Exception("the eval dir cannot be found.")

print("save dir:", data_cfg.save_dir)
if not os.path.exists(data_cfg.save_dir):
    os.makedirs(data_cfg.save_dir)

# set seed
np.random.seed(cfg.seed)
torch.random.manual_seed(cfg.seed)
if use_cuda:
    torch.cuda.manual_seed_all(cfg.seed)

Logger.init_logger(data_cfg)
logger = Logger.get_logger()

eval_labels = data.get_eval_labels(data_cfg.dataset)
label_names = [label[1] for label in eval_labels]


def load_state_dict(model, weights_path):
    """
    if save_dir contains the checkpoint, then the model will load lastest weights
    :param model:
    :param save_dir:
    :return:
    """
    checkpoint = torch.load(weights_path, map_location=device_type)
    model.load_state_dict(checkpoint["state_dict"])
    logger.write("loaded the weights:" + weights_path)
    return checkpoint["epoch"]


def evaluate_model_by_weights(eval_dataloader, transforms, weights_path, logger=None):
    """
    validate model for a epoch
    :param transforms:
    :param eval_dataloader:
    :return:
    """
    # initialize
    model = create_model(cfg.model_type, data_cfg.num_classes)
    epoch = load_state_dict(model, weights_path)
    model = model.to(device)

    evaluate_model(data_cfg, eval_dataloader, transforms, model, epoch, data_cfg.dataset, decode_cfg, device, logger)


def load_weight_paths(weights_dir, start_epoch):
    weight_paths = []

    file_list = os.listdir(weights_dir)
    file_list.sort(reverse=True)
    start_flag = False
    for file in file_list:
        if not start_flag and (start_epoch < 0 or "{}_weights_{:0>8}.pth".format(cfg.model_type, start_epoch) == file):
            start_flag = True
        if start_flag and file.startswith("{}_weights_".format(cfg.model_type)) and file.endswith(".pth"):
            weight_path = os.path.join(weights_dir, file)
            weight_paths.append(weight_path)

    return weight_paths


def eval_weights_dir(weights_dir):
    weight_paths = load_weight_paths(weights_dir, args.start_epoch)
    logger.write("the num of weights file: {}".format(len(weight_paths)))
    for iter_id, weight_path in enumerate(weight_paths):
        if iter_id % 20 == 0:
            evaluate_model_by_weights(eval_dataloader, transforms, weight_path, logger)


if __name__ == "__main__":
    transforms = CommonTransforms(trans_cfg, "val")
    eval_dataloader = data.get_dataloader(data_cfg.batch_size, data_cfg.dataset, data_cfg.eval_dir,
                                          input_size=data_cfg.input_size,
                                          phase=data_cfg.subset, transforms=transforms)
    # eval
    print("start to evaluate...")
    if cfg.weights_dir is None:
        evaluate_model_by_weights(eval_dataloader, transforms, cfg.weights_path, logger)
    else:
        eval_weights_dir(cfg.weights_dir)
    logger.close()