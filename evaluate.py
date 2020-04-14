from __future__ import print_function

from utils.meter import APMeter

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
from models import ERFNet
from utils.mean_ap import eval_map, print_map_summary
from configs import Config
from utils.logger import Logger
from utils import decode
from utils.tranform import CommonTransforms
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
# for modelarts
parser.add_argument("--data_url", required=False, type=str)
parser.add_argument("--init_method", required=False, type=str)
parser.add_argument("--train_url", required=False, type=str)
# parse args
args = parser.parse_args()

cfg = Config(args.cfg_path)
data_cfg = cfg.data
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


def evaluate_model(eval_dataloader, transforms, weights_path):
    """
    validate model for a epoch
    :param transforms:
    :param eval_dataloader:
    :return:
    """
    # initialize
    model = ERFNet(data_cfg.num_classes)
    epoch = load_state_dict(model, weights_path)
    model = model.to(device)

    model.eval()
    num_iter = len(eval_dataloader)

    mAP, eval_results = None, None
    meters = [APMeter(1) for label in eval_labels]
    # foreach the images
    for iter_id, eval_data in enumerate(eval_dataloader):
        # to device
        inputs, targets, infos = eval_data
        inputs = inputs.to(device)
        # forward the models and loss
        with torch.no_grad():
            outputs = model(inputs)
            dets = decode.decode_output(outputs, infos, transforms)
        del inputs
        torch.cuda.empty_cache()

        gt_labels = targets[1]
        # transform the pixel to original image
        gt_polygons = [[transforms.transform_pixel(obj, infos[b_i]) for obj in targets[2][b_i]] for b_i in range(len(targets[2]))]

        mAP, eval_results = eval_map(dets, gt_polygons, gt_labels,eval_labels, meters, print_summary=False, dataset=data_cfg.dataset)

        logger.write("eval for epoch {}:[{}/{}]".format(epoch, iter_id+1, num_iter))
    return epoch, mAP, eval_results


def load_weight_paths(weights_dir):
    weight_paths = []

    file_list = os.listdir(weights_dir)
    file_list.sort(reverse=True)
    for file in file_list:
        if file.startswith("model_weights_") and file.endswith(".pth"):
            weight_path = os.path.join(weights_dir, file)
            weight_paths.append(weight_path)
    return weight_paths


def eval_weights_dir(weights_dir):
    weight_paths = load_weight_paths(weights_dir)
    num_weights = len(weight_paths)
    logger.write("the num of weights file: {}".format(num_weights))
    for i in range(0,num_weights, 5):
        epoch, mAP, eval_result = evaluate_model(eval_dataloader, transforms, weight_paths[i])
        logger.write("epoch {}, mAP:{}".format(epoch, mAP))
        print_map_summary(mAP, eval_result, label_names)

        items = ['recall', 'precision', 'ap']
        logger.open_summary_writer()
        for l_i in range(len(label_names)):
            eval_dict = eval_result[l_i]
            label_name = label_names[l_i]

            for item in items:
                value = eval_dict[item]
                if isinstance(value, np.ndarray):
                    if len(value) == 0:
                        value = 0
                    else:
                        value = value[-1]
                logger.scalar_summary(label_name+"_"+item, value, epoch)
        logger.scalar_summary("mAP", mAP, epoch)
        logger.close_summary_writer()


if __name__ == "__main__":
    transforms = CommonTransforms(data_cfg.input_size, data_cfg.num_classes)
    eval_dataloader = data.get_dataloader(data_cfg.batch_size, data_cfg.dataset, data_cfg.eval_dir,
                                           input_size=data_cfg.input_size,
                                           phase="val", transforms=transforms, from_file=True)
    # eval
    print("start to evaluate...")
    if cfg.weights_dir is None:
        _, mAP, eval_result = evaluate_model(eval_dataloader, transforms, cfg.weights_path)
        print_map_summary(mAP, eval_result, label_names)
    else:
        eval_weights_dir(cfg.weights_dir)
    logger.close()