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
import torch
import os
import numpy as np
import data
from models import ERFNet
from utils.mean_ap import eval_map
from configs import Config
from utils.logger import Logger
from utils import decode
from utils.tranform import CommonTransforms

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


def evaluate_model(eval_dataloader, transforms, weights_path):
    """
    validate model for a epoch
    :param transforms:
    :param eval_dataloader:
    :return:
    """
    # initialize
    model = ERFNet(data_cfg.num_classes)
    load_state_dict(model, weights_path)
    model = model.to(device)

    model.eval()
    num_iter = len(eval_dataloader)

    meters = [APMeter(1) for i in range(data_cfg.num_classes)]
    # foreach the images
    for iter_id, eval_data in enumerate(eval_dataloader):
        # to device
        inputs, targets, infos = eval_data
        inputs = inputs.to(device)
        # forward the models and loss
        with torch.no_grad():
            outputs = model(inputs)
            dets = decode.decode_output(outputs, infos, transforms)

        gt_labels = targets[1]
        # transform the pixel to original image
        gt_polygons = [[transforms.transform_pixel(obj, infos[b_i]) for obj in targets[2][b_i]] for b_i in range(len(targets[2]))]

        # for b_i in range(len(gt_polygons)):
        #     info = infos[b_i]
        #     polygons = gt_polygons[b_i]
        #     det = dets[b_i]
        #
        #     import cv2
        #     from utils.visualize import visualize_instance
        #     img = cv2.imread(info.img_path)
        #     for j in range(len(det)):
        #         det_polys = det[j][-1]
        #         img = visualize_instance(img, [det_polys], mask=True)
        #     save_path = os.path.join(data_cfg.save_dir, "det_{}".format(os.path.basename(info.img_path)))
        #     cv2.imwrite(save_path, img)
        #     img = cv2.imread(info.img_path)
        #     for j in range(len(polygons)):
        #         img = visualize_instance(img, [polygons[j]], mask=True)
        #     save_path = os.path.join(data_cfg.save_dir, "gt_{}".format(os.path.basename(info.img_path)))
        #     cv2.imwrite(save_path, img)
        #     logger.write("detected result saved in {}".format(save_path))
        del inputs
        torch.cuda.empty_cache()
        logger.write("[{}/{}] evaluation".format(iter_id, num_iter))
        mean_ap, eval_result = eval_map(dets, gt_polygons, gt_labels, data_cfg.num_classes
                                        , meters, print_summary=True, dataset=data_cfg.dataset)


if __name__ == "__main__":
    transforms = CommonTransforms(data_cfg.input_size, data_cfg.num_classes)
    eval_dataloader = data.get_dataloader(data_cfg.batch_size, data_cfg.dataset, data_cfg.eval_dir,
                                           input_size=data_cfg.input_size,
                                           phase="val", transforms=transforms, from_file=True)
    # eval
    print("start to evaluate...")
    evaluate_model(eval_dataloader, transforms, cfg.weights_path)
    logger.close()