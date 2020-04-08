from __future__ import print_function

from utils.tranform import CommonTransforms

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
from utils.logger import Logger
from models import ERFNet
import data
from utils import decode
from utils.decode import decode_output
from utils.visualize import visualize_instance
from matplotlib import pyplot as plt
from PIL import Image
from utils import image


# global torch configs for training
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# Tensor type to use, select CUDA or not
torch.set_default_dtype(torch.float32)
use_cuda = torch.cuda.is_available()
device_type ='cuda' if use_cuda else 'cpu'
device = torch.device(device_type)

# load arguments
print("loading the arguments...")
parser = argparse.ArgumentParser(description="training")
# add arguments
parser.add_argument("--save_dir", help="the dir of saving result", dest="save_dir", required=True, type=str)
parser.add_argument("--weights_path", help="the weights path", dest="weights_path", required=True, type=str)
parser.add_argument("--test_dir", help="the dir of test", dest="test_dir", type=str)
parser.add_argument("--test_image", help="the image of test", dest="test_image", type=str)
parser.add_argument("--batch_size", dest="batch_size", default=32, type=int)
parser.add_argument("--input_size", dest="input_size", default=(512, 1024), type=tuple)
parser.add_argument("--seed", dest="seed", default=1, type=int)
parser.add_argument("--num_classes", dest="num_classes", default=-1, type=int)
# parse args
args = parser.parse_args()
args.dataset = "dir"
if args.num_classes == -1:
    args.num_classes = data.get_cls_num(args.dataset)
# validate the arguments
print("test dir:", args.test_dir)
if args.test_dir is not None and not os.path.exists(args.test_dir):
    raise Exception("the train dir cannot be found.")

print("save dir:", args.save_dir)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


# set seed
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

Logger.init_logger(args)
logger = Logger.get_logger()


def load_state_dict(model):
    """
    if save_dir contains the checkpoint, then the model will load lastest weights
    :param model:
    :param save_dir:
    :return:
    """
    checkpoint = torch.load(args.weights_path, map_location=device_type)
    model.load_state_dict(checkpoint["state_dict"])
    logger.write("loaded the weights:" + args.weights_path)


def handle_output(inputs, infos, model, transforms):
    inputs = inputs.to(device)
    # forward the models and loss
    with torch.no_grad():
        outputs = model(inputs)
        dets = decode_output(outputs, infos, transforms)
        for i in range(len(dets)):
            info = infos[i]
            img_path = info.img_path
            name = os.path.basename(img_path)
            det = dets[i]
            logger.write("in {} detected {} objs".format(name, len(det)))
            for j in range(len(det)):
                # logger.write(det)
                img = Image.open(img_path)
                plt.figure(img_path + str(j))
                plt.imshow(img)
                visualize_instance([det[j]])
                save_path = os.path.join(args.save_dir, str(j) + name)
                plt.savefig(save_path)
                logger.write("detected result saved in {}".format(save_path))


def test():
    """
    train the model by the args
    :param args:
    :return:
    """
    # initialize model
    model = ERFNet(args.num_classes)
    load_state_dict(model)
    model = model.to(device)

    # test model
    model.eval()
    transforms = CommonTransforms(args.input_size, args.num_classes)

    decode.device = device
    if args.test_dir is not None:
        # initialize the dataloader by dir
        test_dataloader = data.get_dataloader(args.batch_size, args.dataset, args.test_dir,
                                               input_size=args.input_size,
                                               phase="test", transforms=transforms)
        # foreach the images
        for iter_id, test_data in enumerate(test_dataloader):
            # to device
            inputs, infos = test_data
            handle_output(inputs, infos, model, transforms)
    else:
        img_path = args.test_image
        input_img = image.load_rgb_image(img_path)
        input, _, info = transforms(input_img, img_path=img_path)
        handle_output(input.unsqueeze(0), [info], model, transforms)
        plt.show()
    logger.close()


if __name__ == "__main__":
    # train
    logger.write("start to test...")
    test()
