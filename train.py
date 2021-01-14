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
import os
import time
import numpy as np
import warnings

from models.efficient import input_sizes

warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor
import moxing as mox
mox.file.shift('os', 'mox')

import data
from configs import Config, Configer
from models import EfficientSeg, ComposeLoss
from utils.tranform import CommonTransforms
from utils.logger import Logger
from utils.meter import AverageMeter
from utils.eval_util import evaluate_model

# global torch configs for training
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# Tensor type to use, select CUDA or not
torch.set_default_dtype(torch.float32)
use_cuda = torch.cuda.is_available()
device_type = "cuda" if use_cuda else "cpu"
device = torch.device(device_type)

# load arguments
print("loading the arguments...")
parser = argparse.ArgumentParser(description="training")
# add arguments
parser.add_argument("--cfg_path", help="the file of cfg", dest="cfg_path", default="./configs/train_cfg.yaml", type=str)
# for modelarts
parser.add_argument("--data_url", required=False, type=str)
parser.add_argument("--init_method", required=False, type=str)
parser.add_argument("--train_url", required=False, type=str)
# parse args
args = parser.parse_args()

cfg = Config(args.cfg_path)
data_cfg = cfg.data
opt_cfg = cfg.optimizer
decode_cfg = Config(cfg.decode_cfg_path)
trans_cfg = Configer(configs=cfg.trans_cfg_path)

if data_cfg.num_classes == -1:
    data_cfg.num_classes = data.get_cls_num(data_cfg.dataset)
if isinstance(opt_cfg.lr, str):
    opt_cfg.lr = eval(opt_cfg.lr)

# validate the arguments
print("train dir:", data_cfg.train_dir)
if not os.path.exists(data_cfg.train_dir):
    raise Exception("the train dir cannot be found.")
print("save dir:", data_cfg.save_dir)
if not os.path.exists(data_cfg.save_dir):
    os.makedirs(data_cfg.save_dir)
if data_cfg.dataset not in data.datasetBuildersMap:
    raise Exception("the dataset is not accepted.")

# set seed
np.random.seed(cfg.seed)
torch.random.manual_seed(cfg.seed)
if use_cuda:
    torch.cuda.manual_seed_all(cfg.seed)

Logger.init_logger(data_cfg, type="simple")
logger = Logger.get_logger()
executor = ThreadPoolExecutor(max_workers=3)


def save_checkpoint(model_dict, epoch, best_ap, save_dir, iter_id=None):
    """
    save the check points
    :param model_dict: the best model
    :param epoch: epoch
    :param best_ap: best mAP
    :param save_dir: the checkpoint dir
    :param iter_id: the index of iter
    :return:
    """
    checkpoint = {
        'state_dict': model_dict,
        'epoch': epoch,
        'best_ap': best_ap
    }
    if iter_id is None:
        weight_path = os.path.join(save_dir, "efficient_weights_{:0>8}.pth".format(epoch))
    else:
        weight_path = os.path.join(save_dir, "efficient_weights_{:0>4}_{:0>4}.pth".format(epoch, iter_id))
    # torch.save(best_model_wts, weight_path)
    torch.save(checkpoint, weight_path)
    logger.write("epoch {}, save the weight to {}".format(epoch, weight_path))


def get_optimizer(model, opt):
    """
    initialize the the optimizer
    :param opt:
    :param model:
    :return:
    """
    filter_params = filter(lambda p: p.requires_grad, model.parameters())
    if opt.type == "SGD":
        return torch.optim.SGD(filter_params, lr=opt.lr, momentum=opt.momentum)
    elif opt.type == "Adam":
        return torch.optim.Adam(filter_params, opt.lr, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)
    elif opt.type == "Adadelta":
        return torch.optim.Adadelta(filter_params, lr=opt.lr)

def load_state_dict(model, save_dir, pretrained):
    """
    if save_dir contains the checkpoint, then the model will load lastest weights
    :param model:
    :param save_dir:
    :return:
    """
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        try:
            ret = model.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')
    else:
        file_list = os.listdir(save_dir)
        file_list.sort(reverse=True)
        for file in file_list:
            if file.startswith("efficient_weights_") and file.endswith(".pth"):
                weight_path = os.path.join(save_dir, file)
                checkpoint = torch.load(weight_path, map_location=device_type)
                try:
                    ret = model.load_state_dict(checkpoint["state_dict"], strict=False)
                    print(ret)
                except RuntimeError as e:
                    print('Ignoring ' + str(e) + '"')
                logger.write("loaded the weights:" + weight_path)
                start_epoch = checkpoint["epoch"]
                best_ap = checkpoint["best_ap"] if "best_ap" in checkpoint else 0
                model.init_weight()
                save_checkpoint(model.state_dict(), -1, 0, data_cfg.save_dir)
                return start_epoch+1, best_ap
    # model.init_weight()
    save_checkpoint(model.state_dict(), -1, 0, data_cfg.save_dir)
    return 0, 0


def write_metric(metric, epoch, phase):
    """
    write the metric to logger
    :param metric: the dict of loss
    :param epoch: the epoch
    :param phase: train,val,or test
    :return:
    """
    logger.write('{phase} : [{0}/{1}]|'.format(epoch, cfg.num_epochs, phase=phase), end='')
    logger.open_summary_writer()
    for k, v in metric.items():
        logger.scalar_summary('{phase}/{}'.format(k, phase=phase), v.avg, epoch)
        logger.write('{} {:8f} | '.format(k, v.avg), end='')
    logger.write()
    logger.close_summary_writer()


def train_model_for_epoch(model, train_dataloader, loss_fn, optimizer, epoch):
    """
    train model for a epoch
    :param model:
    :param train_dataloader:
    :param loss_fn:
    :param optimizer:
    :return:
    """
    # prepared
    model.train()
    num_iter = len(train_dataloader) if cfg.max_iter <= 0 else min(len(train_dataloader), cfg.max_iter)
    loss_states = loss_fn.get_loss_states()
    data_time, batch_time = AverageMeter(), AverageMeter()
    running_loss = AverageMeter()
    avg_loss_states = {l: AverageMeter() for l in loss_states}
    start = time.time()
    last = time.time()
    phase = "train"
    # foreach the images
    for iter_id, train_data in enumerate(train_dataloader):
        if iter_id >= num_iter:
            break
        # load data time
        data_time.update(time.time() - last)
        inputs, targets, infos = train_data
        try:
            # to device
            inputs = inputs.to(device)
            # forward the models and loss
            outputs = model(inputs)
            loss, loss_stats = loss_fn(outputs, targets)
            # update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # network time and update time
            batch_time.update(time.time() - last)
            last = time.time()
            # handle the log and accumulate the loss
            # logger.open_summary_writer()
            log_item = '{phase} per epoch: [{0}][{1}/{2}]|Tot: {total:} '.format(
                epoch, iter_id + 1, num_iter, phase=phase, total=last - start)
            for l in avg_loss_states:
                if l in loss_stats:
                    avg_loss_states[l].update(
                        loss_stats[l].item(), inputs.size(0))
                    log_item = log_item + '|{}:{:.4f}'.format(l, avg_loss_states[l].avg)
                    # logger.scalar_summary('{phase}/epoch/{}'.format(l, phase=phase), avg_loss_states[l].avg, epoch* num_iter + iter_id)
            # logger.close_summary_writer()
            running_loss.update(loss.item(), inputs.size(0))
            log_item = log_item + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                      '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)

            logger.write(log_item, level=1)
            del inputs, loss
            torch.cuda.empty_cache()
            if (iter_id + 1) % cfg.save_span == 0:
                executor.submit(save_checkpoint, model.state_dict(), epoch, running_loss.avg, data_cfg.save_dir, iter_id)
        except RuntimeError as e:
            print(infos)
            raise e
    return running_loss, avg_loss_states


def train():
    """
    train the model by the args
    :return:
    """
    # initialize model, optimizer, loss_fn
    model = EfficientSeg(data_cfg.num_classes, compound_coef=cfg.compound_coef,
                                 ratios=eval(cfg.anchors_ratios), scales=eval(cfg.anchors_scales))

    # initialize the dataloader by dir
    train_transforms = CommonTransforms(trans_cfg, model.get_input_size())
    train_dataloader = data.get_dataloader(data_cfg.batch_size, data_cfg.dataset, data_cfg.train_dir,
                                           phase="crops", transforms=train_transforms)

    eval_transforms = CommonTransforms(trans_cfg, model.get_input_size())
    eval_dataloader = data.get_dataloader(data_cfg.batch_size, data_cfg.dataset, data_cfg.train_dir,
                                           phase="val", transforms=eval_transforms)

    start_epoch, best_ap = load_state_dict(model, data_cfg.save_dir, cfg.pretrained_path)
    model = model.to(device)
    optimizer = get_optimizer(model, opt_cfg)
    loss_fn = ComposeLoss(device)

    # train model
    # foreach epoch
    for epoch in range(start_epoch, cfg.num_epochs):
        # each epoch includes two phase: train,val
        train_loss, train_loss_states = train_model_for_epoch(model, train_dataloader, loss_fn, optimizer, epoch)
        write_metric(train_loss_states, epoch, "train")
        executor.submit(save_checkpoint, model.state_dict(), epoch, best_ap, data_cfg.save_dir)

        if epoch >= cfg.start_eval_epoch:
            epoch, mAP, eval_results = evaluate_model(data_cfg, eval_dataloader, model, epoch, data_cfg.dataset, decode_cfg, device, logger)
            # judge the model. if model is greater than current best loss
            if best_ap < mAP:
                best_ap = mAP
    logger.write("the best mAP:{}".format(best_ap))
    logger.close()
    executor.shutdown(wait=True)


if __name__ == "__main__":
    # train
    logger.write("start to train...")
    train()
