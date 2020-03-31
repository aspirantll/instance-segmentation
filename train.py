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
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import moxing as mox
mox.file.shift('os', 'mox')

import data
from configs import Config
from models import ERFNet, ComposeLoss, ClsFocalLoss, AELoss, KPFocalLoss, KPGACLoss, KPLSLoss, WHDLoss
from utils.tranform import TrainTransforms
from utils.logger import Logger
from utils.meter import AverageMeter

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
loss_cfg = cfg.loss

if data_cfg.num_classes == -1:
    data_cfg.num_classes = data.get_cls_num(data_cfg.dataset)
if isinstance(data_cfg.input_size, str):
    data_cfg.input_size = eval(data_cfg.input_size)
if isinstance(opt_cfg.lr, str):
    opt_cfg.lr = eval(opt_cfg.lr)

# validate the arguments
print("train dir:", data_cfg.train_dir)
if not os.path.exists(data_cfg.train_dir):
    raise Exception("the train dir cannot be found.")
print("val dir:", data_cfg.val_dir)
if not os.path.exists(data_cfg.val_dir):
    raise Exception("the val dir cannot be found.")
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


def save_checkpoint(model_dict, epoch, best_loss, save_dir, iter_id=None):
    """
    save the check points
    :param model_dict: the best model
    :param epoch: epoch
    :param best_loss: best loss
    :param save_dir: the checkpoint dir
    :param iter_id: the index of iter
    :return:
    """
    checkpoint = {
        'state_dict': model_dict,
        'epoch': epoch,
        'best_loss': best_loss
    }
    if iter_id is None:
        weight_path = os.path.join(save_dir, "model_weights_{:0>8}.pth".format(epoch))
    else:
        weight_path = os.path.join(save_dir, "model_weights_{:0>4}_{:0>4}.pth".format(epoch, iter_id))
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


def init_loss_fn():
    cls_loss_fn = ClsFocalLoss(device, alpha=loss_cfg.focal_alpha, beta=loss_cfg.focal_beta)
    kp_loss_fn = WHDLoss(device, alpha=loss_cfg.whd_alpha, beta=loss_cfg.whd_beta, th=loss_cfg.kp_threshold)
    ae_loss_fn = AELoss(device, alpha=loss_cfg.ae_alpha, beta=loss_cfg.ae_beta, delta=loss_cfg.ae_delta)
    return ComposeLoss(cls_loss_fn, kp_loss_fn, ae_loss_fn)


def load_state_dict(model, save_dir, pretrained):
    """
    if save_dir contains the checkpoint, then the model will load lastest weights
    :param model:
    :param save_dir:
    :return:
    """
    if pretrained:
        pretrained_dict = torch.load(pretrained, map_location=device_type)
        model_dict = model.state_dict()
        # remove the module suffix and filter the removed layers
        filtered_dict = {}
        for k, v in pretrained_dict["state_dict"].items():
            if k.startswith("module."):
                k = k[7:]
            if k in model_dict:
                filtered_dict[k] = v
        # update the current model
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        model.init_weight()
        executor.submit(save_checkpoint, model.state_dict(), 0, np.inf, data_cfg.save_dir)
        logger.write("loaded the pretrained weights:" + pretrained)
    else:
        file_list = os.listdir(save_dir)
        file_list.sort(reverse=True)
        for file in file_list:
            if file.startswith("model_weights_") and file.endswith(".pth"):
                weight_path = os.path.join(save_dir, file)
                checkpoint = torch.load(weight_path, map_location=device_type)
                model.load_state_dict(checkpoint["state_dict"])
                logger.write("loaded the weights:" + weight_path)
                start_epoch = checkpoint["epoch"]
                best_loss = checkpoint["best_loss"] if "best_loss" in checkpoint else np.inf
                return start_epoch + 1, best_loss
        model.init_weight()
    return 0, np.inf


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


def train_model_for_epoch(model, train_dataloader, loss_fn, optimizer, epoch, debug=False):
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
        logger.open_summary_writer()
        log_item = '{phase} per epoch: [{0}][{1}/{2}]|Tot: {total:} '.format(
            epoch, iter_id, num_iter, phase=phase, total=last - start)
        for l in avg_loss_states:
            if l in loss_stats:
                avg_loss_states[l].update(
                    loss_stats[l].item(), inputs.size(0))
                log_item = log_item + '|{}:{:.4f}'.format(l, avg_loss_states[l].avg)
                logger.scalar_summary('{phase}/epoch/{}'.format(l, phase=phase), avg_loss_states[l].avg, epoch* num_iter + iter_id)
        logger.close_summary_writer()
        running_loss.update(loss.item(), inputs.size(0))
        log_item = log_item + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                  '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)

        logger.write(log_item, level=1)
        del inputs, loss
        torch.cuda.empty_cache()
        if (iter_id + 1) % cfg.save_span == 0:
            executor.submit(save_checkpoint, model.state_dict(), epoch, running_loss.avg, args.save_dir, iter_id)
    avg_loss_states["Total Loss"] = running_loss
    return running_loss, avg_loss_states


def train():
    """
    train the model by the args
    :return:
    """
    # initialize the dataloader by dir
    transforms = TrainTransforms(data_cfg.input_size, data_cfg.num_classes, with_flip=True, with_aug_color=True)
    train_dataloader = data.get_dataloader(data_cfg.batch_size, data_cfg.dataset, data_cfg.train_dir, input_size=data_cfg.input_size,
                                           phase="train", transforms=transforms, from_file=True)

    # initialize model, optimizer, loss_fn
    model = ERFNet(data_cfg.num_classes, fixed_parts=None)
    start_epoch, best_loss = load_state_dict(model, data_cfg.save_dir, cfg.pretrained_path)
    model = model.to(device)
    optimizer = get_optimizer(model, opt_cfg)
    loss_fn = init_loss_fn()

    # train model
    # foreach epoch
    for epoch in range(start_epoch, cfg.num_epochs):
        # each epoch includes two phase: train,val
        train_loss, train_loss_states = train_model_for_epoch(model, train_dataloader, loss_fn, optimizer, epoch)
        executor.submit(save_checkpoint, model.state_dict(), epoch, best_loss, data_cfg.save_dir)
        write_metric(train_loss_states, epoch, "train")
        # val_loss, val_loss_states = val_model_for_epoch(model, val_dataloader, loss_fn, epoch, logger)
        # write_metric(val_loss_states, epoch, "val")
        # judge the model. if model is greater than current best loss
        # if best_loss > train_loss.avg:
        #     best_loss = train_loss.avg
        #     executor.submit(save_checkpoint, model.state_dict(), epoch, best_loss, data_cfg.save_dir)
    logger.write("the best loss:{}".format(best_loss))
    logger.close()
    executor.shutdown(wait=True)


if __name__ == "__main__":
    # train
    logger.write("start to train...")
    train()
