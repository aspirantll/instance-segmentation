__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"
import torch
import data
import numpy as np
from tqdm import tqdm
from utils import decode
from utils.mean_ap import eval_map, print_map_summary
from utils.meter import APMeter


def print_eval_results(epoch, mAP, eval_result, logger, label_names):
    logger.write("eval epoch {}, mAP:{}".format(epoch, mAP))
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
            logger.scalar_summary(label_name + "_" + item, value, epoch)
    logger.scalar_summary("mAP", mAP, epoch)
    logger.close_summary_writer()


def evaluate_model(eval_dataloader, transforms, model, epoch, dataset, decode_cfg, device, logger):
    # initialize
    decode.device = device
    eval_labels = data.get_eval_labels(dataset)
    label_names = [label[1] for label in eval_labels]

    # eval
    model.eval()
    num_iter = len(eval_dataloader)

    mAP, eval_results = None, None
    meters = [APMeter(1) for label in eval_labels]
    # foreach the images
    for iter_id, eval_data in tqdm(enumerate(eval_dataloader), total=num_iter, desc="eval for epoch {}".format(epoch)):
        # to device
        inputs, targets, infos = eval_data
        inputs = inputs.to(device)
        # forward the models and loss
        with torch.no_grad():
            outputs = model(inputs)
            dets = decode.decode_output(outputs, infos, transforms, decode_cfg, device)
        del inputs
        torch.cuda.empty_cache()

        gt_labels = targets[1]
        # transform the pixel to original image
        gt_polygons = [[transforms.transform_pixel(obj, infos[b_i]) for obj in targets[2][b_i]] for b_i in
                       range(len(targets[2]))]

        mAP, eval_results = eval_map(dets, gt_polygons, gt_labels, eval_labels, meters, print_summary=False,
                                     dataset=dataset)

    if logger is not None:
        print_eval_results(epoch, mAP, eval_results, logger, label_names)
    return epoch, mAP, eval_results