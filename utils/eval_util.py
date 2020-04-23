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



import os
import uuid
import pycocotools.mask as mask_util
import cv2
def evaluate_masks(data_cfg, eval_dataloader, transforms, model, epoch, dataset, decode_cfg, device, logger, use_salt=True):
    # initialize
    output_dir = data_cfg.save_dir
    decode.device = device
    eval_labels = data.get_eval_labels(dataset)
    label_names = [label[1] for label in eval_labels]
    label_ids = [label[2] for label in eval_labels]

    # eval
    model.eval()
    num_iter = len(eval_dataloader)

    dets_list = []
    info_list = []
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

        dets_list.extend(dets)
        info_list.extend(infos)

    res_file = os.path.join(
        output_dir, 'segmentations_cityscapes_results')
    if use_salt:
        res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'

    results_dir = os.path.join(output_dir, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    os.environ['CITYSCAPES_DATASET'] = data_cfg.eval_dir
    os.environ['CITYSCAPES_RESULTS'] = output_dir

    # Load the Cityscapes eval script *after* setting the required env vars,
    # since the script reads their values into global variables (at load time).
    import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling \
        as cityscapes_eval

    for i, dets in enumerate(dets_list):
        im_name = info_list[i]['img_path']

        basename = os.path.splitext(os.path.basename(im_name))[0]
        txtname = os.path.join(output_dir, basename + 'pred.txt')
        with open(txtname, 'w') as fid_txt:
            if i % 10 == 0:
                logger.info('i: {}: {}'.format(i, basename))
            for j in range(data_cfg.num_classes):
                clss = label_names[j]
                clss_id = label_ids[j]

                for k in range(len(dets)):
                    center_cls, center_conf, center_indexe, group = dets[k]
                    if center_cls != j:
                        continue
                    score = center_conf
                    mask = mask_util.decode(group)
                    pngname = os.path.join(
                        'results',
                        basename + '_' + clss + '_{}.png'.format(k))
                    # write txt
                    fid_txt.write('{} {} {}\n'.format(pngname, clss_id, score))
                    # save mask
                    cv2.imwrite(os.path.join(output_dir, pngname), mask * 255)
    logger.write('Evaluating...')
    cityscapes_eval.main([])
    return None