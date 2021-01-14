__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"
import os
import json
import cv2
import torch
import data
import numpy as np
from tqdm import tqdm
from utils import decode


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def eval_outputs(data_cfg, dataset, eval_dataloader, model, epoch, decode_cfg, device, logger, use_salt=True):
    decode.device = device
    output_dir = data_cfg.save_dir

    eval_labels = data.get_eval_labels(dataset)
    label_names = [label[1] for label in eval_labels]
    label_ids = [label[2] for label in eval_labels]

    results_dir = os.path.join(output_dir, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # eval
    model.eval()
    num_iter = len(eval_dataloader)

    # foreach the images
    for iter_id, eval_data in tqdm(enumerate(eval_dataloader), total=num_iter,
                                   desc="eval for epoch {}".format(epoch)):
        # to device
        inputs, targets, infos = eval_data
        inputs = inputs.to(device)
        # forward the models and loss
        with torch.no_grad():
            outputs = model(inputs)
            dets, instance_maps = decode.decode_output(inputs, outputs, infos, decode_cfg, device)
        del inputs
        torch.cuda.empty_cache()

        for i in range(len(dets)):
            im_name = infos[i][0]
            img_size = infos[i][1]
            instance_map = instance_maps[i]

            basename = os.path.splitext(os.path.basename(im_name))[0]
            txtname = os.path.join(output_dir, basename + 'pred.txt')
            with open(txtname, 'w') as fid_txt:
                for j in range(data_cfg.num_classes):
                    clss = label_names[j]
                    clss_id = label_ids[j]

                    for k in range(len(dets[i])):
                        center_cls, center_conf, instance_id = dets[i][k]
                        if center_cls != j:
                            continue
                        score = center_conf
                        mask = instance_map == instance_id
                        pngname = os.path.join(
                            'results',
                            basename + '_' + clss + '_{}.png'.format(k))
                        # write txt
                        fid_txt.write('{} {} {}\n'.format(pngname, clss_id, score))
                        # save mask
                        cv2.imwrite(os.path.join(output_dir, pngname), mask * 255)

    os.environ['CITYSCAPES_DATASET'] = data_cfg.eval_dir
    os.environ['CITYSCAPES_RESULTS'] = output_dir

    # Load the Cityscapes eval script *after* setting the required env vars,
    # since the script reads their values into global variables (at load time).
    import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling \
        as cityscapes_eval

    logger.write('Evaluating...')
    cityscapes_eval.main()


def evaluate_model(data_cfg, eval_dataloader, model, epoch, dataset, decode_cfg, device, logger):
    eval_outputs(data_cfg, dataset, eval_dataloader, model, epoch, decode_cfg, device, logger)

