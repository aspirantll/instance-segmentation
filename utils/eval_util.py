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
import uuid
import json
import cv2
import torch
import data
import numpy as np
from tqdm import tqdm
from utils import decode
from utils.image import poly_to_mask


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


def eval_outputs(output_dir, eval_dataloader, transforms, model, epoch, decode_cfg, device, logger):
    dets_path = os.path.join(output_dir, "{}_dets.json".format(epoch))
    infos_path = os.path.join(output_dir, "{}_infos.json".format(epoch))

    if not os.path.exists(dets_path) or not os.path.exists(infos_path):
        decode.device = device

        # eval
        model.eval()
        num_iter = len(eval_dataloader)

        dets_list = []
        info_list = []
        # foreach the images
        for iter_id, eval_data in tqdm(enumerate(eval_dataloader), total=num_iter,
                                       desc="eval for epoch {}".format(epoch)):
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

        logger.write("[{}] finish evaluate step".format(epoch))
        dets_json = json.dumps(dets_list, cls=NpEncoder)
        info_json = json.dumps(info_list, cls=NpEncoder)
        with open(dets_path, 'w') as f:
            f.write(dets_json)
        with open(infos_path, 'w') as f:
            f.write(info_json)
        logger.write("[{}] finish save step".format(epoch))


def evaluate_from_json(data_cfg, epoch, output_dir, dataset, logger, use_salt=True):
    dets_list = json.load(open(os.path.join(output_dir, "{}_dets.json".format(epoch))))
    info_list = json.load(open(os.path.join(output_dir, "{}_infos.json".format(epoch))))

    eval_labels = data.get_eval_labels(dataset)
    label_names = [label[1] for label in eval_labels]
    label_ids = [label[2] for label in eval_labels]

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
        im_name = info_list[i][0]
        img_size = info_list[i][1]

        basename = os.path.splitext(os.path.basename(im_name))[0]
        txtname = os.path.join(output_dir, basename + 'pred.txt')
        with open(txtname, 'w') as fid_txt:
            if i % 10 == 0:
                logger.write('i: {}: {}'.format(i, basename))
            for j in range(data_cfg.num_classes):
                clss = label_names[j]
                clss_id = label_ids[j]

                for k in range(len(dets)):
                    center_cls, center_conf, _, group = dets[k]
                    if center_cls != j:
                        continue
                    score = center_conf
                    mask = poly_to_mask(np.array(group), img_size=img_size)
                    pngname = os.path.join(
                        'results',
                        basename + '_' + clss + '_{}.png'.format(k))
                    # write txt
                    fid_txt.write('{} {} {}\n'.format(pngname, clss_id, score))
                    # save mask
                    cv2.imwrite(os.path.join(output_dir, pngname), mask * 255)
    logger.write('Evaluating...')
    cityscapes_eval.main()


def evaluate_model(data_cfg, eval_dataloader, transforms, model, epoch, dataset, decode_cfg, device, logger):
    eval_outputs(data_cfg.save_dir, eval_dataloader, transforms, model, epoch, decode_cfg, device, logger)
    evaluate_from_json(data_cfg, epoch, data_cfg.save_dir, dataset, logger)

