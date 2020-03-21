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
import numpy as np
from data.dataset import is_train_phase, is_val_phase
from data import coco
from data import cityscapes
from data.dir import DirDatasetBuilder


datasetBuildersMap = {
    "coco": coco.COCODatasetBuilder,
    "cityscapes": cityscapes.CityscapesDatasetBuilder,
    "dir": DirDatasetBuilder
}

datasetClsNumMap = {
    "coco": coco.num_cls,
    "cityscapes": cityscapes.num_cls,
}


def get_cls_num(datatype):
    return datasetClsNumMap[datatype]


def collate_fn_with_label(batch):
    """
    merge the batch inputs
    :param batch:
    :return:
    """
    batch_inputs = [e for e in zip(*batch)]
    input_tensors = torch.stack(batch_inputs[0])
    labels = [e for e in zip(*batch_inputs[1])]
    trans_infos = batch_inputs[2]
    return input_tensors, labels, trans_infos


def collate_fn_without_label(batch):
    """
    merge the batch inputs
    :param batch:
    :return:
    """
    batch_inputs = [e for e in zip(*batch)]
    input_tensors = torch.stack(batch_inputs[0])
    trans_infos = batch_inputs[1]
    return input_tensors, trans_infos


def get_dataloader(batch_size, dataset_type, data_dir, phase, input_size, transforms=None, ann_file=None, num_workers=1, random=False):
    """
    initialize the data loader, and then return a data loader
    :param num_workers: worker num
    :param phase: "train", "test", "val"
    :param input_size: tuple(height * width)
    :param batch_size:
    :param dataset_type:
    :param data_dir:
    :param ann_file:
    :param random:
    :param transforms:
    :return:
    """
    # determine the class of DatasetBuilder by dataset type
    dataset_builder_class = datasetBuildersMap[dataset_type]
    # initialize dataset
    dataset_builder = dataset_builder_class(data_dir, phase, ann_file)
    dataset = dataset_builder.get_dataset(input_size=input_size, transforms=transforms)
    if is_train_phase(phase) or is_val_phase(phase):
        if is_train_phase(phase):
            # initialize sampler
            if random:
                sampler = torch.utils.data.RandomSampler(dataset)
                batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)
                return torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn_with_label
                                                   , num_workers=num_workers)
            else:
                return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_with_label
                                               , num_workers=num_workers)
        else:
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_with_label
                                                   , num_workers=num_workers)
    else:
        collate_fn = collate_fn_without_label
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           collate_fn=collate_fn
                                           , num_workers=num_workers)


if __name__== "__main__":
    data_dir = r"C:\data\cityscapes"
    save_dir = r"C:\data\temp"
    data_type = "cityscapes"
    batch_size = 2
    phase = "train"
    input_size = (512, 512)
    num_cls = 19
    import os.path as op
    from utils.tranform import TrainTransforms
    from PIL import Image
    import cv2
    from matplotlib import pyplot as plt
    from utils.visualize import visualize_box
    import torchvision.transforms.functional as F

    transforms = TrainTransforms(input_size, num_cls, with_flip=True)

    data_loader = get_dataloader(batch_size, data_type, data_dir, phase, input_size, transforms)
    print("length:", len(data_loader))
    for iter_id, train_data in enumerate(data_loader):
        inputs = train_data[0]
        _, centers_list, polygons_list = train_data[1]
        trans_infos = train_data[2]
        for i in range(inputs.shape[0]):
            trans_info = trans_infos[i]
            centers = centers_list[i]
            polygons = polygons_list[i]

            key = iter_id * batch_size + i
            cv_img = cv2.imread(trans_info.img_path)

            polygons = [transforms.transform_pixel(polygon, trans_info) for polygon in polygons]
            centers = transforms.transform_pixel(np.vstack(centers), trans_info)

            img_c = cv2.drawKeypoints(cv_img, cv2.KeyPoint_convert(np.vstack(polygons).reshape((-1, 1, 2))), None,
                                      color=(255, 0, 0))
            cv2.imwrite(op.join(save_dir, "{}_recover_kp.png".format(key)), img_c)

            box_sizes = [polygon.max(0) - polygon.min(0) for polygon in polygons]
            fig = plt.figure(trans_info.img_path)
            pil_img = Image.open(trans_info.img_path)
            plt.imshow(pil_img)
            visualize_box(centers, box_sizes)
            plt.savefig(op.join(save_dir, "{}_box_kp.png".format(key)))
            plt.close(fig)






