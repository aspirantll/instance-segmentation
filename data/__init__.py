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
from data.dataset import is_train_phase, is_val_phase
from data import cityscapes
from data.dir import DirDatasetBuilder


datasetBuildersMap = {
    "cityscapes": cityscapes.CityscapesDatasetBuilder,
    "dir": DirDatasetBuilder
}

datasetClsNumMap = {
    "cityscapes": cityscapes.num_cls
}

datasetEvalLabelMap = {
    "cityscapes": cityscapes.eval_names
}


def get_eval_labels(datatype):
    return datasetEvalLabelMap[datatype]


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
    labels[-1] = torch.stack(labels[-1])
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


def get_dataloader(batch_size, dataset_type, data_dir, phase, input_size, transforms=None
                   , ann_file=None, num_workers=0, random=False, from_file=False, with_label=True):
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
    dataset = dataset_builder.get_dataset(input_size=input_size, transforms=transforms, from_file=from_file)
    if with_label:
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
