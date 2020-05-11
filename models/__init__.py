__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

from . import erfnet, loss, dla

ERFNet = erfnet.ERFNet
DLASeg = dla.DLASeg
ClsFocalLoss = loss.ClsFocalLoss
AELoss = loss.AELoss
ComposeLoss = loss.ComposeLoss
KPFocalLoss = loss.KPFocalLoss
WHLoss = loss.WHLoss
WHDLoss = loss.WHDLoss


model_map = {
    "erf": ERFNet,
    "dla": DLASeg
}


def create_model(model_type, num_classes):
    if model_type not in model_map:
        raise ValueError("model_type must be in {}".format(model_map.keys()))
    heads = {
        "hm_cls": num_classes,
        "hm_kp": 1,
        "ae": 2,
        "wh": 2
    }
    model_class = model_map[model_type]
    return model_class(heads, num_classes)


