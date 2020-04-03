__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

from . import erfnet, loss

ERFNet = erfnet.ERFNet
ClsFocalLoss = loss.ClsFocalLoss
AELoss = loss.AELoss
ComposeLoss = loss.ComposeLoss
KPFocalLoss = loss.KPFocalLoss
KPLSLoss = loss.KPLSLoss
WHDLoss = loss.WHDLoss
