__copyright__ = \
    """
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

import yaml


class Config(object):
    def __init__(self, cfg_path=None, cfg=None):
        if cfg_path is None and cfg is None:
            raise ValueError("the path or cfg must be not None.")
        if cfg is None:
            with open(cfg_path, "r", encoding='utf-8') as f:
                cfg_str = f.read()
                self._cfg = yaml.load(cfg_str)
        else:
            self._cfg = cfg
        # scan the configs, then set attribute
        self.load_dict_as_attributes(self._cfg)

    def load_dict_as_attributes(self, cfg):
        for k, v in cfg.items():
            if isinstance(v, dict):
                sub_obj = Config(cfg=v)
                self.__setattr__(k, sub_obj)
            else:
                self.__setattr__(k, v)

    def __str__(self):
        return str(self.__dict__)
