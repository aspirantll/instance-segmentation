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
import numpy as np


def is_label(filename):
    return filename.endswith("_polygons.json")


if __name__ == "__main__":
    root = r"C:\data\cityscapes\gtFine"
    subset = ["train", "test", "val"]
    labels_roots = [os.path.join(root, sub) for sub in subset]
    max_x, max_y, min_x, min_y = 0, 0, np.inf, np.inf
    for labels_root in labels_roots:
        filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(labels_root)) for f in
                   fn if is_label(f)]
        for fileGT in filenamesGt:
            with open(fileGT, 'r') as f:
                label_data = json.load(f)
                polygons = [np.array(obj["polygon"]) for obj in label_data["objects"]]
                np_arr = np.vstack(polygons)
                cur_max = np_arr.max(0)
                cur_min = np_arr.min(0)
                if max_x < cur_max[0]:
                    max_x = cur_max[0]
                if max_y < cur_max[1]:
                    max_y = cur_max[1]
                if min_x > cur_min[0]:
                    min_x = cur_min[0]
                if min_y > cur_min[0]:
                    min_y = cur_min[0]
    print("max:{}, {}, min:{}, {}".format(max_x, max_y,min_x, min_y))

