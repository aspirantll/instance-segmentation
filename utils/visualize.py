import os

__copyright__ = \
    """
    This code is based on
        # CenterNet (https://github.com/xingyizhou/CenterNet)
    Copyright &copyright © (c) 2020 The Board of xx University.
    All rights reserved.

    This software is covered by China patents and copyright.
    This source code is to be used for academic research purposes only, and no commercial use is allowed.
    """
__authors__ = ""
__version__ = "1.0.0"

from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from PIL import Image
import numpy as np
import seaborn as sns


def visualize_instance(instances):
    ax = plt.gca()
    polygons = []
    color = []
    for obj in instances:
        center_cls, center_confs, center_index, groups = obj
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        poly = np.vstack(groups)
        polygons.append(Polygon(poly))
        color.append(c)

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)


def visualize_objs(objs):
    ax = plt.gca()
    polygons = []
    color = []
    for obj in objs:
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        polygons.append(Polygon(obj.reshape(-1, 2)))
        color.append(c)

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)


def visualize_kp(kps, color=".r"):
    for kp in kps:
        plt.plot(kp[0], kp[1], color)


def visualize_obj_points(kps, centers, path, suffix, colors=(".r", ".b")):
    img = Image.open(path)
    plt.imshow(img)
    visualize_kp(kps, colors[0])
    visualize_kp(centers.reshape(-1, 2), colors[1])


def visualize_mask(mask):
    ax = plt.gca()
    color_mask = np.random.random((1, 3)).tolist()[0]
    img = np.ones((mask.shape[0], mask.shape[1], 3))
    for i in range(3):
        img[:, :, i] = color_mask[i]
    ax.imshow(np.dstack((img, mask * 0.5)))


def visualize_box(centers, box_sizes, center_color='.r'):
    ax = plt.gca()
    polygons = []
    color = []
    for center, box_size in zip(centers, box_sizes):
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        w, h = box_size//2
        c_w, c_h = center
        plt.plot(c_w, c_h, center_color)
        polygons.append(Polygon(np.array([[c_w - w, c_h - h],
                                          [c_w - w, c_h + h],
                                          [c_w + w, c_h + h],
                                          [c_w + w, c_h - h]]).reshape(-1, 2)))
        color.append(c)

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)


def visualize_hm(hm):
    sns.set()
    sns.heatmap(hm, cmap='rainbow')
