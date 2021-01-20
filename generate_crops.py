"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description="generateCrop")
# add arguments
parser.add_argument("--data_root", help="the root dir of cityscape", dest="data_root", default="", type=str)
parser.add_argument("--crop_scale", help="the scale of crop", dest="crop_scale", default=2, type=int)
parser.add_argument("--target", help="the target of crop", dest="target", default="train2", type=str)
# parse args
args = parser.parse_args()


def process(tup):
    im, inst = tup

    image_path = os.path.splitext(os.path.relpath(im, os.path.join(IMAGE_DIR, 'train')))[0]
    image_path = os.path.join(IMAGE_DIR, CROP_TARGET, image_path)
    instance_path = os.path.splitext(os.path.relpath(inst, os.path.join(INSTANCE_DIR, 'train')))[0]
    instance_path = os.path.join(INSTANCE_DIR, CROP_TARGET, instance_path)

    try:  # can't use 'exists' because of threads
        os.makedirs(os.path.dirname(image_path))
        os.makedirs(os.path.dirname(instance_path))
    except FileExistsError:
        pass

    image = Image.open(im)
    instance = Image.open(inst)
    w, h = image.size
    t_w, t_h = w//CROP_SCALE, h//CROP_SCALE

    # loop over instances
    index = 0
    for s_w in range(0, w, t_w):
        if s_w+t_w>=w:
            break
        for s_h in range(0, h, t_h):
            if s_h+t_h>=h:
                break
            im_crop = image.crop((s_w, s_h, s_w + t_w, s_h + t_h))
            instance_crop = instance.crop((s_w, s_h, s_w + t_w, s_h + t_h))

            im_crop.save(image_path + "_{:03d}.png".format(index))
            instance_crop.save(instance_path + "_{:03d}.png".format(index))
            index = index + 1


if __name__ == '__main__':
    # cityscapes dataset
    CITYSCAPES_DIR=args.data_root
    CROP_SCALE=args.crop_scale
    CROP_TARGET = args.target

    IMAGE_DIR=os.path.join(CITYSCAPES_DIR, 'leftImg8bit')
    INSTANCE_DIR=os.path.join(CITYSCAPES_DIR, 'gtFine')

    # load images/instances
    images = glob.glob(os.path.join(IMAGE_DIR, 'train', '*/*.png'))
    images.sort()
    instances = glob.glob(os.path.join(INSTANCE_DIR, 'train', '*/*instanceIds.png'))
    instances.sort()
    process((images[0], instances[0]))

    with Pool(8) as p:
        r = list(tqdm(p.imap(process, zip(images,instances)), total=len(images)))
