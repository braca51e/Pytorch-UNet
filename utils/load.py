#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw
from xml.etree import ElementTree as ET

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i)  for id in ids for i in range(n))


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)

def to_label_imgs(ids, dir, suffix):
    """Read xml file and return vector with normalized locations"""
    for id, pos in ids:

        tree = ET.parse(dir + id + suffix)
        root = tree.getroot()

        list_with_all_boxes = []

        for boxes in root.iter('object'):
            filename = root.find('filename').text

            height_img = int(root.find('size').find('height').text)
            width_img = int(root.find('size').find('width').text)

            ymin, xmin, ymax, xmax = None, None, None, None

            for box in boxes.findall("bndbox"):
                ymin = float(box.find("ymin").text)/height_img
                xmin = float(box.find("xmin").text)/width_img
                ymax = float(box.find("ymax").text)/height_img
                xmax = float(box.find("xmax").text)/width_img

                width = xmax-xmin
                height = ymax-ymin

        list_with_all_boxes.append([xmin, ymin, width, height])

        yield list_with_all_boxes

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    #masks = to_cropped_imgs(ids, dir_mask, '_mask.gif', scale)
    labels = to_label_imgs(ids, dir_mask, '.xml')

    return zip(imgs_normalized, labels)

def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)
