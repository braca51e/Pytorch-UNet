#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image

from .utils import *


def get_ids(dir):
    """Returns a list of the ids in the directory
    and strips the heading zeros"""
    return (f[:-4].lstrip("0") for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.gif', scale)

    return zip(imgs_normalized, masks)

def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)

def to_resized_imgs(ids, dir, suffix):
    for id in ids:
        im = resize_image(Image.open(dir + id + suffix))
        yield im

def load_annotation_information(annotation_filepath):
    """ Load all BBOX and other image labels """
    with open(annotation_filepath) as f:
        data = json.load(f)
    
    # First load image ids, height and width 
    # (will be useful for normalizing bounding boxes)
    img_dimensions = {}
    for i in range(len(data['images'])):
        image_info = data['images'][i]
        img_dimensions[image_info['id']] = [image_info['width'], image_info['height']]

    # Then load all bounding boxes into dictionary
    # Referenced by image id
    bounding_boxes = {}

    for j in range(len(data['annotations'])):
        #Get image information from previous load
        image_annotation = data['annotations'][j]
        image_id = image_annotation['image_id']
        image_width = image_dimensions[image_id][0]
        image_height = image_dimensions[image_id][1]
        
        # Normalize bounding boxes
        new_bbox = normalize_bbox(image_annotation['bbox'], image_width, img_height)

        # Get bounding boxes in dict
        if image_id in bounding_boxes:
            bounding_boxes[image_id] = np.append(bounding_boxes[image_id], new_bbox, axis = 0)
        else:
            bounding_boxes[image_id] = new_bbox
    
    return img_dimensions, bounding_boxes

def get_imgs_and_bboxes(ids, dir_img, dir_bbox, annotation_filepath):
    """  Load & resize images, get labels (bounding boxes) """
    img_dimensions, bounding_boxes = load_annotation_information(annotation_filepath)
    #Resize images to match input format 
    #Normalize and switch to CHW
    imgs = to_resized_imgs(ids, dir_img, '.jpg')
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)
    
    """ TO DO :  
    # Add zeros when less than 100 bounding boxes
    #bboxes = bounding_boxes
    """
    return zip(imgs_normalized, bboxes)
