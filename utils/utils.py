import random
import numpy as np
import cv2

def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def resize_image(pilimg, final_height=640, final_width=640):
    """ Resize image to desired input format - if img is gray, convert to RGB"""
    img = pilimg.resize((final_height, final_width))
    np_img = np.array(img, dtype=np.float32)
    #Convert to RGB if needed so dimensions match
    if len(np_img.shape) < 3 :
        return cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
    else:
        return np_img

def normalize_bbox(bbox, img_width, img_height):
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]

    bbox[0] = x/float(img_width)
    bbox[1] = y/float(img_height)
    bbox[2] = w/float(img_width)
    bbox[3] = h/float(img_height)

    return np.array([bbox])

def to_standard_dimension(bounding_boxes, dimension = 100):
    ''' Make the number of bounding boxes in an image a constant dimension
    Append zero bounding boxes where there were less bounding boxes '''
    null_bbox = np.array([[0.0001,0.0001,0.0001,0.0001]])

    for i in bounding_boxes.keys():
        num_bboxes = bounding_boxes[i].shape[0]
        for j in range(dimension - num_bboxes):
            bounding_boxes[i] = np.append(bounding_boxes[i], null_bbox, axis = 0)
        bounding_boxes[i] = bounding_boxes[i].flatten()
    return bounding_boxes

def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    #Making newH = newW to get a square image
    newW = int(h * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255

def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs

