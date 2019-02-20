import numpy as np
import cv2
import glob
import itertools
import os
import random


# random rotation
def rotate_image_random(img, rotation_index):

    deg_dict = {
        1: 0,
        2: 90,
        3: 180,
        4: 270
    }

    rows = img.shape[0]
    cols = img.shape[1]

    if deg_dict[rotation_index] != 0:
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), deg_dict[rotation_index], 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst
    else:
        return img


# random cropping
def get_random_pos(img_shape, window_shape=(512, 512)):
    """ Extract of 2D random patch of shape window_shape in the image """

    h, w = window_shape
    H, W = img_shape
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


# Change brightness levels
def random_brightness(image):
    # Convert 2 HSV colorspace from BGR colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Generate new random brightness
    rand = random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand*hsv[:, :, 2]
    # Convert back to BGR colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img



if __name__ == "__main__":

    img = cv2.imread("./test.tif")

    # random rotation
    rotation_index = random.randint(1, 4)  # 0, 90, 180, 270 [degree]
    img = rotate_image_random(img, rotation_index)

    # random cropping
    H, W = img.shape[:2]
    x1, x2, y1, y2 = get_random_pos(img_shape=(H, W), window_shape=(300, 300))
    img = img[y1:y2, x1:x2, :]

    # Brightness jitter
    img = random_brightness(img)

