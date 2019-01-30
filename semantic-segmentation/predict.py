# import argparse
import Models, LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import itertools
import os


def make_if_not_exists(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

color_classes_int = { #BGR
    "0": (255, 0, 0),
    "1": (255, 255, 255),
    "2": (255, 255, 0),
    "3": (255, 0, 255),
    "4": (0, 255, 255),
    "5": (0, 255, 0),
    "6": (0, 0, 255),
    "7": (239, 120, 76),
    "8": (247, 238, 179),
    "9": (0, 18, 114),
    "10": (63, 34, 15),
    "11": (0, 0, 0)  # number 11 indicts for nothing
}

n_classes = 12
images_path = "/data/fangwen/results/level3/test_set/rgb_img/"
masks_path = "/data/fangwen/results/level3/test_set/2_mask/"




input_height = 480
input_width = 736

##################
train_mode = "RGB"
##################
if train_mode == "RGB":
    m = Models.Segnet.segnet_indices_pooling(n_classes, input_height=input_height, input_width=input_width,
                                             nchannel=3, pre_train=False)
    m.load_weights("/home/fangwen/ShuFangwen/source/image-segmentation-keras/weights/end_weights.h5")

    f_path = None

elif train_mode == "multi_modality":
    m = Models.Segnet.segnet_indices_pooling(n_classes, input_height=input_height, input_width=input_width,
                                             nchannel=75, pre_train=False)
    m.load_weights("/home/fangwen/ShuFangwen/source/image-segmentation-keras/weights/bestweights.05-0.26.hdf5")

    f_path = "/data/fangwen/results/level3/test_set/"


output_height = input_height
output_width = input_width


assert images_path[-1] == '/'
assert masks_path[-1] == '/'
images = glob.glob(images_path + "*.JPG") + glob.glob(images_path + "*.tif")
images.sort()
masks = glob.glob(masks_path + "*.JPG") + glob.glob(masks_path + "*.tif")
masks.sort()
assert len(images) == len(masks)
for im, mask in zip(images, masks):
    assert (im.split('/')[-1].split(".")[0] == mask.split('/')[-1].split(".")[0])

if f_path is not None:
    assert f_path[-1] == '/'
    f_folders = glob.glob(f_path + "f_*" + "/")
    f_folders.sort()

    for folder_path in f_folders:
        f_imgs = glob.glob(folder_path + "*.JPG") + glob.glob(folder_path + "*.tif")
        f_imgs.sort()
        assert len(images) == len(f_imgs)
        for im, f_img in zip(images, f_imgs):
            assert (im.split('/')[-1].split(".")[0] == f_img.split('/')[-1].split(".")[0])
else:
    f_folders = None

zipped = itertools.cycle(zip(images, masks))

for _ in range(0, len(images)):
    imgName, mask_imName = zipped.next()
    X = LoadBatches.getImageArr(imgName, mask_imName, f_folders, input_width, input_height)
    pr = m.predict(np.array([X]))[0]  # (1, 375000, 12) -> (375000, 12)
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
    seg_img = np.zeros((output_height, output_width, 3))
    for c in range(n_classes-1):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (color_classes_int[str(c)][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (color_classes_int[str(c)][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (color_classes_int[str(c)][2])).astype('uint8')
    seg_img = cv2.resize(seg_img, (input_width, input_height))

    save_path = "/data/fangwen/results/level3/predictions_end"
    make_if_not_exists(save_path)
    cv2.imwrite(os.path.join(save_path, imgName.split("/")[-1]), seg_img)
