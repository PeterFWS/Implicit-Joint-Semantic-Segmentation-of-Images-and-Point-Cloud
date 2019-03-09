import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import cv2
import tifffile

import numpy as np
from PIL import Image
import tensorflow as tf
from PIL import Image, ImageDraw
#import skimage.filters as filters
#
def make_if_not_exists(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)


def chip(img, chip_size=(224, 224), overlap=0.5, nchannel=3, fg=False):
    if nchannel == 3:
        height, width, nchannel = img.shape
        hn, wn = chip_size
        num_chipped = (int(height / (hn * overlap)) - 1) * (int(width / (wn * overlap)) - 1)

        image_patchs = np.zeros((num_chipped, hn, wn, nchannel))

        k = 0
        for i in range(int(height / (hn * overlap)) - 1):
            for j in range(int(width / (wn * overlap)) - 1):
                chip = img[int(hn * i * overlap): int(hn * i * overlap + hn),
                       int(wn * j * overlap): int(wn * j * overlap + wn),
                       :nchannel]
                image_patchs[k] = chip
                k += 1

    elif nchannel == 1:
        height, width = img.shape
        hn, wn = chip_size
        num_chipped = (int(height / (hn * overlap)) - 1) * (int(width / (wn * overlap)) - 1)

        image_patchs = np.zeros((num_chipped, hn, wn))
        flag = np.zeros(num_chipped)

        k = 0
        for i in range(int(height / (hn * overlap)) - 1):
            for j in range(int(width / (wn * overlap)) - 1):
                chip = img[int(hn * i * overlap): int(hn * i * overlap + hn),
                       int(wn * j * overlap): int(wn * j * overlap + wn)]
                image_patchs[k] = chip

                if fg:
                    # create flag
                    chip = chip.ravel()
                    count = 0
                    for _ in range(chip.shape[0]):
                        if chip[_] == 0:
                            count += 1
                    if float(count) / (hn * wn) >= 0.8:
                        # more than 80% pixels are void
                        flag[k] = 1  # this image will not be saved
                k += 1

    if fg:
        return image_patchs, flag
    else:
        return image_patchs


def rotate_image_random(img, rotation_index):
    deg_dict = {
        1: 0,
        2: 90,
        3: 180,
        4: 270
    }

    # rows = img.shape[0]
    # cols = img.shape[1]
    #
    # deg = deg_dict[rotation_index]

    if rotation_index != 1:
        # M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), deg, 1)
        # dst = cv2.warpAffine(img, M, (cols, rows))

        dst = np.rot90(img, rotation_index-1)

        return dst

    else:
        return img


if __name__ == "__main__":

    reference = "/run/user/1003/gvfs/smb-share:server=ifpserv.ifp.uni-stuttgart.de,share=s-platte/ShuFangwen/results/lvl4_nadir/test_set/2_mask"
    data_path = "/run/user/1003/gvfs/smb-share:server=ifpserv.ifp.uni-stuttgart.de,share=s-platte/ShuFangwen/results/lvl4_nadir/test_set"
    folders_list = os.listdir(data_path)
    folders_list.remove("2_mask")
    folders_list.remove("not_use_feature")
    #folders_list.remove("1_pointlabel")

    save_path = "/data1/thesis_fangwen/mix_test"
    make_if_not_exists(save_path)

    size = (480, 480)

    mask_list = os.listdir(reference)
    length = len(mask_list)
    for l in tqdm(range(length)):

        for rotation_index in range(1, 5):

            name = mask_list[l]

            mask_path = os.path.join(reference, name)
            # name = "DSC03717.tif"
            # mask_path ='/data/fangwen/results/level3/test_set/2_mask/DSC03717.tif'
            mask = cv2.imread(mask_path, 0)
            mask = rotate_image_random(mask, rotation_index)

            mask_patchs, flag = chip(mask, chip_size=size, overlap=0.5, nchannel=1, fg=True)

            # based on this flag, we chip other image
            for folder in folders_list:
                folder_path = os.path.join(data_path, folder)
                img_path = os.path.join(folder_path, name)

                if folder_path.split("/")[-1].split("_")[-2] == "f" or folder_path.split("/")[-1].split("_")[-2] == "5":
                    # read index image and feature image
                    img = tifffile.imread(img_path)
                    img = rotate_image_random(img, rotation_index)
                    img_patchs = chip(img, chip_size=size, overlap=0.5, nchannel=1, fg=False)

                elif folder_path.split("/")[-1].split("_")[-2] == "rgb" or folder_path.split("/")[-1].split("_")[
                    -2] == "4":
                    # rgb
                    img = cv2.imread(img_path)
                    img = rotate_image_random(img, rotation_index)
                    img_patchs = chip(img, chip_size=size, overlap=0.5, nchannel=3, fg=False)

                elif folder_path.split("/")[-1].split("_")[-2] == "3":
                    # grey
                    img = cv2.imread(img_path, 0)
                    img = rotate_image_random(img, rotation_index)
                    img_patchs = chip(img, chip_size=size, overlap=0.5, nchannel=1, fg=False)

                for id in range(flag.shape[0]):

                    if flag[id] == 0:
                        # save masks
                        save_mask = os.path.join(save_path, "2_mask")
                        make_if_not_exists(save_mask)
                        cv2.imwrite(os.path.join(save_mask, name.split(".")[-2] + "_" + str(id) + '_r' + str(rotation_index) + ".tif"),
                                    mask_patchs[id])

                        # save other images
                        if folder_path.split("/")[-1].split("_")[-2] == "f" or folder_path.split("/")[-1].split("_")[
                            -2] == "5":
                            save_img = os.path.join(save_path, folder_path.split("/")[-1])
                            make_if_not_exists(save_img)
                            tifffile.imsave(os.path.join(save_img, name.split(".")[-2] + "_" + str(id) + '_r' + str(rotation_index) + ".tif"),
                                            img_patchs[id])

                        if folder_path.split("/")[-1].split("_")[-2] == "rgb" or folder_path.split("/")[-1].split("_")[
                            -2] == "4":
                            save_img = os.path.join(save_path, folder_path.split("/")[-1])
                            make_if_not_exists(save_img)
                            cv2.imwrite(os.path.join(save_img, name.split(".")[-2] + "_" + str(id) + '_r' + str(rotation_index) + ".tif"),
                                        img_patchs[id])

                        elif folder_path.split("/")[-1].split("_")[-2] == "3":
                            save_img = os.path.join(save_path, folder_path.split("/")[-1])
                            make_if_not_exists(save_img)
                            cv2.imwrite(os.path.join(save_img, name.split(".")[-2] + "_" + str(id) + '_r' + str(rotation_index) + ".tif"),
                                        img_patchs[id])
