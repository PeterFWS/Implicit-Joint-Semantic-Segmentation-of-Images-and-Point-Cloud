import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import cv2
import tifffile


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


if __name__ == "__main__":

    reference = "/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_nadir/train_set/2_mask"
    data_path = "/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_nadir/train_set"
    folders_list = os.listdir(data_path)
    folders_list.remove("2_mask")
    folders_list.remove("1_pointlabel")

    save_path = "/data/fangwen/nadir_train"
    make_if_not_exists(save_path)

    mask_list = os.listdir(reference)
    length = len(mask_list)
    for l in tqdm(range(length)):

        name = mask_list[l]

        mask_path = os.path.join(reference, name)
        # name = "DSC03717.tif"
        # mask_path ='/data/fangwen/results/level3/test_set/2_mask/DSC03717.tif'
        mask = cv2.imread(mask_path, 0)
        mask_patchs, flag = chip(mask, chip_size=(480,480), overlap=0.5, nchannel=1, fg=True)

        # based on this flag, we chip other image
        for folder in folders_list:
            folder_path = os.path.join(data_path, folder)
            img_path = os.path.join(folder_path, name)

            if folder_path.split("/")[-1].split("_")[-2] == "f" or folder_path.split("/")[-1].split("_")[-2] == "5":
                # read index image and feature image
                img = tifffile.imread(img_path)
                img_patchs = chip(img, chip_size=(480,480), overlap=0.5, nchannel=1, fg=False)

            elif folder_path.split("/")[-1].split("_")[-2] == "rgb" or folder_path.split("/")[-1].split("_")[-2] == "4":
                # rgb
                img = cv2.imread(img_path)
                img_patchs = chip(img, chip_size=(480,480), overlap=0.5, nchannel=3, fg=False)

            elif folder_path.split("/")[-1].split("_")[-2] == "3":
                # grey
                img = cv2.imread(img_path, 0)
                img_patchs = chip(img, chip_size=(480,480), overlap=0.5, nchannel=1, fg=False)

            for id in range(flag.shape[0]):

                if flag[id] == 0:
                    # save masks
                    save_mask = os.path.join(save_path, "2_mask")
                    make_if_not_exists(save_mask)
                    cv2.imwrite(os.path.join(save_mask, name.split(".")[-2] + "_" + str(id) + ".tif"), mask_patchs[id])

                    # save other images
                    if folder_path.split("/")[-1].split("_")[-2] == "f" or folder_path.split("/")[-1].split("_")[
                        -2] == "5":
                        save_img = os.path.join(save_path, folder_path.split("/")[-1])
                        make_if_not_exists(save_img)
                        tifffile.imsave(os.path.join(save_img, name.split(".")[-2] + "_" + str(id) + ".tif"),
                                        img_patchs[id])

                    if folder_path.split("/")[-1].split("_")[-2] == "rgb" or folder_path.split("/")[-1].split("_")[-2] == "4":
                        save_img = os.path.join(save_path, folder_path.split("/")[-1])
                        make_if_not_exists(save_img)
                        cv2.imwrite(os.path.join(save_img, name.split(".")[-2] + "_" + str(id) + ".tif"),
                                    img_patchs[id])

                    elif folder_path.split("/")[-1].split("_")[-2] == "3":
                        save_img = os.path.join(save_path, folder_path.split("/")[-1])
                        make_if_not_exists(save_img)
                        cv2.imwrite(os.path.join(save_img, name.split(".")[-2] + "_" + str(id) + ".tif"),
                                    img_patchs[id])
