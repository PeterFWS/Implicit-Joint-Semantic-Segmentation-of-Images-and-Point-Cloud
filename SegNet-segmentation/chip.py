import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import cv2

def make_if_not_exists(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)


def chip_image(img, chip_size=(512, 512), overlap=0.5):

    # img = cv2.imread("/home/fangwen/ShuFangwen/data/ImgTexture/Level_0_selected/DSC03146.JPG")

    height, width, nchannel = img.shape
    hn, wn = chip_size
    num_chipped = (int(height / (hn * overlap)) - 1) * (int(width / (wn * overlap)) - 1)

    if nchannel == 3:

        images = np.zeros((num_chipped, hn, wn, nchannel))
        flag = np.zeros(num_chipped)

        k=0
        for i in range(int(height / (hn*overlap)) - 1):
            for j in range(int(width / (wn*overlap)) - 1):

                chip = img[ int(hn*i*overlap) : int(hn*i*overlap+hn),
                            int(wn*j*overlap) : int(wn*j*overlap+wn),
                            :nchannel ]
                images[k] = chip
                k += 1

                # create flag
                chip = chip[:,:,0].ravel()
                count = 0
                for _ in range(chip.shape[0]):
                    if chip[_] == 0:
                        count += 1
                if count / (hn*wn) == 1:
                    flag[k] = 1  # this image will not be trained


        # save_path = "/home/fangwen/ShuFangwen/test_chip"
        # make_if_not_exists(save_path)
        # for i in range(images.shape[0]):
        #     cv2.imwrite(os.path.join(save_path, str(i)+".JPG"), images[i])



    elif nchannel == 1:

        images = np.zeros((num_chipped, hn, wn))
        flag = np.zeros(num_chipped)

        k=0
        for i in range(int(height / (hn*overlap)) - 1):
            for j in range(int(width / (wn*overlap)) - 1):

                chip = img[ int(hn*i*overlap) : int(hn*i*overlap+hn),
                            int(wn*j*overlap) : int(wn*j*overlap+wn) ]
                images[k] = chip
                k += 1

                # create flag
                chip = chip[:,:,0].ravel()
                count = 0
                for _ in range(chip.shape[0]):
                    if chip[_] == 0:
                        count += 1
                if count / (hn*wn) == 1:
                    flag[k] = 1  # this image will not be trained

    return images, flag





