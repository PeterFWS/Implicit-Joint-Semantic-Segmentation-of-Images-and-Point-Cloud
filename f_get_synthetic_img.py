import numpy as np
import my_parameters
import os

import cv2
from tqdm import tqdm
from scipy.interpolate import griddata


def make_if_not_exists(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)


def img_projected(px, py, depth, mylabels, color_classes, img_path, save_path):
    # img_path = "./data/Images/CF013540.jpg"
    print("Generation of synthetic image...")

    save_path_grey = os.path.join(save_path, "grey_train")
    # save_path_color = os.path.join(save_path, "color_train")
    make_if_not_exists(save_path_grey)
    # make_if_not_exists(save_path_color)

    samples_color_value = []
    points = []
    # d = []
    for i in tqdm(range(0, px.shape[1])):
       if my_parameters.width > px[0, i] > 0 and my_parameters.height > py[0, i] > 0:
            samples_color_value.append(int(mylabels[0, i]))
            points.append([px[0, i], py[0, i]])
            # d.append(depth[0,i])

    points = np.array(points)
    samples_color_value = np.array(samples_color_value)
    # d = np.array(d)

    X, Y = np.meshgrid(np.arange(0, my_parameters.width, 1), np.arange(0, my_parameters.height, 1))

    int_im = griddata(points, samples_color_value, (X, Y), method='nearest')
    int_im = int_im.astype(np.uint8)


    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    closing = cv2.morphologyEx(int_im, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(save_path_grey, img_path.split("/")[-1]), closing)

    # img_color_labeled = np.zeros((my_parameters.height, my_parameters.width, 3), np.uint8)

    # for i in range(0, my_parameters.height):
    #     for j in range(0, my_parameters.width):
    #             if int(int_im[i,j]) == 12 or int(int_im[i,j]) == 13:
    #                 int_im[i,j] = 11
    #             r, g, b = my_parameters.color_classes_int[str(int(int_im[i,j]))]
    #             img_color_labeled[i,j,0] = r
    #             img_color_labeled[i,j,1] = g
    #             img_color_labeled[i,j,2] = b

    # cv2.imwrite(os.path.join(save_path_color, img_path.split("/")[-1]).replace(".jpg", "_lable_color.jpg"), img_color_labeled)

