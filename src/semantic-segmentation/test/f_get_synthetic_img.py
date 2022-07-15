import numpy as np
import my_parameters
import os

import cv2
from tqdm import tqdm
from scipy.interpolate import griddata

import time


def make_if_not_exists(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)


def img_projected(px, py, depth, myfeatures, mylabels, myindex, img_path, save_path):
    # img_path = "./data/Images/CF013540.jpg"
    print("Generation of synthetic image... \n")
    start_time = time.time()

    save_path_grey = os.path.join(save_path, "grey_train")
    save_path_color = os.path.join(save_path, "color_label_ground_truth")
    save_path_depth = os.path.join(save_path, "depth")
    save_path_feature1 = os.path.join(save_path, "feature1")
    save_path_feature2 = os.path.join(save_path, "feature2")
    save_path_feature3 = os.path.join(save_path, "feature3")
    save_path_index = os.path.join(save_path, "index")

    make_if_not_exists(save_path_grey)
    make_if_not_exists(save_path_color)
    make_if_not_exists(save_path_depth)
    make_if_not_exists(save_path_feature1)
    make_if_not_exists(save_path_feature2)
    make_if_not_exists(save_path_feature3)
    make_if_not_exists(save_path_index)

    label_value = []
    points = []
    d = []
    feature1 = []
    feature2 = []
    feature3 = []
    id = []
    img_temp = np.zeros((my_parameters.height, my_parameters.width, 3), np.uint8)

    for i in tqdm(range(0, px.shape[1])):
       if my_parameters.width > px[0, i] > 0 and my_parameters.height > py[0, i] > 0:
            label_value.append(int(mylabels[i]))
            points.append([px[0, i], py[0, i]])
            d.append(-depth[0, i])  # depth should > 0 here, we are in aerial image case
            feature1.append(myfeatures[i, 0])
            feature2.append(myfeatures[i, 1])
            feature3.append(myfeatures[i, 2])
            id.append(myindex[i])

            c = my_parameters.color_classes_int[str(int(mylabels[i]))]
            cv2.circle(img_temp, (int(px[0, i]), int(py[0, i])), 5, c, -1)
    cv2.imwrite(os.path.join(save_path_color, img_path.split("/")[-1]).replace(".jpg", "_gt_origin.jpg"), img_temp)

    points = np.array(points)
    label_value = np.array(label_value)
    d = np.array(d)
    id = np.array(id)
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    feature3 = np.array(feature3)

    # ** generate featured image for training
    X, Y = np.meshgrid(np.arange(0, my_parameters.width, 1), np.arange(0, my_parameters.height, 1))

    # * labeled image
    print("Generation of labeled image... \n")
    int_im = griddata(points, label_value, (X, Y), method='nearest').astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    closing = cv2.morphologyEx(int_im, cv2.MORPH_CLOSE, kernel).astype(np.uint8)

    cv2.imwrite(os.path.join(save_path_grey, img_path.split("/")[-1]), closing)

    # * depth image
    print("Generation of depth image... \n")
    depth_im = griddata(points, d, (X, Y), method='nearest')
    closing = cv2.morphologyEx(depth_im, cv2.MORPH_CLOSE, kernel)

    np.save(os.path.join(save_path_depth, img_path.split("/")[-1]).replace(".jpg", ""), closing)

    # * image of feature no.1
    print("Generation of image based on feature no.1 ... \n")
    f1_im = griddata(points, feature1, (X, Y), method='nearest')
    closing = cv2.morphologyEx(f1_im, cv2.MORPH_CLOSE, kernel)

    np.save(os.path.join(save_path_feature1, img_path.split("/")[-1]).replace(".jpg", ""), closing)

    # * image of feature no.2
    print("Generation of image based on feature no.2 ... \n")
    f2_im = griddata(points, feature2, (X, Y), method='nearest')
    closing = cv2.morphologyEx(f2_im, cv2.MORPH_CLOSE, kernel)

    np.save(os.path.join(save_path_feature2, img_path.split("/")[-1]).replace(".jpg", ""), closing)

    # * image of feature no.3
    print("Generation of image based on feature no.3 ... \n")
    f3_im = griddata(points, feature3, (X, Y), method='nearest')
    closing = cv2.morphologyEx(f3_im, cv2.MORPH_CLOSE, kernel)

    np.save(os.path.join(save_path_feature3, img_path.split("/")[-1]).replace(".jpg", ""), closing)

    # * index
    print("Generation of index image ... \n")
    index_im = griddata(points, id, (X, Y), method='nearest').astype(np.uint8)
    closing = cv2.morphologyEx(index_im, cv2.MORPH_CLOSE, kernel)

    np.save(os.path.join(save_path_index, img_path.split("/")[-1]).replace(".jpg", ""), closing)


    # generate ground truth color image (only for visualization, command those code if not needed)
    print("Generation of ground truth color image... \n")
    img_color_labeled = np.zeros((my_parameters.height, my_parameters.width, 3), np.uint8)

    for i in range(0, my_parameters.height):
        for j in range(0, my_parameters.width):
                if int(int_im[i,j]) == 12 or int(int_im[i,j]) == 13:
                    int_im[i,j] = 11
                r, g, b = my_parameters.color_classes_int[str(int(int_im[i,j]))]
                img_color_labeled[i,j,0] = r
                img_color_labeled[i,j,1] = g
                img_color_labeled[i,j,2] = b

    cv2.imwrite(os.path.join(save_path_color, img_path.split("/")[-1]).replace(".jpg", "_gt.jpg"), img_color_labeled)

    duration = time.time() - start_time
    print(duration, "s\n")

