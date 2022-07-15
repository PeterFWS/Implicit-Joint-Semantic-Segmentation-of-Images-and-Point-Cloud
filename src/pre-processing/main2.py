"""
    working on cropping nadir image and small block 5cm point cloud
"""

import numpy as np
import time
import glob
import itertools
import os
from tqdm import tqdm
import utilities
from utilities import make_if_not_exists

from utilities import get_INTER_and_EXTER_Orientations
from utilities import frustum_culling
from utilities import HPR
from utilities import pointcloud2pixelcoord

from utilities import generation_syntheticImg_nadir


path_Imgs = "./data/nadir_imgs_data/Images/"
file_Ori = "./data/nadir_imgs_data/All Camera Orientations/txt/extOri.txt"

file_XYZ = "./data/nadir_imgs_data/features_dense_LiDAR_cloud_5cm/Data_xyz.txt"
file_Features = "./data/nadir_imgs_data/features_dense_LiDAR_cloud_5cm/X.txt"
file_Labels = "./data/nadir_imgs_data/features_dense_LiDAR_cloud_5cm/y.txt"

save_path = "./results/nadir/"
make_if_not_exists(save_path)

print("reading data from txt file...")
start_time = time.time()
pt_xyz = np.loadtxt(file_XYZ)
pt_features = np.loadtxt(file_Features)
pt_labels = np.loadtxt(file_Labels)
duration = time.time() - start_time
print("which costs {0}s\n".format(duration))

index = np.asarray([_ for _ in range(pt_labels.shape[0])]).astype(np.int)




img_list = os.listdir(path_Imgs)
for i in tqdm(range(len(img_list))):


    img_name = img_list[i]

    # Orientation
    ex_data = utilities.get_exterior_orientation(img_name, file_Ori).split("\t")

    # HPR
    Xc, Yc, Zc = map(float, ex_data[1:4])
    myPoints, myIndex = HPR(Xc, Yc, Zc, pt_xyz, index)

    # Projection
    f = -51.6829425484485650 / 1000  # [mm -> m] different sign for aerial image
    pixel_size = 0.0045999880303564 / 1000  # [mm -> m]
    x0 = 5798.5783629179004000  # [pixel] principle point
    y0 = 4358.1365279104657000  # [pixel]

    img_width = 11608
    img_height = 8708

    PhotoID = ex_data[0]
    X, Y, Z, Omega, Phi, Kappa, r11, r12, r13, r21, r22, r23, r31, r32, r33 = map(float, ex_data[1:])
    R = np.matrix([[r11, r12, r13],
                   [r21, r22, r23],
                   [r31, r32, r33]])

    X0 = np.matrix([X, Y, Z]).T

    Rt = np.concatenate((R, -np.dot(R, X0)), axis=1)

    K = np.matrix([[f / pixel_size, 0, x0],
                   [0, -f / pixel_size, y0],
                   [0, 0, 1]])

    px, py = pointcloud2pixelcoord(R, K, Xc, Yc, Zc, myPoints)

    #
    generation_syntheticImg_nadir(px, py, myIndex, pt_labels, pt_features, img_name, save_path, img_width, img_height)




