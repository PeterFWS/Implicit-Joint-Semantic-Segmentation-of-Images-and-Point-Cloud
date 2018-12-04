import numpy as np
import os
import time
import glob
import itertools

import utilities

"""
    ** Global variables
        * test data: level4
"""

path_Imgs = "./data/ImgTexture/Level_4/"
path_Ori = "./data/Ori/Level_4/"

file_XYZ = "./data/features_dense_LiDAR_cloud_5cm/Data_xyz.txt"
file_Features = "./data/features_dense_LiDAR_cloud_5cm/X.txt"
file_Labels = "./data/features_dense_LiDAR_cloud_5cm/y.txt"

save_path = "./result"
utilities.make_if_not_exists(save_path)

"""
    ** Mode
        * [1] pre-processing (input data preparation)
        * [2] training (SegNet/DeepLabver2)
        * [3] prediction (SegNet/DeepLabver2)
        * [4] evaluation (in 2d or 3d space)
"""

Mode = 1  # default value

if Mode == 1:

    print("pre-processing\n")

    pt_xyz = np.loadtxt(file_XYZ)
    pt_features = np.loadtxt(file_Features)
    pt_labels = np.loadtxt(file_Labels)

    index = np.asarray([_ for _ in range(pt_labels.shape[0])])

    try:
        assert path_Imgs[-1] == "/"
    except AssertionError:
        path_Imgs = path_Imgs + "/"

    try:
        assert path_Ori[-1] == "/"
    except AssertionError:
        path_Ori = path_Ori + "/"

    img_list = glob.glob(path_Imgs + "*.tif") + glob.glob(path_Imgs + "*.jpg") + glob.glob(path_Imgs + "*.png") \
               + glob.glob(path_Imgs + "*.jpeg")
    img_list.sort()
    ori_list = glob.glob(path_Ori + "*.ori")
    ori_list.sort()

    assert len(img_list) == len(ori_list)
    for im_, ori_ in zip(img_list, ori_list):
        assert (im_.split('/')[-1].split(".")[0] == ori_.split('/')[-1].split(".")[0])

    zipped = itertools.cycle(zip(img_list, ori_list))

    for _ in range(len(img_list)):
        # img, ori = zipped.__next__()

        file_img = "./data/ImgTexture/Level_4/DSC07248.tif"
        file_ori =  "./data/Ori/Level_4/DSC07248.ori"

        # get interior and exterior orientation
        f, pixel_size_x, pixel_size_y, img_width, img_height, x0, y0, K, R, Xc, Yc, Zc \
            = utilities.get_INTER_and_EXTER_Orientations(file_ori)

        # frustum culling
        assert pixel_size_x == pixel_size_y
        xyz_temp, index_temp = utilities.frustum_culling(Xc, Yc, Zc, f, img_height, img_width, pixel_size_x, pt_xyz, index)

        # hidden point removal
        myPoints, myIndex = utilities.HPR(Xc, Yc, Zc, xyz_temp, index_temp)

        # projection
        px, py = utilities.pointcloud2pixelcoord(R, K, Xc, Yc, Zc, myPoints)

        # generation of synthetic images
        utilities.img_projected(px, py, myIndex, file_img, save_path)

        break


elif Mode == 2:
    print("training\n")
elif Mode == 3:
    print("prediction\n")
elif Mode == 4:
    print("evaluation\n")
