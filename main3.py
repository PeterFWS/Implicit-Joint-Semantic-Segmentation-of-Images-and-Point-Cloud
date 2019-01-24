"""
    label image based on 5cm point cloud
    feature map based on 10cm point cloud

    train_set
"""

import numpy as np
import time
import glob
import itertools
import os
from tqdm import tqdm

from utilities import make_if_not_exists


print("----------------------------pre-processing----------------------------\n")

from utilities import get_INTER_and_EXTER_Orientations
from utilities import frustum_culling
from utilities import HPR
from utilities import pointcloud2pixelcoord

from utilities import generation_syntheticImg_5cmbased
from utilities import generation_syntheticImg_10cmbased


path_Imgs = "./data/ImgTexture/Level_0_selected/"
path_Ori = "./data/Ori/Level_0/"

file_pc_5cm = "./data/data_splits_5cm_onlylabel/train_xyz_y.txt"

file_XYZ = "./data/data_splits_10cm/xyz_10cm_train.txt"
file_Features = "./data/data_splits_10cm/X_10cm_train.txt"
save_path = "./results/level0/train_set/"
make_if_not_exists(save_path)

#
pt_data_5cm = np.loadtxt(file_pc_5cm)
pt_xyz = pt_data_5cm[:, :3]  # (84254800, 3)
pt_labels = pt_data_5cm[:, 3]  # (84254800,)
index = np.asarray([_ for _ in range(pt_labels.shape[0])]).astype(np.int)

pt_xyz2 = np.loadtxt(file_XYZ)  # (9559941, 3)
pt_features = np.loadtxt(file_Features)  # (9559941, 72)
index2 = np.asarray([_ for _ in range(pt_xyz2.shape[0])]).astype(np.int)


path_Imgs_referrence = "/home/fangwen/ShuFangwen/source/image-segmentation-keras/data/train_set/1_pointlabel"

img_list = os.listdir(path_Imgs_referrence)
for i in tqdm(range(len(img_list))):

    img_name = img_list[i]
    ori_name = img_name.replace("tif", "ori")

    # img_name = "DSC04233.tif"
    # ori_name = "DSC04233.ori"


    print("processing image: {0}\n".format(img_name))

    # get interior and exterior orientation
    f, pixel_size, img_width, img_height, K, R, Xc, Yc, Zc = get_INTER_and_EXTER_Orientations(os.path.join(path_Ori,ori_name))

    # 5cm
    myPoints, myIndex = HPR(Xc, Yc, Zc, pt_xyz, index)
    px, py = pointcloud2pixelcoord(R, K, Xc, Yc, Zc, myPoints)
    mask = generation_syntheticImg_5cmbased(px, py, myIndex, pt_labels, img_name, save_path, img_width, img_height)

    # 10cm
    myPoints2, myIndex2 = HPR(Xc, Yc, Zc, pt_xyz2, index2)
    px2, py2 = pointcloud2pixelcoord(R, K, Xc, Yc, Zc, myPoints2)
    generation_syntheticImg_10cmbased(mask, px2, py2, myIndex2, pt_features, img_name, save_path, img_width, img_height)