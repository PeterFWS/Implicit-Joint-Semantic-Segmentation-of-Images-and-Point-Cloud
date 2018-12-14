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
from utilities import generation_syntheticImg


path_reference = "./results/for_LiDAR_5cm/selected_level3/color_labeled_points"

path_Imgs = "./data/ImgTexture/Level_3/"
path_Ori = "./data/Ori/Level_3/"

file_XYZ = "./data/features_dense_LiDAR_cloud_5cm/Data_xyz.txt"
file_Features = "./data/features_dense_LiDAR_cloud_5cm/X.txt"
file_Labels = "./data/features_dense_LiDAR_cloud_5cm/y.txt"

save_path = "./results/for_LiDAR_5cm/selected_level3/label_img"
make_if_not_exists(save_path)

print("reading data from txt file...")
start_time = time.time()
pt_xyz = np.loadtxt(file_XYZ)
pt_features = np.loadtxt(file_Features)
pt_labels = np.loadtxt(file_Labels)
duration = time.time() - start_time
print("which costs {0}s\n".format(duration))

index = np.asarray([_ for _ in range(pt_labels.shape[0])]).astype(np.int)

img_list = os.listdir(path_reference)
for i in tqdm(range(len(img_list))):

    img_name = img_list[i]
    ori_name = img_name.replace("tif", "ori")

    print("processing image: {0}\n".format(img_name))

    # get interior and exterior orientation
    f, pixel_size, img_width, img_height, K, R, Xc, Yc, Zc = get_INTER_and_EXTER_Orientations(os.path.join(path_Ori,ori_name))

    # frustum culling (could be optimized)
    # xyz_temp, index_temp = utilities.frustum_culling(Xc, Yc, Zc, f, img_height, img_width, pixel_size, pt_xyz, index)

    # hidden point removal
    myPoints, myIndex = HPR(Xc, Yc, Zc, pt_xyz, index)

    # projection
    px, py = pointcloud2pixelcoord(R, K, Xc, Yc, Zc, myPoints)

    # generation of synthetic images
    generation_syntheticImg(px, py, myIndex, pt_labels, pt_features, img_name, save_path, img_width, img_height)