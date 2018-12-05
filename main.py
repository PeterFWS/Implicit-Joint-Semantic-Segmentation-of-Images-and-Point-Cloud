import numpy as np
import time
import glob
import itertools

import utilities

"""
    ** Global variables
        * test data: level4
"""

path_Imgs = "./data/ImgTexture/Level_0/"
path_Ori = "./data/Ori/Level_0/"

file_XYZ = "./data/features_dense_LiDAR_cloud_5cm/Data_xyz.txt"
file_Features = "./data/features_dense_LiDAR_cloud_5cm/X.txt"
file_Labels = "./data/features_dense_LiDAR_cloud_5cm/y.txt"

save_path = "./results/for_LiDAR_5cm/"
utilities.make_if_not_exists(save_path)




"""
    ** Mode
        * [1] pre-processing (input data preparation)
        * [2] training (SegNet/DeepLabver2)
        * [3] prediction (SegNet/DeepLabver2)
        * [4] evaluation (in 2d or 3d space)
"""

# Mode = 1  # default value
#
# if Mode == 1:

print("pre-processing...\n")

print("read points cloud from txt file...")

start_time = time.time()

pt_xyz = np.loadtxt(file_XYZ)
# pt_features = np.loadtxt(file_Features)
pt_labels = np.loadtxt(file_Labels)

duration = time.time() - start_time

print("which costs {0}s\n".format(duration))

index = np.asarray([_ for _ in range(pt_labels.shape[0])]).astype(np.int)

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

try:
    assert len(img_list) == len(ori_list)
except AssertionError:


for im_, ori_ in zip(img_list, ori_list):
    assert (im_.split('/')[-1].split(".")[0] == ori_.split('/')[-1].split(".")[0])

zipped = itertools.cycle(zip(img_list, ori_list))

for _ in range(len(img_list)):
    # file_img, file_ori = zipped.__next__()

    file_img = "./data/ImgTexture/Level_0/DSC07102.JPG"
    file_ori = "./data/Ori/Level_0/DSC07102.ori"

    print("processing image: {0}\n".format(file_img.split("/")[-1]))

    # get interior and exterior orientation
    f, pixel_size, img_width, img_height, K, R, Xc, Yc, Zc = utilities.get_INTER_and_EXTER_Orientations(file_ori)

    # frustum culling
    xyz_temp, index_temp = utilities.frustum_culling(Xc, Yc, Zc, f, img_height, img_width, pixel_size, pt_xyz, index)

    # hidden point removal
    myPoints, myIndex = utilities.HPR(Xc, Yc, Zc, xyz_temp, index_temp)

    # projection
    px, py = utilities.pointcloud2pixelcoord(R, K, Xc, Yc, Zc, myPoints)
    # px, py = utilities.pointcloud2pixelcoord(R, K, Xc, Yc, Zc, pt_xyz)

    # generation of synthetic images
    utilities.img_projected(px, py, myIndex, pt_labels, file_img, save_path, img_width, img_height)
    # utilities.img_projected(px, py, index, pt_labels, file_img, save_path, img_width, img_height)

    break



# elif Mode == 2:
#     print("training\n")
# elif Mode == 3:
#     print("prediction\n")
# elif Mode == 4:
#     print("evaluation\n")
