import numpy as np
import time
import os
import utilities
import cv2
from tqdm import tqdm
import math
import tifffile


"""
    ** Code for generation of depth image
"""


# Global varianbles
path_reference = "./results/for_LiDAR_5cm/selected_level3/color_labeled_points"  # where provide a list of visible images
path_Imgs = "./data/ImgTexture/Level_3/"
path_Ori = "./data/Ori/Level_3/"

file_XYZ = "./data/features_dense_LiDAR_cloud_5cm/Data_xyz.txt"

save_path = "./results/for_LiDAR_5cm/selected_level3/depth_img_origin"
utilities.make_if_not_exists(save_path)


# Reading data
print("read points cloud from txt file...")
start_time = time.time()
pt_xyz = np.loadtxt(file_XYZ)
duration = time.time() - start_time
print("which needs {0}s\n".format(duration))

# Processing
img_list = os.listdir(path_reference)  # length = 416, as for 5cm LiDAR pointcloud

for img_name in img_list:

    ori_name = img_name.replace("tif", "ori")
    #img_name = "DSC04106.tif"
    #ori_name = "DSC04106.ori"

    # get interior and exterior orientations
    f, pixel_size, img_width, img_height, K, R, Xc, Yc, Zc = utilities.get_INTER_and_EXTER_Orientations(os.path.join(path_Ori, ori_name))

    # without frustum culling and hidden-point-removal
    px, py = utilities.pointcloud2pixelcoord(R, K, Xc, Yc, Zc, pt_xyz)

    # generation depth image, without interpolation
    img_depth = np.zeros((img_height, img_width, 1), dtype=np.float32)  # 32 bit

    for i in tqdm(range(0, px.shape[1])):
        if img_width > px[0, i] > 0 and img_height > py[0, i] > 0:

            depth = math.sqrt( (Xc - pt_xyz[i, 0])**2 + (Yc - pt_xyz[i, 1])**2 + (Zc - pt_xyz[i, 2])**2 )

            if img_depth[int(py[0, i]), int(px[0, i])] == 0:
                img_depth[int(py[0, i]), int(px[0, i])] = depth

            elif img_depth[int(py[0, i]), int(px[0, i])] != 0 and \
                    depth <= img_depth[int(py[0, i]), int(px[0, i])]:  # only save the closest distance in that pixel
                img_depth[int(py[0, i]), int(px[0, i])] = depth

    # set "nan" to pixel where no point projected
    mask = (img_depth == 0)
    img_depth[mask[:, :] == True] = np.nan

    tifffile.imsave(os.path.join(save_path, img_name+'_dist.tif'), img_depth)

    # test = tifffile.imread(os.path.join(save_path, img_name+'_dist.tif'))
