import numpy as np
import os
import time

# self-define function
import f_get_orientations as f_go
import f_frustum_culling as f_fc
import f_HPR
import f_projection as f_p
import f_get_synthetic_img as f_gsi

def make_if_not_exists(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

# global variables
imgs_path = "./data/Images"
extOri_file = "./data/All Camera Orientations/txt/extOri.txt"
pointcloud_file = "./data/LiDAR_pointcloud_labeled.txt"
save_path = "./data/result"
make_if_not_exists(save_path)


"""
   THE MAIN SCRIPT OF PROJECT:
   
   *Input:
    * images
    * corresponding exterior/interior orientations
    * 3d point cloud with semantic label
    
   *Output:
    * synthetic images
"""


# reading Point Cloud
print("Starting to read point cloud data!\n")
start_time = time.time()
pc_data = np.loadtxt(pointcloud_file)
duration = time.time() - start_time
print("Duration of reading point cloud: {0}s \n".format(duration))

xyz = pc_data[:, :3]
labels = pc_data[:, -1]
features = pc_data[:, 3:6]

index = np.asarray([_ for _ in range(labels.shape[0])])

# processing pip-line
for img_name in os.listdir(imgs_path):
    """
    test example:
    img_name = "CF013540.jpg"
    """
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>processing {0}!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n".format(img_name))
    start_t = time.time()
    
    # get exterior orientation
    ex_data = f_go.get_exterior_orientation(img_name, extOri_file).split("\t")

    # frustum culling
    xyz_temp, label_temp, feature_temp, index_temp = f_fc.frustum_culling(ex_data, xyz, labels, features, index)

    # hidden point removal
    myPoints, mylabels, myfeatures, myindex = f_HPR.HPR(ex_data, xyz_temp, label_temp, feature_temp, index_temp)

    # projection
    px, py, depth = f_p.pointcloud2pixelcoord(ex_data, myPoints)

    # generation of synthetic images
    f_gsi.img_projected(px, py, depth, myfeatures, mylabels, myindex, os.path.join(imgs_path, img_name), save_path)

    d_t = time.time() - start_t
    print("Duration of processing one image: ", d_t, "s\n")
