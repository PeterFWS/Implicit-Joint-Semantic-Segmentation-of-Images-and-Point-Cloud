import os
import numpy as np
import my_parameters

"""
* Render views main program
 * inputs: 
 *     cloud_filename: name of textfile containing point cloud (x y z i r g b) and label filename containing integer labels as in semantic3D
 *     location: output folder location
 *     haslabel: flag with 1 for labeled point clouds and 0 for unlabeled point clouds
 *     downsample_num_points: roughly number of points after downsampling (importance sampling)
 *     haspcd: use pcd file
 *     is_test: for test or validation data
 
 
 need a least 2 input arguments
"""

# create point cloud txt file as format (x y z i r g b)

pointcloud_file = "./LiDAR_pointcloud_labeled.txt"
print(">>>>>>>>>>>>>>>>>>Starting to READ point cloud data!<<<<<<<<<<<<<<<<<<<<< \n")
pc_data = np.loadtxt(pointcloud_file)
print(">>>>>>>>>>>>>>>>>>Down!<<<<<<<<<<<<<<<<<<<<<<<<<<! \n")

xyz = pc_data[:, :3]  # (14129889, 3)
labels = pc_data[:, -1]  # (14129889,)

# print(">>>>>>>>>>>>>>>>>>Starting to SAVE point cloud data for point splatting!<<<<<<<<<<<<<<<<<<<<< \n")
# temp = ""
# with open("./ps_pointcloud.txt", "w") as fp:
#     for i in range(0, xyz.shape[0]):
#         temp = str(xyz[i,0]) + " " + str(xyz[i,1]) + " " + str(xyz[i,2]) + " " + str(i) + str(my_parameters.color_classes[str(labels[i])]).replace("("," ").replace(","," ").replace(")"," ") + "\n"
#         fp.write(temp)
# print(">>>>>>>>>>>>>>>>>>Down!<<<<<<<<<<<<<<<<<<<<<<<<<<! \n")

temp = ""
with open("./labels_int.txt", "w") as fp:
    for i in range(0, labels.shape[0]):
        temp = str(labels[i]).split(".")[0] + "\n"
        fp.write(temp)