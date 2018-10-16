import numpy as np
import time
import os
import my_parameters
import cv2
from tqdm import tqdm

f = my_parameters.f
pixel_size = my_parameters.pixel_size
x0 = my_parameters.x0
y0 = my_parameters.y0

imgs_path = "./Images"
extOri_file = "./extOri_test.txt"


# reading Point Cloud
pointcloud_file = "./LiDAR_pointcloud_labeled.txt"
print(">>>>>>>>>>>>>>>>>>Starting to read point cloud data!<<<<<<<<<<<<<<<<<<<<< \n")
start_time = time.time()
pc_data = np.loadtxt(pointcloud_file)
duration = time.time() - start_time
print("Duration: ", duration, " s \n")
print(">>>>>>>>>>>>>>>>>>Down!<<<<<<<<<<<<<<<<<<<<<<<<<<! \n")

# get 3D point coordinates and labels from point cloud data
xyz = pc_data[:, :3].T  # (3, 14129889)
xyz = np.concatenate((np.mat(xyz), np.full((1, xyz.shape[1]), 1)), axis=0)


img_name = "CF013540"

with open(extOri_file, "r") as fp:
    for line in fp:
        if line.split("\t")[0] == img_name:
            # extOri_text = line

            extOri = line.split("\t")
            PhotoID = extOri[0]
            print("target PhotoID: ", PhotoID, "\n")
            X, Y, Z, Omega, Phi, Kappa, r11, r12, r13, r21, r22, r23, r31, r32, r33 = map(float, extOri[1:])

            # Projection Matrix
            R = np.matrix([[r11, r12, r13],
                           [r21, r22, r23],
                           [r31, r32, r33]])

            X0 = np.matrix([X, Y, Z]).T

            Rt = np.concatenate((R, -np.dot(R, X0)), axis=1)

            K = np.matrix([[f / pixel_size, 0, x0],
                           [0, -f / pixel_size, y0],
                           [0, 0, 1]])

            print("K: \n", K)

            P = np.dot(K, Rt)

            print("P: \n", P)

            # calculate pixel points
            Pix_coor = np.dot(P, xyz)

            Pix_coor[0, :] = Pix_coor[0, :] / Pix_coor[2, :]
            Pix_coor[1, :] = Pix_coor[1, :] / Pix_coor[2, :]

            Pix_coor = Pix_coor.T

            Pix_coor = sorted(Pix_coor, key=lambda x: x[:,2])

img_path = "./Images/CF013540.jpg"
img = cv2.imread(img_path)
print(img.shape)

