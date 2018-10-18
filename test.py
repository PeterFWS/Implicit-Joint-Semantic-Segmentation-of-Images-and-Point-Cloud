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
labels = pc_data[:, -1]

img_name = "CF013540" # (14129889,)

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

            px = Pix_coor[0, :] / Pix_coor[2, :]
            py = Pix_coor[1, :] / Pix_coor[2, :]

            depth = Pix_coor[2, :]





img_path = "./Images/CF013540.jpg"
img = cv2.imread(img_path)
print(img.shape)
img2 = np.zeros(img.shape, np.uint8)
img3 = np.zeros(img.shape, np.uint8)



classes = []
for i in tqdm(range(0, px.shape[1])):
    if my_parameters.width > px[0,i] > 0 and my_parameters.height > py[0,i] > 0:
        c = my_parameters.color_classes[str(labels[i])]
        if labels[i] not in classes:
            classes.append(labels[i])
        cv2.circle(img2, (int(px[0,i]), int(py[0,i])), 1, c, -1)
        cv2.circle(img3, (int(px[0,i]), int(py[0,i])), 10, c, -1)

cv2.imwrite("./2.jpg", img2)
cv2.imwrite("./3.jpg", img3)



with open("./px.txt", "w") as fp1:
    for i in range(0, px.shape[1]):
        fp1.write(str(px[0,i]) + "\n")

with open("./py.txt", "w") as fp2:
    for i in range(0, py.shape[1]):
        fp2.write(str(py[0,i]) + "\n")

with open("./depth.txt", "w") as fp3:
    for i in range(0, depth.shape[1]):
        fp3.write(str(depth[0,i]) + "\n")