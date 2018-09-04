import numpy as np
import time
import os
from tqdm import tqdm

def make_if_not_exists(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

def read_point_cloud(img_name, extOri_file, result_path, xyz, labels):
    # example: img_name = "CF013540"

    extOri_text = []
    with open(extOri_file, "r") as fp:
        for line in fp:
            if line.split("\t")[0] == img_name:
                extOri_text = line

    extOri_text = extOri_text.split("\t")
    PhotoID = extOri_text[0]
    print("target PhotoID: ", PhotoID, "\n")
    X, Y, Z, Omega, Phi, Kappa, r11, r12, r13, r21, r22, r23, r31, r32, r33 = map(float, extOri_text[1:])

    # Projection Matrix
    R = np.matrix([[r11, r12, r13],
                   [r21, r22, r23],
                   [r31, r32, r33]])

    X0 = np.matrix([X, Y, Z]).T

    Rt = np.concatenate((R, -np.dot(R, X0)), axis=1)

    width = 11608
    height = 8708
    f = -50 / 1000  # [mm -> m] different sign for aerial image
    pixel_size = 4.6 / 1000000  # [micrometer -> m]
    x0 = width / 2  # [pixel] principle point
    y0 = height / 2  # [pixel]

    K = np.matrix([[f / pixel_size, 0, x0],
                   [0, -f / pixel_size, y0],
                   [0, 0, 1]])

    P = np.dot(K, Rt)

    # calculate pixel points
    Pix_coor = np.dot(P, xyz)

    # Normalization of pixel points
    px = Pix_coor[0, :] / Pix_coor[2, :]
    py = Pix_coor[1, :] / Pix_coor[2, :]

    save_path = os.path.join(result_path, str(PhotoID))
    make_if_not_exists(save_path)

    np.savetxt(os.path.join(save_path, "px.txt"), px)
    np.savetxt(os.path.join(save_path, "py.txt"), py)
    np.savetxt(os.path.join(save_path, "labels.txt"), labels)


if __name__ == "__main__":

    imgs_path = "./Images"
    result_path = "./Images_projected_pc"
    make_if_not_exists(result_path)
    extOri_file = "./extOri_test.txt"

    # reading Point Cloud
    pointcloud_file = "./LiDAR_pointcloud_labeled.txt"

    print(">>>>>>>>>>>>>>>>>>Starting to read point cloud data!<<<<<<<<<<<<<<<<<<<<< \n")
    start_time = time.time()
    pc_data = np.loadtxt(pointcloud_file)
    duration = time.time() - start_time
    print("Duration: ", duration, " s \n")
    print(">>>>>>>>>>>>>>>>>>Down!<<<<<<<<<<<<<<<<<<<<<<<<<<! \n")

    # get 3D position and labels from point cloud data
    xyz = pc_data[:, :3].T  # (3, 14129889)
    labels = pc_data[:, -1]  # (14129889,)
    xyz = np.concatenate((np.mat(xyz), np.full((1, xyz.shape[1]), 1)), axis=0)  # using homogeneous coord (4, 14129889)

    # processing
    print(">>>>>>>>>>>>>>>>>>Starting to calculate corresponding pixel coord!<<<<<<<<<<<<<<<<<<<<< \n")
    img_list = os.listdir(imgs_path)
    for img_name in tqdm(img_list):
        read_point_cloud(img_name.split(".")[0], extOri_file, result_path, xyz, labels)
    print(">>>>>>>>>>>>>>>>>>Down!<<<<<<<<<<<<<<<<<<<<<<<<<<! \n")
