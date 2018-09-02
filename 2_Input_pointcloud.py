import numpy as np
import time


def read_point_cloud():
    img_test = "./Images/CF013540.jpg"
    extOri_file = "./extOri_test.txt"
    extOri_text = []
    with open(extOri_file, "r") as fp:
        for line in fp:
            extOri_text = line
            break

    extOri_text = extOri_text.split("\t")
    # PhotoID = extOri_text[0]
    X, Y, Z, Omega, Phi, Kappa, r11, r12, r13, r21, r22, r23, r31, r32, r33 = map(float, extOri_text[1:])

    ### Projection Matrix
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

    ### Reading Point Cloud
    pointcloud_file = "./LiDAR_pointcloud_labeled.txt"

    print(">>>>>>>>>>>>>>>>>>Starting to read point cloud data!<<<<<<<<<<<<<<<<<<<<< \n")
    start_time = time.time()
    data = np.loadtxt(pointcloud_file)
    duration = time.time() - start_time
    print(">>>>>>>>>>>>>>>>>>Down!<<<<<<<<<<<<<<<<<<<<<<<<<<! \n")
    print("Duration: ", duration)

    xyz = data[:, :3].T  # (3, 14129889)
    labels = data[:, -1]  # (14129889,)

    ## calculate pixel points
    xyz = np.concatenate((np.mat(xyz), np.full((1, xyz.shape[1]), 1)), axis=0)  # using homogeneous coord (4, 14129889)
    Pix_coor = np.dot(P, xyz)

    ### Normalization of pixel points
    px = Pix_coor[0, :] / Pix_coor[2, :]
    py = Pix_coor[1, :] / Pix_coor[2, :]

    np.savetxt("px.txt", px)
    np.savetxt("py.txt", py)
    np.savetxt("labels.txt", labels)


if __name__ == "__main__":
    read_point_cloud()
