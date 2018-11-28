import numpy as np
import my_parameters
import time

f = my_parameters.f
pixel_size = my_parameters.pixel_size
x0 = my_parameters.x0
y0 = my_parameters.y0




def pointcloud2pixelcoord(ex_data, myPoints):

    print("projection... ")
    start_time = time.time()

    PhotoID = ex_data[0]
    X, Y, Z, Omega, Phi, Kappa, r11, r12, r13, r21, r22, r23, r31, r32, r33 = map(float, ex_data[1:])
    # Projection Matrix
    R = np.matrix([[r11, r12, r13],
                   [r21, r22, r23],
                   [r31, r32, r33]])

    X0 = np.matrix([X, Y, Z]).T

    Rt = np.concatenate((R, -np.dot(R, X0)), axis=1)

    K = np.matrix([[f / pixel_size, 0, x0],
                   [0, -f / pixel_size, y0],
                   [0, 0, 1]])

    # print("K: \n", K)

    P = np.dot(K, Rt)

    # print("P: \n", P)

    # calculate pixel points
    myPoints = np.matrix(myPoints).T
    myPoints = np.concatenate((np.mat(myPoints), np.full((1, myPoints.shape[1]), 1)), axis=0)  # using homogeneous coord (4, 14129889)
    Pix_coor = np.dot(P, myPoints)

    # Normalization of pixel points
    px = Pix_coor[0, :] / Pix_coor[2, :]
    py = Pix_coor[1, :] / Pix_coor[2, :]
    depth = Pix_coor[2, :]

    duration = time.time() - start_time
    print(duration, "s\n")

    return px, py, depth








