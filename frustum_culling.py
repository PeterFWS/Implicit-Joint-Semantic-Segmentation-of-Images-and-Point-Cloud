import math
import numpy as np
import os
import my_parameters
from my_parameters import Vec3
from my_parameters import Plane
from my_parameters import ANG2RAD
import time
import tqdm
import cv2

"""
 * Geometric Approach for View Frustum Culling
  * test data: 3d point clouds
"""


def make_if_not_exists(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

def setCamInternals(angle, ratio, nearD, farD):
    # compute width and height of the near and far plane sections
    tang = float(math.tan(angle * ANG2RAD * 0.5))
    nh = nearD * tang
    nw = nh * ratio
    fh = farD * tang
    fw = fh * ratio

    return nh, nw, fh, fw

# def setCamInternals(nearD, farD):
#     # compute width and height of the near and far plane sections
#     nh = my_parameters.height * my_parameters.pixel_size
#     nw = my_parameters.width * my_parameters.pixel_size
#     fh = nh*farD/nearD
#     fw = nw*farD/nearD
#
#     return nh, nw, fh, fw

def setCamDef(nh, nw, fh, fw, nearD, farD, p=Vec3(), l=Vec3(), u=Vec3()):
    # Input Parameters:
    #  * the position of the camera,
    #  * a point to where the camera is pointing
    #  * the up vector

    # Vec3 dir,nc,fc,X,Y,Z;
    Z = p - l
    Z.normalized()

    X = u * Z
    X.normalized()

    Y = Z * X

    nc = p - Z * nearD
    fc = p - Z * farD

    ntl = nc + Y * nh - X * nw
    ntr = nc + Y * nh + X * nw
    nbl = nc - Y * nh - X * nw
    nbr = nc - Y * nh + X * nw

    ftl = fc + Y * fh - X * fw
    ftr = fc + Y * fh + X * fw
    fbl = fc - Y * fh - X * fw
    fbr = fc - Y * fh + X * fw

    pl_TOP = Plane(ntr,ntl,ftl)
    pl_BOTTOM = Plane(nbl,nbr,fbr)
    pl_LEFT = Plane(ntl,nbl,fbl)
    pl_RIGHT= Plane(nbr,ntr,fbr)
    pl_NEARP= Plane(ntl,ntr,nbr)
    pl_FARP = Plane(ftr,ftl,fbl)

    return pl_TOP, pl_BOTTOM, pl_LEFT, pl_RIGHT, pl_NEARP, pl_FARP


point_status = {
    "INSIDE": 1,
    "OUTSIDE": 0
}


def pointInFrustum(p=Vec3(), pl_TOP=Plane(), pl_BOTTOM=Plane(), pl_LEFT=Plane(), pl_RIGHT=Plane(), pl_NEARP=Plane(), pl_FARP=Plane()):

    result = "INSIDE"

    if pl_TOP.distance(p)<0 \
            or pl_BOTTOM.distance(p)<0 \
            or pl_LEFT.distance(p)<0 \
            or pl_RIGHT.distance(p)<0 \
            or pl_NEARP.distance(p)<0 \
            or pl_FARP.distance(p)<0:
        result = "OUTSIDE"

    return point_status[result]


if __name__ == "__main__":
    """
    test for only one image
    """
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
    print(">>>>>>>>>>>>>>>>>>Down!<<<<<<<<<<<<<<<<<<<<<<<<<< \n")

    # get 3D point coordinates and labels from point cloud data
    xyz = pc_data[:, :3]  # (14129889,3)
    labels = pc_data[-1]
    X_, Y_, Z_ = 513956.3197161849200000, 5426766.6255130861000000, 276.9661760997179300
    PhotoID = "CF013540"


    # start frustum culling
    print(">>>>>>>>>>>>>>>>>>Starting to culling frustum<<<<<<<<<<<<<<<<<<<<<<<<<<")
    nearD = abs(my_parameters.f)
    farD = Z_

    angle = 45
    ratio = my_parameters.width/my_parameters.height

    nh, nw, fh, fw = setCamInternals(angle, ratio, nearD, farD)

    pl_TOP, pl_BOTTOM, pl_LEFT, pl_RIGHT, pl_NEARP, pl_FARP = \
        setCamDef(nh, nw, fh, fw, nearD, farD, Vec3(X_,Y_,Z_), Vec3(0, 0, -1), Vec3(0, 1, 0))


    xyz_3d = []
    for i in range(0, xyz.shape[0]):

        p = Vec3(xyz[i, 0], xyz[i, 1], xyz[i, 2])
        # print(p)

        flag = pointInFrustum(p,
                       pl_TOP, pl_BOTTOM, pl_LEFT,
                       pl_RIGHT, pl_NEARP, pl_FARP)
        # print(flag)

        if flag == 1:
            print(p)
            xyz_3d.append([p.x, p.y, p.z])
    print(">>>>>>>>>>>>>>>>>>Down!<<<<<<<<<<<<<<<<<<<<<<<<<< \n")

    xyz_3d = np.matrix(xyz_3d)


    # 3d to 2d
    print(">>>>>>>>>>>>>>>>>>Starting to project 3d points to 2d<<<<<<<<<<<<<<<<<<<<<<<<<< \n")
    with open(extOri_file, "r") as fp:
        for line in fp:
            if line.split("\t")[0] == PhotoID:
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

                K = np.matrix([[my_parameters.f / my_parameters.pixel_size, 0, my_parameters.x0],
                               [0, -my_parameters.f / my_parameters.pixel_size, my_parameters.y0],
                               [0, 0, 1]])

                P = np.dot(K, Rt)

                # calculate pixel points
                Pix_coor = np.dot(P, xyz_3d)

                # Normalization of pixel points
                px = Pix_coor[0, :] / Pix_coor[2, :]
                py = Pix_coor[1, :] / Pix_coor[2, :]

    img = cv2.imread("./Images/CF013540.jpg")
    print(img.shape)
    img2 = np.zeros(img.shape, np.uint8)
    img3 = np.zeros(img.shape, np.uint8)

    classes = []
    for i in tqdm(range(0, px.shape[0])):
        if my_parameters.width > px[i] > 0 and my_parameters.height > py[i] > 0:
            c = my_parameters.color_classes[str(labels[i])]
            if labels[i] not in classes:
                classes.append(labels[i])
            cv2.circle(img2, (int(px[i]), int(py[i])), 1, c, -1)
            cv2.circle(img3, (int(px[i]), int(py[i])), 10, c, -1)

    cv2.imwrite(os.path.join("./Images_projected_pc", "2.jpg"), img2)
    cv2.imwrite(os.path.join("./Images_projected_pc", "3.jpg"), img3)

    print(">>>>>>>>>>>>>>>>>>Down!<<<<<<<<<<<<<<<<<<<<<<<<<< \n")



