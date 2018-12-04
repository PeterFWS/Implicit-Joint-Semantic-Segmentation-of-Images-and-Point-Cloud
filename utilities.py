import os
import numpy as np
import time

from myClasses import Vec3
from myClasses import Plane

import math
from scipy.spatial import ConvexHull

import cv2
from tqdm import tqdm
from scipy.interpolate import griddata

def make_if_not_exists(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)




"""
    ** return Interior/Exterior Orientations of corresponding image
"""


def get_INTER_and_EXTER_Orientations(file_ori):
    temp = []
    with open(file_ori, "r") as fp:
        for line in fp:
            temp.append(line)

    for i in range(len(temp)):
        if "FocalLength" in temp[i].split("_"):  # [mm->m]
            f = - float(temp[i + 1]) / 1000  # different sign for aerial image
            # print("f: {0}\n".format(f))

        elif "PixelSize" in temp[i].split("_"):  # [mm->m]
            pixel_size_x = float(temp[i + 1].split("\t")[1]) / 1000
            pixel_size_y = float(temp[i + 1].split("\t")[2]) / 1000
            # print("pixel_size_x: {0}\n".format(pixel_size_x))
            # print("pixel_size_y: {0}\n".format(pixel_size_y))

        elif "SensorSize" in temp[i].split("_"):  # [pixel]
            img_width = int(temp[i + 1].split("\t")[1])
            img_height = int(temp[i + 1].split("\t")[2])
            # print("img_width: {0}\n".format(img_width))
            # print("img_height: {0}\n".format(img_height))

        elif "PrincipalPoint" in temp[i].split("_"):  # [pixel]
            x0 = float(temp[i + 1].split("\t")[1])
            y0 = float(temp[i + 1].split("\t")[2])
            # print("x0: {0}\n".format(x0))
            # print("y0: {0}\n".format(y0))

        elif "CameraMatrix" in temp[i].split("_"):  # [ImageCoordinateSystem]
            K = np.matrix([[float(temp[i + 1].split("\t")[1]), float(temp[i + 1].split("\t")[2]),
                            float(temp[i + 1].split("\t")[3])],
                           [float(temp[i + 2].split("\t")[1]), float(temp[i + 2].split("\t")[2]),
                            float(temp[i + 2].split("\t")[3])],
                           [float(temp[i + 3].split("\t")[1]), float(temp[i + 3].split("\t")[2]),
                            float(temp[i + 3].split("\t")[3])]])
            # print("K: {0}\n".format(K))

        elif "RotationMatrix" in temp[i].split("_"):  # [World->ImageCoordinateSystem]
            R = np.matrix([[float(temp[i + 1].split("\t")[1]), float(temp[i + 1].split("\t")[2]),
                            float(temp[i + 1].split("\t")[3])],
                           [float(temp[i + 2].split("\t")[1]), float(temp[i + 2].split("\t")[2]),
                            float(temp[i + 2].split("\t")[3])],
                           [float(temp[i + 3].split("\t")[1]), float(temp[i + 3].split("\t")[2]),
                            float(temp[i + 3].split("\t")[3])]])
            # print("R: {0}\n".format(R))

        elif "TranslationVector" in temp[i].split("_"):  # [WorldCoordinateSystem]
            Xc = float(temp[i + 1].split("\t")[1])
            Yc = float(temp[i + 1].split("\t")[2])
            Zc = float(temp[i + 1].split("\t")[3])
            # print("Xc: {0}\n".format(Xc))
            # print("Yc: {0}\n".format(Yc))
            # print("Zc: {0}\n".format(Zc))

    return f, pixel_size_x, pixel_size_y, img_width, img_height, x0, y0, K, R, Xc, Yc, Zc




"""
    ** Geometric Approach for View Frustum Culling
"""


def setCamInternals(nearD, farD, img_height, img_width, pixel_size):
    # compute width and height of the near and far plane sections
    nh = img_height * pixel_size
    nw = img_width * pixel_size
    fh = nh * farD / nearD
    fw = nw * farD / nearD

    return nh, nw, fh, fw


def setCamDef(nh, nw, fh, fw, nearD, farD, p=Vec3(), l=Vec3(), u=Vec3()):
    # Input Parameters:
    #  * p: the position of the camera,
    #  * l: a point to where the camera is pointing
    #  * u: the up vector

    # Vec3 dir,nc,fc,X,Y,Z;
    Z = p - l
    Z.normalized()

    X = Vec3.cross(u, Z)
    X.normalized()

    Y = Vec3.cross(Z, X)

    nc = p - Vec3.__mul__(Z, nearD)
    fc = p - Vec3.__mul__(Z, farD)

    ntl = nc + Vec3.__mul__(Y, nh) - Vec3.__mul__(X, nw)
    ntr = nc + Vec3.__mul__(Y, nh) + Vec3.__mul__(X, nw)
    nbl = nc - Vec3.__mul__(Y, nh) - Vec3.__mul__(X, nw)
    nbr = nc - Vec3.__mul__(Y, nh) + Vec3.__mul__(X, nw)

    ftl = fc + Vec3.__mul__(Y, fh) - Vec3.__mul__(X, fw)
    ftr = fc + Vec3.__mul__(Y, fh) + Vec3.__mul__(X, fw)
    fbl = fc - Vec3.__mul__(Y, fh) - Vec3.__mul__(X, fw)
    fbr = fc - Vec3.__mul__(Y, fh) + Vec3.__mul__(X, fw)

    pl_TOP = Plane(ntr, ntl, ftl)
    pl_BOTTOM = Plane(nbl, nbr, fbr)
    pl_LEFT = Plane(ntl, nbl, fbl)
    pl_RIGHT = Plane(nbr, ntr, fbr)
    pl_NEARP = Plane(ntl, ntr, nbr)
    pl_FARP = Plane(ftr, ftl, fbl)

    return pl_TOP, pl_BOTTOM, pl_LEFT, pl_RIGHT, pl_NEARP, pl_FARP


point_status = {
    "INSIDE": 1,
    "OUTSIDE": 0
}


def pointInFrustum(p=Vec3(), pl_TOP=Plane(), pl_BOTTOM=Plane(), pl_LEFT=Plane(), pl_RIGHT=Plane(), pl_NEARP=Plane(),
                   pl_FARP=Plane()):
    result = "INSIDE"

    if pl_TOP.distance(p) < 0 \
            or pl_BOTTOM.distance(p) < 0 \
            or pl_LEFT.distance(p) < 0 \
            or pl_RIGHT.distance(p) < 0 \
            or pl_NEARP.distance(p) < 0 \
            or pl_FARP.distance(p) < 0:
        result = "OUTSIDE"

    return point_status[result]


def frustum_culling(Xc, Yc, Zc, f, img_height, img_width, pixel_size, pt_xyz, index):
    print("Culling frustum... ")
    start_time = time.time()

    nearD = abs(f)
    farD = Zc

    nh, nw, fh, fw = setCamInternals(nearD, farD, img_height, img_width, pixel_size)

    pl_TOP, pl_BOTTOM, pl_LEFT, pl_RIGHT, pl_NEARP, pl_FARP = \
        setCamDef(nh, nw, fh, fw, nearD, farD, Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(0, 1, 0))

    xyz_temp = []
    index_temp = []
    for i in range(0, pt_xyz.shape[0]):
        pt = Vec3(pt_xyz[i, 0] - Xc, pt_xyz[i, 1] - Yc, pt_xyz[i, 2] - Zc)
        flag = pointInFrustum(pt,
                              pl_TOP, pl_BOTTOM, pl_LEFT,
                              pl_RIGHT, pl_NEARP, pl_FARP)
        if flag == 1:
            xyz_temp.append([pt_xyz[i, 0], pt_xyz[i, 1], pt_xyz[i, 2]])
            index_temp.append(index[i])

    xyz_temp = np.matrix(xyz_temp)
    index_temp = np.asarray(index_temp)

    duration = time.time() - start_time
    print(duration, "s\n")

    return xyz_temp, index_temp




"""
    ** Hidden Point Removal
"""


def sphericalFlip(points, center, param):
    # Function used to Perform Spherical Flip on the Original Point Cloud
    n = points.shape[0]  # total n points
    points = points - np.repeat(center, n, axis=0)  # Move C to the origin
    normPoints = np.linalg.norm(points, axis=1)  # Normed points, sqrt(x^2 + y^2 + (z-100)^2)
    R = np.repeat(max(normPoints) * np.power(10.0, param), n, axis=0)  # Radius of Sphere

    flippedPointsTemp = 2 * np.multiply(np.repeat((R - normPoints).reshape(n, 1), len(points[0]), axis=1), points)
    flippedPoints = np.divide(flippedPointsTemp, np.repeat(normPoints.reshape(n, 1), len(points[0]),
                                                           axis=1))  # Apply Equation to get Flipped Points
    flippedPoints += points

    return flippedPoints

def convexHull(points):
    # Function used to Obtain the Convex hull
    points = np.append(points, [[0, 0, 0]], axis=0)  # All points plus origin
    hull = ConvexHull(points)  # Visibal points plus possible origin. Use its vertices property.

    return hull

def HPR(Xc, Yc, Zc, xyz_temp, index_temp):
    print("Hidden point removal... ")
    start_time = time.time()

    flag = np.zeros(len(xyz_temp), int)  #  0 - Invisible; 1 - Visible.
    C = np.array([[Xc, Yc, Zc]])  # Center
    flippedPoints = sphericalFlip(xyz_temp, C, math.pi)
    myHull = convexHull(flippedPoints)
    visibleVertex = myHull.vertices[:-1]  # indexes of visible points

    flag[visibleVertex] = 1
    visibleId = np.where(flag == 1)[0]  # indexes of the visible points

    myPoints = []
    myIndex = []
    for i in visibleId:
        myPoints.append([xyz_temp[i, 0], xyz_temp[i, 1], xyz_temp[i, 2]])
        myIndex.append(index_temp[i])

    myPoints = np.matrix(myPoints)
    myIndex = np.asarray(myIndex)

    duration = time.time() - start_time
    print(duration, "s\n")

    return myPoints, myIndex




"""
    ** 3d to 2d projection
"""


def pointcloud2pixelcoord(R, K, Xc, Yc, Zc, myPoints):

    print("projection... ")
    start_time = time.time()

    X0 = np.matrix([Xc, Yc, Zc]).T

    Rt = np.concatenate((R, -np.dot(R, X0)), axis=1)

    P = np.dot(K, Rt)

    # calculate pixel points
    myPoints = myPoints.T
    myPoints = np.concatenate((np.mat(myPoints), np.full((1, myPoints.shape[1]), 1)), axis=0)  # using homogeneous coord (4, 14129889)
    Pix_coor = np.dot(P, myPoints)

    # Normalization of pixel points
    px = Pix_coor[0, :] / Pix_coor[2, :]
    py = Pix_coor[1, :] / Pix_coor[2, :]

    duration = time.time() - start_time
    print(duration, "s\n")

    return px, py




"""
    ** Generation of synthetic images
"""

def img_projected(px, py, depth, myfeatures, mylabels, myindex, img_path, save_path):
    print("Generation of synthetic image... \n")
    start_time = time.time()

    # save_path_grey = os.path.join(save_path, "grey_train")
    save_path_color = os.path.join(save_path, "color_label_ground_truth")
    # save_path_depth = os.path.join(save_path, "depth")
    # save_path_feature1 = os.path.join(save_path, "feature1")
    # save_path_feature2 = os.path.join(save_path, "feature2")
    # save_path_feature3 = os.path.join(save_path, "feature3")
    # save_path_index = os.path.join(save_path, "index")

    # make_if_not_exists(save_path_grey)
    make_if_not_exists(save_path_color)
    # make_if_not_exists(save_path_depth)
    # make_if_not_exists(save_path_feature1)
    # make_if_not_exists(save_path_feature2)
    # make_if_not_exists(save_path_feature3)
    # make_if_not_exists(save_path_index)

    label_value = []
    points = []
    d = []
    feature1 = []
    feature2 = []
    feature3 = []
    id = []
    img_temp = np.zeros((my_parameters.height, my_parameters.width, 3), np.uint8)

    for i in tqdm(range(0, px.shape[1])):
       if my_parameters.width > px[0, i] > 0 and my_parameters.height > py[0, i] > 0:
            label_value.append(int(mylabels[i]))
            points.append([px[0, i], py[0, i]])
            d.append(-depth[0, i])  # depth should > 0 here, we are in aerial image case
            feature1.append(myfeatures[i, 0])
            feature2.append(myfeatures[i, 1])
            feature3.append(myfeatures[i, 2])
            id.append(myindex[i])

            c = my_parameters.color_classes_int[str(int(mylabels[i]))]
            cv2.circle(img_temp, (int(px[0, i]), int(py[0, i])), 5, c, -1)
    cv2.imwrite(os.path.join(save_path_color, img_path.split("/")[-1]).replace(".jpg", "_gt_origin.jpg"), img_temp)

    points = np.array(points)
    label_value = np.array(label_value)
    d = np.array(d)
    id = np.array(id)
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    feature3 = np.array(feature3)

    # ** generate featured image for training
    X, Y = np.meshgrid(np.arange(0, my_parameters.width, 1), np.arange(0, my_parameters.height, 1))

    # * labeled image
    print("Generation of labeled image... \n")
    int_im = griddata(points, label_value, (X, Y), method='nearest').astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    closing = cv2.morphologyEx(int_im, cv2.MORPH_CLOSE, kernel).astype(np.uint8)

    cv2.imwrite(os.path.join(save_path_grey, img_path.split("/")[-1]), closing)

    # * depth image
    print("Generation of depth image... \n")
    depth_im = griddata(points, d, (X, Y), method='nearest')
    closing = cv2.morphologyEx(depth_im, cv2.MORPH_CLOSE, kernel)

    np.save(os.path.join(save_path_depth, img_path.split("/")[-1]).replace(".jpg", ""), closing)

    # * image of feature no.1
    print("Generation of image based on feature no.1 ... \n")
    f1_im = griddata(points, feature1, (X, Y), method='nearest')
    closing = cv2.morphologyEx(f1_im, cv2.MORPH_CLOSE, kernel)

    np.save(os.path.join(save_path_feature1, img_path.split("/")[-1]).replace(".jpg", ""), closing)

    # * image of feature no.2
    print("Generation of image based on feature no.2 ... \n")
    f2_im = griddata(points, feature2, (X, Y), method='nearest')
    closing = cv2.morphologyEx(f2_im, cv2.MORPH_CLOSE, kernel)

    np.save(os.path.join(save_path_feature2, img_path.split("/")[-1]).replace(".jpg", ""), closing)

    # * image of feature no.3
    print("Generation of image based on feature no.3 ... \n")
    f3_im = griddata(points, feature3, (X, Y), method='nearest')
    closing = cv2.morphologyEx(f3_im, cv2.MORPH_CLOSE, kernel)

    np.save(os.path.join(save_path_feature3, img_path.split("/")[-1]).replace(".jpg", ""), closing)

    # * index
    print("Generation of index image ... \n")
    index_im = griddata(points, id, (X, Y), method='nearest').astype(np.uint8)
    closing = cv2.morphologyEx(index_im, cv2.MORPH_CLOSE, kernel)

    np.save(os.path.join(save_path_index, img_path.split("/")[-1]).replace(".jpg", ""), closing)


    # generate ground truth color image (only for visualization, command those code if not needed)
    print("Generation of ground truth color image... \n")
    img_color_labeled = np.zeros((my_parameters.height, my_parameters.width, 3), np.uint8)

    for i in range(0, my_parameters.height):
        for j in range(0, my_parameters.width):
                if int(int_im[i,j]) == 12 or int(int_im[i,j]) == 13:
                    int_im[i,j] = 11
                r, g, b = my_parameters.color_classes_int[str(int(int_im[i,j]))]
                img_color_labeled[i,j,0] = r
                img_color_labeled[i,j,1] = g
                img_color_labeled[i,j,2] = b

    cv2.imwrite(os.path.join(save_path_color, img_path.split("/")[-1]).replace(".jpg", "_gt.jpg"), img_color_labeled)

    duration = time.time() - start_time
    print(duration, "s\n")
