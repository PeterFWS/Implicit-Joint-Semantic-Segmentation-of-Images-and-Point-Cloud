import os
import numpy as np
import time

from myClasses import Vec3
from myClasses import Plane

import math
from scipy.spatial import ConvexHull

import cv2
from scipy.interpolate import griddata
import tifffile

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

    return f, pixel_size_x, img_width, img_height, K, R, Xc, Yc, Zc


# for nadir image
def get_exterior_orientation(img_name, extOri_file):

    print("Looking for exterior orientations of the image: {0} \n".format(img_name))
    with open(extOri_file, "r") as fp:
        for line in fp:
            if line.split("\t")[0] == img_name.split(".")[0]:
                return line


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
    # using homogeneous coord (4, 14129889)
    myPoints = np.concatenate((np.mat(myPoints), np.full((1, myPoints.shape[1]), 1)), axis=0)
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

palette = {  # BGR
    0: (255, 0, 0),      # Powerline
    1: (255, 255, 255),  # Low Vegetation
    2: (255, 255, 0),    # Impervious Surface
    3: (255, 0, 255),    # Vehicles
    4: (0, 255, 255),    # Urban Furniture
    5: (0, 255, 0),      # Roof
    6: (0, 0, 255),      # Facade
    7: (239, 120, 76),   # Bush/Hedge
    8: (247, 238, 179),  # Tree
    9: (0, 18, 114),     # Dirt/Gravel
    10: (63, 34, 15),    # Vertical Surface
    11: (0, 0, 0)        # Void
}


# the original function
def generation_syntheticImg(px, py, myIndex, pt_labels, pt_features, img_name, save_path, img_width, img_height):
    print("Generation of synthetic image... \n")
    start_time = time.time()

    img_mask = np.zeros((img_height, img_width, 1), np.uint8)  # used for making a mask only
    img_temp = np.zeros((img_height, img_width, 3), np.uint8)  # used for point level labeled image only

    label_value = []
    points = []
    id = []

    count = 0
    for i in range(0, px.shape[1]):
        if img_width > px[0, i] > 0 and img_height > py[0, i] > 0:
            # filter the points which are not in the FOV of this image
            points.append([px[0, i], py[0, i]])
            label_value.append(int(pt_labels[myIndex[i]]))
            id.append(myIndex[i])

            r, g, b = palette[int(pt_labels[myIndex[i]])]
            img_temp[int(py[0, i]), int(px[0, i]), 0] = r
            img_temp[int(py[0, i]), int(px[0, i]), 1] = g
            img_temp[int(py[0, i]), int(px[0, i]), 2] = b
            count += 1

            cv2.circle(img_mask, (int(px[0, i]), int(py[0, i])), 3, (255,255,255), -1)

    points = np.array(points)
    label_value = np.array(label_value)
    id = np.array(id)

    if count > 100:  # I suppose at least 20% of pixels in the image should receive 3d point
        # save the point level label image for checking
        folder_path = os.path.join(save_path, "1_pointlabel")
        make_if_not_exists(folder_path)
        cv2.imwrite(os.path.join(folder_path, img_name.split("/")[-1]), img_temp)

        # mask, where no point is projected to pixel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
        closing_img_temp = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel).astype(np.uint8)
        folder_path = os.path.join(save_path, "2_mask")
        make_if_not_exists(folder_path)
        cv2.imwrite(os.path.join(folder_path, img_name.split("/")[-1]), closing_img_temp)

        mask = (closing_img_temp[:,:] != 0)

        # Generation of synthetic image based on different feature
        X, Y = np.meshgrid(np.arange(0, img_width, 1), np.arange(0, img_height, 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

        # * labeled image (grey, as ground truth for training)
        int_im = griddata(points, label_value, (X, Y), method='nearest').astype(np.uint8)
        closing = cv2.morphologyEx(int_im, cv2.MORPH_CLOSE, kernel).astype(np.uint8)
        closing[mask[:, :] == False] = 255  # label value 255 for unlabeled pixel
        folder_path = os.path.join(save_path, "3_greylabel")
        make_if_not_exists(folder_path)
        cv2.imwrite(os.path.join(folder_path, img_name.split("/")[-1]), closing)

        # * labeled image (color, only for visualization)
        img_color_labeled = np.zeros((img_height, img_width, 3), np.uint8)
        for i in range(0, img_height):
                for j in range(0, img_width):
                        if int(int_im[i,j]) == 11 or int(int_im[i,j]) == 12 or int(int_im[i,j])==13:
                            int_im[i,j] = 10

                        if mask[i,j] == True:
                            r, g, b = palette[int(int_im[i, j])]
                            img_color_labeled[i, j, 0] = r
                            img_color_labeled[i, j, 1] = g
                            img_color_labeled[i, j, 2] = b
        folder_path = os.path.join(save_path, "4_colorlabel")
        make_if_not_exists(folder_path)
        cv2.imwrite(os.path.join(folder_path, img_name), img_color_labeled)

        # * index image
        index_im = griddata(points, id, (X, Y), method='nearest').astype(np.float32)
        folder_path = os.path.join(save_path, "5_index")
        make_if_not_exists(folder_path)
        cv2.imwrite(os.path.join(folder_path, img_name.split("/")[-1]), index_im)

        # * feature map
        # here, 20 features will be calculated
        num_features = 20
        features = np.zeros((id.shape[0], num_features), np.float32)

        for i in range(0, features.shape[1]):
            for j in range(0, features.shape[0]):
                features[j, i] = pt_features[id[j], -(i + 1)]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        for i in range(0, features.shape[1]):
            f_im = griddata(points, features[:,i], (X, Y), method='linear').astype(np.float32)
            closing = cv2.morphologyEx(f_im, cv2.MORPH_CLOSE, kernel)
            closing[mask[:, :] == False] = np.nan
            folder_path = os.path.join(save_path, "f_"+str(i))
            make_if_not_exists(folder_path)
            tifffile.imsave(os.path.join(folder_path, img_name.split("/")[-1]), closing)

    duration = time.time() - start_time
    print(duration, "s\n")


# used in "main2", cropping image strategy of nadir image
def generation_syntheticImg_nadir(px, py, myIndex, pt_labels, pt_features, img_name, save_path, img_width, img_height):

    img_temp = np.zeros((img_height, img_width, 3), np.uint8)

    label_value = []
    points = []
    id = []

    f_nDOM = []
    f_I = []
    f_Q_Pulse = []
    f_prior = []

    f_Q_N1 = []
    f_Q_N2 = []
    f_Q_N3 = []

    f_Q_ES_1 = []
    f_Q_ES_2 = []
    f_Q_ES_3 = []

    count = 0
    for i in range(0, px.shape[1]):
        if img_width > px[0, i] > 0 and img_height > py[0, i] > 0:
            # filter the points which are not in the FOV of this image
            points.append([px[0, i], py[0, i]])
            label_value.append(int(pt_labels[myIndex[i]]))
            id.append(myIndex[i])

            f_nDOM.append(pt_features[myIndex[i], -4])
            f_I.append(pt_features[myIndex[i], -3])
            f_Q_Pulse.append(pt_features[myIndex[i], -2])
            f_prior.append(pt_features[myIndex[i], -1])

            f_Q_N3.append(pt_features[myIndex[i], -5])
            f_Q_N2.append(pt_features[myIndex[i], -6])
            f_Q_N1.append(pt_features[myIndex[i], -7])

            f_Q_ES_3.append(pt_features[myIndex[i], -8])
            f_Q_ES_2.append(pt_features[myIndex[i], -9])
            f_Q_ES_1.append(pt_features[myIndex[i], -10])

            c = palette[int(pt_labels[myIndex[i]])]
            cv2.circle(img_temp, (int(px[0, i]), int(py[0, i])), 5, c, -1)
            count += 1

    points = np.array(points)
    label_value = np.array(label_value)
    id = np.array(id)

    f_nDOM = np.array(f_nDOM)
    f_I = np.array(f_I)
    f_Q_Pulse = np.array(f_Q_Pulse)
    f_prior = np.array(f_prior)

    f_Q_N3 = np.array(f_Q_N3)
    f_Q_N2 = np.array(f_Q_N2)
    f_Q_N1 = np.array(f_Q_N1)

    f_Q_ES_3 = np.array(f_Q_ES_3)
    f_Q_ES_2 = np.array(f_Q_ES_2)
    f_Q_ES_1 = np.array(f_Q_ES_1)

    if count > 100:  # I suppose at least 20% of pixels in the image should receive 3d point
        folder_path = os.path.join(save_path, "1_pointlabel")
        make_if_not_exists(folder_path)
        cv2.imwrite(os.path.join(folder_path, img_name.split("/")[-1]), img_temp)

        # Generation of synthetic image based on different feature
        X, Y = np.meshgrid(np.arange(0, img_width, 1), np.arange(0, img_height, 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

        # * labeled image (grey, as ground truth for training)
        int_im = griddata(points, label_value, (X, Y), method='nearest').astype(np.uint8)
        closing = cv2.morphologyEx(int_im, cv2.MORPH_CLOSE, kernel).astype(np.uint8)
        folder_path = os.path.join(save_path, "3_greylabel")
        make_if_not_exists(folder_path)
        cv2.imwrite(os.path.join(folder_path, img_name.split("/")[-1]), closing)

        # * labeled image (color, only for visualization)
        img_color_labeled = np.zeros((img_height, img_width, 3), np.uint8)
        for i in range(0, img_height):
                for j in range(0, img_width):
                        if int(int_im[i,j]) == 11 or int(int_im[i,j]) == 12 or int(int_im[i,j])==13:
                            int_im[i,j] = 10

                        r, g, b = palette[int(int_im[i, j])]
                        img_color_labeled[i, j, 0] = r
                        img_color_labeled[i, j, 1] = g
                        img_color_labeled[i, j, 2] = b
        folder_path = os.path.join(save_path, "4_colorlabel")
        make_if_not_exists(folder_path)
        cv2.imwrite(os.path.join(folder_path, img_name), img_color_labeled)

        # * index image
        index_im = griddata(points, id, (X, Y), method='nearest').astype(np.float32)
        folder_path = os.path.join(save_path, "5_index")
        make_if_not_exists(folder_path)
        cv2.imwrite(os.path.join(folder_path, img_name.split("/")[-1]), index_im)

        # # <entry val="_nDOM" format="3.4" invalidValue="0" externalType="float" />
        f_nDOM_im = griddata(points, f_nDOM, (X, Y), method='linear').astype(np.float32)
        closing = cv2.morphologyEx(f_nDOM_im, cv2.MORPH_CLOSE, kernel)
        folder_path = os.path.join(save_path, "f_nDOM")
        make_if_not_exists(folder_path)
        tifffile.imsave(os.path.join(folder_path, img_name.split("/")[-1]), closing)

        # <entry val="_I" format="4.4" invalidValue="0" externalType="float" />
        f_I_im = griddata(points, f_I, (X, Y), method='nearest').astype(np.float32)
        closing = cv2.morphologyEx(f_I_im, cv2.MORPH_CLOSE, kernel)
        folder_path = os.path.join(save_path, "f_I")
        make_if_not_exists(folder_path)
        tifffile.imsave(os.path.join(folder_path, img_name.split("/")[-1]), closing)

        # <entry val="_Q_Pulse" format="1.4" invalidValue="1" externalType="float" />
        f_Q_Pulse_im = griddata(points, f_Q_Pulse, (X, Y), method='nearest').astype(np.float32)
        closing = cv2.morphologyEx(f_Q_Pulse_im, cv2.MORPH_CLOSE, kernel)
        folder_path = os.path.join(save_path, "f_Q_Pulse")
        make_if_not_exists(folder_path)
        tifffile.imsave(os.path.join(folder_path, img_name.split("/")[-1]), closing)

        # <entry val="_prior" format="1.4" invalidValue="1" externalType="float" />
        f_prior_im = griddata(points, f_prior, (X, Y), method='nearest').astype(np.float32)
        closing = cv2.morphologyEx(f_prior_im, cv2.MORPH_CLOSE, kernel)
        folder_path = os.path.join(save_path, "f_prior")
        make_if_not_exists(folder_path)
        tifffile.imsave(os.path.join(folder_path, img_name.split("/")[-1]), closing)

        # <entry val="_Q_N3" format="1.4" invalidValue="1" externalType="float" />
        f_Q_N3_im = griddata(points, f_Q_N3, (X, Y), method='nearest').astype(np.float32)
        closing = cv2.morphologyEx(f_Q_N3_im, cv2.MORPH_CLOSE, kernel)
        folder_path = os.path.join(save_path, "f_Q_N3")
        make_if_not_exists(folder_path)
        tifffile.imsave(os.path.join(folder_path, img_name.split("/")[-1]), closing)

        # <entry val="_Q_N2" format="1.4" invalidValue="1" externalType="float" />
        f_Q_N2_im = griddata(points, f_Q_N2, (X, Y), method='nearest').astype(np.float32)
        closing = cv2.morphologyEx(f_Q_N2_im, cv2.MORPH_CLOSE, kernel)
        folder_path = os.path.join(save_path, "f_Q_N2")
        make_if_not_exists(folder_path)
        tifffile.imsave(os.path.join(folder_path, img_name.split("/")[-1]), closing)

        # <entry val="_Q_N1" format="1.4" invalidValue="1" externalType="float" />
        f_Q_N1_im = griddata(points, f_Q_N1, (X, Y), method='nearest').astype(np.float32)
        closing = cv2.morphologyEx(f_Q_N1_im, cv2.MORPH_CLOSE, kernel)
        folder_path = os.path.join(save_path, "f_Q_N1")
        make_if_not_exists(folder_path)
        tifffile.imsave(os.path.join(folder_path, img_name.split("/")[-1]), closing)

        # <entry val="_Q_ES_3" format="1.4" invalidValue="0" externalType="float" />
        f_Q_ES_3_im = griddata(points, f_Q_ES_3, (X, Y), method='nearest').astype(np.float32)
        closing = cv2.morphologyEx(f_Q_ES_3_im, cv2.MORPH_CLOSE, kernel)
        folder_path = os.path.join(save_path, "f_Q_ES_3")
        make_if_not_exists(folder_path)
        tifffile.imsave(os.path.join(folder_path, img_name.split("/")[-1]), closing)

        # entry val="_Q_ES_2" format="1.4" invalidValue="0" externalType="float" />
        f_Q_ES_2_im = griddata(points, f_Q_ES_2, (X, Y), method='nearest').astype(np.float32)
        closing = cv2.morphologyEx(f_Q_ES_2_im, cv2.MORPH_CLOSE, kernel)
        folder_path = os.path.join(save_path, "f_Q_ES_2")
        make_if_not_exists(folder_path)
        tifffile.imsave(os.path.join(folder_path, img_name.split("/")[-1]), closing)

        # <entry val="_Q_ES_1" format="1.4" invalidValue="0" externalType="float" />
        f_Q_ES_1_im = griddata(points, f_Q_ES_1, (X, Y), method='nearest').astype(np.float32)
        closing = cv2.morphologyEx(f_Q_ES_1_im, cv2.MORPH_CLOSE, kernel)
        folder_path = os.path.join(save_path, "f_Q_ES_1")
        make_if_not_exists(folder_path)
        tifffile.imsave(os.path.join(folder_path, img_name.split("/")[-1]), closing)


# * the 2 functions blow are used in "main3", generation of 5cm-based label image, 10cm-based feature map
# 5cm only generate index image and label image
def generation_syntheticImg_5cmbased(px, py, myIndex, pt_labels, img_name, save_path, img_width, img_height):

    img_mask = np.zeros((img_height, img_width, 1), np.uint8)  # used for making a mask only
    img_temp = np.zeros((img_height, img_width, 3), np.uint8)  # used for point level labeled image only

    # index_im2 = np.zeros((img_height, img_width), np.float32)  # point index, without interpolation

    label_value = []
    points = []
    id = []

    count = 0
    for i in range(0, px.shape[1]):
        if img_width > px[0, i] > 0 and img_height > py[0, i] > 0:
            # filter the points which are not in the FOV of this image
            points.append([px[0, i], py[0, i]])
            label_value.append(int(pt_labels[myIndex[i]]))
            id.append(myIndex[i])

            r, g, b = palette[int(pt_labels[myIndex[i]])]
            img_temp[int(py[0, i]), int(px[0, i]), 0] = r
            img_temp[int(py[0, i]), int(px[0, i]), 1] = g
            img_temp[int(py[0, i]), int(px[0, i]), 2] = b
            count += 1

            cv2.circle(img_mask, (int(px[0, i]), int(py[0, i])), 3, (255,255,255), -1)  # level3
            # cv2.circle(img_mask, (int(px[0, i]), int(py[0, i])), 6, (255, 255, 255), -1)  # level0

            # index_im2[int(py[0, i]), int(px[0, i])] = myIndex[i]  # point index, without interpolation


    points = np.array(points)
    label_value = np.array(label_value)
    id = np.array(id)

    if count > 100:  # I suppose at least 20% of pixels in the image should receive 3d point
        # save the point level label image for checking
        folder_path = os.path.join(save_path, "1_pointlabel")
        make_if_not_exists(folder_path)
        cv2.imwrite(os.path.join(folder_path, img_name.split("/")[-1]), img_temp)

        # mask, where no point is projected to pixel
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))  # level3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))  # level0
        closing_img_temp = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel).astype(np.uint8)
        folder_path = os.path.join(save_path, "2_mask")
        make_if_not_exists(folder_path)
        cv2.imwrite(os.path.join(folder_path, img_name.split("/")[-1]), closing_img_temp)

        mask = (closing_img_temp[:,:] != 0)

        # Generation of synthetic image based on different feature
        X, Y = np.meshgrid(np.arange(0, img_width, 1), np.arange(0, img_height, 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))  # level3
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # level0

        # * labeled image (grey, as ground truth for training)
        int_im = griddata(points, label_value, (X, Y), method='nearest').astype(np.uint8)
        closing = cv2.morphologyEx(int_im, cv2.MORPH_CLOSE, kernel).astype(np.uint8)
        closing[mask[:, :] == False] = 255  # label value 255 for unlabeled pixel
        folder_path = os.path.join(save_path, "3_greylabel")
        make_if_not_exists(folder_path)
        cv2.imwrite(os.path.join(folder_path, img_name.split("/")[-1]), closing)

        # * labeled image (color, only for visualization)
        img_color_labeled = np.zeros((img_height, img_width, 3), np.uint8)
        for i in range(0, img_height):
                for j in range(0, img_width):
                        if int(int_im[i,j]) == 11 or int(int_im[i,j]) == 12 or int(int_im[i,j])==13:
                            int_im[i,j] = 10
                            print("!!!!")

                        if mask[i,j] == True:
                            r, g, b = palette[int(int_im[i, j])]
                            img_color_labeled[i, j, 0] = r
                            img_color_labeled[i, j, 1] = g
                            img_color_labeled[i, j, 2] = b
        folder_path = os.path.join(save_path, "4_colorlabel")
        make_if_not_exists(folder_path)
        cv2.imwrite(os.path.join(folder_path, img_name), img_color_labeled)

        # * index image
        index_im = griddata(points, id, (X, Y), method='nearest').astype(np.float32)
        folder_path = os.path.join(save_path, "5_index")
        make_if_not_exists(folder_path)
        tifffile.imsave(os.path.join(folder_path, img_name.split("/")[-1]), index_im)

        # # * index image, without interpolation
        # folder_path = os.path.join(save_path, "6_pointindex")
        # make_if_not_exists(folder_path)
        # tifffile.imsave(os.path.join(folder_path, img_name.split("/")[-1]), index_im2)

        return mask


# 10cm only generate feature map, here, 72 feature maps
def generation_syntheticImg_10cmbased(mask, px2, py2, myIndex2, pt_features, img_name, save_path, img_width, img_height):

    points2 = []
    id2 = []

    count = 0
    for i in range(0, px2.shape[1]):
        if img_width > px2[0, i] > 0 and img_height > py2[0, i] > 0:
            # filter the points which are not in the FOV of this image
            points2.append([px2[0, i], py2[0, i]])
            id2.append(myIndex2[i])
            count += 1

    points2 = np.array(points2)
    id2 = np.array(id2)

    if count > 100:  # I suppose at least 20% of pixels in the image should receive 3d point
        # Generation of synthetic image based on different feature
        X, Y = np.meshgrid(np.arange(0, img_width, 1), np.arange(0, img_height, 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # * feature map
        features = np.zeros((id2.shape[0], pt_features.shape[1]), np.float32)

        for i in range(0, features.shape[1]):
            for j in range(0, features.shape[0]):
                features[j, i] = pt_features[id2[j], i]

        for i in range(0, features.shape[1]):
            f_im = griddata(points2, features[:,i], (X, Y), method='linear').astype(np.float32)
            closing = cv2.morphologyEx(f_im, cv2.MORPH_CLOSE, kernel)
            closing[mask[:, :] == False] = np.nan
            folder_path = os.path.join(save_path, "f_"+str(i))
            make_if_not_exists(folder_path)
            tifffile.imsave(os.path.join(folder_path, img_name.split("/")[-1]), closing)