import math
import numpy as np
import os
import my_parameters
from my_parameters import Vec3
from my_parameters import Plane
from my_parameters import ANG2RAD
import time

"""
* Parameters:
    Vec3 ntl,ntr,nbl,nbr,ftl,ftr,fbl,fbr;
    float nearD, farD, ratio, angle,tang;
    float nw,nh,fw,fh;
"""

def make_if_not_exists(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

# def setCamInternals(angle, ratio, nearD, farD):
#     tang = float(math.tan(angle * ANG2RAD * 0.5))
#     nh = nearD * tang
#     nw = nh * ratio
#     fh = farD * tang
#     fw = fh * ratio
#
#     return nh, nw, fh, fw

def setCamInternals(nearD, farD):
    nh = my_parameters.height * my_parameters.pixel_size
    nw = my_parameters.width * my_parameters.pixel_size
    fh = nh*farD/nearD
    fw = nw*farD/nearD

    return nh, nw, fh, fw

def setCamDef(nh, nw, fh, fw, nearD, farD, p=Vec3(), l=Vec3(), u=Vec3()):
    # Vec3 dir,nc,fc,X,Y,Z;
    Z = p - l
    Z.normalize()

    X = u * Z
    X.normalize()

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

def pointInFrustum(p=Vec3(),
                   pl_TOP=Plane(), pl_BOTTOM=Plane(), pl_LEFT=Plane(),
                   pl_RIGHT=Plane(), pl_NEARP=Plane(), pl_FARP=Plane()):
    result = "INSIDE"

    if pl_TOP.distance(p)<0 \
        or pl_BOTTOM.distance(p)<0 \
        or pl_LEFT.distance(p)<0 \
        or pl_RIGHT.distance(p)<0 \
        or pl_NEARP.distance(p)<0 \
        or pl_FARP.distance(p)<0:
        return "OUTSIDE"

    return point_status[result]



if __name__ == "__main__":
    """
    test for only one image
    """
    mgs_path = "./Images"
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

    # get 3D point coordinates and labels from point cloud data
    xyz = pc_data[:, :3]  # (14129889,3)

    X_, Y_, Z_ = 513956.3197161849200000, 5426766.6255130861000000, 276.9661760997179300
    nearD = abs(my_parameters.f)
    farD = Z_

    nh, nw, fh, fw = setCamInternals(nearD, farD)

    pl_TOP, pl_BOTTOM, pl_LEFT, pl_RIGHT, pl_NEARP, pl_FARP = \
        setCamDef(nh, nw, fh, fw, nearD, farD, p=Vec3(X_, Y_, Z_), l=Vec3(0,0,1), u=Vec3())

