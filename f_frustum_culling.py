import numpy as np
import my_parameters
from my_parameters import Vec3
from my_parameters import Plane
import time

"""
 * Geometric Approach for View Frustum Culling
  * test data: 3d point clouds
"""

def setCamInternals(nearD, farD):
    # compute width and height of the near and far plane sections
    nh = my_parameters.height * my_parameters.pixel_size
    nw = my_parameters.width * my_parameters.pixel_size
    fh = nh*farD/nearD
    fw = nw*farD/nearD

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

    nc = p - Vec3.__mul__(Z , nearD)
    fc = p - Vec3.__mul__(Z , farD)

    ntl = nc + Vec3.__mul__(Y , nh) - Vec3.__mul__(X , nw)
    ntr = nc + Vec3.__mul__(Y , nh) + Vec3.__mul__(X , nw)
    nbl = nc - Vec3.__mul__(Y , nh) - Vec3.__mul__(X , nw)
    nbr = nc - Vec3.__mul__(Y , nh) + Vec3.__mul__(X , nw)

    ftl = fc + Vec3.__mul__(Y , fh) - Vec3.__mul__(X , fw)
    ftr = fc + Vec3.__mul__(Y , fh) + Vec3.__mul__(X , fw)
    fbl = fc - Vec3.__mul__(Y , fh) - Vec3.__mul__(X , fw)
    fbr = fc - Vec3.__mul__(Y , fh) + Vec3.__mul__(X , fw)

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



def frustum_culling(ex_data, xyz, labels, features, index):
    """
    Main
    """
    print("Culling frustum... ")
    start_time = time.time()

    X_, Y_, Z_ = map(float, ex_data[1:4])
    nearD = abs(my_parameters.f)
    farD = Z_

    nh, nw, fh, fw = setCamInternals(nearD, farD)

    pl_TOP, pl_BOTTOM, pl_LEFT, pl_RIGHT, pl_NEARP, pl_FARP = \
        setCamDef(nh, nw, fh, fw, nearD, farD, Vec3(0,0,0), Vec3(0,0,1), Vec3(0, 1, 0))

    xyz_temp = []
    label_temp = []
    feature_temp = []
    index_temp = []
    for i in range(0, xyz.shape[0]):
        pt = Vec3(xyz[i, 0]-X_, xyz[i, 1]-Y_, xyz[i, 2]-Z_)
        flag = pointInFrustum(pt,
                       pl_TOP, pl_BOTTOM, pl_LEFT,
                       pl_RIGHT, pl_NEARP, pl_FARP)
        if flag == 1:
            xyz_temp.append([xyz[i, 0], xyz[i, 1], xyz[i, 2]])
            label_temp.append(labels[i])
            feature_temp.append([_ for _ in features[i, :]])
            index_temp.append(index[i])

    xyz_temp = np.matrix(xyz_temp)
    label_temp = np.asarray(label_temp)
    feature_temp = np.asarray(feature_temp)
    index_temp = np.asarray(index_temp)

    duration = time.time() - start_time
    print(duration, "s\n")

    return xyz_temp, label_temp, feature_temp, index_temp




