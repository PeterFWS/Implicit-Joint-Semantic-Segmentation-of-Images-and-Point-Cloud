import math
import numpy as np
from scipy.spatial import ConvexHull
import time

"""
 * Hidden Point Removal
"""

'''
Function used to Perform Spherical Flip on the Original Point Cloud
'''
def sphericalFlip(points, center, param):
    n = points.shape[0]  # total n points
    points = points - np.repeat(center, n, axis=0)  # Move C to the origin
    normPoints = np.linalg.norm(points, axis=1)  # Normed points, sqrt(x^2 + y^2 + (z-100)^2)
    R = np.repeat(max(normPoints) * np.power(10.0, param), n, axis=0)  # Radius of Sphere

    flippedPointsTemp = 2 * np.multiply(np.repeat((R - normPoints).reshape(n, 1), len(points[0]), axis=1), points)
    flippedPoints = np.divide(flippedPointsTemp, np.repeat(normPoints.reshape(n, 1), len(points[0]),
                                                           axis=1))  # Apply Equation to get Flipped Points
    flippedPoints += points

    return flippedPoints


'''
Function used to Obtain the Convex hull
'''
def convexHull(points):
    points = np.append(points, [[0, 0, 0]], axis=0)  # All points plus origin
    hull = ConvexHull(points)  # Visibal points plus possible origin. Use its vertices property.

    return hull


def HPR(ex_data, xyz_temp, label_temp, feature_temp, index_temp):
    """
    Main
    """
    print("Hidden point removal... ")
    start_time = time.time()

    flag = np.zeros(len(xyz_temp),int)  #  0 - Invisible; 1 - Visible.
    X_, Y_, Z_ = map(float, ex_data[1:4])
    C = np.array([[X_, Y_, Z_]])  # Center
    flippedPoints = sphericalFlip(xyz_temp, C, math.pi)
    myHull = convexHull(flippedPoints)
    visibleVertex = myHull.vertices[:-1]  # indexes of visible points

    flag[visibleVertex] = 1
    visibleId = np.where(flag == 1)[0]  # indexes of the visible points

    myPoints = []
    mylabels = []
    myfeatures = []
    myindex = []
    for i in visibleId:
        myPoints.append([xyz_temp[i, 0], xyz_temp[i, 1], xyz_temp[i, 2]])
        mylabels.append(label_temp[i])
        myfeatures.append([_ for _ in feature_temp[i, :]])
        myindex.append(index_temp[i])

    myPoints = np.matrix(myPoints)
    mylabels = np.asarray(mylabels)
    myfeatures = np.asarray(myfeatures)
    myindex = np.asarray(myindex)

    duration = time.time() - start_time
    print(duration, "s\n")

    return myPoints, mylabels, myfeatures, myindex




