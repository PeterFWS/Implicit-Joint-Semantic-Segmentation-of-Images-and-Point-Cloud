import math
import numpy as np
from scipy.spatial import ConvexHull



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


def HPR(ex_data, my_xyz, labels):
    """
    Main
    """
    print("Starting HPR\n")

    flag = np.zeros(len(my_xyz),int)  #  0 - Invisible; 1 - Visible.
    X_, Y_, Z_ = map(float, ex_data[1:4])
    C = np.array([[X_, Y_, Z_]])  # Center
    flippedPoints = sphericalFlip(my_xyz, C, math.pi)
    myHull = convexHull(flippedPoints)
    visibleVertex = myHull.vertices[:-1]  # indexes of visible points

    flag[visibleVertex] = 1
    visibleId = np.where(flag == 1)[0]  # indexes of the visible points

    myPoints = []
    mylabels = []
    for i in visibleId:
        myPoints.append([my_xyz[i, 0], my_xyz[i, 1], my_xyz[i, 2]])
        mylabels.append(labels[0, i])

    myPoints = np.matrix(myPoints)
    mylabels = np.matrix(mylabels)

    return myPoints, mylabels




