import csv
import math
import numpy as np
import scipy as sp
# import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
# from mpl_toolkits.mplot3d import Axes3D
import cv2
import my_parameters

import time

'''
Function used to Import csv Points
'''
def importPoints(fileName):
    p = np.loadtxt(fileName)
    return p


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





# Main
print(">>>>>>>>>>>>>>>>>>Starting to load data!<<<<<<<<<<<<<<<<<<<<< \n")
start_time = time.time()
Points = importPoints('hpr_pointsdata.txt')
myPoints = Points[:, 0:3]
labels = Points[:, -1]
duration = time.time() - start_time
print("Duration: ", duration, " s \n")
print(">>>>>>>>>>>>>>>>>>Down!<<<<<<<<<<<<<<<<<<<<<<<<<<! \n")

# Use a flag array indicating visibility, most efficent in speed and memory
print(">>>>>>>>>>>>>>>>>>Starting to process data!<<<<<<<<<<<<<<<<<<<<< \n")
start_time = time.time()
flag = np.zeros(len(myPoints),int)  #  0 - Invisible; 1 - Visible.
C = np.array([[513956.31971618492, 5426766.6255130861, 276.9661760997179300]])  # Center
flippedPoints = sphericalFlip(myPoints, C, math.pi)
myHull = convexHull(flippedPoints)
visibleVertex = myHull.vertices[:-1]  # indexes of visible points
duration = time.time() - start_time
print("Duration: ", duration, " s \n")
print(">>>>>>>>>>>>>>>>>>Down!<<<<<<<<<<<<<<<<<<<<<<<<<<! \n")

flag[visibleVertex] = 1
visibleId = np.where(flag == 1)[0]  # indexes of the invisible points

img_path = "./CF013540.jpg"
img = cv2.imread(img_path)
print(img.shape)
img2 = np.zeros(img.shape, np.uint8)
img3 = np.zeros(img.shape, np.uint8)

for i in visibleId:
    c = my_parameters.color_classes[str(labels[i])]
    cv2.circle(img2, (int(Points[i, 3]), int(Points[i, 4])), 5, c, -1)
    cv2.circle(img3, (int(Points[i, 3]), int(Points[i, 4])), 10, c, -1)

cv2.imwrite("./2.jpg", img2)
cv2.imwrite("./3.jpg", img3)




