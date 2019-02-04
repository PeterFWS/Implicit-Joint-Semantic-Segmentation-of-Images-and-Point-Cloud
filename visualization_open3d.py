import numpy as np
from open3d import *


pcd_train = read_point_cloud("/home/fangwen/ShuFangwen/data/data_splits/xyz_10cm_train.txt", format='xyz')
pcd_test = read_point_cloud("/home/fangwen/ShuFangwen/data/data_splits/xyz_10cm_test.txt", format='xyz')
pcd_val = read_point_cloud("/home/fangwen/ShuFangwen/data/data_splits/xyz_10cm_val.txt", format='xyz')

pcd_train.paint_uniform_color([1, 0.706, 0])

pcd_test.paint_uniform_color([0, 255, 0])

pcd_val.paint_uniform_color([0, 0, 255])
draw_geometries([pcd_train, pcd_test, pcd_val])





import cv2
def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


image = cv2.imread('/home/fangwen/ShuFangwen/source/image-segmentation-keras/data/train_set/rgb_img/DSC04320.tif')
overlay = cv2.imread('/home/fangwen/ShuFangwen/source/image-segmentation-keras/data/train_set/4_colorlabel/DSC04320.tif')
output = image.copy()

for alpha in frange(0, 1, 0.1):
    cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, output)
    cv2.imwrite("./overlay_" + str(alpha) + ".tif", output)