import numpy as np
from open3d import *


pcd_train = read_point_cloud("/home/fangwen/ShuFangwen/data/data_splits/xyz_10cm_train.txt", format='xyz')
pcd_test = read_point_cloud("/home/fangwen/ShuFangwen/data/data_splits/xyz_10cm_test.txt", format='xyz')
pcd_val = read_point_cloud("/home/fangwen/ShuFangwen/data/data_splits/xyz_10cm_val.txt", format='xyz')

pcd_train.paint_uniform_color([1, 0.706, 0])

pcd_test.paint_uniform_color([0, 255, 0])

pcd_val.paint_uniform_color([0, 0, 255])
draw_geometries([pcd_train, pcd_test, pcd_val])