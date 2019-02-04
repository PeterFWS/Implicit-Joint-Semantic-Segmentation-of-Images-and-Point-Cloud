import numpy as np
import time
import glob
import itertools
import os
from tqdm import tqdm
import math
import cv2

file_Features = "./data/data_splits_10cm/X_10cm_train.txt"
pt_features = np.loadtxt(file_Features)  # (9559941, 72)

file_format = "./data/data_splits_5cm_onlylabel/feature format.txt"

# # obtain invalid value of each feature
# fm = []
# with open(file_format, "r") as fp:
#     for line in fp:
#         invalid_value = int(line[line.find("invalidValue") + 14])
#         if invalid_value == 9:
#             fm.append(90)
#         else:
#             fm.append(invalid_value)

# # calculate mean value and standard deviation of each feature
# mean = []
# sdv = []
# for i in tqdm(range(pt_features.shape[1])):
#     # invalid_value = fm[i]
#     data = pt_features[:, i]  # (9559941,)
#
#     # temp = []
#     # for x in data:
#     #     # if int(x) != int(invalid_value):
#     #         temp.append(x)
#     # temp = np.asarray(temp)
#     m = np.mean(data)
#     sd = np.std(data)
#
#     mean.append(m)
#     sdv.append(sd)
#
# mean = np.asarray(mean)
# sdv = np.asarray(sdv)
#
# np.savetxt('./mean.out', mean)
# np.savetxt('./sdv.out', sdv)

# max and min value
max = []
min = []
for i in tqdm(range(pt_features.shape[1])):
    invalid_value = fm[i]
    data = pt_features[:, i]  # (9559941,)

    temp = []
    for x in data:
        if int(x) != int(invalid_value):
            temp.append(x)
    temp = np.asarray(temp)
    max.append(np.amax(temp))
    min.append(np.amin(temp))

np.savetxt('./max.out', max)
np.savetxt('./min.out', min)

########  RGB
path_level3_img = "/home/fangwen/ShuFangwen/source/image-segmentation-keras/data/train_set/rgb_img"
img_list = os.listdir(path_level3_img)

# R = []
# G = []
# B = []
# for name in img_list:
#     img = cv2.imread(os.path.join(path_level3_img, name))
#     temp_B = img[:,:,0].ravel()
#     temp_G = img[:,:,1].ravel()
#     temp_R = img[:,:,2].ravel()
#
#     R.append(temp_R)
#     G.append(temp_G)
#     B.append(temp_B)
#
# R_all = np.concatenate([_ for _ in R])
# G_all = np.concatenate([_ for _ in G])
# B_all = np.concatenate([_ for _ in B])
#
# mean_R = np.mean(R_all)
# mean_G = np.mean(G_all)
# mean_B = np.mean(B_all)
#
# std_R = np.std(R_all)
# std_G = np.std(G_all)
# std_B = np.std(B_all)

# H = []
# S = []
# V = []
# for name in img_list:
#     img = cv2.imread(os.path.join(path_level3_img, name))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     temp_H = img[:,:,0].ravel() # B
#     temp_S = img[:,:,1].ravel() # G
#     temp_V = img[:,:,2].ravel() # R
#
#     H.append(temp_H)
#     S.append(temp_S)
#     V.append(temp_V)
#
# H_all = np.concatenate([_ for _ in H])
# S_all = np.concatenate([_ for _ in S])
# V_all = np.concatenate([_ for _ in V])
#
# mean_H = np.mean(H_all)
# mean_S = np.mean(S_all)
# mean_V = np.mean(V_all)
#
# std_H = np.std(H_all)
# std_S = np.std(S_all)
# std_V = np.std(V_all)