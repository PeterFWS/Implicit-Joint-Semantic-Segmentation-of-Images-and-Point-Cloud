import os
import shutil


## clean not_good imgs
# path = "/home/fangwen/ShuFangwen/source/image-segmentation-keras/data/validation_set/"
# folder_list = os.listdir(path)
# folder_list.remove("not_good")
# folder_list.remove("4_colorlabel")
#
# notgood_img_list = os.listdir("/home/fangwen/ShuFangwen/source/image-segmentation-keras/data/validation_set/not_good/4_colorlabel")
# for folder_name in folder_list:
#     src = os.path.join(path, folder_name)
#     dst = os.path.join("/home/fangwen/ShuFangwen/source/image-segmentation-keras/data/validation_set/not_good", folder_name)
#     if not os.path.exists(dst):
#         os.makedirs(dst)
#     for img_name in notgood_img_list:
#         src2 = os.path.join(src, img_name)
#         dst2 = os.path.join(dst, img_name)
#         shutil.move(src2, dst2)


## copy corresponding BGR images
# img_list = os.listdir("/home/fangwen/ShuFangwen/source/image-segmentation-keras/data/level3_nadir/test_set/2_mask")
# for img_name in img_list:
#     src = os.path.join("/home/fangwen/ShuFangwen/data/Nadir_level3_level5/ImgTexture/Level_3", img_name)
#     dst = os.path.join("/home/fangwen/ShuFangwen/source/image-segmentation-keras/data/level3_nadir/test_set/rgb_img", img_name)
#     shutil.copyfile(src, dst)


# clean redundant imgs
# reference = os.listdir("/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_nadir/validation_set/f_68")
# folder_list = os.listdir("/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_nadir/validation_set")
# folder_list.remove("f_68")
#
# for folder in folder_list:
#     folder_path = os.path.join("/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_nadir/validation_set", folder)
#     img_list = os.listdir(folder_path)
#
#     for name in img_list:
#         if name not in reference:
#             img_path = os.path.join(folder_path, name)
#             os.remove(img_path)
#             print(img_path)


## calculate class_weights
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.utils import class_weight

label_img_list = os.listdir("/data/fangwen/results/level3_nadir/chip_train_set/3_greylabel")

# all_label = []
num_classes = np.zeros(12, int)
for i in tqdm(range(len(label_img_list))):
    name = label_img_list[i]
    path = os.path.join("/data/fangwen/results/level3_nadir/chip_train_set/3_greylabel", name)
    img = cv2.imread(path).ravel()

    for j in range(len(img)):
        if img[j] != 255:
            num_classes[img[j]] += 1
        elif img[j] == 255:
            num_classes[11] += 1

    # all_label.append(img)

total = sum(num_classes)

frequency = num_classes / float(total)

inverse = 1/ frequency

# y_train = np.concatenate([p for p in all_label])
#
# class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.array([0,1,2,3,4,5,6,7,8,9,10,255]),
#                                                  y_train)




