import os
import shutil


# #### not good
# path = "/home/fangwen/ShuFangwen/source/image-segmentation-keras/data/test_set/"
# folder_list = os.listdir(path)
# folder_list.remove("not_good")
# folder_list.remove("1_pointlabel")
#
# notgood_img_list = os.listdir("/home/fangwen/ShuFangwen/source/image-segmentation-keras/data/test_set/not_good/1_pointlabel")
# for folder_name in folder_list:
#
#     src = os.path.join(path, folder_name)
#     dst = os.path.join("/home/fangwen/ShuFangwen/source/image-segmentation-keras/data/test_set/not_good/", folder_name)
#     if not os.path.exists(dst):
#         os.makedirs(dst)
#
#     for img_name in notgood_img_list:
#         src2 = os.path.join(src, img_name)
#         dst2 = os.path.join(dst, img_name)
#         shutil.move(src2,dst2)
#
#
# #### rgb
img_list = os.listdir("/data/fangwen/results/level3/test_set/1_pointlabel")
for img_name in img_list:
    src = os.path.join("/data/fangwen/data/ImgTexture/Level_3", img_name)
    dst = os.path.join("/data/fangwen/results/level3/test_set/rgb_img", img_name)
    shutil.copyfile(src, dst)


# #####
# img_list_f = os.listdir("/data/fangwen/results/level3/test_set/f_0")
# img_list_index = os.listdir("/data/fangwen/results/level3/test_set/5_index")
#
# for name in img_list_index:
#     if name not in img_list_f:
#         src = os.path.join("/data/fangwen/results/level3/test_set/5_index", name)
#         dst = os.path.join("/data/fangwen/results/level3/test_set/rest", name)
#         shutil.move(src, dst)