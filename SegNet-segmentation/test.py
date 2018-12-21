import os
import shutil


img_list = os.listdir("/home/fangwen/ShuFangwen/source/image-segmentation-keras/data/test/2_mask")


for img_name in img_list:
    src = os.path.join("/home/fangwen/ShuFangwen/data/ImgTexture/Level_3", img_name)
    dst = os.path.join("/home/fangwen/ShuFangwen/source/image-segmentation-keras/data/test/rgb_img", img_name)
    shutil.copyfile(src, dst)

