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
# img_list = os.listdir("/data/fangwen/results/validation_set/2_mask")
# for img_name in img_list:
#     src = os.path.join("/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/data/Nadir_level3_level4_level5/ImgTexture/Level_4", img_name)
#     dst = os.path.join("/data/fangwen/results/validation_set/rgb_img", img_name)
#     shutil.copyfile(src, dst)
#
#
# # clean redundant imgs
# import os
# import shutil
# reference = os.listdir("/data/fangwen/oblique_train/rgb_img")
# folder_list = os.listdir("/data/fangwen/oblique_train")
# folder_list.remove("rgb_img")
#
# for folder in folder_list:
#     folder_path = os.path.join("/data/fangwen/oblique_train", folder)
#     img_list = os.listdir(folder_path)
#
#     for name in img_list:
#         if name not in reference:
#             img_path = os.path.join(folder_path, name)
#             os.remove(img_path)
#             print(img_path)
#
#
# #--------------------------------------------------------    calculate class_weights
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.utils import class_weight
#
label_img_list = os.listdir("/data/fangwen/mix_train/3_greylabel")
#
# # all_label = []
num_classes = np.zeros(12, int)
for i in tqdm(range(len(label_img_list))):
    name = label_img_list[i]
    path = os.path.join("/data/fangwen/mix_train/3_greylabel", name)
    img = cv2.imread(path).ravel()

    for j in range(len(img)):
        if img[j] != 255:
            num_classes[img[j]] += 1
        elif img[j] == 255:
            num_classes[11] += 1
#
#     # all_label.append(img)
#
total = sum(num_classes)
# #
frequency = num_classes / float(total)
# #
# inverse = 1/ frequency

class_weights = np.median(frequency)/frequency
#
# # y_train = np.concatenate([p for p in all_label])
# #
# # class_weights = class_weight.compute_class_weight('balanced',
# #                                                  np.array([0,1,2,3,4,5,6,7,8,9,10,255]),
# #                                                  y_train)
#
#
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter
from open3d import *

def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0, 'left': 0.8}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')
#
#
N = 12
ind = np.arange(N)
width = 0.35
fig, ax = plt.subplots()
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
rects1 = ax.bar(ind - width/2, num_classes[:], width,
                color='SkyBlue', label='per pixel label in training-set imagey')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of pixels')
ax.set_title('Total number of each of classes ')
ax.set_xticks(ind)
ax.set_xticklabels(('PowerLine', 'Low Vegetation', 'Impervious Surface', 'Vehicles', 'Urban Furniture',
                    'Roof', 'Facade', 'Bush/Hedge', 'Tree', 'Dirt/Gravel', 'Vertical Surface', 'Void'))
ax.legend()


autolabel(rects1, "left")

plt.show()


#------------------------------------------------------ Data augmentation
import os
import cv2
import shutil
from tqdm import tqdm

def make_if_not_exists(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

path = "/data/fangwen/mix_train/3_greylabel"

path2 = "/data/fangwen/mix_train"
floder_list = os.listdir(path2)

save_path = "/data/fangwen/sythetic_train"
make_if_not_exists(save_path)

img_list = os.listdir(path)
for _ in tqdm(range(len(img_list))):
    name = img_list[_]
    img_path = os.path.join(path, name)
    img = cv2.imread(img_path, 0).ravel()
    count_0 = 0 # powerline
    # count_3 = 0 # car
    # count_4 = 0 # urban furniture
    # count_6 = 0 # facade
    # count_10 = 0 # vertical surface
    for i in range(len(img)):
        if img[i] == 0:
            count_0 += 1
        # elif img[i] == 3:
        #     count_3 += 1
        # elif img[i] == 4:
        #     count_4 += 1
        # elif img[i] == 6:
        #     count_6 += 1
        # elif img[i] == 10:
        #     count_10 += 1
    if count_0 >= 20:
        for floder in floder_list:
            folder_path = os.path.join(path2, floder)
            f_img_path = os.path.join(folder_path, name)
            save_path2 = os.path.join(save_path, floder)
            make_if_not_exists(save_path2)
            shutil.copyfile(f_img_path, os.path.join(save_path2, name.split(".")[-2]+"_pwl"+".tif"))
    # elif count_3 >= 20:
    #     for floder in floder_list:
    #         folder_path = os.path.join(path2, floder)
    #         f_img_path = os.path.join(folder_path, name)
    #         save_path2 = os.path.join(save_path, floder)
    #         make_if_not_exists(save_path2)
    #         shutil.copyfile(f_img_path, os.path.join(save_path2, name.split(".")[-2]+"_vehicle"+".tif"))
    # elif count_4 >= 20:
    #     for floder in floder_list:
    #         folder_path = os.path.join(path2, floder)
    #         f_img_path = os.path.join(folder_path, name)
    #         save_path2 = os.path.join(save_path, floder)
    #         make_if_not_exists(save_path2)
    #         shutil.copyfile(f_img_path, os.path.join(save_path2, name.split(".")[-2]+"_urbanfurniture"+".tif"))
    # elif count_6 >= 20:
    #     for floder in floder_list:
    #         folder_path = os.path.join(path2, floder)
    #         f_img_path = os.path.join(folder_path, name)
    #         save_path2 = os.path.join(save_path, floder)
    #         make_if_not_exists(save_path2)
    #         shutil.copyfile(f_img_path, os.path.join(save_path2, name.split(".")[-2]+"_facade"+".tif"))
    # elif count_10 >= 20:
    #     for floder in floder_list:
    #         folder_path = os.path.join(path2, floder)
    #         f_img_path = os.path.join(folder_path, name)
    #         save_path2 = os.path.join(save_path, floder)
    #         make_if_not_exists(save_path2)
    #         shutil.copyfile(f_img_path, os.path.join(save_path2, name.split(".")[-2]+"_vertical_surface"+".tif"))



# resize nadir image
# import os
# import cv2
# from tqdm import tqdm
# import tifffile
#
#
# def make_if_not_exists(dirPath):
#     if not os.path.exists(dirPath):
#         os.makedirs(dirPath)
#
# path = "/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_nadir/train_set/2_mask"
# img_list = os.listdir(path)
#
# path2 = "/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_nadir/train_set"
# folder_list = os.listdir(path2)
# folder_list.remove("1_pointlabel")
#
#
# save_path = "/data/fangwen/sythetic_nadir_resized"
# make_if_not_exists(save_path)
#
# for _ in tqdm(range(len(img_list))):
#     name = img_list[_]
#
#     for folder in folder_list:
#         folder_path = os.path.join(path2, folder)
#
#         img_path = os.path.join(folder_path, name)
#
#         if folder_path.split("/")[-1].split("_")[-2] == "rgb" or folder_path.split("/")[-1].split("_")[-2] == "4":
#             img = cv2.imread(img_path, 1)
#
#         elif folder_path.split("/")[-1].split("_")[-2] == "2" or folder_path.split("/")[-1].split("_")[-2] == "3":
#             img = cv2.imread(img_path, 0)
#
#         else:
#             img = tifffile.imread(img_path)
#
#
#         img = cv2.resize(img, (480, 480))
#
#         save_path2 = os.path.join(save_path, folder)
#         make_if_not_exists(save_path2)
#
#         if folder_path.split("/")[-1].split("_")[-2] == "rgb" or folder_path.split("/")[-1].split("_")[-2] == "4":
#             img = cv2.imwrite(os.path.join(save_path2, name.split(".")[-2]+"_resized"+".tif"), img)
#
#         elif folder_path.split("/")[-1].split("_")[-2] == "2" or folder_path.split("/")[-1].split("_")[-2] == "3":
#             img = cv2.imwrite(os.path.join(save_path2, name.split(".")[-2] + "_resized" + ".tif"), img)
#
#         else:
#             img = tifffile.imsave(os.path.join(save_path2, name.split(".")[-2] + "_resized" + ".tif"), img)







# ----------------------------------------------------------------------------create grey label image
# import os
# import cv2
# from tqdm import tqdm
# import numpy as np
# palette = {  # BGR
#     0: (255, 0, 0),      # Powerline
#     1: (255, 255, 255),  # Low Vegetation
#     2: (255, 255, 0),    # Impervious Surface
#     3: (255, 0, 255),    # Vehicles
#     4: (0, 255, 255),    # Urban Furniture
#     5: (0, 255, 0),      # Roof
#     6: (0, 0, 255),      # Facade
#     7: (239, 120, 76),   # Bush/Hedge
#     8: (247, 238, 179),  # Tree
#     9: (0, 18, 114),     # Dirt/Gravel
#     10: (63, 34, 15),    # Vertical Surface
#     255: (0, 0, 0)        # Void
# }
#
# invert_palette = {v: k for k, v in palette.items()}
#
# def convert_to_color(arr_2d, palette=palette):
#     """ Numeric labels to RGB-color encoding """
#     arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
#
#     for c, i in palette.items():
#         m = arr_2d == c
#         arr_3d[m] = i
#
#     return arr_3d
#
#
# def convert_from_color(arr_3d, palette=invert_palette):
#     """ RGB-color encoding to grayscale labels """
#     arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
#
#     for c, i in palette.items():
#         m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
#         arr_2d[m] = i
#
#     return arr_2d
#
#
# color_label_img_list = os.listdir("/data/fangwen/Merged_dataset_oblique_nadir_480x480/3_greylabel")
#
# for _ in tqdm(range(len(color_label_img_list))):
#     name = color_label_img_list[_]
#
#     img = cv2.imread(os.path.join("/data/fangwen/Merged_dataset_oblique_nadir_480x480/3_greylabel", name), 0)
#
#     img = convert_to_color(img)
#
#     cv2.imwrite(os.path.join("/data/fangwen/Merged_dataset_oblique_nadir_480x480/test", name), img)




# --------------------------------------------------create a color bar
# import matplotlib as mpl
# import numpy as np
# import sys
# import matplotlib.pyplot as plt
#
# def make_cmap(colors, position=None, bit=False):
#     '''
#     make_cmap takes a list of tuples which contain RGB values. The RGB
#     values may either be in 8-bit [0 to 255] (in which bit must be set to
#     True when called) or arithmetic [0 to 1] (default). make_cmap returns
#     a cmap with equally spaced colors.
#     Arrange your tuples so that the first color is the lowest value for the
#     colorbar and the last is the highest.
#     position contains values from 0 to 1 to dictate the location of each color.
#     '''
#
#     bit_rgb = np.linspace(0, 1, 256)
#     if position == None:
#         position = np.linspace(0, 1, len(colors))
#     else:
#         if len(position) != len(colors):
#             sys.exit("position length must be the same as colors")
#         elif position[0] != 0 or position[-1] != 1:
#             sys.exit("position must start with 0 and end with 1")
#     if bit:
#         for i in range(len(colors)):
#             colors[i] = (bit_rgb[colors[i][0]],
#                          bit_rgb[colors[i][1]],
#                          bit_rgb[colors[i][2]])
#     cdict = {'red': [], 'green': [], 'blue': []}
#     for pos, color in zip(position, colors):
#         cdict['red'].append((pos, color[2], color[2]))
#         cdict['green'].append((pos, color[1], color[1]))
#         cdict['blue'].append((pos, color[0], color[0]))
#
#     cmap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 256)
#     return cmap
#
# def discrete_cmap(N, base_cmap=None):
#     """Create an N-bin discrete colormap from the specified input map"""
#
#     # Note that if base_cmap is a string or None, you can simply do
#     #    return plt.cm.get_cmap(base_cmap, N)
#     # The following works for string, None, or a colormap instance:
#
#     base = plt.cm.get_cmap(base_cmap)
#     color_list = base(np.linspace(0, 1, N))
#     cmap_name = base.name + str(N)
#
#     return base.from_list(cmap_name, color_list, N)
#
#
# colors = [(255, 0, 0),      # Powerline
# (255, 255, 255),  # Low Vegetation
# (255, 255, 0),    # Impervious Surface
# (255, 0, 255),    # Vehicles
# (0, 255, 255),    # Urban Furniture
# (0, 255, 0),      # Roof
# (0, 0, 255),      # Facade
# (239, 120, 76),   # Bush/Hedge
# (247, 238, 179),  # Tree
# (0, 18, 114),     # Dirt/Gravel
# (63, 34, 15),    # Vertical Surface
# (0, 0, 0)]        # Void
#
# my_cmap = make_cmap(colors, bit=True)
#
# # fig = plt.figure()
# # plt.pcolor(np.random.rand(0,11), cmap=my_cmap)
# # plt.colorbar(ticks=range(12))
# # plt.clim(0, 11)
# # plt.show()
#
# N = 12
#
# x = np.random.randn(40)
# y = np.random.randn(40)
# c = np.random.randint(N, size=40)
#
# # Edit: don't use the default ('jet') because it makes @mwaskom mad...
# plt.scatter(x, y, c=c, s=50, cmap=discrete_cmap(N, my_cmap))
# plt.colorbar(ticks=range(N))
# plt.clim(-0.5, N - 0.5)
# plt.show()
#
#

