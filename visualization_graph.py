import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter
from open3d import *

pt_data_train = np.loadtxt("/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/data/Data_11_1_19_5cm/train_xyz_y.txt")
pt_data_test = np.loadtxt("/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/data/Data_11_1_19_5cm/val_xyz_y.txt")
pt_data_val = np.loadtxt("/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/data/Data_11_1_19_5cm/test_xyz_y.txt")


data = np.concatenate((pt_data_train[:, :3], pt_data_test[:, :3], pt_data_val[:, :3]))
label = np.concatenate((pt_data_train[:, 3], pt_data_test[:, 3], pt_data_val[:, 3]))


l1 = np.loadtxt("/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/data/features_dense_LiDAR_cloud_10cm/data_splits/y_10cm_train.txt")
l2 = np.loadtxt("/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/data/features_dense_LiDAR_cloud_10cm/data_splits/y_10cm_val.txt")
l3 = np.loadtxt("/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/data/features_dense_LiDAR_cloud_10cm/data_splits/y_10cm_test.txt")
label2 = np.concatenate((l1, l2, l3))


# 11 classes
num_eachClass = np.zeros(11, dtype=int)
num_eachClass2 = np.zeros(11, dtype=int)

for i in tqdm(range(label.shape[0])):
    num_eachClass[int(label[i])] += 1

for i in tqdm(range(label2.shape[0])):
    num_eachClass2[int(label[i])] += 1

###########
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

N = 11
ind = np.arange(N)
width = 0.35
fig, ax = plt.subplots()
labels = ax.get_xticklabels()
plt.setp(labels, rotation=45, horizontalalignment='right')
rects1 = ax.bar(ind - width/2, num_eachClass[:], width,
                color='SkyBlue', label='5cm density LiDAR point cloud')
rects2 = ax.bar(ind + width/2, num_eachClass2[:], width,
                color='IndianRed', label='10cm density LiDAR point cloud')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of points')
ax.set_title('Total number of points of each class ')
ax.set_xticks(ind)
ax.set_xticklabels(('PowerLine', 'Low Vegetation', 'Impervious Surface', 'Vehicles', 'Urban Furniture',
                    'Roof', 'Facade', 'Bush/Hedge', 'Tree', 'Dirt/Gravel', 'Vertical Surface'))
ax.legend()


autolabel(rects1, "left")
autolabel(rects2, "right")

plt.show()