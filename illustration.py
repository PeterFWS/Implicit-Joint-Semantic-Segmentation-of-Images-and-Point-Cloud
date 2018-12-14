import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

file_Labels1 = "./data/features_dense_LiDAR_cloud_5cm/y.txt"
pt_labels1 = np.loadtxt(file_Labels1)

file_Labels2 = "./data/features_dense_LiDAR_cloud_10cm/y.txt"
pt_labels2 = np.loadtxt(file_Labels2)

# 11 classes
num_eachClass = np.zeros((2,11), dtype=int)

for i in tqdm(range(pt_labels1.shape[0])):
    num_eachClass[0, int(pt_labels1[i])] += 1

for i in tqdm(range(pt_labels2.shape[0])):
    num_eachClass[1, int(pt_labels2[i])] += 1



# data = {'PowerLine': num_eachClass[0,0],
#         'Low Vegetation': num_eachClass[0,1],
#         'Impervious Surface': num_eachClass[0,2],
#         'Vehicles': num_eachClass[0,3],
#         'Urban Furniture': num_eachClass[0,4],
#         'Roof': num_eachClass[0,5],
#         'Facade': num_eachClass[0,6],
#         'Bush/Hedge': num_eachClass[0,7],
#         'Tree': num_eachClass[0,8],
#         'Dirt/Gravel': num_eachClass[0,9],
#         'Vertical Surface': num_eachClass[0,10]}
#
# group_data = list(data.values())
# group_names = list(data.keys())
# # group_mean = np.mean(group_data)
#
# plt.rcParams.update({'figure.autolayout': True})
# plt.style.use('fivethirtyeight')
# fig, ax = plt.subplots()
# ax.barh(group_names, group_data)
# labels = ax.get_xticklabels()
# plt.setp(labels, rotation=45, horizontalalignment='right')
# ax.set(xlabel='Total number of points', ylabel='Classes',
#        title='Number of points of each class (10cm LiDAR)')
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
rects1 = ax.bar(ind - width/2, num_eachClass[0,:], width,
                color='SkyBlue', label='5cm LiDAR')
rects2 = ax.bar(ind + width/2, num_eachClass[1,:], width,
                color='IndianRed', label='10cm LiDAR')


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