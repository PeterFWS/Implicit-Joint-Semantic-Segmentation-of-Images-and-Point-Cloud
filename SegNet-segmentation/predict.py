# import argparse
import Models, LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import itertools


color_classes_int = {
    "0": (0, 0, 0),
    "1": (255, 255, 255),  # 
    "2": (255, 255, 0),  # 
    "3": (255, 0, 255),  # 
    "4": (0, 255, 255),  # 
    "5": (0, 255, 0),  # 
    "6": (0, 0, 255),  # 
    "7": (239, 120, 76),  # 
    "8": (247, 238, 179),  #
    "9": (0, 18, 114),  #
    "10": (63, 34, 15)
}

n_classes = 11
images_path = "./data/test/images_prepped_test/"
depth_path = "./data/test/depth_prepped_test/"
f1_path = "./data/test/f1_prepped_test/"
f2_path = "./data/test/f2_prepped_test/"
f3_path = "./data/test/f3_prepped_test/"

input_height = 2048  # 8708
input_width = 2048  # 11608

m = Models.Segnet.segnet(n_classes, input_height=input_height, input_width=input_width)
m.load_weights("./weights/ex1" + "." + "weights")
m.compile(loss='categorical_crossentropy',
          optimizer='adadelta',
          metrics=['accuracy'])

output_height = input_height
output_width = input_width

images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
images.sort()

depth_im = glob.glob(depth_path + "*.npy")
depth_im.sort()

f1_im = glob.glob(f1_path + "*.npy")
f1_im.sort()

f2_im = glob.glob(f2_path + "*.npy")
f2_im.sort()

f3_im = glob.glob(f3_path + "*.npy")
f3_im.sort()

assert len(images) == len(depth_im)
assert len(images) == len(f1_im)
assert len(images) == len(f2_im)
assert len(images) == len(f3_im)
for im, dep in zip(images, depth_im):
    assert (im.split('/')[-1].split(".")[0] == dep.split('/')[-1].split(".")[0])
for im, f1 in zip(images, f1_im):
    assert (im.split('/')[-1].split(".")[0] == f1.split('/')[-1].split(".")[0])
for im, f2 in zip(images, f2_im):
    assert (im.split('/')[-1].split(".")[0] == f2.split('/')[-1].split(".")[0])
for im, f3 in zip(images, f3_im):
    assert (im.split('/')[-1].split(".")[0] == f3.split('/')[-1].split(".")[0])

zipped = itertools.cycle(zip(images, depth_im, f1_im, f2_im, f3_im))

for _ in range(0, len(images)):
    imgName, dep_imName, f1, f2, f3= zipped.next()
    outName = imgName.replace(images_path, "./data/predictions/")
    X = LoadBatches.getImageArr(imgName, dep_imName, f1, f2, f3, input_width, input_height)
    pr = m.predict(np.array([X]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
    seg_img = np.zeros((output_height, output_width, 3))
    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (color_classes_int[str(c)][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (color_classes_int[str(c)][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (color_classes_int[str(c)][2])).astype('uint8')
    seg_img = cv2.resize(seg_img, (input_width, input_height))
    cv2.imwrite(outName, seg_img)
