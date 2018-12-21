# import argparse
import Models, LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import itertools


color_classes_int = { #BGR
    "0": (255, 0, 0),
    "1": (255, 255, 255),  # 
    "2": (255, 255, 0),  # 
    "3": (255, 0, 255),  # 
    "4": (0, 255, 255),  # 
    "5": (0, 255, 0),  # 
    "6": (0, 0, 255),  # 
    "7": (239, 120, 76),  # 
    "8": (247, 238, 179),  #
    "9": (0, 18, 114),  #
    "10": (63, 34, 15),
    "11": (0, 0, 0)
}

n_classes = 12
images_path = "./data/test/rgb_img/"
masks_path = "./data/test/2_mask/"

input_height = 500
input_width = 750

m = Models.Segnet.segnet_level3(n_classes, input_height=input_height, input_width=input_width)
m.load_weights("/home/fangwen/ShuFangwen/source/image-segmentation-keras/weights/best_model.h5")
m.compile(loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])

output_height = input_height
output_width = input_width

images = glob.glob(images_path + "*.JPG") + glob.glob(images_path + "*.tif")
images.sort()

masks = glob.glob(masks_path + "*.JPG") + glob.glob(masks_path + "*.tif")
masks.sort()


assert len(images) == len(masks)
for im, mask in zip(images, masks):
    assert (im.split('/')[-1].split(".")[0] == mask.split('/')[-1].split(".")[0])


zipped = itertools.cycle(zip(images,masks))

for _ in range(0, len(images)):
    imgName, mask_imName = zipped.next()
    outName = imgName.replace(images_path, "./data/predictions/")
    # X = LoadBatches.getImageArr(imgName, dep_imName, f1, f2, f3, input_width, input_height)
    X = LoadBatches.getImageArr(imgName, mask_imName, input_width, input_height)
    pr = m.predict(np.array([X]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
    seg_img = np.zeros((output_height, output_width, 3))
    for c in range(n_classes-1):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (color_classes_int[str(c)][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (color_classes_int[str(c)][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (color_classes_int[str(c)][2])).astype('uint8')
    seg_img = cv2.resize(seg_img, (input_width, input_height))
    cv2.imwrite(outName, seg_img)
