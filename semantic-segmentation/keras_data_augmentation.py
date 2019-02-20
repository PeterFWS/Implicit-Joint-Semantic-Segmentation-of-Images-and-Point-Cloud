from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import glob
import itertools
import tifffile
import os
from sklearn.utils import shuffle
import random


def getImageArr(im_path, mask_path, f_folders, width, height, imgNorm="normalization", rotation_index=None,
                random_crop=None):
    # read mask
    mask = cv2.imread(mask_path, 0)
    # read rgb image
    img = cv2.imread(im_path, 1).astype(np.float32)

    # img[mask[:, :] == 0] = 0  # masking after normalization!

    return img


def getSegmentationArr(path, nClasses, width, height, rotation_index=None, random_crop=None):
    seg_labels = np.zeros((height, width, nClasses))

    img = cv2.imread(path, 0)
    mask = (img[:, :] == 255)
    img[mask[:, :] == True] = 11

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)  # on-hot coding

    seg_labels = np.reshape(seg_labels, (width * height, nClasses)).astype(int)

    return seg_labels


def imageSegmentationGenerator(images_path, segs_path, mask_path,
                               batch_size, n_classes, input_height, input_width,
                               output_height, output_width):
    images=[]
    segmentations=[]
    masks=[]

    for diff_path in range(len(images_path)):
        assert images_path[diff_path][-1] == '/'
        assert segs_path[diff_path][-1] == '/'
        assert mask_path[diff_path][-1] == '/'
        images += glob.glob(images_path[diff_path] + "*.JPG") + glob.glob(images_path[diff_path] + "*.tif")
        images.sort()
        segmentations += glob.glob(segs_path[diff_path] + "*.JPG") + glob.glob(segs_path[diff_path] + "*.tif")
        segmentations.sort()
        masks += glob.glob(mask_path[diff_path] + "*.JPG") + glob.glob(mask_path[diff_path] + "*.tif")
        masks.sort()


    # else:
    f_folders = None

    zipped = itertools.cycle(zip(images, segmentations, masks))

    while True:
        X = []
        Y = []

        for _ in range(batch_size):
            im, seg, mk = zipped.next()

            rotation_index = None
            random_crop = None

            X.append(getImageArr(im, mk, f_folders, input_width, input_height, rotation_index, random_crop))
            Y.append(getSegmentationArr(seg, n_classes, output_width, output_height, rotation_index, random_crop))

        yield np.array(X), np.array(Y)







if __name__ == "__main__":

    image_gen = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.01,
                                   zoom_range=[0.9, 1.25],
                                   horizontal_flip=True,
                                   vertical_flip=False,
                                   fill_mode='reflect',
                                   data_format='channels_last',
                                   brightness_range=[0.5, 1.5])


    def create_aug_gen(in_gen):

        for in_x, in_y in in_gen:
            g_x = image_gen.flow(255 * in_x, in_y,
                                 batch_size=in_x.shape[0])
            x, y = next(g_x)

        yield x / 255.0, y

    train_images_path = ["/data/fangwen/test/rgb_img/"]
    train_segs_path = ["/data/fangwen/test/3_greylabel/"]
    train_mask_path = ["/data/fangwen/test/2_mask/"]

    input_height = 480
    input_width = 480
    n_classes = 12  # 11 classes + 1 un-classified class
    train_batch_size = 1


    train_gen = imageSegmentationGenerator(train_images_path, train_segs_path, train_mask_path,
                                           train_batch_size, n_classes, input_height, input_width,
                                           output_height=input_height, output_width=input_width)

    cur_gen = create_aug_gen(train_gen)
    t_x, t_y = next(cur_gen)

    print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
    print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())
