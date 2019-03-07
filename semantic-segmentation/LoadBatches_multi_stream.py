import numpy as np
import cv2
import glob
import itertools
import tifffile
import os
from sklearn.utils import shuffle
import random
from skimage import exposure
from imgaug import augmenters as iaa


def random_brightness(image):
    # Convert 2 HSV colorspace from BGR colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Generate new random brightness
    rand = random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand * hsv[:, :, 2]
    # Convert back to BGR colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


def rotate_image_random(img, rotation_index):
    deg_dict = {
        1: 0,
        2: 90,
        3: 180,
        4: 270
    }

    if rotation_index != 1:
        dst = np.rot90(img, rotation_index - 1)
        return dst
    else:
        return img


def get_random_pos(img_shape, window_shape=(512, 512)):
    """ Extract of 2D random patch of shape window_shape in the image """
    h, w = window_shape
    H, W = img_shape
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def getImageArr(im_path, mask_path, f_folders, width, height, imgNorm="normalization", rotation_index=None, random_crop=None,
                data_aug=False):
    # read mask
    mask = cv2.imread(mask_path, 0)
    # read rgb image
    img = cv2.imread(im_path, 1).astype(np.float32)

    if data_aug is not False:
        # Contrast stretching
        if np.random.random() < 0.5:
            p2, p98 = np.percentile(img, (2, 98))
            img = exposure.rescale_intensity(img, in_range=(p2, p98))

        # Brightness jitter
        else:
            img = random_brightness(img)

    attacth_HSV = False
    if attacth_HSV is not False:
        # transfer color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    if imgNorm == "sub_mean":
        # Mean subtraction: sub mean value of the statistic result from VGG, aiming at zero centered
        # https://gist.github.com/ksimonyan/211839e770f7b538e2d8
        img[:, :, 0] -= 86.036
        img[:, :, 1] -= 104.410
        img[:, :, 2] -= 106.238

    elif imgNorm == "normalization":
        # normalize BGR imagery
        mean_B = 86.036
        mean_G = 104.410
        mean_R = 106.238
        std_B = 35.627
        std_G = 36.160
        std_R = 42.036
        img[:, :, 0] = (img[:, :, 0] - mean_B) / std_B
        img[:, :, 1] = (img[:, :, 1] - mean_G) / std_G
        img[:, :, 2] = (img[:, :, 2] - mean_R) / std_R

        if attacth_HSV is not False:
            # normalize HSV imagery
            mean_H = 44.849
            mean_S = 65.800
            mean_V = 111.447
            std_H = 34.782
            std_S = 36.319
            std_V = 39.702
            hsv[:, :, 0] = (hsv[:, :, 0] - mean_H) / std_H
            hsv[:, :, 1] = (hsv[:, :, 1] - mean_S) / std_S
            hsv[:, :, 2] = (hsv[:, :, 2] - mean_V) / std_V

    elif imgNorm == "divide":
        img = img / 255.0

    if attacth_HSV is not False:
        img = np.dstack((img, hsv))

    img[mask[:, :] == 0] = 0  # masking after normalization!


    if f_folders is None:

        if rotation_index is not None:
            img = rotate_image_random(img, rotation_index)

        if random_crop is not None:
            x1, x2, y1, y2 = random_crop
            img = img[y1:y2, x1:x2, :]

        return img

    elif f_folders is not None:
        # for folder_path in f_folders:
        # # tell nadir or oblique image
        #     if folder_path.split("/")[-4] == im_path.split("/")[-4]:
        #     f_img_path = os.path.join(folder_path, im_path.split('/')[-1])
        #     f_img = tifffile.imread(f_img_path).astype(np.float32)
        #     where_are_NaNs = np.isnan(f_img)
        #     f_img[where_are_NaNs] = 0.0
        #     # according to Michael, no further normalization is need for feature map
        #     f_img[mask[:, :] == 0] = 0.0  # "nan" actually was set where mask==0 # masking after normalization!
        #
        # img = np.dstack((img, f_img))

        ## folder_path = "/data/fangwen/mix_train/f_68/"  # nDSM
        count = 0
        for folder_path in f_folders:
            # print(folder_path)
            if folder_path.split('/')[-2] == "f_68" or folder_path.split('/')[-2] == "f_5" or folder_path.split('/')[-2] == "f_15":
                f_img_path = os.path.join(folder_path, im_path.split('/')[-1])
                f_img = tifffile.imread(f_img_path).astype(np.float32)
                where_are_NaNs = np.isnan(f_img)
                f_img[where_are_NaNs] = 0.0
                f_img[mask[:, :] == 0] = 0.0  # "nan" actually was set where mask==0 # masking after normalization!

                if count == 0:
                    img = f_img
                elif count > 0:
                    img = np.dstack((img, f_img))

                count += 1

        return img


def getSegmentationArr(path, nClasses, width, height, rotation_index=None, random_crop=None):
    seg_labels = np.zeros((height, width, nClasses))

    img = cv2.imread(path, 0)
    mask = (img[:, :] == 255)
    img[mask[:, :] == True] = 11

    if rotation_index is not None:
        img = rotate_image_random(img, rotation_index)

    if random_crop is not None:
        x1, x2, y1, y2 = random_crop
        img = img[y1:y2, x1:x2]

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)  # on-hot coding

    seg_labels = np.reshape(seg_labels, (width * height, nClasses)).astype(int)

    return seg_labels


def imageSegmentationGenerator(images_path, segs_path, mask_path,
                               f_path,
                               batch_size, n_classes, input_height, input_width,
                               output_height, output_width, data_aug=False):
    images = []
    segmentations = []
    masks = []

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

    assert len(images) == len(segmentations)
    assert len(images) == len(masks)

    # shuffle whole dataset ordering, without disrupting the mapping
    images, segmentations, masks = shuffle(images, segmentations, masks, random_state=0)

    for im, seg in zip(images, segmentations):
        assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])
    for im, mask in zip(images, masks):
        assert (im.split('/')[-1].split(".")[0] == mask.split('/')[-1].split(".")[0])

    # train_mode = "multi_modality"
    if f_path is not None:
        f_folders = []
        for diff_path in range(len(f_path)):
            assert f_path[diff_path][-1] == '/'
            f_folders += glob.glob(f_path[diff_path] + "f_*" + "/")
            f_folders.sort()
    # for folder_path in f_folders:
    # 	f_imgs = glob.glob(folder_path + "*.JPG") + glob.glob(folder_path + "*.tif")
    # 	f_imgs.sort()
    # 	assert len(images) == len(f_imgs)
    # 	for im, f_img in zip(images, f_imgs):
    # 		assert (im.split('/')[-1].split(".")[0] == f_img.split('/')[-1].split(".")[0])

    else:
        f_folders = None

    zipped = itertools.cycle(zip(images, segmentations, masks))

    while True:
        X = []
        X2 = []
        Y = []

        for _ in range(batch_size):
            im, seg, mk = zipped.next()

            rotation_index = None
            random_crop = None

            X.append(getImageArr(im, mk, None, input_width, input_height, rotation_index, random_crop,
                                 data_aug=True))

            X2.append(getImageArr(im, mk, f_folders, input_width, input_height, rotation_index, random_crop,
                                 data_aug=False))

            Y.append(getSegmentationArr(seg, n_classes, output_width, output_height, rotation_index, random_crop))

        yield [np.array(X), np.array(X2)], np.array(Y)
