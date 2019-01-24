import numpy as np
import cv2
import glob
import itertools
import tifffile
import os

from chip import chip_image


def getImageArr(im_path, mask_path, f_folders, width, height):
    # read mask
    mask = cv2.imread(mask_path, 0)  # grey img

    # read rgb image
    img = cv2.imread(im_path, 1).astype(np.float32)
    img = img / 255.0  # normalization
    img[mask[:, :] == 0] = 0

    count = 0
    # read features
    for folder_path in f_folders:
        f_path = os.path.join(folder_path, im_path.split('/')[-1])
        f_img = tifffile.imread(f_path).astype(np.float32)
        where_are_NaNs = np.isnan(f_img)
        f_img[where_are_NaNs] = 0.0
        max = np.amax(f_img)
        min = np.amin(f_img)
        f_img = (f_img - min) / (max - min)  # normalization
        f_img[mask[:, :] == 0] = 0.0

        if count == 0:
            temp = np.dstack((img, f_img))
        if count > 0:
            temp = np.dstack((temp, f_img))
        count += 1

    temp = temp[:480, :736, :]

    return temp


def getSegmentationArr(path, nClasses, width, height):
    seg_labels = np.zeros((height, width, nClasses))

    img = cv2.imread(path, 0)
    mask = (img[:, :] == 255)
    img[mask[:, :] == True] = 11
    img = img[:480, :736]

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    seg_labels = np.reshape(seg_labels, (width * height, nClasses))

    return seg_labels


def imageSegmentationGenerator(images_path, segs_path, mask_path,
                               f_path,
                               batch_size, n_classes, input_height, input_width,
                               output_height, output_width):
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'
    assert mask_path[-1] == '/'

    assert f_path[-1] == '/'

    images = glob.glob(images_path + "*.JPG") + glob.glob(images_path + "*.tif")
    images.sort()

    segmentations = glob.glob(segs_path + "*.JPG") + glob.glob(segs_path + "*.tif")
    segmentations.sort()

    masks = glob.glob(mask_path + "*.JPG") + glob.glob(mask_path + "*.tif")
    masks.sort()

    f_folders = glob.glob(f_path + "f_*" + "/")
    f_folders.sort()

    #
    assert len(images) == len(segmentations)
    assert len(images) == len(masks)

    for im, seg in zip(images, segmentations):
        assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])
    for im, mask in zip(images, masks):
        assert (im.split('/')[-1].split(".")[0] == mask.split('/')[-1].split(".")[0])

    for folder_path in f_folders:
        f_imgs = glob.glob(folder_path + "*.JPG") + glob.glob(folder_path + "*.tif")
        f_imgs.sort()
        assert len(images) == len(f_imgs)
        for im, f_img in zip(images, f_imgs):
            assert (im.split('/')[-1].split(".")[0] == f_img.split('/')[-1].split(".")[0])

    zipped = itertools.cycle(zip(images, segmentations, masks))

    while True:

        im, seg, mask = zipped.next()
        img_chips, flag = chip_image(im)
        seg_chips, flag1 = chip_image(seg)
        mask_chips, flag1 = chip_image(mask)

        X = []
        Y = []
        for _ in range(batch_size):
            # im, seg, mask = zipped.next()

            X.append(getImageArr(im, mask, f_folders, input_width, input_height))
            Y.append(getSegmentationArr(seg, n_classes, output_width, output_height))
        yield np.array(X), np.array(Y)


