import numpy as np
import cv2
import glob
import itertools
import tifffile
import os
from sklearn.utils import shuffle
import random

def rotate_image_random(img, rotation_index):

	deg_dict = {
		1: 0,
		2: 90,
		3: 180,
		4: 270
	}

	rows = img.shape[0]
	cols = img.shape[1]

	if deg_dict[rotation_index] != 0:
		M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), deg_dict[rotation_index], 1)
		dst = cv2.warpAffine(img, M, (cols, rows))
		return dst
	else:
		return img

def getImageArr(im_path, mask_path, f_folders, width, height, imgNorm="sub_mean", rotation_index=None):
	# read mask
	mask = cv2.imread(mask_path, 0)
	# read rgb image
	img = cv2.imread(im_path, 1).astype(np.float32)

	##################
	attacth_HSV = False
	##################
	if attacth_HSV is not False:
		# transfer color space
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

	if imgNorm == "sub_mean":
		# Mean subtraction: sub mean value of the statistic result from VGG, aiming at zero centered
		# https://gist.github.com/ksimonyan/211839e770f7b538e2d8
		img[:, :, 0] -= 103.939
		img[:, :, 1] -= 116.779
		img[:, :, 2] -= 123.68
	elif imgNorm == "normalization":
		"""
			% Level3 based on training set
			mean_B = 93.55575820215054
			mean_G = 109.91111718537634
			mean_R = 116.77154769032258
			std_B = 45.41826739260701
			std_G = 45.72733518845752
			std_R = 50.29461861852277
			
			mean_H = 41.07548142795699
			mean_S = 64.7442760051613
			mean_V = 121.00639388043011
			std_H = 40.400934438326466
			std_S = 38.57465872094216
			std_V = 49.27051219866851
			
			% mixed data
			
		"""
		# normalize BGR imagery
		mean_B = 93.555
		mean_G = 109.911
		mean_R = 116.771
		std_B = 45.418
		std_G = 45.727
		std_R = 50.294
		img[:, :, 0] = (img[:, :, 0] - mean_B) / std_B
		img[:, :, 1] = (img[:, :, 0] - mean_G) / std_G
		img[:, :, 2] = (img[:, :, 0] - mean_R) / std_R

		if attacth_HSV is not False:
			# normalize HSV imagery
			mean_H = 41.075
			mean_S = 64.744
			mean_V = 121.006
			std_H = 40.400
			std_S = 38.574
			std_V = 49.270
			hsv[:, :, 0] = (hsv[:, :, 0] - mean_H) / std_H
			hsv[:, :, 1] = (hsv[:, :, 0] - mean_S) / std_S
			hsv[:, :, 2] = (hsv[:, :, 0] - mean_V) / std_V
	elif imgNorm == "divide":
		img = img / 255.0

	if attacth_HSV is not False:
		img = np.dstack((img, hsv))

	img[mask[:, :] == 0] = 0  # masking after normalization!


	# train_mode = "multi_modality"
	if f_folders is not None:
		count = 0
		for folder_path in f_folders:
			f_path = os.path.join(folder_path, im_path.split('/')[-1])
			f_img = tifffile.imread(f_path).astype(np.float32)
			where_are_NaNs = np.isnan(f_img)
			f_img[where_are_NaNs] = 0.0
			# according to Michael, no further normalization is need for feature map
			f_img[mask[:, :] == 0] = 0.0  # "nan" actually was set where mask==0 # masking after normalization!

			img = np.dstack((img, f_img))

		# folder_path = f_folders[65]  # nDSM
		# f_path = os.path.join(folder_path, im_path.split('/')[-1])
		# f_img = tifffile.imread(f_path).astype(np.float32)
		# where_are_NaNs = np.isnan(f_img)
		# f_img[where_are_NaNs] = 0.0
		# f_img[mask[:, :] == 0] = 0.0  # "nan" actually was set where mask==0 # masking after normalization!
		# img = np.dstack((img, f_img))

	if rotation_index is not None:
		img = rotate_image_random(img, rotation_index)

	return img


def getSegmentationArr(path, nClasses, width, height, rotation_index=None):

	seg_labels = np.zeros((height, width, nClasses))

	img = cv2.imread(path, 0)
	mask = (img[:,:] ==255)
	img[mask[:,:]==True] = 11

	if rotation_index is not None:
		img = rotate_image_random(img, rotation_index)

	for c in range(nClasses):
		seg_labels[:, :, c] = (img == c).astype(int)  # on-hot coding

	seg_labels = np.reshape(seg_labels, (width * height, nClasses))

	return seg_labels


def imageSegmentationGenerator(images_path, segs_path, mask_path,
							   f_path,
							   batch_size, n_classes, input_height, input_width,
							   output_height, output_width):
	"""
		images_path = "/data/fangwen/results/level3_nadir/chip_train_set/rgb_img/"
		segs_path = "/data/fangwen/results/level3_nadir/chip_train_set/3_greylabel/"
		mask_path ="/data/fangwen/results/level3_nadir/chip_train_set/2_mask/"
		f_path = None
		input_width = 224
		input_height = 224
		output_width =224
		output_height=224
		n_classes=12
		f_folders = None
		batch_size = 8
	"""

	assert images_path[-1] == '/'
	assert segs_path[-1] == '/'
	assert mask_path[-1] == '/'

	images = glob.glob(images_path + "*.JPG") + glob.glob(images_path + "*.tif")
	images.sort()
	segmentations = glob.glob(segs_path + "*.JPG") + glob.glob(segs_path + "*.tif")
	segmentations.sort()
	masks = glob.glob(mask_path + "*.JPG") + glob.glob(mask_path + "*.tif")
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
		assert f_path[-1] == '/'
		f_folders = glob.glob(f_path + "f_*" + "/")
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
		Y = []
		for _ in range(batch_size):
			im, seg, mask = zipped.next()
			# add random rotation
			rotation_index = random.randint(1, 4)  # 0, 90, 180, 270 [degree]
			X.append(getImageArr(im, mask, f_folders, input_width, input_height, rotation_index))
			Y.append(getSegmentationArr(seg, n_classes, output_width, output_height, rotation_index))

		yield np.array(X), np.array(Y)
