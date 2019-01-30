import numpy as np
import cv2
import glob
import itertools
import tifffile
import os


def getImageArr(im_path, mask_path, f_folders, width, height, imgNorm="sub_mean"):
	# read mask
	mask = cv2.imread(mask_path, 0)

	# read rgb image
	img = cv2.imread(im_path, 1).astype(np.float32)
	img[mask[:,:]==0] = 0
	if imgNorm == "sub_mean":
		# Mean subtraction: sub mean value of the statistic result from VGG, aiming at zero centered
		# https://gist.github.com/ksimonyan/211839e770f7b538e2d8
		img[:, :, 0] -= 103.939
		img[:, :, 1] -= 116.779
		img[:, :, 2] -= 123.68
	elif imgNorm == "normalization":
		"""
			% Level3
			mean_B = 118.78483640465971
			mean_G = 110.6080243351727
			mean_R = 91.54471911465359
			std_B = 48.938716478634376
			std_G = 44.827248561407906
			std_R = 44.48971564535396
		"""
		mean_B = 118.784
		mean_G = 110.608
		mean_R = 91.544
		std_B = 48.938
		std_G = 44.827
		std_R = 44.489
		img[:, :, 0] = (img[:, :, 0] - mean_B) / std_B
		img[:, :, 1] = (img[:, :, 0] - mean_G) / std_G
		img[:, :, 2] = (img[:, :, 0] - mean_R) / std_R
	elif imgNorm == "divide":
		img = img / 255.0


	if f_folders is not None:
		count = 0
		for folder_path in f_folders:
			f_path = os.path.join(folder_path, im_path.split('/')[-1])
			f_img = tifffile.imread(f_path).astype(np.float32)
			where_are_NaNs = np.isnan(f_img)
			f_img[where_are_NaNs] = 0.0
			f_img[mask[:,:]==0] = 0.0  # "nan" actually was set where mask==0
			if imgNorm == "normalization":
				id = int(folder_path.split("/")[-1].split("_")[-1])
				mean = np.loadtxt("./data/mean.out")[id]
				std = np.loadtxt("./data/sdv.out")[id]
				f_img = (f_img - mean) / std
			elif imgNorm == "divide":
				id = int(folder_path.split("/")[-1].split("_")[-1])
				max = np.loadtxt("./data/max.out")[id]
				min = np.loadtxt("./data/min.out")[id]
				f_img = (f_img - min) / (max - min)

			if count == 0:
				temp = np.dstack((img, f_img))
			if count > 0:
				temp = np.dstack((temp, f_img))
			count += 1

		temp = temp[:480, :736, :]
		return temp

	else:
		img = img[:480, :736, :]
		return img


def getSegmentationArr(path, nClasses, width, height):

	seg_labels = np.zeros((height, width, nClasses))

	img = cv2.imread(path, 0)
	mask = (img[:,:] ==255)
	img[mask[:,:]==True] = 11
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
	images = glob.glob(images_path + "*.JPG") + glob.glob(images_path + "*.tif")
	images.sort()
	segmentations = glob.glob(segs_path + "*.JPG") + glob.glob(segs_path + "*.tif")
	segmentations.sort()
	masks = glob.glob(mask_path + "*.JPG") + glob.glob(mask_path + "*.tif")
	masks.sort()

	#
	assert len(images) == len(segmentations)
	assert len(images) == len(masks)
	for im, seg in zip(images, segmentations):
		assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])
	for im, mask in zip(images, masks):
		assert (im.split('/')[-1].split(".")[0] == mask.split('/')[-1].split(".")[0])


	if f_path is not None:
		assert f_path[-1] == '/'
		f_folders = glob.glob(f_path + "f_*" + "/")
		f_folders.sort()
		for folder_path in f_folders:
			f_imgs = glob.glob(folder_path + "*.JPG") + glob.glob(folder_path + "*.tif")
			f_imgs.sort()
			assert len(images) == len(f_imgs)
			for im, f_img in zip(images, f_imgs):
				assert (im.split('/')[-1].split(".")[0] == f_img.split('/')[-1].split(".")[0])
	else:
		f_folders = None


	zipped = itertools.cycle(zip(images, segmentations, masks))

	while True:
		X = []
		Y = []
		for _ in range(batch_size):
			im, seg, mask = zipped.next()
			X.append(getImageArr(im, mask, f_folders, input_width, input_height))
			Y.append(getSegmentationArr(seg, n_classes, output_width, output_height))
		yield np.array(X), np.array(Y)
