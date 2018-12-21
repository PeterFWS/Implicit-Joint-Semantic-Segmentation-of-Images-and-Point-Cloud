import numpy as np
import cv2
import glob
import itertools
import tifffile


def getImageArr(im_path, mask_path, width, height):

	img = cv2.imread(im_path, 1).astype(np.float32)  # RGB img
	mask = cv2.imread(mask_path, 0)  # grey img
	img = img / 255.0
	img[mask[:,:]==0] = 0


	# resize
	# img = cv2.resize(img, (width, height)).astype(np.float32)

	# img_temp = np.zeros((height,width,106),dtype=np.float32)
	# for i in range(img_temp.shape[2]):
	# 	img_temp[:,:,i] = f_ndom
	# a = np.dstack((img, img_temp))
	# print a.shape
	return img

def getImageArr_predict(im_path, width, height):

	img = cv2.imread(im_path, 1).astype(np.float32)  # RGB img
	img = img / 255.0
	return img

def getSegmentationArr(path, nClasses, width, height):
	seg_labels = np.zeros((height, width, nClasses))
	try:
		img = cv2.imread(path, 0)
		mask = (img[:,:] ==255)
		img[mask[:,:]==True] = 11
		# resize
		# img = cv2.resize(img, (width, height))

		for c in range(nClasses):
			seg_labels[:, :, c] = (img == c).astype(int)

	except Exception, e:
		print e

	seg_labels = np.reshape(seg_labels, (width * height, nClasses))
	# print "seg_labels", seg_labels.shape
	return seg_labels



def imageSegmentationGenerator(images_path, segs_path, mask_path,
							   batch_size, n_classes, input_height, input_width,
							   output_height, output_width):

	assert images_path[-1] == '/'
	assert segs_path[-1] == '/'
	assert mask_path[-1] == '/'

	images = glob.glob(images_path + "*.JPG") + glob.glob(images_path + "*.tif")
	images.sort()

	segmentations = glob.glob(segs_path + "*.JPG") + glob.glob(segs_path + "*.tif")
	segmentations.sort()

	masks = glob.glob(mask_path + "*.JPG")+ glob.glob(mask_path + "*.tif")
	masks.sort()

	assert len(images) == len(segmentations)
	assert len(images) == len(masks)

	for im, seg in zip(images, segmentations):
		assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])

	for im, mask in zip(images, masks):
		assert (im.split('/')[-1].split(".")[0] == mask.split('/')[-1].split(".")[0])

	zipped = itertools.cycle(zip(images, segmentations, masks))

	while True:
		X = []
		Y = []
		for _ in range(batch_size):
			im, seg, mask = zipped.next()
			X.append(getImageArr(im, mask, input_width, input_height))
			Y.append(getSegmentationArr(seg, n_classes, output_width, output_height))
		yield np.array(X), np.array(Y)
