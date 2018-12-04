import numpy as np
import cv2
import glob
import itertools


def getImageArr(im_path, dep_path, f1_path, f2_path, f3_path, width, height):
	img = cv2.imread(im_path, 1)
	img = cv2.resize(img, (width, height)).astype(np.float32)
	img = img / 255.0

	dep = np.load(dep_path)
	mean = np.mean(dep, dtype=np.float32)
	# s = np.std(dep, dtype=np.float32)
	s = np.amax(dep) - np.amin(dep)
	dep = cv2.resize(dep, (width, height)).astype(np.float32)
	dep = (dep - mean) / s

	f1 = np.load(f1_path)
	mean = np.mean(f1, dtype=np.float32)
	s = np.amax(f1) - np.amin(f1)
	f1 = cv2.resize(f1, (width, height)).astype(np.float32)
	f1 = (f1 - mean) / s

	f2 = np.load(f2_path)
	mean = np.mean(f2, dtype=np.float32)
	s = np.amax(f2) - np.amin(f2)
	f2 = cv2.resize(f2, (width, height)).astype(np.float32)
	f2 = (f2 - mean) / s

	f3 = np.load(f3_path)
	mean = np.mean(f3, dtype=np.float32)
	s = np.amax(f3) - np.amin(f3)
	f3 = cv2.resize(f3, (width, height)).astype(np.float32)
	f3 = (f3 - mean) / s

	return np.dstack((img, dep, f1, f2, f3))  # (512, 512, 7)


def getSegmentationArr(path, nClasses, width, height):
	seg_labels = np.zeros((height, width, nClasses))
	try:
		img = cv2.imread(path, 0)
		img = cv2.resize(img, (width, height))

		for c in range(nClasses):
			seg_labels[:, :, c] = (img == c).astype(int)

	except Exception, e:
		print e

	seg_labels = np.reshape(seg_labels, (width * height, nClasses))
	# print "seg_labels", seg_labels.shape
	return seg_labels


def imageSegmentationGenerator(images_path, segs_path, depth_path, feature1_path, feature2_path, feature3_path,
							   batch_size, n_classes, input_height, input_width,
							   output_height, output_width):
	assert images_path[-1] == '/'
	assert segs_path[-1] == '/'
	assert depth_path[-1] == '/'
	assert feature1_path[-1] == '/'
	assert feature2_path[-1] == '/'
	assert feature3_path[-1] == '/'

	images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
	images.sort()

	segmentations = glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg")
	segmentations.sort()

	depth_im = glob.glob(depth_path + "*.npy")
	depth_im.sort()

	f1_im = glob.glob(feature1_path + "*.npy")
	f1_im.sort()

	f2_im = glob.glob(feature2_path + "*.npy")
	f2_im.sort()

	f3_im = glob.glob(feature3_path + "*.npy")
	f3_im.sort()

	assert len(images) == len(segmentations)
	assert len(images) == len(depth_im)
	assert len(images) == len(f1_im)
	assert len(images) == len(f2_im)
	assert len(images) == len(f3_im)

	for im, seg in zip(images, segmentations):
		assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])

	for im, dep in zip(images, depth_im):
		assert (im.split('/')[-1].split(".")[0] == dep.split('/')[-1].split(".")[0])

	for im, f1 in zip(images, f1_im):
		assert (im.split('/')[-1].split(".")[0] == f1.split('/')[-1].split(".")[0])

	for im, f2 in zip(images, f2_im):
		assert (im.split('/')[-1].split(".")[0] == f2.split('/')[-1].split(".")[0])

	for im, f3 in zip(images, f3_im):
		assert (im.split('/')[-1].split(".")[0] == f3.split('/')[-1].split(".")[0])

	zipped = itertools.cycle(zip(images, segmentations, depth_im, f1_im, f2_im, f3_im))

	while True:
		X = []
		Y = []
		for _ in range(batch_size):
			im, seg, d, f1, f2, f3 = zipped.next()
			X.append(getImageArr(im, d, f1, f2, f3, input_width, input_height))
			Y.append(getSegmentationArr(seg, n_classes, output_width, output_height))

		yield np.array(X), np.array(Y)
