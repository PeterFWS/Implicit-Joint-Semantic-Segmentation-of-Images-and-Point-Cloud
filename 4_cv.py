import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./1.jpg")
kernel = np.ones((5,5), np.uint8)

# erosion = cv2.erode(img, kernel, iterations=1)
# dilation = cv2.dilate(erosion, kernel, iterations=1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

cv2.imwrite("result.jpg", closing)