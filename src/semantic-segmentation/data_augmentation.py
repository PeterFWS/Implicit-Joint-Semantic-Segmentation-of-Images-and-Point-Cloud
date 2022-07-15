import random
from imgaug import augmenters as iaa
import cv2
import numpy as np
from skimage import exposure


# Change brightness levels
def random_brightness(image):
    # Convert 2 HSV colorspace from BGR colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Generate new random brightness
    rand = random.uniform(0.3, 1.0)
    hsv[:, :, 2] = rand*hsv[:, :, 2]
    # Convert back to BGR colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img



if __name__ == "__main__":


    img = cv2.imread("/data/fangwen/CF012463_0_r1.tif")

    i = 0
    while True:
        # Contrast stretching
        if np.random.random() < 0.5:
            p2, p98 = np.percentile(img, (2, 98))
            img = exposure.rescale_intensity(img, in_range=(p2, p98))

        # Brightness jitter
        elif np.random.random() < 0.5:
            img = random_brightness(img)

        cv2.imwrite("/data/fangwen/test" + str(i) + ".tif", img)
        i += 1


