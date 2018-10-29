import numpy as np
from tqdm import tqdm
import cv2
import os
import my_parameters


def img_projected(px, py, labels, color_classes, img_path, save_path):
    # img_path = "./Images/CF013540.jpg"
    img = cv2.imread(img_path)
    print(img.shape)
    img2 = np.zeros(img.shape, np.uint8)

    classes = []
    for i in tqdm(range(0, px.shape[1])):
        if my_parameters.width > px[0, i] > 0 and my_parameters.height > py[0, i] > 0:
            c = color_classes[str(labels[0, i])]
            if labels[0, i] not in classes:
                classes.append(labels[0, i])
            cv2.circle(img2, (int(px[0, i]), int(py[0, i])), 5, c, -1)

    cv2.imwrite(os.path.join(save_path, img_path.split("/")[-1]), img2)

    print("\n classes: ", classes, " img name: ", img_path.split("/")[-1], "\n")


        
