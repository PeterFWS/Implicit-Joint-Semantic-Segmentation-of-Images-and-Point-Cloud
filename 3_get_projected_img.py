import numpy as np
from tqdm import tqdm
import cv2
import os


def img_projected(px, py, labels, color_classes, img_path, data_path):

    width = 11608
    height = 8708

    img = cv2.imread(img_path)
    print(img.shape)
    img2 = np.zeros(img.shape, np.uint8)
    img3 = np.zeros(img.shape, np.uint8)

    classes = []
    for i in tqdm(range(0, px.shape[0])):
        if width > px[i] > 0 and height > py[i] > 0:
            c = color_classes[str(labels[i])]
            if labels[i] not in classes:
                classes.append(labels[i])
            cv2.circle(img2, (int(px[i]), int(py[i])), 1, c, -1)
            cv2.circle(img3, (int(px[i]), int(py[i])), 10, c, -1)

    cv2.imwrite(os.path.join(data_path, "2.jpg"), img2)
    cv2.imwrite(os.path.join(data_path, "3.jpg"), img3)

    print("\n classes: ", classes, " img name: ", img_path.split("/")[-1], "\n")


if __name__ == "__main__":

    color_classes = {"1.0": (255, 255, 255),  # white
                     "2.0": (255, 255, 0),  # yellow
                     "3.0": (255, 0, 255),  # fashion red
                     "4.0": (0, 255, 255),  # Cyan
                     "5.0": (0, 255, 0),  # green
                     "6.0": (0, 0, 255),  # blue
                     "7.0": (239, 120, 76),  # some orange
                     "8.0": (247, 238, 179),  # some yellow
                     "9.0": (0, 18, 114),  # some blue
                     "10.0": (63, 34, 15),  # some brown
                     "11.0": (143, 67, 61)  # some red
                     }

    projected_pc_path = "./Images_projected_pc"
    imgs_path = "./Images"
    img_list = os.listdir(imgs_path)

    for img_name in tqdm(img_list):

        data_path = os.path.join(projected_pc_path, img_name.split(".")[0])
        img_path = os.path.join(imgs_path, img_name)

        px = np.loadtxt(os.path.join(data_path, "px.txt"))
        py = np.loadtxt(os.path.join(data_path, "py.txt"))
        labels = np.loadtxt(os.path.join(data_path, "labels.txt"))

        img_projected(px, py, labels, color_classes, img_path, data_path)
