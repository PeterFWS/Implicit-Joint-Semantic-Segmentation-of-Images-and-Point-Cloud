# from PIL import Image
# from PIL import ImageDraw
import numpy as np
import time
from tqdm import tqdm
import cv2


def img_projected():
    print("start to load \n")
    start_time = time.time()
    px = np.loadtxt("./px.txt")
    py = np.loadtxt("./py.txt")
    labels = np.loadtxt("./labels.txt")
    duration = time.time() - start_time
    print("down! \n")
    print("duration: ", duration, " s")

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

    width = 11608
    height = 8708

    # img = Image.open('./Images/CF013540.jpg')
    # img2 = Image.new("RGB", img.size)
    # draw = ImageDraw.Draw(img2)
    img = cv2.imread('./Images/CF013540.jpg')
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
    cv2.imwrite("./2.jpg", img2)
    cv2.imwrite("./3.jpg", img3)
            # draw.point((px[i], py[i]), fill=c)
    # img2.save("./CF013540_labeled.jpg", 'JPEG')
    print("classes: ", classes)


if __name__ == "__main__":
    img_projected()
