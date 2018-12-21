import numpy as np
from PIL import Image
from tqdm import tqdm
import os


def make_if_not_exists(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)


def chip_image(img, chip_size=(300, 300), nchannel=3):
    """
    Segment an image into NxWxH chips

    Args:
        img : Array of image to be chipped
        chip_size : A list of (width,height) dimensions for chips

    Outputs:
        An ndarray of shape (N,W,H,3) where N is the number of chips,
            W is the width per chip, and H is the height per chip.

    """
    if nchannel == 3:
        width, height, _ = img.shape

        wn, hn = chip_size
        images = np.zeros((int(width / wn) * int(height / hn), wn, hn, nchannel))
        k = 0
        for i in tqdm(range(int(width / wn))):
            for j in range(int(height / hn)):
                chip = img[wn * i:wn * (i + 1), hn * j:hn * (j + 1), :nchannel]
                images[k] = chip

                k = k + 1

    elif nchannel == 1:
        width, height = img.shape

        wn, hn = chip_size
        images = np.zeros((int(width / wn) * int(height / hn), wn, hn))
        k = 0
        for i in tqdm(range(int(width / wn))):
            for j in range(int(height / hn)):
                chip = img[wn * i:wn * (i + 1), hn * j:hn * (j + 1)]
                images[k] = chip

                k = k + 1



    return images.astype(np.uint8)


if __name__ == "__main__":

    # chip RGB image (*, *, 3)
    # Imgdata_path = "./data/train/pre_images"
    # for img_name in os.listdir(Imgdata_path):
    #     arr = np.array(Image.open(os.path.join(Imgdata_path, img_name)))
    #     chip_size = (512, 512)
    #     img = chip_image(arr, chip_size, 3)
    #     print(img.shape)
    #
    #     chipresult = "./data/train/croppedImg/"
    #     make_if_not_exists(chipresult)
    #     for index in range(img.shape[0]):
    #         temp = img[index]
    #         r = Image.fromarray(temp[:, :, 0]).convert('L')
    #         g = Image.fromarray(temp[:, :, 1]).convert('L')
    #         b = Image.fromarray(temp[:, :, 2]).convert('L')
    #         image = Image.merge("RGB", (r, g, b))
    #         image.save(chipresult +img_name.replace(".jpg", "_") + str(index) + ".jpg", 'JPEG')

    # chip label image (*, *, 1)
    Labeldata_path = "./data/train/pre_labels"
    for img_name in os.listdir(Labeldata_path):
        arr = np.array(Image.open(os.path.join(Labeldata_path, img_name)))
        chip_size = (512, 512)
        img = chip_image(arr, chip_size, 1)
        print(img.shape)

        chipresult = "./data/train/croppedLabel/"
        make_if_not_exists(chipresult)
        for index in range(img.shape[0]):
            temp = Image.fromarray(img[index]).convert('L')
            temp.save(chipresult +img_name.replace(".jpg", "_") + str(index) + ".jpg", 'JPEG')




