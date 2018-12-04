import numpy as np
from PIL import Image
from tqdm import tqdm


def chip_image(img, chip_size=(300, 300)):
    """
    Segment an image into NxWxH chips

    Args:
        img : Array of image to be chipped
        chip_size : A list of (width,height) dimensions for chips

    Outputs:
        An ndarray of shape (N,W,H,3) where N is the number of chips,
            W is the width per chip, and H is the height per chip.

    """
    width, height, _ = img.shape
    wn, hn = chip_size
    images = np.zeros((int(width / wn) * int(height / hn), wn, hn, 3))
    k = 0
    for i in tqdm(range(int(width / wn))):
        for j in range(int(height / hn)):
            chip = img[wn * i:wn * (i + 1), hn * j:hn * (j + 1), :3]
            images[k] = chip

            k = k + 1

    return images.astype(np.uint8)


if __name__ == "__main__":
    arr = np.array(Image.open("./Images/CF013540.jpg"))
    chip_size = (300, 300)
    img = chip_image(arr, chip_size)
    print(img.shape)

    chipresult = "./chipresult/"
    for index in range(img.shape[0]):
        a = img[index]
        r = Image.fromarray(a[:, :, 0]).convert('L')
        g = Image.fromarray(a[:, :, 1]).convert('L')
        b = Image.fromarray(a[:, :, 2]).convert('L')
        image = Image.merge("RGB", (r, g, b))
        image.save(chipresult + str(index) + ".jpg", 'jpg')


