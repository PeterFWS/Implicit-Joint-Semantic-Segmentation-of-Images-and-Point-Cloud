import os
from tqdm import tqdm

def get_extOri(imgs_path, extOri_file, extOri_save_file):

    """
    Exterior Orientation of images
    Camera: Phase One
    """

    img_list = os.listdir(imgs_path)

    if os.path.exists(extOri_save_file):
        os.remove(extOri_save_file)

    print(">>>>>>>>>>>>>>>>>>find corresponding exterior orientations of each single image<<<<<<<<<<<<<<<<<<<<<< \n")
    with open(extOri_file, "r") as fp:
        for line in fp:
            for img_name in tqdm(img_list):
                if line.split("\t")[0] == img_name.split(".")[0]:
                    with open(extOri_save_file, "a") as fp1:
                        fp1.write(line)
    print(">>>>>>>>>>>>>>>>>>Down!<<<<<<<<<<<<<<<<<<<<<<<<<<<< \n")


if __name__ == "__main__":

    imgs_path = "./Images"
    extOri_file = "./All Camera Orientations/txt/extOri.txt"
    extOri_save_file = "./extOri_test.txt"

    get_extOri(imgs_path, extOri_file, extOri_save_file)
