import os
from tqdm import tqdm

def get_exterior_orientation(imgs_path, extOri_file, extOri_save_file):

    """
    Exterior Orientation of images

    $CAMERA
      $TYPE : iXU-RS1000_(50mm)
      $DATE : 19:07:49 17/08/2018
      $BRAND : Custom
      $KIND : CCDFrame
      $CCD_INTERIOR_ORIENTATION :
         217.3923957723146400       -0.0000000000000000    5798.5783629179004000
           0.0000000000    -217.3913442725533700    4358.1365279104657000
      $CCD_COLUMNS : 11608
      $CCD_ROWS : 8708
      $PIXEL_REFERENCE : CenterTopLeft
      $FOCAL_LENGTH :    51.6829425484485650
      $PRINCIPAL_POINT_PPA :     0.000000     0.000000
      $DISTORTION_TYPE :  Polynomial
      $RADIAL_COEFFS :
                          0.0000000000000000             -0.0000145693931767                  0.0000000040590362                  -0.0000000000000411
                          -0.0000000000000001             0.0000000000000000                  0.0000000000000000                  0.0000000000000000
      $DECENTRE_COEFFS :
                          0.0000034332895555             0.0000043572785450                  0.0000000000000000                  0.0000000000000000
      $GPS_ANTENNA_OFFSET :     0.000000     0.000000     0.000000
      $CAMERA_MOUNT_ROTATION :     0.000000
    $END
    $ADJUSTED_POINTS
    $END_POINTS

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
    print("Save exterior orientations in file: extOri_test.txt")
    print(">>>>>>>>>>>>>>>>>>Down!<<<<<<<<<<<<<<<<<<<<<<<<<<<< \n")


if __name__ == "__main__":

    imgs_path = "./Images"
    extOri_file = "./All Camera Orientations/txt/extOri.txt"
    extOri_save_file = "./extOri_test.txt"

    get_exterior_orientation(imgs_path, extOri_file, extOri_save_file)
