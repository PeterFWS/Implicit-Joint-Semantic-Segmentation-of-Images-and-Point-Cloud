"""
return Exterior Orientations of corresponding image
"""

def get_exterior_orientation(img_name, extOri_file):

    print("Looking for exterior orientations of the image: {0} \n".format(img_name))
    with open(extOri_file, "r") as fp:
        for line in fp:
            if line.split("\t")[0] == img_name.split(".")[0]:
                return line

