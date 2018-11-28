# import argparse
import Models , LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import random


color_classes_int = {
    "0": (0, 0, 0),
    "1": (255, 255, 255),  # 
    "2": (255, 255, 0),  # 
    "3": (255, 0, 255),  # 
    "4": (0, 255, 255),  # 
    "5": (0, 255, 0),  # 
    "6": (0, 0, 255),  # 
    "7": (239, 120, 76),  # 
    "8": (247, 238, 179),  #
    "9": (0, 18, 114),  #
    "10": (63, 34, 15)
}

n_classes = 11
images_path = "data/images_prepped_test/"

input_height = 2048#8708
input_width = 2048#11608
epoch_number = 5

m = Models.Segnet.segnet( n_classes , input_height=input_height, input_width=input_width   )
m.load_weights(  "./weights/ex1" + "." + str(  epoch_number-1 )  )
m.compile(loss='categorical_crossentropy',
      optimizer= 'adadelta' ,
      metrics=['accuracy'])

output_height = input_height
output_width = input_width

images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
images.sort()

#colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]

for imgName in images:
    outName = imgName.replace( images_path ,  "data/predictions/" )
    X = LoadBatches.getImageArr(imgName , input_width  , input_height)
    pr = m.predict( np.array([X]) )[0]
    pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )
    seg_img = np.zeros( ( output_height , output_width , 3  ) )
    for c in range(n_classes):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( color_classes_int[str(c)][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( color_classes_int[str(c)][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( color_classes_int[str(c)][2] )).astype('uint8')
    seg_img = cv2.resize(seg_img  , (input_width , input_height ))
    cv2.imwrite(  outName , seg_img )

