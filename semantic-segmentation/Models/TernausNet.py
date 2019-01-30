from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.applications.vgg16 import VGG16
from keras import backend as K
import numpy as np

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=False):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def get_unet_pre_trained(nClasses, input_height=480, input_width=736, nchannel=3):

	input_shape = (input_height, input_width, nchannel)
	inputs = Input(shape=input_shape)

	vgg_model = VGG16(input_tensor=inputs, weights='imagenet',include_top=False)

    layers = dict([(layer.name, layer) for layer in vgg_model.layers])

    vgg_top = layers['block5_conv3'].output
    # Now getting bottom layers for multi-scale skip-layers
    block1_conv2 = layers['block1_conv2'].output
    block2_conv2 = layers['block2_conv2'].output
    block3_conv3 = layers['block3_conv3'].output
    block4_conv3 = layers['block4_conv3'].output


    u_mid = Conv2D(512, (3, 3), activation='relu', padding='same')(vgg_top)
    u_mid = Conv2D(512, (3, 3), activation='relu', padding='same')(u_mid)


    # expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(u_mid)
    u6 = concatenate([u6, block4_conv3])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, block3_conv3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, block2_conv2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, block1_conv2], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
    outputs = Reshape((nClasses,input_height*input_width))(outputs)
    outputs = Permute((2,1))(outputs)

    model = Model(inputs=vgg_model.input, outputs=[outputs])

    for layer in model.layers[:18]:
        layer.trainable = False

    model.summary()
return model