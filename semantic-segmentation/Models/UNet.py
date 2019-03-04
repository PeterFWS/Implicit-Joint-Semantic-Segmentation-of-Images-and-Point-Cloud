# todo upgrade to keras 2.0
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2DTranspose, Conv2D, Activation, MaxPooling2D, UpSampling2D, Reshape, ZeroPadding2D, Permute, Dropout, concatenate
from keras.layers.normalization import BatchNormalization
from keras.applications import vgg16

# def U_Net(nClasses, input_height=480, input_width=736, nchannel=3, pre_train=False):
#
#     input_shape = (input_height, input_width, nchannel)
#
#     inputs = Input(shape=input_shape)
#
#     # norm = BatchNormalization()(inputs)
#
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
#     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#
#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
#
#
#     up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
#     conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
#     conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
#     conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
#
#     up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
#     conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
#     conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
#     conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
#
#     up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
#     conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
#     conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
#
#     up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
#     conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
#     conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
#
#     conv10 = Conv2D(nClasses, (1, 1), activation='relu', padding='same')(conv9)
#     conv10 = Reshape((nClasses,input_height*input_width))(conv10)
#     conv10 = Permute((2,1))(conv10)
#
#     conv11 = Activation('softmax')(conv10)
#
#     if pre_train is not False:
#         vgg_weight_path = "./data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
#         model = Model(inputs=inputs, outputs=conv11, name="UNet")
#         model.load_weights(vgg_weight_path, by_name=True)
#         print("Load VGG weight success..\n")
#     # freeze the VGG layers
#     # for layer in model.layers[:45]:
#     # 	if str(layer.name).split("_")[-2] == "conv2d":
#     # 		layer.trainable = False
#     else:
#         print("no pre-trained.. \n")
#         model = Model(inputs=inputs, outputs=conv11, name="UNet")
#
#     # model = Model(input=inputs, output=conv11)
#
#     print(model.summary())
#
#     return model
#




def UNet(nClasses, input_height, input_width, nchannel=3):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, nchannel))

    vgg_streamlined = vgg16.VGG16(
        include_top=False,
        weights='imagenet', input_tensor=img_input)
    assert isinstance(vgg_streamlined, Model)


    o = UpSampling2D((2, 2))(vgg_streamlined.output)
    o = concatenate([vgg_streamlined.get_layer(
        name="block4_pool").output, o], axis=-1)
    o = Conv2D(512, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([vgg_streamlined.get_layer(
        name="block3_pool").output, o], axis=-1)
    o = Conv2D(256, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([vgg_streamlined.get_layer(
        name="block2_pool").output, o], axis=-1)
    o = Conv2D(128, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = concatenate([vgg_streamlined.get_layer(
        name="block1_pool").output, o], axis=-1)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = UpSampling2D((2, 2))(o)
    o = Conv2D(64, (3, 3), padding="same")(o)
    o = BatchNormalization()(o)

    o = Conv2D(nClasses, (1, 1), padding="same")(o)
    o = BatchNormalization()(o)
    o = Activation("relu")(o)

    o = Reshape((-1, nClasses))(o)
    o = Activation("softmax")(o)

    model = Model(inputs=img_input, outputs=o)
    return model


