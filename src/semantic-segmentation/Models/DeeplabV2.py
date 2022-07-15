from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, ZeroPadding2D, Dropout, Activation, UpSampling2D

# WEIGHTS_PATH = 'deeplabV2_weights_tf.h5'
# VOC2012_CLASSES_COUNT = 21


def DeeplabV2(input_shape=(480,480,3),
              upsampling=8,
              apply_softmax=True,
              nclasses=12):

    img_input = Input(shape=input_shape)

    # Block 1
    h = ZeroPadding2D(padding=(1, 1))(img_input)
    h = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv1_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv1_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 2
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name='conv2_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name='conv2_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 3
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', name='conv3_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', name='conv3_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', name='conv3_3')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 4
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', name='conv4_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', name='conv4_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', name='conv4_3')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # Block 5 -TODO - Might be incorrect
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    p5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # branching for Atrous Spatial Pyramid Pooling - Until here -14 layers
    # hole = 6
    b1 = ZeroPadding2D(padding=(6, 6))(p5)
    b1 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(6, 6), activation='relu', name='fc6_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Conv2D(filters=21, kernel_size=(1, 1), activation='relu', name='fc8_voc12_1')(b1)

    # hole = 12
    b2 = ZeroPadding2D(padding=(12, 12))(p5)
    b2 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(12, 12), activation='relu', name='fc6_2')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_2')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Conv2D(filters=21, kernel_size=(1, 1), activation='relu', name='fc8_voc12_2')(b2)

    # hole = 18
    b3 = ZeroPadding2D(padding=(18, 18))(p5)
    b3 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(18, 18), activation='relu', name='fc6_3')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_3')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Conv2D(filters=21, kernel_size=(1, 1), activation='relu', name='fc8_voc12_3')(b3)

    # hole = 24
    b4 = ZeroPadding2D(padding=(24, 24))(p5)
    b4 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(24, 24), activation='relu', name='fc6_4')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_4')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Conv2D(filters=nclasses, kernel_size=(1, 1), activation='relu', name='fc8_voc12_4')(b4)

    s = Add()([b1, b2, b3, b4])
    logits = UpSampling2D(size=upsampling, interpolation='bilinear')(s)

    if apply_softmax:
        out = Activation('softmax')(logits)
    else:
        out = logits

    model = Model(img_input, out, name='deeplabV2')



    return model