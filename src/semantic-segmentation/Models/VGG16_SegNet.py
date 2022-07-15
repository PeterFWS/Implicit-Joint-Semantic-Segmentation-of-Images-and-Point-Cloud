from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout, Reshape, Permute, UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.applications.vgg16 import VGG16
from keras import backend as K

def VGG16_Segnet(nClasses, input_height=224, input_width=224, nchannel=3, dropout=True):
	"""
nClasses=12
input_height=224
input_width=224
nchannel = 3
	"""

	if dropout == True:
		rate = 0.4
	else:
		rate = 0

	input_shape = (input_height, input_width, nchannel)
	inputs = Input(shape=input_shape)

	# Encoder of VGG16
	vgg_model = VGG16(input_tensor=inputs, weights='imagenet', include_top=False)
	vgg_output = vgg_model.output # <tf.Tensor 'block5_pool/MaxPool:0' shape=(?, 15, 23, 512) dtype=float32>

	# Decoder block[5]
	up_1 = UpSampling2D(size=(2, 2))(vgg_output)

	conv_14 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same')(up_1)
	conv_14 = BatchNormalization()(conv_14)
	conv_14 = Activation('relu')(conv_14)
	conv_14 = Dropout(rate)(conv_14)

	conv_15 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same')(conv_14)
	conv_15 = BatchNormalization()(conv_15)
	conv_15 = Activation('relu')(conv_15)
	conv_15 = Dropout(rate)(conv_15)

	conv_16 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same')(conv_15)
	conv_16 = BatchNormalization()(conv_16)
	conv_16 = Activation('relu')(conv_16)
	conv_16 = Dropout(rate)(conv_16)

	# Decoder block[4]
	up_2 = UpSampling2D(size=(2, 2))(conv_16)

	conv_17 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same')(up_2)
	conv_17 = BatchNormalization()(conv_17)
	conv_17 = Activation('relu')(conv_17)
	conv_17 = Dropout(rate)(conv_17)

	conv_18 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same')(conv_17)
	conv_18 = BatchNormalization()(conv_18)
	conv_18 = Activation('relu')(conv_18)
	conv_18 = Dropout(rate)(conv_18)

	conv_19 = Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding='same')(conv_18)
	conv_19 = BatchNormalization()(conv_19)
	conv_19 = Activation('relu')(conv_19)
	conv_19 = Dropout(rate)(conv_19)

	# Decoder block[3]
	up_3 = UpSampling2D(size=(2, 2))(conv_19)

	conv_20 = Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding='same')(up_3)
	conv_20 = BatchNormalization()(conv_20)
	conv_20 = Activation('relu')(conv_20)
	conv_20 = Dropout(rate)(conv_20)

	conv_21 = Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding='same')(conv_20)
	conv_21 = BatchNormalization()(conv_21)
	conv_21 = Activation('relu')(conv_21)
	conv_21 = Dropout(rate)(conv_21)

	conv_22 = Conv2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='same')(conv_21)
	conv_22 = BatchNormalization()(conv_22)
	conv_22 = Activation('relu')(conv_22)
	conv_22 = Dropout(rate)(conv_22)

	# Decoder block[2]
	up_4 = UpSampling2D(size=(2, 2))(conv_22)

	conv_23 = Conv2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='same')(up_4)
	conv_23 = BatchNormalization()(conv_23)
	conv_23 = Activation('relu')(conv_23)
	conv_23 = Dropout(rate)(conv_23)

	conv_24 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same')(conv_23)
	conv_24 = BatchNormalization()(conv_24)
	conv_24 = Activation('relu')(conv_24)
	conv_24 = Dropout(rate)(conv_24)

	# Decoder block[1]
	up_5 = UpSampling2D(size=(2, 2))(conv_24)

	conv_25 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same')(up_5)
	conv_25 = BatchNormalization()(conv_25)
	conv_25 = Activation('relu')(conv_25)
	conv_25 = Dropout(rate)(conv_25)


	conv_26 = Conv2D(filters=nClasses, kernel_size=(1,1), strides=(1, 1), padding='valid')(conv_25)
	conv_26 = BatchNormalization()(conv_26)

	conv_26 = Reshape((input_shape[0] * input_shape[1], nClasses), 
					  input_shape=(input_shape[0], input_shape[1], nClasses))(conv_26)

	# Output layer in the end
	outputs = Activation('softmax')(conv_26)

	model = Model(inputs=vgg_model.input, outputs=outputs, name="VGG16_SegNet")

	for layer in model.layers[:19]:
	    layer.trainable = False

	print("\n")
	for layer in model.layers:
		print(layer.name, layer.trainable)
	print("\n")