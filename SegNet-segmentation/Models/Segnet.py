# todo upgrade to keras 2.0
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, UpSampling2D, Reshape, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import orthogonal, he_normal
from keras.regularizers import l2


def segnet_level3(nClasses, input_height=500, input_width=750):

	input_shape = (input_height, input_width, 3)

	# ------------------------------------------------------------------------------------------------------------ #
	# ----------------------------------------------Encoder------------------------------------------------------- #
	# ------------------------------------------------------------------------------------------------------------ #
	# Encoder block[1]
	inputs = Input(shape=input_shape)
	conv_1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(inputs)
	# 'channels_last' corresponds to inputs with shape (batch, height, width, channels)
	conv_1 = BatchNormalization()(conv_1)
	conv_1 = Activation('relu')(conv_1)

	conv_2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(conv_1)
	conv_2 = BatchNormalization()(conv_2)
	conv_2 = Activation('relu')(conv_2)

	temp = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(conv_2)

	# Encoder block[2]
	conv_3 = Conv2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(temp)
	conv_3 = BatchNormalization()(conv_3)
	conv_3 = Activation('relu')(conv_3)

	conv_4 = Conv2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(conv_3)
	conv_4 = BatchNormalization()(conv_4)
	conv_4 = Activation('relu')(conv_4)

	temp = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv_4)

	# Encoder block[3]
	conv_5 = Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(temp)
	conv_5 = BatchNormalization()(conv_5)
	conv_5 = Activation('relu')(conv_5)

	conv_6 = Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(conv_5)
	conv_6 = BatchNormalization()(conv_6)
	conv_6 = Activation('relu')(conv_6)

	conv_7 = Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(conv_6)
	conv_7 = BatchNormalization()(conv_7)
	conv_7 = Activation('relu')(conv_7)

	temp = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv_7)

	# Encoder block[4]
	conv_8 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(temp)
	conv_8 = BatchNormalization()(conv_8)
	conv_8 = Activation('relu')(conv_8)

	conv_9 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(conv_8)
	conv_9 = BatchNormalization()(conv_9)
	conv_9 = Activation('relu')(conv_9)

	conv_10 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(conv_9)
	conv_10 = BatchNormalization()(conv_10)
	conv_10 = Activation('relu')(conv_10)

	temp = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv_10)

	# Encoder block[5]
	conv_11 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(temp)
	conv_11 = BatchNormalization()(conv_11)
	conv_11 = Activation('relu')(conv_11)

	conv_12 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(conv_11)
	conv_12 = BatchNormalization()(conv_12)
	conv_12 = Activation('relu')(conv_12)

	conv_13 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(conv_12)
	conv_13 = BatchNormalization()(conv_13)
	conv_13 = Activation('relu')(conv_13)

	temp = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv_13)

	print("Build encoder done..")

	# ------------------------------------------------------------------------------------------------------------ #
	# ----------------------------------------------Decoder------------------------------------------------------- #
	# ------------------------------------------------------------------------------------------------------------ #
	# Decoder block[5]
	temp = UpSampling2D(size=(2, 2))(temp)
	temp = ZeroPadding2D(((1, 0), (0, 0)))(temp)  #  ((top_pad, bottom_pad), (left_pad, right_pad))

	conv_14 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(temp)
	conv_14 = BatchNormalization()(conv_14)
	conv_14 = Activation('relu')(conv_14)

	conv_15 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(conv_14)
	conv_15 = BatchNormalization()(conv_15)
	conv_15 = Activation('relu')(conv_15)

	conv_16 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(conv_15)
	conv_16 = BatchNormalization()(conv_16)
	conv_16 = Activation('relu')(conv_16)

	# Decoder block[4]
	temp = UpSampling2D(size=(2, 2))(conv_16)
	temp = ZeroPadding2D(((0, 0), (1, 0)))(temp)

	conv_17 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(temp)
	conv_17 = BatchNormalization()(conv_17)
	conv_17 = Activation('relu')(conv_17)

	conv_18 = Conv2D(filters=512, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(conv_17)
	conv_18 = BatchNormalization()(conv_18)
	conv_18 = Activation('relu')(conv_18)

	conv_19 = Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(conv_18)
	conv_19 = BatchNormalization()(conv_19)
	conv_19 = Activation('relu')(conv_19)

	# Decoder block[3]
	temp = UpSampling2D(size=(2, 2))(conv_19)
	temp = ZeroPadding2D(((1, 0), (1, 0)))(temp)

	conv_20 = Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(temp)
	conv_20 = BatchNormalization()(conv_20)
	conv_20 = Activation('relu')(conv_20)

	conv_21 = Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(conv_20)
	conv_21 = BatchNormalization()(conv_21)
	conv_21 = Activation('relu')(conv_21)

	conv_22 = Conv2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(conv_21)
	conv_22 = BatchNormalization()(conv_22)
	conv_22 = Activation('relu')(conv_22)

	# Decoder block[2]
	temp = UpSampling2D(size=(2, 2))(conv_22)
	temp = ZeroPadding2D(((0, 0), (1, 0)))(temp)

	conv_23 = Conv2D(filters=128, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(temp)
	conv_23 = BatchNormalization()(conv_23)
	conv_23 = Activation('relu')(conv_23)

	conv_24 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(conv_23)
	conv_24 = BatchNormalization()(conv_24)
	conv_24 = Activation('relu')(conv_24)

	# Decoder block[1]
	temp = UpSampling2D(size=(2, 2))(conv_24)

	conv_25 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same', data_format='channels_last',
					kernel_initializer='glorot_uniform')(temp)
	conv_25 = BatchNormalization()(conv_25)
	conv_25 = Activation('relu')(conv_25)


	conv_26 = Conv2D(filters=nClasses, kernel_size=(1,1), strides=(1, 1), padding='valid', data_format='channels_last',
					 kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.005))(conv_25)
	conv_26 = BatchNormalization()(conv_26)

	conv_26 = Reshape((input_shape[0] * input_shape[1], nClasses))(conv_26)

	# Output layer in the end
	outputs = Activation('softmax')(conv_26)

	print("Build decoder done..")

	model = Model(inputs=inputs, outputs=outputs, name="SegNet")

	print("model.output_shape: ", model.output_shape, "\n")

	return model







