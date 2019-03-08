from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Reshape, Dropout, concatenate, Average
from keras.layers.normalization import BatchNormalization
from keras.constraints import MaxNorm

from keras.utils import plot_model
from mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D

def testNet(nClasses, input_height=480, input_width=480):

	"""
	nClasses=12
input_height=480
input_width=480
	:param nClasses:
	:param input_height:
	:param input_width:
	:return:
	"""

	input_shape = (input_height, input_width, 3)
	kernel = 3
	pool_size = (2, 2)
	output_mode = "softmax"

	auxiliary_input = Input(shape=(input_height,input_width, 3), name='aux_input')

	conv_1_ = Conv2D(64, (kernel, kernel), padding="same")(auxiliary_input)
	conv_1_ = BatchNormalization()(conv_1_)
	conv_1_ = Activation("relu")(conv_1_)
	conv_2_ = Conv2D(64, (kernel, kernel), padding="same")(conv_1_)
	conv_2_ = BatchNormalization()(conv_2_)
	conv_2_a = Activation("relu")(conv_2_)

	pool_1_, mask_1_ = MaxPoolingWithArgmax2D(pool_size)(conv_2_a)

	conv_3_ = Conv2D(128, (kernel, kernel), padding="same")(pool_1_)
	conv_3_ = BatchNormalization()(conv_3_)
	conv_3_ = Activation("relu")(conv_3_)
	conv_4_ = Conv2D(128, (kernel, kernel), padding="same")(conv_3_)
	conv_4_ = BatchNormalization()(conv_4_)
	conv_4_a = Activation("relu")(conv_4_)

	pool_2_, mask_2_ = MaxPoolingWithArgmax2D(pool_size)(conv_4_a)

	conv_5_ = Conv2D(256, (kernel, kernel), padding="same")(pool_2_)
	conv_5_ = BatchNormalization()(conv_5_)
	conv_5_ = Activation("relu")(conv_5_)
	conv_6_ = Conv2D(256, (kernel, kernel), padding="same")(conv_5_)
	conv_6_ = BatchNormalization()(conv_6_)
	conv_6_ = Activation("relu")(conv_6_)
	conv_7_ = Conv2D(256, (kernel, kernel), padding="same")(conv_6_)
	conv_7_ = BatchNormalization()(conv_7_)
	conv_7_a = Activation("relu")(conv_7_)

	pool_3_, mask_3_ = MaxPoolingWithArgmax2D(pool_size)(conv_7_a)

	conv_8_ = Conv2D(512, (kernel, kernel), padding="same")(pool_3_)
	conv_8_ = BatchNormalization()(conv_8_)
	conv_8_ = Activation("relu")(conv_8_)
	conv_9_ = Conv2D(512, (kernel, kernel), padding="same")(conv_8_)
	conv_9_ = BatchNormalization()(conv_9_)
	conv_9_ = Activation("relu")(conv_9_)
	conv_10_ = Conv2D(512, (kernel, kernel), padding="same")(conv_9_)
	conv_10_ = BatchNormalization()(conv_10_)
	conv_10_a = Activation("relu")(conv_10_)

	pool_4_, mask_4_ = MaxPoolingWithArgmax2D(pool_size)(conv_10_a)


	conv_11_ = Conv2D(512, (kernel, kernel), padding="same")(pool_4_)
	conv_11_ = BatchNormalization()(conv_11_)
	conv_11_ = Activation("relu")(conv_11_)
	conv_12_ = Conv2D(512, (kernel, kernel), padding="same")(conv_11_)
	conv_12_ = BatchNormalization()(conv_12_)
	conv_12_ = Activation("relu")(conv_12_)
	conv_13_ = Conv2D(512, (kernel, kernel), padding="same")(conv_12_)
	conv_13_ = BatchNormalization()(conv_13_)
	conv_13_a = Activation("relu")(conv_13_)

	# pool_5_, mask_5_ = MaxPoolingWithArgmax2D(pool_size)(conv_13_)

	## ---------------------------------------------------------------------------------------##

	main_input = Input(shape=(input_height,input_width, 3), name='main_input')

	conv_1 = Conv2D(64, (kernel, kernel), padding="same")(main_input)
	conv_1 = BatchNormalization()(conv_1)
	conv_1 = Activation("relu")(conv_1)
	conv_2 = Conv2D(64, (kernel, kernel), padding="same")(conv_1)
	conv_2 = BatchNormalization()(conv_2)
	conv_2 = Average()([Activation("relu")(conv_2), conv_2_a])

	pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

	conv_3 = Conv2D(128, (kernel, kernel), padding="same")(pool_1)
	conv_3 = BatchNormalization()(conv_3)
	conv_3 = Activation("relu")(conv_3)
	conv_4 = Conv2D(128, (kernel, kernel), padding="same")(conv_3)
	conv_4 = BatchNormalization()(conv_4)
	conv_4 = Average()([Activation("relu")(conv_4), conv_4_a])

	pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

	conv_5 = Conv2D(256, (kernel, kernel), padding="same")(pool_2)
	conv_5 = BatchNormalization()(conv_5)
	conv_5 = Activation("relu")(conv_5)
	conv_6 = Conv2D(256, (kernel, kernel), padding="same")(conv_5)
	conv_6 = BatchNormalization()(conv_6)
	conv_6 = Activation("relu")(conv_6)
	conv_7 = Conv2D(256, (kernel, kernel), padding="same")(conv_6)
	conv_7 = BatchNormalization()(conv_7)
	conv_7 = Average()([Activation("relu")(conv_7), conv_7_a])

	pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)


	conv_8 = Conv2D(512, (kernel, kernel), padding="same")(pool_3)
	conv_8 = BatchNormalization()(conv_8)
	conv_8 = Activation("relu")(conv_8)
	conv_9 = Conv2D(512, (kernel, kernel), padding="same")(conv_8)
	conv_9 = BatchNormalization()(conv_9)
	conv_9 = Activation("relu")(conv_9)
	conv_10 = Conv2D(512, (kernel, kernel), padding="same")(conv_9)
	conv_10 = BatchNormalization()(conv_10)
	conv_10 = Average()([Activation("relu")(conv_10), conv_10_a])

	pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)


	conv_11 = Conv2D(512, (kernel, kernel), padding="same")(pool_4)
	conv_11 = BatchNormalization()(conv_11)
	conv_11 = Activation("relu")(conv_11)
	conv_12 = Conv2D(512, (kernel, kernel), padding="same")(conv_11)
	conv_12 = BatchNormalization()(conv_12)
	conv_12 = Activation("relu")(conv_12)
	conv_13 = Conv2D(512, (kernel, kernel), padding="same")(conv_12)
	conv_13 = BatchNormalization()(conv_13)
	conv_13 = Activation("relu")(conv_13)
	conv_13 = Average()([Activation("relu")(conv_13), conv_13_a])

	pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)

	# Decoder
	unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

	conv_14 = Conv2D(512, (kernel, kernel), padding="same")(unpool_1)
	conv_14 = BatchNormalization()(conv_14)
	conv_14 = Activation("relu")(conv_14)
	conv_15 = Conv2D(512, (kernel, kernel), padding="same")(conv_14)
	conv_15 = BatchNormalization()(conv_15)
	conv_15 = Activation("relu")(conv_15)
	conv_16 = Conv2D(512, (kernel, kernel), padding="same")(conv_15)
	conv_16 = BatchNormalization()(conv_16)
	conv_16 = Activation("relu")(conv_16)


	unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

	conv_17 = Conv2D(512, (kernel, kernel), padding="same")(unpool_2)
	conv_17 = BatchNormalization()(conv_17)
	conv_17 = Activation("relu")(conv_17)
	conv_18 = Conv2D(512, (kernel, kernel), padding="same")(conv_17)
	conv_18 = BatchNormalization()(conv_18)
	conv_18 = Activation("relu")(conv_18)
	conv_19 = Conv2D(256, (kernel, kernel), padding="same")(conv_18)
	conv_19 = BatchNormalization()(conv_19)
	conv_19 = Activation("relu")(conv_19)



	unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

	conv_20 = Conv2D(256, (kernel, kernel), padding="same")(unpool_3)
	conv_20 = BatchNormalization()(conv_20)
	conv_20 = Activation("relu")(conv_20)
	conv_21 = Conv2D(256, (kernel, kernel), padding="same")(conv_20)
	conv_21 = BatchNormalization()(conv_21)
	conv_21 = Activation("relu")(conv_21)
	conv_22 = Conv2D(128, (kernel, kernel), padding="same")(conv_21)
	conv_22 = BatchNormalization()(conv_22)
	conv_22 = Activation("relu")(conv_22)



	unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

	conv_23 = Conv2D(128, (kernel, kernel), padding="same")(unpool_4)
	conv_23 = BatchNormalization()(conv_23)
	conv_23 = Activation("relu")(conv_23)
	conv_24 = Conv2D(64, (kernel, kernel), padding="same")(conv_23)
	conv_24 = BatchNormalization()(conv_24)
	conv_24 = Activation("relu")(conv_24)

	unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

	conv_25 = Conv2D(64, (kernel, kernel), padding="same")(unpool_5)
	conv_25 = BatchNormalization()(conv_25)
	conv_25 = Activation("relu")(conv_25)

	conv_26 = Conv2D(nClasses, (1, 1), padding="valid")(conv_25)
	conv_26 = BatchNormalization()(conv_26)

	conv_26 = Reshape(
		(input_shape[0] * input_shape[1], nClasses),
		input_shape=(input_shape[0], input_shape[1], nClasses))(conv_26)

	outputs = Activation(output_mode)(conv_26)

	model = Model(inputs=[main_input, auxiliary_input], outputs=outputs, name="SegNet")

	# plot_model(model, to_file='/data/fangwen/model.png')

	print(model.summary())

	return model



def testNet2(nClasses, input_height=480, input_width=480):

	"""
	nClasses=12
input_height=480
input_width=480
	:param nClasses:
	:param input_height:
	:param input_width:
	:return:
	"""

	input_shape = (input_height, input_width, 3)
	kernel = 3
	pool_size = (2, 2)
	output_mode = "softmax"

	main_input = Input(shape=(input_height, input_width, 3), name='main_input')
	auxiliary_input = Input(shape=(input_height,input_width, 3), name='aux_input')

	# Encoder

	conv_1 = Conv2D(64, (kernel, kernel), padding="same")(main_input)
	conv_1 = BatchNormalization()(conv_1)
	conv_1 = Activation("relu")(conv_1)
	conv_2 = Conv2D(64, (kernel, kernel), padding="same")(conv_1)
	conv_2 = BatchNormalization()(conv_2)
	conv_2 = Activation("relu")(conv_2)

	pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

	conv_3 = Conv2D(128, (kernel, kernel), padding="same")(pool_1)
	conv_3 = BatchNormalization()(conv_3)
	conv_3 = Activation("relu")(conv_3)
	conv_4 = Conv2D(128, (kernel, kernel), padding="same")(conv_3)
	conv_4 = BatchNormalization()(conv_4)
	conv_4 = Activation("relu")(conv_4)

	pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

	conv_5 = Conv2D(256, (kernel, kernel), padding="same")(pool_2)
	conv_5 = BatchNormalization()(conv_5)
	conv_5 = Activation("relu")(conv_5)
	conv_6 = Conv2D(256, (kernel, kernel), padding="same")(conv_5)
	conv_6 = BatchNormalization()(conv_6)
	conv_6 = Activation("relu")(conv_6)
	conv_7 = Conv2D(256, (kernel, kernel), padding="same")(conv_6)
	conv_7 = BatchNormalization()(conv_7)
	conv_7 = Activation("relu")(conv_7)

	pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)
	# pool_3 = Dropout(rate)(pool_3)


	conv_8 = Conv2D(512, (kernel, kernel), padding="same")(pool_3)
	conv_8 = BatchNormalization()(conv_8)
	conv_8 = Activation("relu")(conv_8)
	conv_9 = Conv2D(512, (kernel, kernel), padding="same")(conv_8)
	conv_9 = BatchNormalization()(conv_9)
	conv_9 = Activation("relu")(conv_9)
	conv_10 = Conv2D(512, (kernel, kernel), padding="same")(conv_9)
	conv_10 = BatchNormalization()(conv_10)
	conv_10 = Activation("relu")(conv_10)

	pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)
	# pool_4 = Dropout(rate)(pool_4)


	conv_11 = Conv2D(512, (kernel, kernel), padding="same")(pool_4)
	conv_11 = BatchNormalization()(conv_11)
	conv_11 = Activation("relu")(conv_11)
	conv_12 = Conv2D(512, (kernel, kernel), padding="same")(conv_11)
	conv_12 = BatchNormalization()(conv_12)
	conv_12 = Activation("relu")(conv_12)
	conv_13 = Conv2D(512, (kernel, kernel), padding="same")(conv_12)
	conv_13 = BatchNormalization()(conv_13)
	conv_13 = Activation("relu")(conv_13)

	pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)

	print("Build enceder done..")

	# Decoder
	unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

	conv_14 = Conv2D(512, (kernel, kernel), padding="same")(unpool_1)
	conv_14 = BatchNormalization()(conv_14)
	conv_14 = Activation("relu")(conv_14)
	conv_15 = Conv2D(512, (kernel, kernel), padding="same")(conv_14)
	conv_15 = BatchNormalization()(conv_15)
	conv_15 = Activation("relu")(conv_15)
	conv_16 = Conv2D(512, (kernel, kernel), padding="same")(conv_15)
	conv_16 = BatchNormalization()(conv_16)
	conv_16 = Activation("relu")(conv_16)


	unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

	conv_17 = Conv2D(512, (kernel, kernel), padding="same")(unpool_2)
	conv_17 = BatchNormalization()(conv_17)
	conv_17 = Activation("relu")(conv_17)
	conv_18 = Conv2D(512, (kernel, kernel), padding="same")(conv_17)
	conv_18 = BatchNormalization()(conv_18)
	conv_18 = Activation("relu")(conv_18)
	conv_19 = Conv2D(256, (kernel, kernel), padding="same")(conv_18)
	conv_19 = BatchNormalization()(conv_19)
	conv_19 = Activation("relu")(conv_19)
	# conv_19 = Dropout(rate)(conv_19)


	unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

	conv_20 = Conv2D(256, (kernel, kernel), padding="same")(unpool_3)
	conv_20 = BatchNormalization()(conv_20)
	conv_20 = Activation("relu")(conv_20)
	conv_21 = Conv2D(256, (kernel, kernel), padding="same")(conv_20)
	conv_21 = BatchNormalization()(conv_21)
	conv_21 = Activation("relu")(conv_21)
	conv_22 = Conv2D(128, (kernel, kernel), padding="same")(conv_21)
	conv_22 = BatchNormalization()(conv_22)
	conv_22 = Activation("relu")(conv_22)
	# conv_22 = Dropout(rate)(conv_22)


	unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

	conv_23 = Conv2D(128, (kernel, kernel), padding="same")(unpool_4)
	conv_23 = BatchNormalization()(conv_23)
	conv_23 = Activation("relu")(conv_23)
	conv_24 = Conv2D(64, (kernel, kernel), padding="same")(conv_23)
	conv_24 = BatchNormalization()(conv_24)
	conv_24 = Activation("relu")(conv_24)

	unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

	conv_25 = Conv2D(64, (kernel, kernel), padding="same")(unpool_5)
	conv_25 = BatchNormalization()(conv_25)
	conv_25 = Activation("relu")(conv_25)



	## -----------------------------------------------------------------------------------------------#

	conv_1_ = Conv2D(64, (kernel, kernel), padding="same")(auxiliary_input)
	conv_1_ = BatchNormalization()(conv_1_)
	conv_1_ = Activation("relu")(conv_1_)
	conv_2_ = Conv2D(64, (kernel, kernel), padding="same")(conv_1_)
	conv_2_ = BatchNormalization()(conv_2_)
	conv_2_ = Activation("relu")(conv_2_)

	pool_1_, mask_1_ = MaxPoolingWithArgmax2D(pool_size)(conv_2_)

	conv_3_ = Conv2D(128, (kernel, kernel), padding="same")(pool_1_)
	conv_3_ = BatchNormalization()(conv_3_)
	conv_3_ = Activation("relu")(conv_3_)
	conv_4_ = Conv2D(128, (kernel, kernel), padding="same")(conv_3_)
	conv_4_ = BatchNormalization()(conv_4_)
	conv_4_ = Activation("relu")(conv_4_)

	pool_2_, mask_2_ = MaxPoolingWithArgmax2D(pool_size)(conv_4_)

	conv_5_ = Conv2D(256, (kernel, kernel), padding="same")(pool_2_)
	conv_5_ = BatchNormalization()(conv_5_)
	conv_5_ = Activation("relu")(conv_5_)
	conv_6_ = Conv2D(256, (kernel, kernel), padding="same")(conv_5_)
	conv_6_ = BatchNormalization()(conv_6_)
	conv_6_ = Activation("relu")(conv_6_)
	conv_7_ = Conv2D(256, (kernel, kernel), padding="same")(conv_6_)
	conv_7_ = BatchNormalization()(conv_7_)
	conv_7_ = Activation("relu")(conv_7_)

	pool_3_, mask_3_ = MaxPoolingWithArgmax2D(pool_size)(conv_7_)
	# pool_3 = Dropout(rate)(pool_3)


	conv_8_ = Conv2D(512, (kernel, kernel), padding="same")(pool_3_)
	conv_8_ = BatchNormalization()(conv_8_)
	conv_8_ = Activation("relu")(conv_8_)
	conv_9_ = Conv2D(512, (kernel, kernel), padding="same")(conv_8_)
	conv_9_ = BatchNormalization()(conv_9_)
	conv_9_ = Activation("relu")(conv_9_)
	conv_10_ = Conv2D(512, (kernel, kernel), padding="same")(conv_9_)
	conv_10_ = BatchNormalization()(conv_10_)
	conv_10_ = Activation("relu")(conv_10_)

	pool_4_, mask_4_ = MaxPoolingWithArgmax2D(pool_size)(conv_10_)
	# pool_4 = Dropout(rate)(pool_4)


	conv_11_ = Conv2D(512, (kernel, kernel), padding="same")(pool_4_)
	conv_11_ = BatchNormalization()(conv_11_)
	conv_11_ = Activation("relu")(conv_11_)
	conv_12_ = Conv2D(512, (kernel, kernel), padding="same")(conv_11_)
	conv_12_ = BatchNormalization()(conv_12_)
	conv_12_ = Activation("relu")(conv_12_)
	conv_13_ = Conv2D(512, (kernel, kernel), padding="same")(conv_12_)
	conv_13_ = BatchNormalization()(conv_13_)
	conv_13_ = Activation("relu")(conv_13_)

	pool_5_, mask_5_ = MaxPoolingWithArgmax2D(pool_size)(conv_13_)

	print("Build enceder done..")

	# Decoder
	unpool_1_ = MaxUnpooling2D(pool_size)([pool_5_, mask_5_])

	conv_14_ = Conv2D(512, (kernel, kernel), padding="same")(unpool_1_)
	conv_14_ = BatchNormalization()(conv_14_)
	conv_14_ = Activation("relu")(conv_14_)
	conv_15_ = Conv2D(512, (kernel, kernel), padding="same")(conv_14_)
	conv_15_ = BatchNormalization()(conv_15_)
	conv_15_ = Activation("relu")(conv_15_)
	conv_16_ = Conv2D(512, (kernel, kernel), padding="same")(conv_15_)
	conv_16_ = BatchNormalization()(conv_16_)
	conv_16_ = Activation("relu")(conv_16_)


	unpool_2_ = MaxUnpooling2D(pool_size)([conv_16_, mask_4_])

	conv_17_ = Conv2D(512, (kernel, kernel), padding="same")(unpool_2_)
	conv_17_ = BatchNormalization()(conv_17_)
	conv_17_ = Activation("relu")(conv_17_)
	conv_18_ = Conv2D(512, (kernel, kernel), padding="same")(conv_17_)
	conv_18_ = BatchNormalization()(conv_18_)
	conv_18_ = Activation("relu")(conv_18_)
	conv_19_ = Conv2D(256, (kernel, kernel), padding="same")(conv_18_)
	conv_19_ = BatchNormalization()(conv_19_)
	conv_19_ = Activation("relu")(conv_19_)
	# conv_19 = Dropout(rate)(conv_19)


	unpool_3_ = MaxUnpooling2D(pool_size)([conv_19_, mask_3_])

	conv_20_ = Conv2D(256, (kernel, kernel), padding="same")(unpool_3_)
	conv_20_ = BatchNormalization()(conv_20_)
	conv_20_ = Activation("relu")(conv_20_)
	conv_21_ = Conv2D(256, (kernel, kernel), padding="same")(conv_20_)
	conv_21_ = BatchNormalization()(conv_21_)
	conv_21_ = Activation("relu")(conv_21_)
	conv_22_ = Conv2D(128, (kernel, kernel), padding="same")(conv_21_)
	conv_22_ = BatchNormalization()(conv_22_)
	conv_22_ = Activation("relu")(conv_22_)
	# conv_22 = Dropout(rate)(conv_22)


	unpool_4_ = MaxUnpooling2D(pool_size)([conv_22_, mask_2_])

	conv_23_ = Conv2D(128, (kernel, kernel), padding="same")(unpool_4_)
	conv_23_ = BatchNormalization()(conv_23_)
	conv_23_ = Activation("relu")(conv_23_)
	conv_24_ = Conv2D(64, (kernel, kernel), padding="same")(conv_23_)
	conv_24_ = BatchNormalization()(conv_24_)
	conv_24_ = Activation("relu")(conv_24_)

	unpool_5_ = MaxUnpooling2D(pool_size)([conv_24_, mask_1_])

	conv_25_ = Conv2D(64, (kernel, kernel), padding="same")(unpool_5_)
	conv_25_ = BatchNormalization()(conv_25_)
	conv_25_ = Activation("relu")(conv_25_)


	## --------------------------------------------------#

	temp = Average()([conv_25, conv_25_])

	conv_26 = Conv2D(nClasses, (1, 1), padding="valid")(temp)
	conv_26 = BatchNormalization()(conv_26)

	conv_26 = Reshape(
		(input_shape[0] * input_shape[1], nClasses),
		input_shape=(input_shape[0], input_shape[1], nClasses))(conv_26)

	outputs = Activation(output_mode)(conv_26)

	model = Model(inputs=[main_input, auxiliary_input], outputs=outputs, name="SegNet")

	print(model.summary())

	return model





