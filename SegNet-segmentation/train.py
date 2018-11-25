import Models 
import LoadBatches

# Global parameters
train_images_path = "./data/images_prepped_train/"
train_segs_path = "./data/annotations_prepped_train/"
save_weights_path = "./weights/ex1"

input_height = 244
input_width = 244
n_classes = 11
train_batch_size = 1
epochs = 1

validate = False
if validate:
	val_images_path = "./data/images_prepped_test/"
	val_segs_path = "./data/annotations_prepped_test/"
	val_batch_size = 2


m = Models.Segnet.segnet2(n_classes, input_height=input_height, input_width=input_width)
m.compile(loss='categorical_crossentropy',
      optimizer= "adadelta" ,
      metrics=['accuracy'])

# load_weights = ""
# if len( load_weights ) > 0:
# 	m.load_weights(load_weights)


print "Model output shape" ,  m.output_shape

# output_height = m.outputHeight
# output_width = m.outputWidth

G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , input_height , input_width   )


if validate:
	G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , input_height , input_width   )

if not validate:
	for ep in range( epochs ):
		m.fit_generator( G , 512  , epochs=1 )
		m.save_weights( save_weights_path + "." + str( ep ) )
		m.save( save_weights_path + ".model." + str( ep ) )
else:
	for ep in range( epochs ):
		m.fit_generator( G , 512  , validation_data=G2 , validation_steps=200 ,  epochs=1 )
		m.save_weights( save_weights_path + "." + str( ep )  )
		m.save( save_weights_path + ".model." + str( ep ) )


