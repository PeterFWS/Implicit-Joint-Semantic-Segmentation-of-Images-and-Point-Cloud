# coding=utf-8
import Models
import LoadBatches
import os
from keras import optimizers

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

# Global parameters for training
train_images_path = "./data/train/training_set/rgb_imgs/"
train_segs_path = "./data/train/training_set/3_greylabel/"
train_mask_path = "./data/train/training_set/2_mask/"

save_weights_path = "./weights/"

input_height = 500
input_width = 750
n_classes = 12  # 11 classes + 1 un-classified class
train_batch_size = 1


# Compile model
m = Models.Segnet.segnet_level3(n_classes, input_height=input_height, input_width=input_width)
m.compile(loss="categorical_crossentropy",
          optimizer='adadelta',
          metrics=["accuracy"])
# m.load_weights("./data/vgg16_weights_th_dim_ordering_th_kernels.h5")

# Generator of input data
G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_mask_path,
                                           train_batch_size, n_classes, input_height, input_width,
                                           output_height=input_height, output_width=input_width)


# Train
validate = True
if validate:
    val_images_path = "./data/train/validation_set/rgb_imgs/"
    val_segs_path = "./data/train/validation_set/3_greylabel/"
    val_mask_path = "./data/train/validation_set/2_mask/"
    val_batch_size = 1

    G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_mask_path,
                                                val_batch_size, n_classes, input_height,input_width,
                                                output_height=input_height, output_width=input_width)
    """
    In Keras, we can implement early stopping as a callback function. Callbacks are functions that can be applied 
    at certain stages of the training process, such as at the end of each epoch. Specifically, in our solution, 
    we included EarlyStopping(monitor='val_loss', patience=2) to define that we wanted to monitor the test (
    validation) loss at each epoch and after the test loss has not improved after 2 epochs, training is 
    interrupted. However, since we set patience=2, we won’t get the best model, but the model two epochs after the 
    best model. Therefore, optionally, we can include a second operation, ModelCheckpoint which saves the model to a 
    file after every checkpoint (which can be useful in case a multi-day training session is interrupted for some 
    reason. Helpful for us, if we set save_best_only=True then ModelCheckpoint will only save the best model. 
    """
    callbacks = [EarlyStopping(monitor='val_loss', patience=20),
                 ModelCheckpoint(filepath=os.path.join(save_weights_path, 'best_model.h5'), monitor='val_loss',
                                 save_best_only=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001),
                 TensorBoard(log_dir=os.path.join('/home/fangwen/ShuFangwen/source/image-segmentation-keras/board'),
                             histogram_freq=0, write_graph=True, write_images=True)]

    m.fit_generator(G, steps_per_epoch=1513//train_batch_size, callbacks=callbacks,
                    validation_data=G2, validation_steps=440//val_batch_size, epochs=10000)
    m.save_weights(save_weights_path + "endtest.weights")
    m.save(save_weights_path + "endtest.model.weights")

else:
    m.fit_generator(G, steps_per_epoch=1513//train_batch_size, epochs=1)
    m.save_weights(save_weights_path + ".weights")
    m.save(save_weights_path + ".model.weights")


