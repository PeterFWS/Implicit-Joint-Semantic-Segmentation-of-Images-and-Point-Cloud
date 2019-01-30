# coding=utf-8
import Models
import LoadBatches
import os
from keras import optimizers

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

# Global parameters for training
# TODO: attention!!! path has to end with "/"
train_images_path = "./data/train_set/rgb_img/"
train_segs_path = "./data/train_set/3_greylabel/"
train_mask_path = "./data/train_set/2_mask/"
save_weights_path = "./weights/"

input_height = 480
input_width = 736
n_classes = 12  # 11 classes + 1 un-classified class
train_batch_size = 1


##################
train_mode = "RGB"
##################

if train_mode == "RGB":
    m = Models.Segnet.segnet_indices_pooling(n_classes, input_height=input_height, input_width=input_width,
                                             nchannel=3, pre_train=True)
    m.compile(loss="categorical_crossentropy",
              optimizer=optimizers.SGD(lr=0.01, momentum=0.9),
              metrics=["accuracy"])
    train_f_path = None
    val_f_path = None
    G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_mask_path,
                                               train_f_path,
                                               train_batch_size, n_classes, input_height, input_width,
                                               output_height=input_height, output_width=input_width)

elif train_mode == "multi_modality":
    m = Models.Segnet.segnet_indices_pooling(n_classes, input_height=input_height, input_width=input_width,
                                             nchannel=75, pre_train=False)
    m.compile(loss="categorical_crossentropy",
              optimizer=optimizers.SGD(lr=0.01, momentum=0.9),
              metrics=["accuracy"])
    train_f_path = "./data/train_set/"
    val_f_path = "./data/validation_set/"
    G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_mask_path,
                                               train_f_path,
                                               train_batch_size, n_classes, input_height, input_width,
                                               output_height=input_height, output_width=input_width)

##################
validate = True
##################
if validate:
    val_images_path = "./data/validation_set/rgb_img/"
    val_segs_path = "./data/validation_set/3_greylabel/"
    val_mask_path = "./data/validation_set/2_mask/"

    val_batch_size = 1

    G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_mask_path,
                                                val_f_path,
                                                val_batch_size, n_classes, input_height,input_width,
                                                output_height=input_height, output_width=input_width)

    callbacks = [EarlyStopping(monitor='val_loss', patience=20),
                 ModelCheckpoint(filepath=os.path.join(save_weights_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                 monitor='val_loss', save_best_only=False), 
                 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.000001),
                 TensorBoard(log_dir=os.path.join('/home/fangwen/ShuFangwen/source/image-segmentation-keras/board'),
                             histogram_freq=0, write_graph=True, write_images=True)]

    m.fit_generator(G,
                    steps_per_epoch=1550//train_batch_size,
                    callbacks=callbacks,
                    validation_data=G2,
                    validation_steps=594//val_batch_size,
                    epochs=10000)
    m.save_weights(save_weights_path + "end_weights.h5")
    # m.save(save_weights_path + "endtest.model.weights")


else:
    m.fit_generator(G, steps_per_epoch=1513//train_batch_size, epochs=1)
    m.save_weights(save_weights_path + ".weights")
    m.save(save_weights_path + ".model.weights")


