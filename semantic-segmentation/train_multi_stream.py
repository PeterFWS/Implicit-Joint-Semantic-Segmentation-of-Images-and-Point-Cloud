# coding=utf-8
import Models
import LoadBatches
import os
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import numpy as np

# Global parameters
# TODO: attention!!! path has to end with "/"
# TODO: In case you have data saved in different folder, we use "list" to save path here
train_images_path = ["/data/fangwen/mix_train2/rgb_img/"]
train_segs_path = ["/data/fangwen/mix_train2/3_greylabel/"]
train_mask_path = ["/data/fangwen/mix_train2/2_mask/"]

save_weights_path = "./weights/"

train_f_path = ["/data/fangwen/mix_train2/"]


input_height = 480
input_width = 480
n_classes = 12  # 11 classes + 1 un-classified class
batch_size = 2

validate = True


m = Models.testNet.testNet(nClasses=n_classes, input_height=480, input_width=480)

m.compile(loss="categorical_crossentropy",
          optimizer=optimizers.SGD(lr=0.01, momentum=0.9),
          metrics=["accuracy"])

G_rgb = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_mask_path,
                                           train_f_path,
                                           batch_size, n_classes, input_height, input_width,
                                           output_height=input_height, output_width=input_width, data_aug=True)

G_f = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_mask_path,
                                           train_f_path,
                                           batch_size, n_classes, input_height, input_width,
                                           output_height=input_height, output_width=input_width, data_aug=False)

if validate:
    val_images_path = ["/data/fangwen/mix_validation2/rgb_img/"]
    val_segs_path = ["/data/fangwen/mix_validation2/3_greylabel/"]
    val_mask_path = ["/data/fangwen/mix_validation2/2_mask/"]
    val_f_path = ["/data/fangwen/mix_validation2/"]


    G2_rgb = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_mask_path,
                                                val_f_path,
                                                batch_size, n_classes, input_height,input_width,
                                                output_height=input_height, output_width=input_width, data_aug=False)

    G2_f = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_mask_path,
                                                val_f_path,
                                                batch_size, n_classes, input_height,input_width,
                                                output_height=input_height, output_width=input_width, data_aug=False)

    callbacks = [EarlyStopping(monitor='val_loss', patience=20),
                 ModelCheckpoint(filepath=os.path.join(save_weights_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                    monitor='val_loss', save_best_only=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001),
                 TensorBoard(log_dir='./board', histogram_freq=0, write_graph=True, write_images=True)]

    class_weights = np.array([201.791,  # Powerline
                              0.0701,  # Low Vegetation
                              0.122,  # Impervious Surface
                              3.126,  # Vehicles
                              0.924,  # Urban Furniture
                              0.186,  # Roof
                              1.434,  # Facade
                              1.088,  # Bush/Hedge
                              0.138,  # Tree
                              0.117,  # Dirt/Gravel
                              10.019,  # Vertical Surface
                              0.001]  # Void
                             )

    m.fit_generator({'main_input': G_rgb, 'aux_input': G_rgb},
                    steps_per_epoch=6555//batch_size,
                    callbacks=callbacks,
                    validation_data={'main_input': G2_rgb, 'aux_input': G2_rgb},
                    validation_steps=590//batch_size,
                    epochs=10000,
                    class_weight=class_weights)
    m.save_weights(save_weights_path + "end_weights.hdf5")

else:
    m.fit_generator(G, steps_per_epoch=15447//batch_size, epochs=1)
    m.save_weights(save_weights_path + "end_weights.hdf5")