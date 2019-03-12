# coding=utf-8
import Models
import LoadBatches_multi_stream
import os
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import numpy as np

# Global parameters
# TODO: attention!!! path has to end with "/"
# TODO: In case you have data saved in different folder, we use "list" to save path here
train_images_path = ["/data/fangwen/mix_train/rgb_img/"]
train_segs_path = ["/data/fangwen/mix_train/3_greylabel/"]
train_mask_path = ["/data/fangwen/mix_train/2_mask/"]

save_weights_path = "./weights/"


input_height = 480
input_width = 480
n_classes = 12  # 11 classes + 1 un-classified class
batch_size = 1

validate = True


m = Models.testNet.testNet(nClasses=n_classes, input_height=480, input_width=480)

m.compile(loss="categorical_crossentropy",
          optimizer=optimizers.SGD(lr=0.001, momentum=0.9),
          metrics=["accuracy"])

G1 = LoadBatches_multi_stream.imageSegmentationGenerator(train_images_path, train_segs_path, train_mask_path,
                                           ["/data/fangwen/mix_train/"],
                                           batch_size, n_classes, input_height, input_width,
                                           output_height=input_height, output_width=input_width, data_aug=True)

if validate:
    val_images_path = ["/data/fangwen/mix_validation/rgb_img/"]
    val_segs_path = ["/data/fangwen/mix_validation/3_greylabel/"]
    val_mask_path = ["/data/fangwen/mix_validation/2_mask/"]



    G2 = LoadBatches_multi_stream.imageSegmentationGenerator(val_images_path, val_segs_path, val_mask_path,
                                                ["/data/fangwen/mix_validation/"],
                                                batch_size, n_classes, input_height,input_width,
                                                output_height=input_height, output_width=input_width, data_aug=False)

    callbacks = [EarlyStopping(monitor='val_loss', patience=15),
                 ModelCheckpoint(filepath=os.path.join(save_weights_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                    monitor='val_loss', save_best_only=False),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001),
                 TensorBoard(log_dir='./board', histogram_freq=0, write_graph=True, write_images=True)]

    class_weights = np.array([209.763,  # Powerline
                              0.202,  # Low Vegetation
                              0.305,  # Impervious Surface
                              8.547,  # Vehicles
                              2.462,  # Urban Furniture
                              0.657,  # Roof
                              4.267,  # Facade
                              2.088,  # Bush/Hedge
                              0.417,  # Tree
                              0.502,  # Dirt/Gravel
                              7.249,  # Vertical Surface
                              0.192]  # Void
                             )

    # class_weights = np.array([4.34011007e+03, 4.19371021e+00, 6.32311425e+00, 1.76848514e+02,
    #        5.09546817e+01, 1.36017262e+01, 8.83042539e+01, 4.32105208e+01,
    #        8.64246763e+00, 1.03978730e+01, 1.50001822e+02, 3.97847443e+00])


    m.fit_generator(G1,
                    steps_per_epoch=15922//batch_size,
                    callbacks=callbacks,
                    validation_data=G2,
                    validation_steps=3197//batch_size,
                    epochs=10000,
                    class_weight=class_weights)

    m.save_weights(save_weights_path + "end_weights.hdf5")
