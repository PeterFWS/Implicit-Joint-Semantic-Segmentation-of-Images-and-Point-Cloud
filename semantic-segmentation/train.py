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
train_images_path = ["/data/fangwen/mix_train/rgb_img/"]
train_segs_path = ["/data/fangwen/mix_train/3_greylabel/"]
train_mask_path = ["/data/fangwen/mix_train/2_mask/"]
save_weights_path = "./weights/"

input_height = 480
input_width = 480
n_classes = 12  # 11 classes + 1 un-classified class
batch_size = 2

# train_mode = "multi_modality"
validate = True

# compile model
# if train_mode == "BGR":
train_f_path = None
val_f_path = None

# elif train_mode == "multi_modality":
# train_f_path = ["/data/fangwen/mix_train/"]
# val_f_path = ["/data/fangwen/mix_validation/"]


# m = Models.Segnet.segnet_indices_pooling(n_classes,
#                                          input_height=input_height, input_width=input_width,
#                                          nchannel=4, pre_train=False)

m = Models.PSPnet.PSPNet50(input_shape=(480, 480, 3), n_labels=n_classes)

m.compile(loss="categorical_crossentropy",
          optimizer=optimizers.SGD(lr=0.01, momentum=0.9),
          metrics=["accuracy"])

G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_mask_path,
                                           train_f_path,
                                           batch_size, n_classes, input_height, input_width,
                                           output_height=input_height, output_width=input_width, data_aug=True)

if validate:
    val_images_path = ["/data/fangwen/mix_validation/rgb_img/"]

    val_segs_path = ["/data/fangwen/mix_validation/3_greylabel/"]

    val_mask_path = ["/data/fangwen/mix_validation/2_mask/"]


    G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_mask_path,
                                                val_f_path,
                                                batch_size, n_classes, input_height,input_width,
                                                output_height=input_height, output_width=input_width, data_aug=False)

    callbacks = [EarlyStopping(monitor='val_loss', patience=20),
                 ModelCheckpoint(filepath=os.path.join(save_weights_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                    monitor='val_loss', save_best_only=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001),
                 TensorBoard(log_dir='./board', histogram_freq=0, write_graph=True, write_images=True)]

    # used for weighting the loss function (during training only)
    # mix of nadir/ oblique imagery
    # num_classes
    # array([631779, 653834019, 433645557, 15504741, 53812335, 201591354,
    #        31051623, 63456546, 317269386, 263706855, 18279714, 689206491])

    # inverse of frequency
    # array([4.34011007e+03, 4.19371021e+00, 6.32311425e+00, 1.76848514e+02,
    #        5.09546817e+01, 1.36017262e+01, 8.83042539e+01, 4.32105208e+01,
    #        8.64246763e+00, 1.03978730e+01, 1.50001822e+02, 3.97847443e+00])
    #
    # ignore void, inverse of frequency
    # array([3.24921200e+03, 3.13961013e+00, 4.73378287e+00, 1.32397175e+02,
    #        3.81470886e+01, 1.01828966e+01, 6.61087476e+01, 3.23494429e+01,
    #        6.47016069e+00, 7.78434034e+00, 1.12298470e+02])

    # median frequency balancing
    # array([2.09763145e+02, 2.02687450e-01, 3.05604307e-01, 8.54731788e+00,
    #        2.46270581e+00, 6.57389056e-01, 4.26785904e+00, 2.08842048e+00,
    #        4.17701663e-01, 5.02542681e-01, 7.24978246e+00, 1.92284826e-01])

    # ignore void, median frequency balancing
    # array([1.00441050e+02, 9.70529892e-02, 1.46332748e-01, 4.09271887e+00,
    #        1.17921934e+00, 3.14778113e-01, 2.04358226e+00, 1.00000000e+00,
    #        2.00008412e-01, 2.40632903e-01, 3.47141897e+00])

    # median frequency balancing
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

    # # median frequency balancing, no void
    # class_weights = np.array([100.441,  # Powerline
    #                           0.097,  # Low Vegetation
    #                           0.146,  # Impervious Surface
    #                           4.092,  # Vehicles
    #                           1.179,  # Urban Furniture
    #                           0.314,  # Roof
    #                           2.043,  # Facade
    #                           1.000,  # Bush/Hedge
    #                           0.200,  # Tree
    #                           0.240,  # Dirt/Gravel
    #                           3.471]  # Vertical Surface
    #                          )

    m.fit_generator(G,
                    steps_per_epoch=15922//batch_size,
                    callbacks=callbacks,
                    validation_data=G2,
                    validation_steps=3197//batch_size,
                    epochs=10000,
                    class_weight=class_weights)
    m.save_weights(save_weights_path + "end_weights.hdf5")

else:
    m.fit_generator(G, steps_per_epoch=15447//batch_size, epochs=1)
    m.save_weights(save_weights_path + "end_weights.hdf5")