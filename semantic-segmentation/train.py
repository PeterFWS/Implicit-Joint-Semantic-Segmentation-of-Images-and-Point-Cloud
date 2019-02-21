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
train_batch_size = 2

train_mode = "BGR"
validate = True

# compile model
if train_mode == "BGR":
    nchannel = 3
    pre_train = True

    train_f_path = None
    val_f_path = None

elif train_mode == "multi_modality":
    nchannel = 4
    pre_train = False

    train_f_path = ["/data/fangwen/mix_train/"]
    val_f_path = ["/data/fangwen/mix_validation/"]


m = Models.Segnet.segnet_indices_pooling(n_classes, input_height=input_height, input_width=input_width,
                                             nchannel=nchannel, pre_train=pre_train)

# m = Models.UNet.U_Net(n_classes, input_height=input_height, input_width=input_width, nchannel=nchannel)

# m = Models.TernausNet.get_unet_pre_trained(n_classes, input_height=input_height, input_width=input_width, nchannel=nchannel)

m.compile(loss="categorical_crossentropy",
          optimizer=optimizers.SGD(lr=0.1, decay=0.0005, momentum=0.9),
          metrics=["accuracy"])

G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_mask_path,
                                           train_f_path,
                                           train_batch_size, n_classes, input_height, input_width,
                                           output_height=input_height, output_width=input_width)

if validate:
    val_images_path = ["/data/fangwen/mix_validation/rgb_img/"]

    val_segs_path = ["/data/fangwen/mix_validation/3_greylabel/"]

    val_mask_path = ["/data/fangwen/mix_validation/2_mask/"]

    val_batch_size = train_batch_size

    G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_mask_path,
                                                val_f_path,
                                                val_batch_size, n_classes, input_height,input_width,
                                                output_height=input_height, output_width=input_width)

    callbacks = [EarlyStopping(monitor='val_loss', patience=50),
                 ModelCheckpoint(filepath=os.path.join(save_weights_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                    monitor='val_loss', save_best_only=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001),
                 TensorBoard(log_dir='./board', histogram_freq=0, write_graph=True, write_images=True)]

    # used for weighting the loss function (during training only)
    # oblique image only
    # class_weights = np.array([193.29048864, 0.35519734, 0.60095986, 10.93489853, 4.12920855, 0.96278714, 3.96056851,
    #                  2.64414337, 0.62389212, 0.91866959, 7.50542126, 0.37203563])


    # num_classes
    # array([631779, 653834019, 433645557, 15504741, 53812335, 201591354,
    #        31051623, 63456546, 317269386, 263706855, 18279714, 689206491])

    # mix of nadir/ oblique imagery
    # inverse of frequency
    # class_weights = array([4.34011007e+03, 4.19371021e+00, 6.32311425e+00, 1.76848514e+02,
    #        5.09546817e+01, 1.36017262e+01, 8.83042539e+01, 4.32105208e+01,
    #        8.64246763e+00, 1.03978730e+01, 1.50001822e+02, 3.97847443e+00])

    class_weights = np.array([4340.11,  # Powerline
                              4.19,  # Low Vegetation
                              6.32,  # Impervious Surface
                              176.84,  # Vehicles
                              50.95,  # Urban Furniture
                              13.60,  # Roof
                              88.30,  # Facade
                              43.21,  # Bush/Hedge
                              8.64,  # Tree
                              10.39,  # Dirt/Gravel
                              150.00,  # Vertical Surface
                              1.0]  # Void
                             )

    m.fit_generator(G,
                    steps_per_epoch=3967//train_batch_size,
                    callbacks=callbacks,
                    validation_data=G2,
                    validation_steps=789//val_batch_size,
                    epochs=10000,
                    class_weight=class_weights)
    m.save_weights(save_weights_path + "end_weights.hdf5")

else:
    m.fit_generator(G, steps_per_epoch=1513//train_batch_size, epochs=1)
    m.save_weights(save_weights_path + "end_weights.hdf5")