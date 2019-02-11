# coding=utf-8
import Models
import LoadBatches
import os
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import numpy as np

# Global parameters
# TODO: attention!!! path has to end with "/"
train_images_path = ["/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_nadir/train_set/rgb_img/",
                     "/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_oblique/train_set/rgb_img/"]

train_segs_path = ["/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_nadir/train_set/3_greylabel/",
                   "/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_oblique/train_set/3_greylabel/"]

train_mask_path = ["/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_nadir/train_set/2_mask/",
                   "/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_oblique/train_set/2_mask/"]

save_weights_path = "./weights/"

input_height = 512
input_width = 512
n_classes = 12  # 11 classes + 1 un-classified class
train_batch_size = 8

# Training mode
# TODO: define your training mode
train_mode = "multi_modality"
validate = True

if train_mode == "BGR":
    m = Models.Segnet.segnet_indices_pooling(n_classes, input_height=input_height, input_width=input_width,
                                             nchannel=3, pre_train=True)
    train_f_path = None
    val_f_path = None

elif train_mode == "multi_modality":
    m = Models.Segnet.segnet_indices_pooling(n_classes, input_height=input_height, input_width=input_width,
                                             nchannel=75, pre_train=False)
    train_f_path = ["/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_nadir/train_set/",
                    "/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_oblique/train_set/"]

    val_f_path = ["/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_nadir/validation_set/",
                  "/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_oblique/validation_set/"]


# compile model
m.compile(loss="categorical_crossentropy",
          optimizer=optimizers.SGD(lr=0.01, decay=0.0005, momentum=0.9),
          metrics=["accuracy"])
G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_mask_path,
                                           train_f_path,
                                           train_batch_size, n_classes, input_height, input_width,
                                           output_height=input_height, output_width=input_width)

if validate:
    val_images_path = ["/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_nadir/validation_set/rgb_img/",
                       "/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_oblique/validation_set/rgb_img/"]

    val_segs_path = ["/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_nadir/validation_set/3_greylabel/",
                     "/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_oblique/validation_set/3_greylabel/"]

    val_mask_path = ["/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_nadir/validation_set/2_mask/",
                     "/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_oblique/validation_set/2_mask/"]

    val_batch_size = train_batch_size

    G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_mask_path,
                                                val_f_path,
                                                val_batch_size, n_classes, input_height,input_width,
                                                output_height=input_height, output_width=input_width)

    callbacks = [EarlyStopping(monitor='val_loss', patience=50),
                 ModelCheckpoint(filepath=os.path.join(save_weights_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                 monitor='val_loss', save_best_only=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001),
                 TensorBoard(log_dir=os.path.join('/home/fangwen/ShuFangwen/source/image-segmentation-keras/board'),
                             histogram_freq=0, write_graph=True, write_images=True)]

    # used for weighting the loss function (during training only)
    # oblique image only
    # class_weights = np.array([193.29048864, 0.35519734, 0.60095986, 10.93489853, 4.12920855, 0.96278714, 3.96056851,
    #                  2.64414337, 0.62389212, 0.91866959, 7.50542126, 0.37203563])


    # num_classes
    # array([   2261700, 4607688099, 3408115581,  123947364,  389877027,
    #        1361819178,  160966275,  363013626, 2057536023, 1950714060,
    #          96768021, 1306817526])
    # mix of nadir/ oblique imagery
    # inverse of frequency
    # class_weights = np.array([6.99894968e+03, 3.43545920e+00, 4.64465600e+00, 1.27711667e+02,
    #    4.06013265e+01, 1.16238079e+01, 9.83406274e+01, 4.36058686e+01,
    #    7.69343735e+00, 8.11473337e+00, 1.63582187e+02, 1.21130335e+01])
    class_weights = np.array([6998.94968,  # Powerline
                              3.43545920,  # Low Vegetation
                              4.64465600,  # Impervious Surface
                              127.711667,  # Vehicles
                              40.6013265,  # Urban Furniture
                              11.6238079,  # Roof
                              98.3406274,  # Facade
                              43.6058686,  # Bush/Hedge
                              7.69343735,  # Tree
                              8.11473337,  # Dirt/Gravel
                              163.582187,  # Vertical Surface
                              12.1130335]  # Void
                             )

    m.fit_generator(G,
                    steps_per_epoch=3090//train_batch_size,
                    callbacks=callbacks,
                    validation_data=G2,
                    validation_steps=1071//val_batch_size,
                    epochs=10000,
                    class_weight=class_weights)
    m.save_weights(save_weights_path + "end_weights.hdf5")

else:
    m.fit_generator(G, steps_per_epoch=1513//train_batch_size, epochs=1)
    m.save_weights(save_weights_path + "end_weights.hdf5")