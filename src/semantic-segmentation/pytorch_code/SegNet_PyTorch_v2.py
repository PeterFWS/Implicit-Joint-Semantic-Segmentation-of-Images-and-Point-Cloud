# coding: utf-8
#
# Semantic segmentation of aerial images with deep networks.
# This script presents a straightforward PyTorch implementation of a 
# Fully Convolutional Network for semantic segmentation of aerial images. 
# More specifically, we aim to automatically perform scene 
# interpretation of images taken from a plane or a satellite by 
# classifying every pixel into several land cover classes.
# We are going to use the [SegNet architecture]
# (http://mi.eng.cam.ac.uk/projects/segnet/) to segment aerial images. 
# The weights can be trained on images from the
# [ISPRS 2D Semantic Labeling dataset]
# (http://www2.isprs.org/commissions/comm3/wg4/results.html) over the 
# cities of Vaihingen (no RGB) and Potsdam (amongst others RGB).
# Considered classes: roads, buildings, low veg., trees, cars, clutter
# 
# This work is a PyTorch implementation of the baseline presented in the
# git repository: https://github.com/nshaud/DeepNetsForEO, 17.06.2018. 
# The git repository bases on the work: 
# ["Beyond RGB: Very High Resolution Urban Remote Sensing With 
# Multimodal Deep Networks "]
# (https://hal.archives-ouvertes.fr/hal-01636145), *Nicolas Audebert*, 
# *Bertrand Le Saux* and *Sébastien Lefèvre*, ISPRS Journal, 2018.
# Modified work, current git repository:
# https://bitbucket.org/laupheimer/deepnetsforeo/

### TODO:
# - You should create a function that stores all settings and results into a file.
#   Like train ids, training time, achieved accuracy, etc., ...
# - You should make it possible to use pretrained weights on VGG16

### Imports and Stuff
import my_functions as mf

import os
import sys
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener
from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
import torch.utils.data as data
import torch.optim as optim
import torch.nn.init


#### Parameters
# There are several parameters than can be tuned to use this script with
# different datasets. The default parameters are suitable for the ISPRS
# dataset, but you can change them to work with your data.
# e.g.
#   * Binary classification: `N_CLASSES = 2`
#   * Multi-spectral data (e.g. IRRGB): `IN_CHANNELS = 4`
#   * New folder naming convention : `DATA_FOLDER = MAIN_FOLDER + 'sentinel2/sentinel2_img_{}.tif'`
flag_test = True  # TODO: True: pytorch mode, False: normal mode
if flag_test:  # TODO: import Global_Variables(_Test) also in my_functions.py!!!
    import Global_Variables_Test as GLOB
else:
    import Global_Variables as GLOB


########################################################################
################################ CHECKS ################################
########################################################################
img2bePredicted = io.imread(GLOB.PREDICTION_DATA_FOLDER.format(GLOB.predict_ids[0]))
if img2bePredicted.shape[-1] > 3:
    sys.exit('Error! Image that should be predicted has to have less than 4 channels.')
else:
    print('Image that should be predicted has', img2bePredicted.shape[-1],  'channels. Image can be processed. \n')

### TESTING AREA
#test_img = 1/255 * np.asarray(io.imread(PREDICTION_DATA_FOLDER.format(predict_ids[0])), dtype='float32')
#t1 = test_img[:,:,:3]
#io.imsave('test1.tif', t1)
#t2 = test_img[:,:,1:4]
#io.imsave('test2.tif', t2)

#T = 1/255 * np.asarray(io.imread('/data/Ortho-flights08-17-5cmGSD.tif'), dtype='uint16')
#T1 = T[:,:,:3]
#io.imsave('test_Ortho.tif', T1)
#scipy.misc.toimage(test_img[:,:,0:3], cmin=0.0, cmax=255).save('outfile.jpg')



########################################################################
############################## PROCESSING ##############################
########################################################################
print("\n\n------------- SETTINGS ---------------\n")
print("Tiles (IDs) for training: ", GLOB.train_ids)
print("Tiles (IDs) for testing: ", GLOB.test_ids)
print("File to be predicted/segmented: ", GLOB.PREDICTION_DATA_FOLDER.format(GLOB.predict_ids[0]))
print("Testing Mode?: ", str(flag_test))



print("\n\n --------------- NETWORK PREPARATION -----------------\n")
### Instantiate the network
# We can now instantiate the network using the specified parameters. 
# By default, the weights will be initialized using the [He policy]
# (https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf).
print("Instantiate net...")
net = mf.SegNet()
print("Done. \n")


#### Download pretrained VGG-16 weights from PyTorch and load into net 
# This step is optional but it makes the network converge faster. 
# We skip the weights from VGG-16 that have no counterpart in SegNet.
print("Download VGG16 weights and load into net...")
vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
    weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
mapped_weights = {}
for k_vgg, k_segnet in zip(vgg16_weights.keys(), net.state_dict().keys()):
    if "features" in k_vgg:
        mapped_weights[k_segnet] = vgg16_weights[k_vgg]
        print("Mapping {} to {}".format(k_vgg, k_segnet))
try:
    #TODO: Does not work atm...
    net.load_state_dict(mapped_weights)
    print("Loaded VGG-16 weights in SegNet!\n")
except:
    # Ignore missing keys
    print("\nCould not load VGG-16 weights in SegNet!\n")
    pass

print("Done.\n")


### Load Network on GPU
print("Load net on GPU...")
net.cuda()
print("Done. \n")



print("\n\n --------------- DATA LOADING AND PREPARATION -----------------\n")
### Loading the data 
# We now create a train/pytorch split. If you want to use another dataset,
# you have to adjust the method to collect all filenames. 
# In our case, we specify a fixed train/pytorch split for the demo.
print("Load data...\n")
if GLOB.DATASET == 'Potsdam':
    print('Potsdam data set used. \n')
    all_files = sorted(glob(GLOB.LABEL_FOLDER.replace('{}', '*'))) # TODO: sorted does not work correctly
    all_ids = ["_".join(f.split('_')[6:8]) for f in all_files] # TODO: [6:8] has to be changed, when data structure changes (LABEL_FOLDER)
elif GLOB.DATASET == 'Vaihingen':
    print('Vaihingen data set used. \n')
    all_files = sorted(glob(GLOB.LABEL_FOLDER.replace('{}', '*')))
    all_ids = [f.split('area')[-1].split('.')[0] for f in all_files]

train_set = mf.ISPRS_dataset(GLOB.train_ids, cache=GLOB.CACHE)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=GLOB.BATCH_SIZE)
print("Done.\n")



print("\n\n --------------- DESIGNING THE OPTIMIZER-----------------\n")
### Designing the optimizer
# We use the standard Stochastic Gradient Descent algorithm to optimize 
# the network's weights. The encoder is trained at half the learning 
# rate of the decoder, as we rely on the pre-trained VGG-16 weights. 
# We use the ``torch.optim.lr_scheduler`` to reduce the learning rate by 
# 10 after 25, 35 and 45 epochs.
base_lr = 0.01
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params': [value], 'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we could have VGG-16 weights as initialization, if it worked)
        params += [{'params': [value], 'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
print("Done.\n")



### Training the network
# Let's train the network for NUM_EPOCHS epochs.
# Depending on your GPU, this might take from a few hours (Titan Pascal)
# to a full day (old K20).
print("\n\n------------- TRAINING ---------------\n")
mf.train(net, optimizer, GLOB.NUM_EPOCHS, train_loader, scheduler, weights=GLOB.WEIGHTS,
         save_epoch=GLOB.NUM_SAVE_EPOCHS, filename=GLOB.SAVE_TRAINED_WEIGHTS)
print("Done.\n")


print("\n\n --------------- TESTING/INFERENCE-----------------\n")
### Loading trained weights
# Now that the training has ended, we can load the final weights and 
# pytorch the network using a reasonable stride, e.g. half or a quarter of
# the window size. Inference time depends on the chosen stride, e.g. a 
# step size of 32 (75% overlap) will take ~15 minutes, but no overlap 
# will take only one minute or two.
print("Loading trained weights from file:")
net.load_state_dict(torch.load(GLOB.LOAD_TRAINED_WEIGHTS))
print(GLOB.LOAD_TRAINED_WEIGHTS)
print("Done. \n")


### Testing the network
print("\n\n------------- TESTING ---------------\n")
print("Predicting segmentation (ground truth given)...")
all_acc, all_acc_e, all_preds, _, _ = mf.test(net, GLOB.test_ids, output_all=True, stride=GLOB.STRIDE,
                                              batch_size=GLOB.BATCH_SIZE, window_size=GLOB.WINDOW_SIZE)
print("Done.\n")

# Saving the results
# We can save the resulting tiles for qualitative assessment.
print("\n\n------------- SAVING GT INFERENCE ---------------\n")
print("Save segmented images...\n")
for p, id_ in tqdm(zip(all_preds, GLOB.test_ids)):
    img = mf.convert_to_color(p)
    #plt.imshow(img) and plt.show()
    io.imsave('inference_gt_tile_{}.png'.format(id_), img)
print("Done.\n")


### Predicting
print("Predicting segmentation...")
all_preds = mf.predictOrtho(net, GLOB.predict_ids, stride=GLOB.STRIDE,
                            batch_size=GLOB.BATCH_SIZE, window_size=GLOB.WINDOW_SIZE)
print("Done.\n")

# Saving the results
# We can save the resulting tiles for qualitative assessment.
print("\n\n------------- SAVING ---------------\n")
print("Save segmented images...\n")
for p, id_ in tqdm(zip(all_preds, GLOB.predict_ids)):
    img = mf.convert_to_color(p)
    #plt.imshow(img) and plt.show()
    io.imsave('inference_tile_{}.tif'.format(id_), img)
print("Done.\n")


print("\n\n-------------------------------------------\n")
print("------------- END OF SCRIPT ---------------\n")
print("-------------------------------------------\n")

