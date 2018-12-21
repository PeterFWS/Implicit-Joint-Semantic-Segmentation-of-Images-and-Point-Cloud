#### Parameters
# There are several parameters than can be tuned to use this script with
# different datasets. The default parameters are suitable for the ISPRS
# dataset, but you can change them to work with your data.
# e.g.
#   * Binary classification: `N_CLASSES = 2`
#   * Multi-spectral data (e.g. IRRGB): `IN_CHANNELS = 4`
#   * New folder naming convention : `DATA_FOLDER = MAIN_FOLDER + 'sentinel2/sentinel2_img_{}.tif'`
import torch

WINDOW_SIZE = (256, 256)  # Patch size
STRIDE = 32  # Stride for testing
IN_CHANNELS = 3  # Number of input channels (e.g. RGB)
FOLDER = "/data/ISPRS_BENCHMARK_DATASETS/"  # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10  # Number of samples in a mini-batch
NUM_EPOCHS = 50  # TODO: Number of epochs (training)
NUM_SAVE_EPOCHS = 50 # TODO: Number of saving iterations

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]  # Label names
N_CLASSES = len(LABELS)  # Number of classes
WEIGHTS = torch.ones(N_CLASSES)  # Weights for class balancing
CACHE = True  # Store the dataset in-memory

DATASET = 'Potsdam'  # TODO Vaihingen, Potsdam

SAVE_TRAINED_WEIGHTS = './segnet_final.pth'  # TODO: save trained weights
#LOAD_TRAINED_WEIGHTS = []
#LOAD_TRAINED_WEIGHTS.append('./trained_Potsdam_20180726/segnet_final.pth')  # TODO: after training weights are stored, load this or other pretrained file

# Prediction based on several weight files
import os
weights_list = []
for file in os.listdir("./trained_Potsdam_20180726/"):
    if file.startswith("segnet256_"):
        weights_list.append(file)
LOAD_TRAINED_WEIGHTS = sorted(weights_list)

PREDICTION_DATA_FOLDER = "/data/Orthophoto/Ortho_flights08-17-v{}_3Channel.tif"  # size: 5413 x 16316 x 3px
predict_ids = ['1']  # TODO: version1: 1 (3 Channels, saved by io.imsave), version2: 2 (3 Channels, saved by scipy.misc.toimage)


if DATASET == 'Potsdam':
    MAIN_FOLDER = FOLDER + 'Potsdam/'
    DATA_FOLDER = MAIN_FOLDER + '2_Ortho_RGB/top_potsdam_{}_RGB.tif'
    LABEL_FOLDER = MAIN_FOLDER + '5_Labels_all/top_potsdam_{}_label.tif'  # '5_Labels_for_participants/top_potsdam_{}_label.tif'
    ERODED_FOLDER = MAIN_FOLDER + '5_Labels_all_noBoundary/top_potsdam_{}_label_noBoundary.tif'  # '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'
    train_ids = ['6_13', '2_10', '7_12', '5_10', '2_11', '6_9', '4_13', '5_15', '7_8', '3_10', '4_10', '7_10',
                 '4_12', '6_8', '2_14', '5_12', '6_10', '3_14', '6_7', '3_12', '5_13', '4_11', '7_13', '2_13',
                 '6_12', '4_15', '7_7', '5_11', '6_15', '7_11', '2_12', '6_11', '5_14', '3_13']  # train_ids.sort()
    test_ids = ['7_9', '3_11', '4_14', '6_14']
elif DATASET == 'Vaihingen':
    MAIN_FOLDER = FOLDER + 'Vaihingen/ISPRS_semantic_labeling_Vaihingen/'
    DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'
    train_ids = ['24', '30', '1', '15', '3', '23', '26', '7', '31', '14', '20', '6', '13', '28', '4', '33', '17', '29',
                 '10', '21', '32', '16', '8', '34', '22', '11', '38', '2', '27']
    test_ids = ['5', '21', '12', '37']


# ISPRS color palette
palette = {0: (255, 255, 255),  # Impervious surfaces (white)
           1: (0, 0, 255),  # Buildings (blue)
           2: (0, 255, 255),  # Low vegetation (cyan)
           3: (0, 255, 0),  # Trees (green)
           4: (255, 255, 0),  # Cars (yellow)
           5: (255, 0, 0),  # Clutter (red)
           6: (0, 0, 0)}  # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}
print("\nISPRS color palette used.")