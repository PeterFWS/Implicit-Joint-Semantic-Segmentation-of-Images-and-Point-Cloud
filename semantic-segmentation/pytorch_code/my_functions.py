########################################################################
######################### FUNCTIONS AND CLASSES ########################
########################################################################

### TODO:
# Many functions make use of global variables as it was a quick and dirty reassembling of existing code.
# We should change that.


### Visualizing the dataset
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import random
import itertools
# from IPython.display import clear_output
# Matplotlib
import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib qt4')
# Torch imports
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler
from torch.autograd import Variable
from skimage import io
from tqdm import tqdm_notebook as tqdm

flag_test = False  # TODO: True: pytorch mode, False: normal mode
if flag_test:  # TODO: import Global_Variables(_Test) also in SegNet_PyTorch_v2.py!!!
    import Global_Variables_Test as GLOB
else:
    import Global_Variables as GLOB


def convert_to_color(arr_2d, palette=GLOB.palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color(arr_3d, palette=GLOB.invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


### Utils
def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(predictions, gts, label_values=GLOB.LABELS, filepath=None):
    cm = confusion_matrix(
        gts,
        predictions,
        range(len(label_values)))

    # print("Confusion matrix :")
    # print(cm)
    # print("---")

    # print and plot confusion matrix
    if filepath is None:
        print('No file path given. Default file path for storing the confusion matrix is used: /home/dominik/deepnetsforeo/')
        plot_confusion_matrix(cm, class_names=GLOB.LABELS, filepath='/home/dominik/deepnetsforeo/',
                              filename='confusion_matrix.png', flag_normalize=True, title='Confusion Matrix',
                              cmap='plt.cm.Blues')
    else:
        plot_confusion_matrix(cm, class_names=GLOB.LABELS, filepath=filepath,
                              filename='confusion_matrix.png', flag_normalize=True, title='Confusion Matrix',
                              cmap='plt.cm.Blues')

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = np.trace(cm)  # sum([cm[x][x] for x in range(len(cm))]) # main diagonal elements
    accuracy *= 100 / float(total)  # overall accuracy
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))

    # # Compute F1 score
    # F1Score = np.zeros(len(label_values))
    # for i in range(len(label_values)):
    #     try:
    #         F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
    #     except:
    #         # Ignore exception if there is no element in class i for pytorch set
    #         pass
    # print("F1Score :")
    # for l_id, score in enumerate(F1Score):
    #     print("{}: {}".format(label_values[l_id], score))

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: " + str(kappa))
    print("---")

    return accuracy


# Dataset class
# We define a PyTorch dataset (```torch.utils.data.Dataset```) that
# loads all the tiles in memory and performs random sampling. Tiles are
# stored in memory on the fly. The dataset also performs random data
# augmentation (horizontal and vertical flips) and normalizes the data
# in [0, 1].
class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=GLOB.DATA_FOLDER, label_files=GLOB.LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = [GLOB.DATA_FOLDER.format(id) for id in ids]
        self.label_files = [GLOB.LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        # Default epoch size is 10 000 samples
        return 10000

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = 1 / 255 * np.asarray(io.imread(self.data_files[random_idx]).transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, GLOB.WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        data_p, label_p = self.data_augmentation(data_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))


# Network definition
# We can now define the Fully Convolutional network based on the SegNet
# architecture. We could use any other network as drop-in replacement,
# provided that the output has dimensions `(N_CLASSES, W, H)` where
# `W` and `H` are the sliding window dimensions (i.e. the network should
# preserve the spatial dimensions).
class SegNet(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(self, in_channels=GLOB.IN_CHANNELS, out_channels=GLOB.N_CLASSES):
        super(SegNet, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)

        self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(512)

        self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(256)

        self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(128)

        self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(64)

        self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(64, out_channels, 3, padding=1)

        self.apply(self.weight_init)

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)

        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)

        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)

        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)

        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)
        #------------------------------------------------------------------------------------------------------------#
        # ------------------------------------------------------------------------------------------------------------#
        # Decoder block 5
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))

        # Decoder block 4
        x = self.unpool(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))

        # Decoder block 3
        x = self.unpool(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))

        # Decoder block 2
        x = self.unpool(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        # Decoder block 1
        x = self.unpool(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = F.log_softmax(self.conv1_1_D(x),
                          dim=1)  # original version without 'dim=x'; update: dim=1 und dim=-3 performen richtig gut
        return x


def save_figure(fig, path, filename):
    """ Save figure fig in directory specified by path under filename """

    # create desired subfolder (TODO maybe: use createNewSubdirectory)
    if not os.path.exists(path):
        os.makedirs(path)

    # save figure in desired subfolder
    fig.savefig(os.path.join(path, filename), bbox_inches='tight',
                dpi=300)  # TODO: set dpi (300dpi is good default choice)

    plt.close(fig)
    return 0


## lambda functions
format_float_2f = lambda x: "%.2f" % x


def plot_confusion_matrix(conf_mat, class_names, filepath, filename,
                          flag_normalize=False, title='Confusion Matrix',
                          cmap='plt.cm.Blues'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    # Arguments
        conf_mat: output of sklearn.confusion_matrix
        class_names: list of strings
        flag_normalize: flag for controlling normalization of confusion matrix
        cmap: string (!) defining the color style (string in order to
            avoid problems with matplotlib.use)

    # Returns
        ---

    adopted by source:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    # evaluate string of colormap --> matplotlib.colors.<?>
    cmap = eval(cmap)

    # normalization
    if flag_normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix created.")
        v_min = 0.0
        v_max = 1.0
    else:
        print('Confusion Matrix, without Normalization created.')
        v_min = 0.0
        v_max = 100.0

    # np.set_printoptions(precision=2)
    # print(conf_mat)

    # plot content of confusion matrix
    fig = plt.figure()
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i + 0.075, format_float_2f(conf_mat[i, j]),  # +0.075 for centering text within tiles
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

    # add any other things you want to the figure (after colorbar!)
    cax = plt.imshow(conf_mat, interpolation='nearest', vmin=v_min, vmax=v_max, cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')

    # add colorbar
    cbar = plt.colorbar(orientation="vertical", ticks=[v_min, v_max / 2, v_max])
    if flag_normalize:
        cbar.ax.set_yticklabels(['0.0', '0.5', '1.0'])  # vertical colorbar
    else:
        cbar.ax.set_yticklabels(['0.0', '50.0', '100.0'])  # vertical colorbar

    # save figure
    save_figure(fig, filepath, filename)


def test(net, test_ids, output_all=False, stride=GLOB.WINDOW_SIZE[0], batch_size=GLOB.BATCH_SIZE, window_size=GLOB.WINDOW_SIZE):
    # Use the network on the pytorch set
    test_images = (1 / 255 * np.asarray(io.imread(GLOB.DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(GLOB.LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)  # RGB image
    eroded_labels = (convert_from_color(io.imread(GLOB.ERODED_FOLDER.format(id))) for id in
                     test_ids)  # grayscale image with class indices as grayvalues
    all_preds = []
    all_gts = []
    all_gts_e = []

    # Switch the network to inference mode
    net.eval()

    for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (GLOB.N_CLASSES,))

        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(
                tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                     leave=False)):
            # # Display in progress results
            # if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
            #         _pred = np.argmax(pred, axis=-1)
            #         fig = plt.figure()
            #         fig.add_subplot(1,3,1)
            #         plt.imshow(np.asarray(255 * img, dtype='uint8'))
            #         fig.add_subplot(1,3,2)
            #         plt.imshow(convert_to_color(_pred))
            #         fig.add_subplot(1,3,3)
            #         plt.imshow(gt)
            #         clear_output()
            #         plt.show()

            # Build the tensor
            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(),
                                     requires_grad=True)  # original, instead of requires_grad: volatile = True

            # Do the inference
            with torch.no_grad():  # original: without this line (only THIS line) and activate volatile = True command above
                outs = net(image_patches)
                outs = outs.data.cpu().numpy()

            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out
            del (outs)

        pred = np.argmax(pred, axis=-1)

        # Display the result
        # clear_output()
        # fig = plt.figure()
        # fig.add_subplot(1,3,1)
        # plt.imshow(np.asarray(255 * img, dtype='uint8'))
        # fig.add_subplot(1,3,2)
        # plt.imshow(convert_to_color(pred))
        # fig.add_subplot(1,3,3)
        # plt.imshow(gt)
        # plt.show()

        all_preds.append(pred)
        gt = convert_from_color(gt)  # convert to grayscale image with class indices
        all_gts.append(gt)
        all_gts_e.append(gt_e)

        # clear_output()
        # Compute some metrics
        print("\nCalculate some metrics for one image:")
        print("Ground truth:")
        metrics(pred.ravel(),
                gt.ravel())  # input: flattened arrays

        print("Ground truth (eroded; some pixels are neglected --> see ISPRS webpage):")
        metrics(pred.ravel(),
                gt_e.ravel())  # input: flattened arrays

    print("Calculate overall metrics (ground truth)...")
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate(
                           [p.ravel() for p in all_gts]))  # p contains a single grayscale ground truth img (6000x6000)
    # np.concatenate([p.ravel() for p in all_preds]): flattening of prediction images and concatenating into long 1D array
    # np.concatenate([p.ravel() for p in all_gts]): flattening of ground truth images and concatenating into long 1D array

    print("Calculate overall metrics (eroded ground truth)...")
    accuracy_e = metrics(np.concatenate([p.ravel() for p in all_preds]),
                         np.concatenate([p.ravel() for p in
                                         all_gts_e]))  # p contains a single grayscale ground truth img (6000x6000)
    # np.concatenate([p.ravel() for p in all_preds]): flattening of prediction images and concatenating into long 1D array
    # np.concatenate([p.ravel() for p in all_gts_e]): flattening of eroded ground truth images and concatenating into long 1D array

    if output_all:
        return accuracy, accuracy_e, all_preds, all_gts, all_gts_e
    else:
        return accuracy, accuracy_e


def predict(net, predict_ids, stride=GLOB.WINDOW_SIZE[0], batch_size=GLOB.BATCH_SIZE, window_size=GLOB.WINDOW_SIZE):
    # Use the network on the pytorch set
    predict_images = (1 / 255 * np.asarray(io.imread(GLOB.DATA_FOLDER.format(id)), dtype='float32') for id in predict_ids)
    all_preds = []

    # Switch the network to inference mode
    net.eval()

    for img in tqdm(predict_images, total=len(predict_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (GLOB.N_CLASSES,))

        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(
                tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                     leave=False)):
            # Display in progress results
            # if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
            #         _pred = np.argmax(pred, axis=-1)
            #         fig = plt.figure()
            #         fig.add_subplot(1,3,1)
            #         plt.imshow(np.asarray(255 * img, dtype='uint8'))
            #         fig.add_subplot(1,3,2)
            #         plt.imshow(convert_to_color(_pred))
            #         fig.add_subplot(1,3,3)
            #         plt.imshow(gt)
            #         clear_output()
            #         plt.show()

            # Build the tensor
            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(),
                                     requires_grad=True)  # original, instead of requires_grad: volatile = True

            # Do the inference
            with torch.no_grad():  # original: without this line (only THIS line) and activate volatile = True command above
                outs = net(image_patches)
                outs = outs.data.cpu().numpy()

            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out
            del (outs)

        pred = np.argmax(pred, axis=-1)

        # Display the result
        # clear_output()
        # fig = plt.figure()
        # fig.add_subplot(1,3,1)
        # plt.imshow(np.asarray(255 * img, dtype='uint8'))
        # fig.add_subplot(1,3,2)
        # plt.imshow(convert_to_color(pred))
        # fig.add_subplot(1,3,3)
        # plt.imshow(gt)
        # plt.show()

        all_preds.append(pred)

    return all_preds


def predictOrtho(net, predict_ids, stride=GLOB.WINDOW_SIZE[0], batch_size=GLOB.BATCH_SIZE, window_size=GLOB.WINDOW_SIZE):
    predict_images = (np.asarray(io.imread(GLOB.PREDICTION_DATA_FOLDER.format(id))[:, :, 0:3], dtype='float32') for id in predict_ids)
    all_preds = []

    # Switch the network to inference mode
    net.eval()

    for img in tqdm(predict_images, total=len(predict_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (GLOB.N_CLASSES,))

        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(
                tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                     leave=False)):
            # # Display in progress results
            # if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
            #     _pred = np.argmax(pred, axis=-1)
            #     fig = plt.figure()
            #     fig.add_subplot(1,3,1)
            #     plt.imshow(np.asarray(255 * img, dtype='uint8'))
            #     fig.add_subplot(1,3,2)
            #     plt.imshow(convert_to_color(_pred))
            #     fig.add_subplot(1,3,3)
            #     plt.imshow(gt)
            #     clear_output()
            #     plt.show()

            # Build the tensor
            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(),
                                     requires_grad=True)  # original, instead of requires_grad: volatile = True

            # Do the inference
            with torch.no_grad():  # original: without this line (only THIS line) and activate volatile = True command above
                outs = net(image_patches)
                outs = outs.data.cpu().numpy()

            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out
            del (outs)

        pred = np.argmax(pred, axis=-1)

        # Display the result
        # clear_output()
        # fig = plt.figure()
        # fig.add_subplot(1,2,1)
        # plt.imshow(np.asarray(255 * img, dtype='uint8'))
        # fig.add_subplot(1,2,2)
        # plt.imshow(convert_to_color(pred))
        # plt.show()

        all_preds.append(pred)

    return all_preds


def train(net, optimizer, epochs, train_loader=None, scheduler=None, weights=GLOB.WEIGHTS, save_epoch=5, filename=GLOB.SAVE_TRAINED_WEIGHTS):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    criterion = nn.NLLLoss(weight=weights)  # original: NLLLoss2d
    iter_ = 0

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            loss = CrossEntropy2d(output, target, weight=weights)
            loss.backward()
            optimizer.step()

            losses[
                iter_] = loss.data.item()  # original: loss.data[0] with 'UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number'

            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                # clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data.item(), accuracy(pred,
                                                                                     gt)))  # original: loss.data[0] with 'UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number'

                # plt.plot(mean_losses[:iter_]) and plt.show()
                # fig = plt.figure()
                # fig.add_subplot(131)
                # plt.imshow(rgb)
                # plt.title('RGB')
                # fig.add_subplot(132)
                # plt.imshow(convert_to_color(gt))
                # plt.title('Ground truth')
                # fig.add_subplot(133)
                # plt.title('Prediction')
                # plt.imshow(convert_to_color(pred))
                # plt.show()
            iter_ += 1

            del (data, target, loss)

        if e % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            acc, _ = test(net, GLOB.test_ids, output_all=False, stride=min(GLOB.WINDOW_SIZE))
            torch.save(net.state_dict(), './segnet256_epoch{}_{}'.format(e, acc))
    torch.save(net.state_dict(), filename)
    print("Save trained weights to file", filename, "...")

