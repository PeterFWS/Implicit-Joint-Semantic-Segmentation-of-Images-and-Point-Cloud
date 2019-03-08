import numpy as np
import cv2
import glob
import itertools
import os
from tqdm import tqdm
import tifffile
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# Global values
LABELS = ['PowerLine', 'Low Vegetation', 'Impervious Surface', 'Vehicles', 'Urban Furniture',
          'Roof', 'Facade', 'Bush/Hedge', 'Tree', 'Dirt/Gravel', 'Vertical Surface', 'Void']

palette = {  # BGR
    0: (255, 0, 0),      # Powerline
    1: (255, 255, 255),  # Low Vegetation
    2: (255, 255, 0),    # Impervious Surface
    3: (255, 0, 255),    # Vehicles
    4: (0, 255, 255),    # Urban Furniture
    5: (0, 255, 0),      # Roof
    6: (0, 0, 255),      # Facade
    7: (239, 120, 76),   # Bush/Hedge
    8: (247, 238, 179),  # Tree
    9: (0, 18, 114),     # Dirt/Gravel
    10: (63, 34, 15),    # Vertical Surface
    11: (0, 0, 0)        # Void
}

invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def save_figure(fig, path, filename):
    """ Save figure fig in directory specified by path under filename """

    # create desired subfolder (TODO maybe: use createNewSubdirectory)
    if not os.path.exists(path):
        os.makedirs(path)

    # save figure in desired subfolder
    fig.savefig(os.path.join(path, filename), bbox_inches='tight',
                dpi=300)  # TODO: set dpi (300dpi is good default choice)

    # plt.close(fig)
    # return 0


format_float_2f = lambda x: "%.2f" % x


def plot_confusion_matrix(conf_mat, class_names, filepath, filename,
                          flag_normalize=False, title='Confusion Matrix',
                          cmap='plt.cm.Blues'):

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

    # save_figure(fig, filepath, filename)


# Evaluation in 2D image space
def evaluation_2d(path_predictions, path_groundtruths, path_mask, ignore_void=False):

    # reading data
    pred_imgs = glob.glob(path_predictions + "*.JPG") + glob.glob(path_predictions + "*.tif")
    pred_imgs.sort()
    gt_imgs = glob.glob(path_groundtruths + "*.JPG") + glob.glob(path_groundtruths + "*.tif")
    gt_imgs.sort()
    masks = glob.glob(path_mask + "*.JPG") + glob.glob(path_mask + "*.tif")
    masks.sort()
    assert len(pred_imgs) == len(pred_imgs)
    assert len(pred_imgs) == len(masks)
    for pred_img, gt_img in zip(pred_imgs, gt_imgs):
        assert (pred_img.split('/')[-1].split(".")[0] == gt_img.split('/')[-1].split(".")[0])
    for pred_img, mask in zip(pred_imgs, masks):
        assert (pred_img.split('/')[-1].split(".")[0] == mask.split('/')[-1].split(".")[0])

    # convert data into a 1-D array
    all_preds = []
    all_gts = []

    for i in tqdm(range(0, len(pred_imgs))):
        pred_img = convert_from_color(cv2.imread(pred_imgs[i], 1))
        gt_img = cv2.imread(gt_imgs[i], 0)[:pred_img.shape[0], :pred_img.shape[1]]
        gt_img[(gt_img == 255)] = 11
        mask = cv2.imread(masks[i], 0)[:pred_img.shape[0], :pred_img.shape[1]]
        mask = mask.ravel()

        if ignore_void is not False:
            index = []  # index of void pixels
            [index.append(_) for _ in range(mask.shape[0]) if mask[_] == 0]
            pred_img = np.delete(pred_img.ravel(), index)
            gt_img = np.delete(gt_img.ravel(), index)
        else:
            pred_img = pred_img.ravel()
            gt_img = gt_img.ravel()

        all_preds.append(pred_img)
        all_gts.append(gt_img)

    print("Calculate overall metrics...")
    predictions = np.concatenate([p for p in all_preds])
    gts = np.concatenate([p for p in all_gts])

    cm = confusion_matrix(y_true=gts, y_pred=predictions, labels=range(len(LABELS)))

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = np.trace(cm)  # sum([cm[x][x] for x in range(len(cm))]) # main diagonal elements
    accuracy *= 100 / float(total)  # overall accuracy
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: " + str(kappa))

    report = classification_report(y_true=gts, y_pred=predictions, labels=range(len(LABELS)), target_names=LABELS)
    print(report)

    filepath = "/home/fangwen/ShuFangwen/source/image-segmentation-keras"
    plot_confusion_matrix(cm, class_names=LABELS, filepath=filepath, filename='confusion_matrix.png',
                          flag_normalize=True, title='Normalized Confusion Matrix of 2D space evaluation',
                          cmap='plt.cm.Blues')


# Evaluation in 3D object space
def evaluation_3d(path_predictions, path_groundtruths, path_mask, path_index, path_pointcloud_label, save_3D=False):

    # read image data, sorted, and check if those data is matched to each other
    pred_imgs = glob.glob(path_predictions + "*.JPG") + glob.glob(path_predictions + "*.tif")
    pred_imgs.sort()
    gt_imgs = glob.glob(path_groundtruths + "*.JPG") + glob.glob(path_groundtruths + "*.tif")
    gt_imgs.sort()
    masks = glob.glob(path_mask + "*.JPG") + glob.glob(path_mask + "*.tif")
    masks.sort()
    point_indexs = glob.glob(path_index + "*.JPG") + glob.glob(path_index + "*.tif")
    point_indexs.sort()
    assert len(pred_imgs) == len(pred_imgs)
    assert len(pred_imgs) == len(masks)
    assert len(pred_imgs) == len(point_indexs)
    for pred_img, gt_img in zip(pred_imgs, gt_imgs):
        assert (pred_img.split('/')[-1].split(".")[0] == gt_img.split('/')[-1].split(".")[0])
    for pred_img, mask in zip(pred_imgs, masks):
        assert (pred_img.split('/')[-1].split(".")[0] == mask.split('/')[-1].split(".")[0])
    for pred_img, index in zip(pred_imgs, point_indexs):
        assert (pred_img.split('/')[-1].split(".")[0] == index.split('/')[-1].split(".")[0])

    data = np.loadtxt(path_pointcloud_label)
    gt_label_3d = data[:, -1]

    pt_recorder = []
    for i in range(gt_label_3d.shape[0]):
        # create a lists of list, each single list save the predicted label, used for majority vote
        pt_recorder.append([])
    pt_label_majority = np.zeros(gt_label_3d.shape[0])  # save the label after majority vote

    for i in tqdm(range(0, len(pred_imgs))):
        # read predicted img
        pred_img = convert_from_color(cv2.imread(pred_imgs[i], 1))

        # read mask (where, invalid pixel value = 0, valid pixel value = 255)
        mask = cv2.imread(masks[i], 0)[0:pred_img.shape[0], 0:pred_img.shape[1]]
        mask = mask.ravel()

        # read point index img
        pt_index = tifffile.imread(point_indexs[i])[0:pred_img.shape[0], 0:pred_img.shape[1]].astype(np.uint32)

        # In 3D space, you are evaluating on points, not pixels, so you have to delete those invalid pixels first
        # index, where the pixel is invalid and corresponding pixel will be deleted
        index_ignore = []
        [index_ignore.append(j) for j in range(mask.shape[0]) if mask[j] == 0]
        pt_index = np.delete(pt_index.ravel(), index_ignore)
        pred_img = np.delete(pred_img.ravel(), index_ignore)

        # since one 3d point shows up in different image, save the predicted label from different image
        for k in range(pt_index.shape[0]):
            if pt_index[k] != 0:
                # l_gt = label_3d[pt_index[k]]  # the gt label of this 3d point
                try:
                    l_pred = pred_img[k]  # the predicted label of this point in the predicted image
                    pt_recorder[pt_index[k]].append(l_pred)  # save the predicted label into corresponding list
                except IndexError:
                    continue

    # index2, where 3d point is not projected
    index_ignore2 = []
    # majority vote in 3d object space
    for i in range(len(pt_recorder)):
        if len(pt_recorder[i]) == 0:
            # no predicted label saved, so this point is not projected
            index_ignore2.append(i)
            continue
        else:
            recorder = np.zeros((12, 1))  # 12 classes
            # find the value which shows up most
            for predicted_label in pt_recorder[i]:
                recorder[predicted_label, 0] += 1
            amax_label = np.argmax(recorder)  # the majority label of this point
            pt_label_majority[i] = amax_label
    index_ignore2 = np.asarray(index_ignore2)

    precentage = float(index_ignore2.shape[0]) / float(gt_label_3d.shape[0]) * 100
    print("{}% 3D points (5cm density point cloud) are not evaluated due to occlusion".format(precentage))

    predictions = np.delete(pt_label_majority, index_ignore2)
    gts = np.delete(gt_label_3d, index_ignore2)

    cm = confusion_matrix(y_true=gts, y_pred=predictions, labels=range(len(LABELS)))

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = np.trace(cm)  # sum([cm[x][x] for x in range(len(cm))]) # main diagonal elements
    accuracy *= 100 / float(total)  # overall accuracy
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: " + str(kappa))

    report = classification_report(y_true=gts, y_pred=predictions, labels=range(len(LABELS)), target_names=LABELS)
    print(report)

    filepath = "/home/fangwen/ShuFangwen/source/image-segmentation-keras"
    plot_confusion_matrix(cm, class_names=LABELS, filepath=filepath, filename='confusion_matrix.png',
                          flag_normalize=True, title='Normalized Confusion Matrix of 3D space evaluation',
                          cmap='plt.cm.Blues')

    if save_3D is not False:
        # save the semantic result, including the points which are not projected, with void label=255
        gt_xyz = data[:, :3]
        for i in tqdm(range(index_ignore2.shape[0])):
            pt_label_majority[index_ignore2[i]] = 255
        saved_data_predicted = np.concatenate((gt_xyz, np.asmatrix(pt_label_majority).astype(np.uint8).T), axis=1)
        np.savetxt("./saved_data_predicted.txt", saved_data_predicted)

        # save the semantic result, only those points are projected and evaluated
        gts_xyz_part = np.delete(gt_xyz, index_ignore2, axis=0)
        saved_data_predicted_part = np.concatenate((gts_xyz_part, np.asmatrix(predictions).astype(np.uint8).T), axis=1)
        np.savetxt("./saved_data_predicted_part.txt", saved_data_predicted_part)


if __name__ == "__main__":

    path_predictions = "/data/fangwen/predictions_baseline5/"
    path_groundtruths = "/data/fangwen/mix_test/3_greylabel/"
    path_mask = "/data/fangwen/mix_test/2_mask/"
    path_index = "/data/fangwen/mix_test/5_index/"

    path_pointcloud_label = "/data/fangwen/test_xyz_y.txt"

    # evaluation_2d(path_predictions, path_groundtruths, path_mask, ignore_void=True)

    evaluation_3d(path_predictions, path_groundtruths, path_mask, path_index, path_pointcloud_label, save_3D=False)


