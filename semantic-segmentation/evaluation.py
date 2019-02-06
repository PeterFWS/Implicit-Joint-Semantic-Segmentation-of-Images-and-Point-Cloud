import numpy as np
import cv2
import glob
import itertools
import os
from tqdm import tqdm
import tifffile
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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

    # save figure
    # save_figure(fig, filepath, filename)

# Global values
color_classes_int = {  #BGR
    "0": (255, 0, 0),
    "1": (255, 255, 255),
    "2": (255, 255, 0),
    "3": (255, 0, 255),
    "4": (0, 255, 255),
    "5": (0, 255, 0),
    "6": (0, 0, 255),
    "7": (239, 120, 76),
    "8": (247, 238, 179),
    "9": (0, 18, 114),
    "10": (63, 34, 15),
    "11": (0, 0, 0)  # number 11 indicts for nothing
}

LABELS = ['PowerLine', 'Low Vegetation', 'Impervious Surface', 'Vehicles', 'Urban Furniture',
          'Roof', 'Facade', 'Bush/Hedge', 'Tree', 'Dirt/Gravel', 'Vertical Surface']

# Evaluation in 2D image space
def evaluation_2d(path_predictions, path_groundtruths, path_mask):

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

    all_preds = []
    all_gts = []
    for i in tqdm(range(0, len(pred_imgs))):
        pred_img = cv2.imread(pred_imgs[i], 1)
        gt_img = cv2.imread(gt_imgs[i], 0)[0:pred_img.shape[0], 0:pred_img.shape[1]]
        gt_img[(gt_img == 255)] = 11

        mask = cv2.imread(masks[i], 0)[0:pred_img.shape[0], 0:pred_img.shape[1]]
        mask = mask.ravel()
        index = []
        [index.append(i) for i in range(mask.shape[0]) if mask[i] == 0]

        # convert rgb img into grey image
        for c in color_classes_int:
            pred_img[(pred_img == color_classes_int[str(c)])] = c
        pred_img = pred_img[:,:,0]

        pred_img = np.delete(pred_img.ravel(), index)
        gt_img = np.delete(gt_img.ravel(), index)

        all_preds.append(pred_img)
        all_gts.append(gt_img)

    print("Calculate overall metrics...")
    predictions = np.concatenate([p for p in all_preds])
    gts = np.concatenate([p for p in all_gts])

    cm = confusion_matrix(gts, predictions, range(len(LABELS)))

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = np.trace(cm)  # sum([cm[x][x] for x in range(len(cm))]) # main diagonal elements
    accuracy *= 100 / float(total)  # overall accuracy
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))

    print 'Report : '
    report = classification_report(gts, predictions)
    print report

    filepath = "/home/fangwen/ShuFangwen/source/image-segmentation-keras"
    plot_confusion_matrix(cm, class_names=LABELS, filepath=filepath, filename='confusion_matrix.png',
                          flag_normalize=True, title='Confusion Matrix',
                          cmap='plt.cm.Blues')

# Evaluation in 3D object space
def evaluation_3d(path_predictions, path_groundtruths, path_mask, path_index, path_pointcloud_label):

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
    gt_xyz = data[:, :3]
    gt_label_3d = data[:, -1]

    # processing
    pt_recorder = []
    for i in range(gt_label_3d.shape[0]):
        pt_recorder.append([])  # create a list of lists, each single list save the predicted label, used for majority vote
    pt_label_majority = np.zeros(gt_label_3d.shape[0])  # save the label after majority vote

    for i in tqdm(range(0, len(pred_imgs))):
        # read predicted img (rgb)
        pred_img = cv2.imread(pred_imgs[i], 1)
        for c in color_classes_int:  # convert rgb img into grey label image, same format as ground truth image
            pred_img[(pred_img == color_classes_int[str(c)])] = c
        pred_img = pred_img[:,:,0]

        # read mask (where, invalid pixel value = 0, valid pixel value = 255)
        mask = cv2.imread(masks[i], 0)[0:pred_img.shape[0], 0:pred_img.shape[1]]
        mask = mask.ravel()
        index_ignore = []  # index, where the pixel is invalid and it will be deleted before calculating confusion matrix
        [index_ignore.append(j) for j in range(mask.shape[0]) if mask[j] == 0]

        # read point index img, find corresponding 3d point and label value from 3d point cloud of each valid pixel
        pt_index = tifffile.imread(point_indexs[i])[0:pred_img.shape[0], 0:pred_img.shape[1]].astype(np.uint32)
        pt_index = np.delete(pt_index.ravel(), index_ignore)

        pred_img = np.delete(pred_img.ravel(), index_ignore)

        # since one 3d point shows up in different image, save the predicted label from different image
        for k in range(pt_index.shape[0]):
            if pt_index[k] != 0:
                # l_gt = label_3d[pt_index[k]]  # the gt label of this 3d point
                l_pred = pred_img[k]  # the predicted label of this point in the predicted image
                pt_recorder[pt_index[k]].append(l_pred)  # save the predicted label into corresponding list

    # majority vote
    index_ignore2 = []
    for i in range(len(pt_recorder)):
        if len(pt_recorder[i]) == 0:
            # this point is not shown in any image
            index_ignore2.append(i)
            continue
        else:
            recorder = np.zeros((12, 1))  # 12 classes
            # find the value which shows up most
            for lb in pt_recorder[i]:
                recorder[lb, 0] += 1
            amax_label = np.argmax(recorder)  # this will be the true predicted label of this point
            pt_label_majority[i] = amax_label
    index_ignore2 = np.asarray(index_ignore2)

    precentage = float(index_ignore2.shape[0]) / float(gt_label_3d.shape[0]) * 100
    print("how many points in test-set will not be evaluated: {}%".format(precentage))


    # for i in tqdm(range(index_ignore2.shape[0])):
    #     pt_label_majority[index_ignore2[i]] = 255
    #
    # saved_data_predicted = np.concatenate((gt_xyz, np.asmatrix(pt_label_majority).astype(np.uint8).T), axis=1)
    #
    # np.savetxt("./saved_data_predicted.txt",saved_data_predicted)



    predictions = np.delete(pt_label_majority, index_ignore2)
    gts = np.delete(gt_label_3d, index_ignore2)

    # gts_xyz_new = np.delete(gt_xyz, index_ignore2, axis=0)
    # saved_data_predicted_part = np.concatenate((gts_xyz_new, np.asmatrix(predictions).astype(np.uint8).T), axis=1)
    # np.savetxt("./saved_data_predicted_part.txt", saved_data_predicted_part)

    cm = confusion_matrix(gts, predictions, range(len(LABELS)))

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = np.trace(cm)  # sum([cm[x][x] for x in range(len(cm))]) # main diagonal elements
    accuracy *= 100 / float(total)  # overall accuracy
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))

    print 'Report : '
    report = classification_report(gts, predictions)
    print report

    filepath = "/home/fangwen/ShuFangwen/source/image-segmentation-keras"
    plot_confusion_matrix(cm, class_names=LABELS, filepath=filepath, filename='confusion_matrix.png',
                          flag_normalize=True, title='Confusion Matrix',
                          cmap='plt.cm.Blues')



if __name__ == "__main__":

    path_predictions = "/home/fangwen/ShuFangwen/source/image-segmentation-keras/data/predictions_1_baseline/"
    path_groundtruths = "/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_oblique/test_set/3_greylabel/"
    path_mask = "/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_oblique/test_set/2_mask/"
    path_index = "/run/user/1001/gvfs/smb-share:server=141.58.125.9,share=s-platte/ShuFangwen/results/level3_oblique/test_set/5_index/"

    path_pointcloud_label = "/home/fangwen/ShuFangwen/data/data_splits_5cm_onlylabel/train_xyz_y.txt"



    start_time = time()

    # evaluation_2d(path_predictions, path_groundtruths, path_mask)
    evaluation_3d(path_predictions, path_groundtruths, path_mask, path_index, path_pointcloud_label)

    duration = time() - start_time
    print("duration: {}s".format(duration))

