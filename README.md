# Implicit Joint Semantic Segmentation of Images and Point Cloud
##### keywords: Deep learning, semantic segmentation, ALS point cloud, aerial imagery, multi-modal features

Website at ifp, Uni Stuttgart: https://www.ifp.uni-stuttgart.de/lehre/masterarbeiten/581-shu/

```
Master Thesis at Insititute of Photogrammetry (Ifp), University of Stuttgart, Germany
Author: Fangwen Shu
Betreuer: M.Sc. Dominik Laupheimer
Prüfer: Prof. Norbert Haala

framework: keras 2.0
pre-processing: python3
semantic segmentation: python2

End in 01.04.2019
```

## Configuration of conda env

./conda_env/py2.yml <br>
./conda_env/py3.yml <br>

```
conda env export > py2.yml
conda env create -f py2.yml
```


## Pre-processing part

#### ./main.py

pre-processing code for aerial imagery and LiDAR point cloud, including 3D-2D projection, frustum culling, 
Hidden-point-removal (HPR), gird interpolation and operator of Morphology.


#### ./utilities.py

functions of each of algorithms implemented in pre-processing.


#### ./myClasses.py

some classes related to frustum culling, traslated from C++ code, detailed explanation in OpenGL.


#### ./visualization.py

functions used to visualize data. 


#### ./statistics.py

functions used to calculate statistic information of the data.


#### ./Generation_depth_img.py

code for generating depth image.

## Semantic segmentation part

#### ./version_playground

Old code backup, including point splatting achieved in C++ if you needed.<br>


#### ./semantic-segmentation/Models/

* testNet (multi-stream costume CNN based on SegNet, early or late fusion, multi-input stream)
* SegNet (main model used in thesis)
* PSPnet (runnable, one of baselines)
* FCN8,32 (runnable)
* U-net (runnable)
* TernausNet (runnable)


#### ./semantic-segmentation/board/

where you save tensorboard file.


#### ./semantic-segmentation/data/

where you save train/validation/test-set and VGG pre-treained weights.


#### ./semantic-segmentation/weights/

where you save trained weights.


#### ./semantic-segmentation/pytorch_code/

some dirty code of SegNet and pre-processing implemented in pytorch. 

### Code for training/prediction/evaluation:

#### ./semantic-segmentation/train.py
#### ./semantic-segmentation/prediction.py

train and prediction your data.


#### ./semantic-segmentation/LoadBatches.py

data generator with pre-processing such as normalization, random rotation, random cropping <br>
randomly brightness jitter, contrast normalization..


#### ./semantic-segmentation/chip.py

cropping images if you need.


#### ./semantic-segmentation/evaluation.py

evaluation semantic result in 2D and 3D space.

