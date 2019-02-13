## Implicit Joint Semantic Segmentation of Images and Point Cloud
##### keywords: Deep learning, semantic segmentation, ALS point cloud, aerial imagery
```
Master Thesis in University of Stuttgart, Insititute of Photogrammetry (Ifp), Germany
Author: Fangwen Shu
Betreuer: M.Sc. Dominik Laupheimer
Prüfer: apl. Prof. Dr.-Ing. Norbert Haala

framework: keras 2.0
pre-processing: python3
semantic segmentation: python2

End in 01.04.2018
```
## Implemented Pipeline
In order to avoid feeding noisy or non-uniformly sampled point cloud into 3D CNN, in this work, a novel fusion of the labeled LiDAR point cloud and oriented aerial imagery in 2D space is hypothesized, in this way, we can leverage image-based semantic segmentation and create a multi-view, multi-modal and multi-scale segmentation classifier. Thereafter through a fast back-projection of the 2D semantic result to 3D point cloud, we achieve a joint semantic segmentation of imagery and point cloud. <br>
The proposed method is validated against to our own dataset: the oriented high-resolution oblique and nadir aerial imagery of village Hessigheim, Germany, captured by unmanned aerial vehicle (UAV), as well as the LiDAR point cloud obtained by the airborne laser scanning (ALS) device. The high resolution aerial images offer views of diverse urban scene, with useful geometric characteristics derived from point cloud, it is a potential combination to set up a big dataset for training a well-engineered deep CNN. 

<img src="https://github.com/PeterFWS/masterThesis_BK/blob/master/imgs/pipline.png" width="700">
<img src="https://github.com/PeterFWS/masterThesis_BK/blob/master/imgs/list_of_classes.PNG" width="700">


## Configuration of conda env

./conda_env/py2.yml <br>
./conda_env/py3.yml <br>

```
conda env export > py2.yml
conda env create -f py2.yml
```


## Pre-processing part
```
./main.py
```
pre-processing code for aerial imagery and LiDAR point cloud, including 3D-2D projection, frustum culling, 
Hidden-point-removal (HPR), gird interpolation and operator of Morphology.

```
./utilities.py
```
functions of each of algorithms implemented in pre-processing.

```
./myClasses.py
```
some classes related to frustum culling, traslated from C++ code, detailed explanation in OpenGL.

```
./visualization.py
```
functions used to visualize data. 

```
./statistics.py
```
functions used to calculate statistic information of the data.

```
./Generation_depth_img.py
```
code for generating depth image.

## Deep learning part
```
./version_playground
```
Old code backup, including point splatting achieved in C++ if you needed.<br>

```
./semantic-segmentation/Models/
```
* SegNet (main model used in thesis)
* U-net (not test)
* TernausNet (not test)

```
./semantic-segmentation/board/
```
where you save tensorboard file.

```
./semantic-segmentation/data/
```
where you save train/validation/test-set and VGG pre-treained weights.

```
./semantic-segmentation/weights/
```
where you save trained weights.

```
./semantic-segmentation/pytorch_code/
```
some dirty code of SegNet and pre-processing implemented in pytorch. 

### Code:
```
./semantic-segmentation/train.py
./semantic-segmentation/prediction.py
```
train and prediction your data.

```
./semantic-segmentation/LoadBatches.py
```
data generator with pre-processing such as normalization, random rotation, random cropping

```
./semantic-segmentation/chip.py
```
cropping images if you need.

```
./semantic-segmentation/evaluation.py
```
evaluation semantic result in 2D and 3D space.

