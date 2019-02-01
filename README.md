MasterThesis in University of Stuttgart, Germany
Insititute of Photogrammetry (Ifp)

Topic: Implicit Joint Semantic Segmentation of Images and Point Cloud
keywords: semantic segmantation, CNN, LiDAR point cloud, aerial imagery
author: Fangwen Shu
Betreuer: M.Sc. Dominik Laupheimer
Prüfer: apl. Prof. Dr.-Ing. Norbert Haala

framework: keras 2.0
pre-processing: python3
semantic segmentation: python2

End in 01.04.2018

#---------------------------Configuration of conda env----------------------------#

conda_env/py2.yml and py3.yml

#---------------------------Pre-processing part-----------------------------------#
[1] main*.py 
pre-processing code for aerial imagery and LiDAR point cloud, 
including projection, frustum culling, Hidden-point-removal (HPR),
gird interpoaltion and operator of Mophology.

[2] utilities.py
functions of each of algorithms implemented in pre-processing.

[3] myClasses.py
some classes related to frustum culling, detailed explanation in OpenGL.

[4] visualization.py
functions used to visualize data.

[5] statistics.py
functions used to calculate statistic value of the data.

[6] Generation_depth_img.py
code for generating depth image.

#---------------------------Deep learning part-----------------------------------#
folder: version_playground
Old code backup, including point splatting achieved in C++ if you needed.

folder: semantic-segmentation
[1] folder Models: 
1. SegNet
2. U-net
3. TernausNet

[2] folder board:
save tensorboard file.

[3] folder data:
where you save train/validation/test set and VGG pre-treained weights.

[4] folder weights:
where you save trained weights.

[5] pytorch_code:
some dirty code of SegNet and pre-processing implemented in pytorch. 

Code:
[1] train.py and prediction.py
train and prediction your data.

[2] LoadBatches.py
loading data with pre-processing such as normalization.

[3] chip.py
cropping images if you need.

[4] evaluation.py
evaluation semantic result in 2D and 3D space.

