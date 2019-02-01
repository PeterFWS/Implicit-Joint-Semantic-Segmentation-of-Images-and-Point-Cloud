# Implicit Joint Semantic Segmentation of Images and Point Cloud
MasterThesis in University of Stuttgart,Insititute of Photogrammetry (Ifp), Germany<br>
keywords: semantic segmantation, CNN, LiDAR point cloud, aerial imagery<br>
Author: Fangwen Shu<br>
Betreuer: M.Sc. Dominik Laupheimer<br>
Prüfer: apl. Prof. Dr.-Ing. Norbert Haala<br>

framework: keras 2.0<br>
pre-processing: python3<br>
semantic segmentation: python2<br>

End in 01.04.2018<br>

#---------------------------Configuration of conda env----------------------------#<br>

conda_env/py2.yml and py3.yml<br>

#---------------------------Pre-processing part-----------------------------------#<br>
[1] main*.py <br>
pre-processing code for aerial imagery and LiDAR point cloud, <br>
including projection, frustum culling, Hidden-point-removal (HPR),<br>
gird interpoaltion and operator of Mophology.

[2] utilities.py <br>
functions of each of algorithms implemented in pre-processing.

[3] myClasses.py <br>
some classes related to frustum culling, detailed explanation in OpenGL.

[4] visualization.py <br>
functions used to visualize data. 

[5] statistics.py <br>
functions used to calculate statistic value of the data.

[6] Generation_depth_img.py <br>
code for generating depth image.

#---------------------------Deep learning part-----------------------------------#<br>
folder: version_playground<br>
Old code backup, including point splatting achieved in C++ if you needed.<br>

folder: semantic-segmentation<br>
[1] folder Models: 
1. SegNet
2. U-net
3. TernausNet

[2] folder board:<br>
save tensorboard file.

[3] folder data:<br>
where you save train/validation/test set and VGG pre-treained weights.

[4] folder weights:<br>
where you save trained weights.

[5] pytorch_code:<br>
some dirty code of SegNet and pre-processing implemented in pytorch. 

Code:<br>
[1] train.py and prediction.py <br>
train and prediction your data.

[2] LoadBatches.py <br>
loading data with pre-processing such as normalization.

[3] chip.py <br>
cropping images if you need.

[4] evaluation.py <br>
evaluation semantic result in 2D and 3D space.

