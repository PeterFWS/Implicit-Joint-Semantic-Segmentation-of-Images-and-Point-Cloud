To use it you need to set the paths to the point cloud files in render_views.cpp. 

The rest is sort of described in the comment in render_views.cpp just before the main function.

It is quite slow but you should be able to speed it up by parallellizing the loop in processDepthImageMeanShift in src/point_cloud_to_depth.cpp.




* inputs: 

**Basic information:
 *	[1]clound_filename: 
 		name of textfile containing point cloud (x y z i r g b) and label filename containing integer labels as in semantic3D
 *	[2]location: output folder location
 *	[3]haslabel: flag with 1 for labeled point clouds and 0 for unlabeled point clouds

**Paramaters related to Mean-shift pip-line
 *	[4]lim
 *	[5]cluster_val_threshold
 *	[6]num_iterations
 *	[7]cluster_width

**Parameters related to splatting radius and image size
 *	[8]k_guss
 *	[9]rows
 *	[10]cols


 export OMP_NUM_THREADS=4

./render_point_views "ps_pointcloud" "/home/fangwen/masThesis/point_splatting/result" 1 -5 0.001f 300 1000.0f 40 8000 8000