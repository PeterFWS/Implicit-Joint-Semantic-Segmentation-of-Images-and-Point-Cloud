To use it you need to set the paths to the point cloud files in render_views.cpp. 

The rest is sort of described in the comment in render_views.cpp just before the main function.

It is quite slow but you should be able to speed it up by parallellizing the loop in processDepthImageMeanShift in src/point_cloud_to_depth.cpp.




* inputs: 
 *     clound_filename: name of textfile containing point cloud (x y z i r g b) and label filename containing integer labels as in semantic3D
 *     location: output folder location
 *     haslabel: flag with 1 for labeled point clouds and 0 for unlabeled point clouds

 *     lim
 *     cluster_val_threshold
 *     num_iterations
 *     cluster_width

./render_point_views "ps_pointcloud" "/home/fangwen/masThesis/point_splatting/result" 1 -5 0.01f 30 0.1f