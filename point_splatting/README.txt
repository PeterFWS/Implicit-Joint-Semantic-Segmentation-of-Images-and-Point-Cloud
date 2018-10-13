To use it you need to set the paths to the point cloud files in render_views.cpp. 

The rest is sort of described in the comment in render_views.cpp just before the main function.

It is quite slow but you should be able to speed it up by parallellizing the loop in processDepthImageMeanShift in src/point_cloud_to_depth.cpp.




Run the code in /build
./render_point_views "ps_pointcloud" "/home/fangwen/masThesis/point_splatting/result"
