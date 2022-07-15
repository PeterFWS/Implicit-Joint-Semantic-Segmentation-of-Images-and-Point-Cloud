/*
 * Code for downsampling of point cloud
 * 
 */

#ifndef CLOUD_DOWNSAMPLING_H
#define CLOUD_DOWNSAMPLING_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class CloudDownsampler
{
  public:
    CloudDownsampler();
    ~CloudDownsampler();

    void downsampleRandom(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_out, float rate);
    void downsampleRadius(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_out, float rate);
    void downsampleUniform(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_out, float rate);

};

#endif
