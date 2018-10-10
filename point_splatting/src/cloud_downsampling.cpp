/*
 * Code for downsampling of point cloud
 * 
 */
#include "cloud_downsampling.h"
#include <pcl/io/ply_io.h>
//#include <pcl/kdtree/kdtree_flann.h>
#include <math.h>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

CloudDownsampler::CloudDownsampler()
{

}
CloudDownsampler::~CloudDownsampler()
{

}

void CloudDownsampler::downsampleRandom(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_out, float rate)
{
  srand (time(NULL));
  unsigned int num_points_out = 0;
  unsigned int num_points = cloud->points.size();
  uint8_t* mask = new uint8_t[num_points]; 
  double prob = 1.0;
  for(unsigned int i = 0; i < num_points; i++)
  {
    int randnum = rand() % 10000;
    bool keep = randnum > (int)((1.0-prob*rate) * 10000);
    
    if(keep)
    {
      mask[i] = 1;
      num_points_out++;     
    }
    else
    {
      mask[i] = 0;
    }
  }

  std::cout<<"num_points_out = "<<num_points_out<<std::endl;

  cloud_out->resize(num_points_out);

  unsigned int p_cnt = 0;
  for(unsigned long int i = 0; i < num_points; i++)
  {
    if(mask[i])
    {
      cloud_out->points[p_cnt] = cloud->points[i];
      p_cnt++;
    }
  }
}

void CloudDownsampler::downsampleRadius(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_out, float rate)
{
  srand (time(NULL));
  unsigned int num_points_out = 0;
  unsigned int num_points = cloud->points.size();
  uint8_t* mask = new uint8_t[num_points]; 
  float* rsqr = new float[num_points];
  double c = 0.0;
  double tmp_rsqr, max_rsqr;
  max_rsqr = 0.0;
  for(unsigned int i = 0; i < num_points; i++)
  {
    tmp_rsqr = cloud->points[i].x*cloud->points[i].x+cloud->points[i].y*cloud->points[i].y+cloud->points[i].z*cloud->points[i].z;
    //if(max_rsqr < tmp_rsqr)
    //   max_rsqr = tmp_rsqr;

    rsqr[i] = tmp_rsqr;
    c += tmp_rsqr > 30.0 ? 30.0 : tmp_rsqr;
  } 

  c = c/(double)num_points;
  //float d = 1.0/(2*c);
  std::cout<<"c = "<<c<<std::endl;
  //std::cout<<"c = "<<c<<", d = "<<d<<std::endl;
  for(unsigned int i = 0; i < num_points; i++)
  {
    double prob = rsqr[i]/c;
    //double prob2 = prob
    int randnum = rand() % 10000;
    bool keep = randnum > (int)((1.0-prob*rate) * 10000);
    
    if(keep)
    {
      mask[i] = 1;
      num_points_out++;     
    }
    else
    {
      mask[i] = 0;
    }
  }
  
  std::cout<<"num_points_out = "<<num_points_out<<std::endl;

  cloud_out->resize(num_points_out);

  unsigned int p_cnt = 0;
  for(unsigned long int i = 0; i < num_points; i++)
  {
    if(mask[i])
    {
      cloud_out->points[p_cnt] = cloud->points[i];
      p_cnt++;
    }
  }
}

void CloudDownsampler::downsampleUniform(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_out, float rate)
{
  srand (time(NULL));
  unsigned int num_points_out = 0;
  unsigned int num_points = cloud->points.size();
  uint8_t* mask = new uint8_t[num_points]; 
  double prob = 1.0;
  for(unsigned int i = 0; i < num_points; i++)
  {
    int randnum = rand() % 10000;
    bool keep = randnum > (int)((1.0-prob*rate) * 10000);
    
    if(keep)
    {
      mask[i] = 1;
      num_points_out++;     
    }
    else
    {
      mask[i] = 0;
    }
  }

  std::cout<<"num_points_out = "<<num_points_out<<std::endl;

  cloud_out->resize(num_points_out);

  unsigned int p_cnt = 0;
  for(unsigned long int i = 0; i < num_points; i++)
  {
    if(mask[i])
    {
      cloud_out->points[p_cnt] = cloud->points[i];
      p_cnt++;
    }
  }
}


