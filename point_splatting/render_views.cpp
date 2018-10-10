#include <iostream>
#include "include/point_cloud_to_depth.h"
#include "include/cloud_downsampling.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>

/**********************************************************
 * global variables
 **********************************************************/

std::string labels_location = "/home/fangwen/masThesis/point_splatting/";
std::string clouds_location = "/home/fangwen/masThesis/point_splatting/";

/**********************************************************
 * Help functions
 **********************************************************/

void savePCD(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, std::string name)
{
  std::string filename = clouds_location + name + ".pcd";
  std::cout<<"save cloud to "<<filename<<std::endl;  
  pcl::io::savePCDFile( filename, *cloud, true );
}

unsigned int readToArray(std::string config_file_name, float** files_to_process, unsigned int row_size)
{
  std::cout<<"read file: "<<config_file_name<<std::endl;
	std::ifstream file_init(config_file_name.c_str());

	std::string line;
	int i = 0;
	
	unsigned int lines_count = 0;
	while (std::getline(file_init, line))
	  ++lines_count;
	  
	file_init.close();
	
	std::ifstream file(config_file_name.c_str());
	*files_to_process = new float[lines_count*row_size];
	
	while (std::getline(file, line))
	{
	  float value;
	  std::stringstream ss(line);

		for(unsigned int c = 0; c < row_size; c++)
		{
		  if(ss >> value)
		  {
			  (*files_to_process)[row_size*i+c] = value;
			}
			else
			  std::cout<<"unable to read value, this will crash \n";
		}

    ++i;
	}
	file.close();
	
  return lines_count;
}

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr readPointCloudFromPCD(std::string dataset)
{
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
  std::string filename = clouds_location + dataset + ".pcd";

  if(pcl::io::loadPCDFile( filename, *cloud ) == -1)
    std::cout<<"unable to read "<<filename<<std::endl;

  return cloud;
}

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr readPointCloudFromTxt(std::string dataset, bool haslabel, unsigned int downsample_num_points, bool save)
{
  float* label;
  unsigned int label_size = 0;
  if(haslabel)
	  label_size = readToArray(labels_location + dataset + ".labels", &label, 1);
	else
	  std::cout<<"dataset has no labels\n";
  
	float* pc;
	unsigned int pc_size = readToArray(clouds_location + dataset + ".txt", &pc, 7);
	std::cout<<"Files loaded"<<std::endl;
  std::cout<<"pc_size = "<<pc_size<<" label size = "<<label_size<<std::endl;
  if(label_size != pc_size)
  {
    std::cout<<"the number of labels are inconsistent with the number of points \n";
    //return -1;
  }  

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
	cloud->width    = pc_size;
	cloud->height   = 1;
	cloud->is_dense = false;
	unsigned int res_points = (cloud->width * cloud->height);
	cloud->points.resize (res_points);

  unsigned int pcnt = 0;
	std::cout<<"generate point cloud... \n";
	for(unsigned int i = 0; i < pc_size; i++)
	{
	  cloud->points[pcnt].x =pc[7*i+0];
	  cloud->points[pcnt].y =pc[7*i+1];
	  cloud->points[pcnt].z =pc[7*i+2];
	  cloud->points[pcnt].r =pc[7*i+4];
	  cloud->points[pcnt].g =pc[7*i+5];
	  cloud->points[pcnt].b =pc[7*i+6];
	  
	  if(haslabel)
	    cloud->points[pcnt].a =label[1*i+0];
	  else
	    cloud->points[pcnt].a = 0;

    pcnt++;
	}
	
	std::cout<<"res_points = "<<res_points<<" pcnt = "<<pcnt<<std::endl;

  std::cout<<"delete pc\n";
	delete[] pc;
 
  
	if(haslabel)
  {
    std::cout<<"delete labels\n";
	  delete[] label;
  }
  float downsamplerate = (float)downsample_num_points/(float)res_points;
  if(downsamplerate<1.0f && downsamplerate > 0.0f)
  {
    std::cout<<"Downsample cloud rate: "<<downsamplerate<<std::endl;
    CloudDownsampler ds;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ds_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
    
    ds.downsampleRadius(cloud, ds_cloud, downsamplerate);
    if(save)
      savePCD(ds_cloud, dataset);
  
    return ds_cloud;
  }
  else
  {
    std::cout<<"return full cloud\n";
    if(save)
      savePCD(cloud, dataset);

    return cloud;
  }
}

void saveEigenMatrix(std::string filename, Eigen::MatrixXf m)
{
  std::ofstream file(filename.c_str(), std::ofstream::out);
  if (file.is_open())
  {
    file << m;
  }
  else
    std::cout<<"unable to open file: "<<filename<<std::endl;
}

void saveNumPointsImage(std::string filename, unsigned int* num_points, unsigned int size_im)
{
	std::ofstream file(filename.c_str(), std::ofstream::binary);
  if (file.is_open())
  {
    char* buffer = reinterpret_cast<char*>(num_points);
    file.write(buffer, sizeof(unsigned int) * size_im);
  }
  else
    std::cout<<"unable to open file: "<<filename<<std::endl;
}

void savePointsImage(std::string filename, std::vector< std::vector< unsigned int > >* points)
{
	std::ofstream file(filename.c_str(), std::ofstream::binary);
  if (file.is_open())
  {
    for(unsigned int i = 0; i < points->size(); i++)
    {
      char* buffer = reinterpret_cast<char*>(&(*points)[i][0]);
      file.write(buffer, sizeof(unsigned int) * (*points)[i].size());
    }
  }
  else
    std::cout<<"unable to open file: "<<filename<<std::endl;
}

void saveFloatImage(std::string filename, float* num_points, unsigned int size_im)
{
	std::ofstream file(filename.c_str(), std::ofstream::binary);
  if (file.is_open())
  {
    char* buffer = reinterpret_cast<char*>(num_points);
    file.write(buffer, sizeof(float) * size_im);
  }
  else
    std::cout<<"unable to open filetransform: "<<filename<<std::endl;
}

/**********************************************************
 * Render views main program
 * inputs: 
 *     clound_filename: name of textfile containing point cloud (x y z i r g b) and label filename containing integer labels as in semantic3D
 *     location: output folder location
 *     haslabel: flag with 1 for labeled point clouds and 0 for unlabeled point clouds
 *     downsample_num_points: roughly number of points after downsampling (importance sampling)
 *     haspcd: use pcd file
 *     is_test: for test or validation data
 * outputs:
 *     depth images: binary files containing float arrays, 
 *     label images: png, 
 *     RGB images: png
 *     camera pose matrices: .txt files with [R t],
 *     number of points images: integer array (column major) with number of points per pixel,
 *     point id images: integer array with global point id (row index in cloud_filename textfile) ordered per pixel column major 
 **********************************************************/

int main(int argc, char** argv)
{
  /*****************************
   * Parse config
   *****************************/
  if(argc < 3)
  {
    std::cout<<"need a least 2 input arguments\n";
    return -1;
  }
  std::cout<<"read args, argc = "<<argc<<std::endl;
	std::string clound_filename = argv[1];
  std::string location = argv[2];

	bool haslabel = true;
	if(argc > 3)
	{
	  std::string haslabelstring = argv[3];
    std::cout<<"has label string: "<<haslabelstring<<std::endl;
	  if(haslabelstring.compare("0") == 0)
	    haslabel = false;
  
    std::cout<<"has label = "<<haslabel<<std::endl;
	}
	
	unsigned int downsample_num_points = 100000000;
	if(argc > 4)
	{
	  std::string myString = argv[4];
    std::istringstream buffer(myString);
	  buffer >> downsample_num_points;
	  if(downsample_num_points<0)
	    downsample_num_points = 100000000;
	    
	  std::cout<<"downsample_num_points = "<<downsample_num_points<<std::endl;
	}
	
  std::string is_test_string = "1";
  bool is_test = false;
  if(argc > 5)
	{
    is_test_string = argv[5];
	  if(is_test_string.compare("1") == 0)
	    is_test = true;
  }

  std::string use_pcd_string = "0";
  bool use_pcd = false;
  if(argc > 6)
	{
    use_pcd_string = argv[6];
	  if(use_pcd_string.compare("1") == 0)
	    use_pcd = true;
  }



  /***********************************
   * Init parameters and read data
   ***********************************/
	//rotation in degree
	float rot_x=0;
	float rot_y=0;
	float rot_z=0;
	//translation	
	double trans_x=513956.3197161849200000;
	double trans_y=5426766.6255130861000000;
	double trans_z=276.9661760997179300;

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);

  std::cout<<"reading cloud from txt\n";
  if(use_pcd)
    cloud = readPointCloudFromPCD(clound_filename);
  else
    cloud = readPointCloudFromTxt(clound_filename, haslabel, downsample_num_points, true);

  if(cloud->size() <= 0)
  {
    std::cout << "could not read file\n";
    return -1;
  }

  std::cout<<"cloud size = "<<cloud->points.size()<<std::endl;
  std::string sizefile = location + "/" + clound_filename + "cloud_size.txt";
  std::ofstream numfile(sizefile.c_str() , std::ofstream::out);
  if(numfile.is_open())
  {
    numfile << cloud->points.size();
    numfile.close();
  }
  else
    std::cout<<"unable to open file " + sizefile<<std::endl;
  
	double* intr_mat = new double[9];
  for(unsigned int i = 0; i < 9;i++)
    intr_mat[i] = 0.0;

	int rows = 8708;
	int cols = 11608;

	double f = 51.6829425484485650/0.0045999880303564;
	double x0 = 5798.5783629179004000; 
	double y0 = 4358.1365279104657000;


	intr_mat[0] = f;
	intr_mat[2] = x0;
	intr_mat[4] = -f;
	intr_mat[5] = y0;	
	intr_mat[8] = 1.0;
  
  double* dist_coeff = NULL;
	
	float lim = std::pow(10,-5); 
	float cluster_val_threshold = 0.01f;
	unsigned int num_iterations = 30;
	float cluster_width = 0.1f;
	


  /***************************************************
   * render point cloud from specific camera view
   ***************************************************/

  Eigen::Matrix4f tot_transform;

  pointCloudProjection::PointCloudToDepthBase point_cloud_projector(intr_mat, dist_coeff, rows, cols);

point_cloud_projector.addPointCloud(cloud);
  
std::cout<<"transform cloud.. \n";
  tot_transform = point_cloud_projector.transform(rot_x,rot_y,rot_z,trans_x,trans_y,trans_z);


std::cout<<tot_transform<<std::endl;



  //depth image
  float* depth_im = point_cloud_projector.getDepthImageMeanShift(cluster_val_threshold, lim, num_iterations, cluster_width, tot_transform);

  float* R_im = point_cloud_projector.getRimage();
  float* G_im = point_cloud_projector.getGimage();
  float* B_im = point_cloud_projector.getBimage();

	int* label_im = point_cloud_projector.getLabelimage();

  cv::Mat label_image = cv::Mat(rows, cols, CV_32S, label_im);
  cv::Mat R_image = cv::Mat(rows, cols, CV_32F, R_im);
  cv::Mat G_image = cv::Mat(rows, cols, CV_32F, G_im);
  cv::Mat B_image = cv::Mat(rows, cols, CV_32F, B_im);

  std::vector<int> compression_params;
compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

std::cout<<"save image: "<<location + "/" + clound_filename + "/depth/" + clound_filename + "depth"+"im_num"+".png\n";

saveFloatImage(location + "/" + clound_filename + "/depth/" + clound_filename + "depth"+"im_num"+".png", depth_im, rows * cols);

  imwrite(location + "/" + clound_filename + "/label/" + clound_filename + "label"+"im_num"+".png", label_image, compression_params);

  std::vector<cv::Mat> RGB_im;	
  RGB_im.push_back(B_image);
  RGB_im.push_back(G_image);
  RGB_im.push_back(R_image);

  cv::Mat color;
  cv::merge(RGB_im,color);

  imwrite(location + "/" + clound_filename + "/rgb/" + clound_filename + "RGB" + "im_num" + ".png", color, compression_params);

  saveEigenMatrix(location + "/" + clound_filename + "/transform/" + clound_filename + "Transform" + "im_num" + ".txt", tot_transform);

if(is_test)
{
	  unsigned int* num_points = point_cloud_projector.getNumSources();
    std::vector< std::vector< unsigned int > >* points = point_cloud_projector.getVisablePoints();
    saveNumPointsImage(location + "/" + clound_filename + "/visability/" + clound_filename + "numPoints" + "im_num" +".bin", num_points, rows * cols);
    savePointsImage(location + "/" + clound_filename + "/visability/" + clound_filename + "Points" + "im_num" +".bin", points);
  points->clear();
  delete[] num_points;
}
// ss.clear();	
// im_num.clear();
delete[] R_im;
delete[] G_im;
delete[] B_im;
delete[] label_im;
delete[] depth_im;





  /***************************************************
   * render point cloud from different camera views
   ***************************************************/
	// for(int iter = 0; iter < tot_iter; iter++)
	// {

	//   int c = iter % num_angles;
	//   int u = iter / num_angles;
	//   std::cout<< iter <<'\n';

  
	//   Eigen::Matrix4f tot_transform;
	//   pointCloudProjection::PointCloudToDepthBase point_cloud_projector(intr_mat, dist_coeff, rows, cols);

 //    point_cloud_projector.addPointCloud(cloud);
      
 //    std::cout<<"transform cloud.. \n";
	//   tot_transform = point_cloud_projector.transform(rot_x,c*rot_angle,rot_z,trans_x,trans_y,trans_z);
	//   if (u == 2)
	//   {
	// 	  tot_transform = point_cloud_projector.transform(-M_PI/8,0,0,0,0,0) * tot_transform;
	//   }
 //    else if(u == 3)
 //    {
 //      tot_transform = point_cloud_projector.transform(-M_PI/8-M_PI/12 ,0,0,0,0,0) * tot_transform;
 //    }
	//   else if(u == 1)
	// 	  tot_transform = point_cloud_projector.transform(0,0,0,0,0,7) * tot_transform;
 //    else if(u == 4)
 //      tot_transform = point_cloud_projector.transform(-M_PI/12 ,0,0,0,0,0) * tot_transform;

 //    std::cout<<tot_transform<<std::endl;

	//   ss << iter;

	//   std::string im_num = ss.str();
	//   ss.str("");

	//   //depth image
	//   float* depth_im = point_cloud_projector.getDepthImageMeanShift(cluster_val_threshold, lim, num_iterations, cluster_width, tot_transform);

	//   float* R_im = point_cloud_projector.getRimage();
	//   float* G_im = point_cloud_projector.getGimage();
	//   float* B_im = point_cloud_projector.getBimage();

 //   	int* label_im = point_cloud_projector.getLabelimage();

	//   cv::Mat label_image = cv::Mat(rows, cols, CV_32S, label_im);
	//   cv::Mat R_image = cv::Mat(rows, cols, CV_32F, R_im);
	//   cv::Mat G_image = cv::Mat(rows, cols, CV_32F, G_im);
	//   cv::Mat B_image = cv::Mat(rows, cols, CV_32F, B_im);

	//   std::vector<int> compression_params;
 //    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
 //  	compression_params.push_back(9);

 //    std::cout<<"save image: "<<location + "/" + clound_filename + "/depth/" + clound_filename + "depth"+im_num+".png\n";

 //    saveFloatImage(location + "/" + clound_filename + "/depth/" + clound_filename + "depth"+im_num+".png", depth_im, rows * cols);
    
	//   imwrite(location + "/" + clound_filename + "/label/" + clound_filename + "label"+im_num+".png", label_image, compression_params);

	//   std::vector<cv::Mat> RGB_im;	
	//   RGB_im.push_back(B_image);
	//   RGB_im.push_back(G_image);
	//   RGB_im.push_back(R_image);

	//   cv::Mat color;
	//   cv::merge(RGB_im,color);
	
	//   imwrite(location + "/" + clound_filename + "/rgb/" + clound_filename + "RGB" + im_num + ".png", color, compression_params);
	
	//   saveEigenMatrix(location + "/" + clound_filename + "/transform/" + clound_filename + "Transform" + im_num + ".txt", tot_transform);

 //    if(is_test)
 //    {
 //  	  unsigned int* num_points = point_cloud_projector.getNumSources();
	//     std::vector< std::vector< unsigned int > >* points = point_cloud_projector.getVisablePoints();
	//     saveNumPointsImage(location + "/" + clound_filename + "/visability/" + clound_filename + "numPoints" + im_num +".bin", num_points, rows * cols);
	//     savePointsImage(location + "/" + clound_filename + "/visability/" + clound_filename + "Points" + im_num +".bin", points);
 //      points->clear();
 //      delete[] num_points;
 //    }
 //    ss.clear();	
 //    im_num.clear();
 //    delete[] R_im;
 //    delete[] G_im;
 //    delete[] B_im;
 //    delete[] label_im;
 //    delete[] depth_im;
    
	// }







	std::cout<<"Done!"<<'\n';

	return 0;
}
