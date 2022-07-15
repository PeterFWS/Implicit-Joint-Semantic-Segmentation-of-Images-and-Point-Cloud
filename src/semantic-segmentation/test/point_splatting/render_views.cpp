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




/*****************************
	* Modified based on origin code:
		* no down-sample needed
		* use full point cloud
		* re-arrange the code, looks better for me

	* by Fangwen Shu
*****************************/

pcl::PointCloud<pcl::PointXYZRGBA>::Ptr readPointCloudFromTxt(std::string dataset, bool haslabel, bool save)
{
	/*****************************
	* main()
	* cloud = readPointCloudFromTxt(clound_filename, haslabel, true);
	* clound_filename = "ps_pointcloud"
	*****************************/

	// Get label size
	float* label;
	unsigned int label_size = 0;
	if(haslabel)
		label_size = readToArray(labels_location + dataset + ".labels", &label, 1);
	else
		std::cout<<"dataset has no labels\n";

	// Get point cloud size
	float* pc;
	unsigned int pc_size = readToArray(clouds_location + dataset + ".txt", &pc, 7);
	std::cout<<"Files loaded"<<std::endl;
	std::cout<<"pc_size = "<<pc_size<<" label size = "<<label_size<<std::endl;

	if(label_size != pc_size)
	{
		std::cout<<"the number of labels are inconsistent with the number of points \n";
	}  


	// Generate point cloud
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
	cloud->width    = pc_size; // total number od points
	cloud->height   = 1; // it is set to 1 for unorganized datasets 
	cloud->is_dense = false;
	unsigned int res_points = (cloud->width * cloud->height);
	cloud->points.resize (res_points);

	unsigned int pcnt = 0;
	std::cout<<"generate point cloud... \n";

//********************
// float r_ = 0.0;
// float g_ = 0.0;
// float b_ = 0.0;
// float* px;
// unsigned int px_size = 0;
// px_size = readToArray("/home/fangwen/masThesis/px.txt", &px, 1);

// float* py;
// unsigned int py_size = 0;
// py_size = readToArray("/home/fangwen/masThesis/py.txt", &py, 1);


// cv::Mat test_image = cv::Mat(8708, 11608, CV_8UC3, cv::Scalar(0, 0, 0));
//********************

	for(unsigned int i = 0; i < pc_size; i++)
	{
		cloud->points[pcnt].x = pc[7*i+0];
		cloud->points[pcnt].y = pc[7*i+1];
		cloud->points[pcnt].z = pc[7*i+2];
		cloud->points[pcnt].r = pc[7*i+4];
		cloud->points[pcnt].g = pc[7*i+5];
		cloud->points[pcnt].b = pc[7*i+6];

		if(haslabel)
			cloud->points[pcnt].a =label[1*i+0];
		else 	
			cloud->points[pcnt].a = 0;

		pcnt++;

//********************
		// if(px[i] > 0 && px[i] < 11608 && py[i] > 0 && py[i] < 8708) 
		// {	
		// 	std::cout << px[i] << " , " << py[i] << std::endl;

		// 	r_ = pc[7*i+4];
		// 	g_ = pc[7*i+5];
		// 	b_ = pc[7*i+6];

		// 	cv::circle(test_image, cv::Point(int(px[i]), int(py[i])), 5, cv::Scalar(b_,g_,r_), cv::FILLED);
		// }
//*******************

	}

	std::cout<<"delete pc\n";
	delete[] pc;
 
	if(haslabel)
	{
		std::cout<<"delete labels\n" << std::endl;
		delete[] label;
	}

    // if(save)
    // 	savePCD(cloud, dataset);



//********************
	// imwrite("/home/fangwen/masThesis/point_splatting/result/ps_pointcloud/test.jpg", test_image);

	// std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
//********************


    return cloud;
}


/**********************************************************
 * Render views main program
 * inputs: 
 *     clound_filename: name of textfile containing point cloud (x y z i r g b) and label filename containing integer labels as in semantic3D
 *     location: output folder location
 *     haslabel: flag with 1 for labeled point clouds and 0 for unlabeled point clouds

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

	std::string clound_filename = argv[1]; // "ps_pointcloud"
	std::string location = argv[2]; // "/home/fangwen/masThesis/point_splatting/result"

	bool haslabel = true;
	if(argc > 3)
	{
		std::string haslabelstring = argv[3];
		std::cout<<"has label string: "<<haslabelstring<<std::endl;

		if(haslabelstring.compare("0") == 0)
			haslabel = false;

		std::cout<<"has label = "<<haslabel<<std::endl;
	}


	/***********************************
	* Init parameters and read data
	***********************************/
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);

	std::cout<<"reading cloud from txt\n";
	
	cloud = readPointCloudFromTxt(clound_filename, haslabel, true);

	if(cloud->size() <= 0)
	{
		std::cout << "could not read file\n";
		return -1;
	}


  	// Get point cloud size 
	std::cout<<"cloud size = "<<cloud->points.size()<<std::endl;

	// Define intrinsic matrix
	float* intr_mat_K = new float[9];
	for(unsigned int i = 0; i < 9;i++)
		intr_mat_K[i] = 0.0;

	int rows = std::stoi(argv[9]); // height 8708
	int cols = std::stoi(argv[10]); // width 11608

	float f = -51.6829425484485650/1000; // m
	float pixel_size = 0.0045999880303564/1000; //m
	float x0 = 5798.5783629179004000;  // [pixel] principle point
	float y0 = 4358.1365279104657000;  // [pixel]

	intr_mat_K[0] = f/pixel_size;
	intr_mat_K[2] = x0;
	intr_mat_K[4] = -f/pixel_size;
	intr_mat_K[5] = y0;	
	intr_mat_K[8] = 1.0;
	
	// Define some parameters
	double* dist_coeff = NULL;
		
	// float lim = std::pow(10,-5);
	// float cluster_val_threshold = 0.01f;
	// unsigned int num_iterations = 30;
	// float cluster_width = 0.1f;
	float lim = std::pow(10,std::stof(argv[4]));
	float cluster_val_threshold = std::stof(argv[5]);
	unsigned int num_iterations = std::stoul(argv[6]);
	float cluster_width = std::stof(argv[7]);

	int k_guss = std::stoi(argv[8]); // splatting radius

	std::cout << "lim: " << lim << std::endl;
	std::cout << "cluster_val_threshold: " << cluster_val_threshold << std::endl;
	std::cout << "num_iterations: " << num_iterations << std::endl;
	std::cout << "cluster_width: " << cluster_width << std::endl;
		
	std::stringstream ss;	
	int iter = 0;

	/***************************************************
	* render point cloud (from one specific view)
	***************************************************/
	pointCloudProjection::PointCloudToDepthBase point_cloud_projector(intr_mat_K, dist_coeff, rows, cols, k_guss);
	point_cloud_projector.addPointCloud(cloud);

	// Define extrinsic matrix (camera pose) information	
	std::cout <<  "Corresponding Image: CF013540.jpg" << std::endl; 
	// float trans_x = 513956.31971618492;
	// float trans_y = 5426766.6255130861;
	// float trans_z = 276.96617609971793;

	// float r11 = 0.9950306608836720;
	// float r12 = 0.0816887604073615;
	// float r13 = 0.0569291693643232;

	// float r21 = -0.0814123153947630;
	// float r22 = 0.9966547730117062;
	// float r23 = -0.0071622856022397;

	// float r31 = -0.0573238066030756;
	// float r32 = 0.0024919582847845;
	// float r33 = 0.9983525285891952;
	      
	Eigen::MatrixXf tot_transform(3,4);
	// Here only project point cloud into a specific camera view

	// std::cout<<"transform cloud.. \n";

	// tot_transform = point_cloud_projector.transform(trans_x,trans_y,trans_z, r11, r12, r13, r21, r22, r23, r31, r32, r33); 


	ss << iter;

	std::string im_num = ss.str();
	ss.str("");

	
	//depth image
	float* depth_im = point_cloud_projector.getDepthImageMeanShift(cluster_val_threshold, lim, num_iterations, cluster_width, tot_transform);

	float* R_im = point_cloud_projector.getRimage();
	float* G_im = point_cloud_projector.getGimage();
	float* B_im = point_cloud_projector.getBimage();

   	// int* label_im = point_cloud_projector.getLabelimage();

	// cv::Mat label_image = cv::Mat(rows, cols, CV_32S, label_im);
	cv::Mat R_image = cv::Mat(rows, cols, CV_32F, R_im);
	cv::Mat G_image = cv::Mat(rows, cols, CV_32F, G_im);
	cv::Mat B_image = cv::Mat(rows, cols, CV_32F, B_im);

	// std::vector<int> compression_params;
	// compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	// compression_params.push_back(9);

    // std::cout<<"save image: "<<location + "/" + clound_filename + "/depth/" + clound_filename + "depth"+im_num+".jpg\n";

    // saveFloatImage(location + "/" + clound_filename + "/depth/" + clound_filename + "depth"+im_num+".jpg", depth_im, rows * cols);
    
	// imwrite(location + "/" + clound_filename + "/label/" + clound_filename + "label"+im_num+".jpg", label_image);

	std::vector<cv::Mat> RGB_im;	
	RGB_im.push_back(B_image);
	RGB_im.push_back(G_image);
	RGB_im.push_back(R_image);

	cv::Mat color;
	cv::merge(RGB_im,color);

	imwrite(location + "/" + clound_filename + "/rgb/" + clound_filename + "RGB" + im_num + ".jpg", color);

	// saveEigenMatrix(location + "/" + clound_filename + "/transform/" + clound_filename + "Transform" + im_num + ".txt", tot_transform);

    
    ss.clear();	
    im_num.clear();
    delete[] R_im;
    delete[] G_im;
    delete[] B_im;
    // delete[] label_im;
    delete[] depth_im;
    

    std::cout<<"Down!"<<std::endl;

	return 0;
}