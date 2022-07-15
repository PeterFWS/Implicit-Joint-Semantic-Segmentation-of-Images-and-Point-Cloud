#include "point_cloud_to_depth.h"
#include <math.h>
#include <algorithm>    // std::sort
#include <Eigen/Dense>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <opencv2/opencv.hpp>

#include "omp.h" 

namespace pointCloudProjection
{

	bool cmp( const MeanShiftClusterColor* p1, const MeanShiftClusterColor* p2 )
	{
		return *p1 > *p2;
	}


	/*****************************
		* Modified based on origin code

		* by Fangwen Shu
	*****************************/
	PointCloudToDepthBase::PointCloudToDepthBase(float* intrinsic_camera_matrix, double* distortion_coefficients, unsigned int in_rows, unsigned int in_cols, int k_g)
	{
	    //allocate vector memory
	    for(unsigned int i = 0; i < in_rows*in_cols; i++)
	    {
	      std::vector<DepthImagePoint> tmp;
	      depth_image_points_.push_back(tmp);
	    }

	    rows_ = in_rows;
	    cols_ = in_cols;

	    k_guss_ = k_g;

	    fx_ = intrinsic_camera_matrix[0];
	    fy_ = intrinsic_camera_matrix[4];
	    cx_ = intrinsic_camera_matrix[2];
	    cy_ = intrinsic_camera_matrix[5];

	    std::cout<<"\n";

		std::cout<<"Initializing point cloud to depth mapper\n";
		std::cout<<"focal length: \n"<<" fx = "<<fx_<<" fy = "<<fy_<<std::endl;
		std::cout<<"principle points: \n"<<" cx = "<<cx_<<" cy = "<<cy_<<std::endl;
		std::cout<<"size of image: \n"<<" rows = "<<rows_<<" cols = "<<cols_<<std::endl;

		std::cout<<"\n";
	    
		if(distortion_coefficients != NULL)
		{
      		length_dist_coeff_ = 8;
			dist_coeff_ = new float[length_dist_coeff_];
			for(unsigned int i = 0; i < length_dist_coeff_; i++)
			{
        		dist_coeff_[i] = double(distortion_coefficients[i]);
			}
		}
		else
			length_dist_coeff_ = 0;

		cloud_id_ = 0;
	}

	PointCloudToDepthBase::~PointCloudToDepthBase()
	{
	    depth_image_points_.clear();
	}

	float* PointCloudToDepthBase::getClusterValImage()
	{
		return cluster_val_image_;
	}

	float* PointCloudToDepthBase::getRimage()
	{
		return R_image_;
	}

	float* PointCloudToDepthBase::getGimage()
	{
		return G_image_;
	}
	float* PointCloudToDepthBase::getBimage()
	{
		return B_image_;
	}

	int* PointCloudToDepthBase::getLabelimage()
	{
		return label_image_;
	}

	unsigned int* PointCloudToDepthBase::getNumSources()
	{
		return num_sources_;
	}

	std::vector< std::vector < unsigned int > >* PointCloudToDepthBase::getVisablePoints()
	{
		return &visible_points_;
	}



	void PointCloudToDepthBase::addPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
	{
		std::cout<<"adding point cloud..\n";
		big_cloud_.push_back(cloud);

		//  big_cloud_.size() = 1
		//  big_cloud_[0]->points.size() = 14129889

		std::vector<unsigned char> tmp_mask(cloud->points.size(),1);
		cluster_point_mask_.push_back(tmp_mask);
	}


	float* PointCloudToDepthBase::getDepthImageMeanShift(float cluster_val_threshold, float lim, unsigned int num_iterations, float cluster_width, Eigen::MatrixXf transform)
	{
		std::vector< std::vector< unsigned int > > mask; 
		for(unsigned int c = 0; c < big_cloud_.size(); c++)
		{
			std::vector<unsigned int> mask_in(big_cloud_[c]->points.size(),1);
			mask.push_back(mask_in);
		}

		projectCloud(mask, transform);
		processDepthImageMeanShift(cluster_val_threshold, lim, num_iterations, cluster_width);
		return depth_image_;
	}





	/*****************************
		* Modified based on origin code
			* modify to fit aerial image situation

		* by Fangwen Shu
	*****************************/
	Eigen::MatrixXf PointCloudToDepthBase::transform(float trans_x, float trans_y, float trans_z, float r11, float r12, float r13, float r21, float r22, float r23, float r31, float r32, float r33)
	{	
		Eigen::MatrixXf R(3,3); 
		Eigen::MatrixXf K(3,3);
		Eigen::MatrixXf X0(3,1);
		Eigen::MatrixXf Rt(3,4);

		Eigen::MatrixXf temp(3,1);

		Eigen::MatrixXf P(3,4);


		// define R
		R(0,0) = r11;
		R(0,1) = r12;
		R(0,2) = r13;
		R(1,0) = r21;
		R(1,1) = r22;
		R(1,2) = r23;
		R(2,0) = r31;
		R(2,1) = r32;
		R(2,2) = r33;

		X0(0,0) = trans_x;
		X0(1,0) = trans_y;
		X0(2,0) = trans_z;


		//define Rt
		temp = - R*X0; // 3x1
		Rt(0,0) = r11;
		Rt(0,1) = r12;
		Rt(0,2) = r13;
		Rt(1,0) = r21;
		Rt(1,1) = r22;
		Rt(1,2) = r23;
		Rt(2,0) = r31;
		Rt(2,1) = r32;
		Rt(2,2) = r33;

		Rt(0,3) = temp(0,0);
		Rt(1,3) = temp(1,0);
		Rt(2,3) = temp(2,0);

		// define K
		K(0,0) = fx_;
		K(0,1) = 0.0;
		K(0,2) = cx_;
		K(1,0) = 0.0;
		K(1,1) = fy_;
		K(1,2) = cy_;
		K(2,0) = 0.0;
		K(2,1) = 0.0;
		K(2,2) = 1.0;

		P = K*Rt;

		std::cout << "K: \n" << K << std::endl;
		std::cout << "Rt: \n" << Rt << std::endl;

		std::cout << "P: \n" << P << std::endl;

		return P;
	}

	unsigned int PointCloudToDepthBase::readToArray(std::string config_file_name, float** files_to_process, unsigned int row_size)
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

	void PointCloudToDepthBase::projectCloud(std::vector< std::vector< unsigned int > >& mask, Eigen::MatrixXf transform)
	{	
		std::cout<<"projectCloud... \n";

		float proj_point[2];
		float distorted_point[2];
		float pixel_coordinates[2];
		unsigned int pts_cnt = 0;

		Eigen::MatrixXf xyz(4,1); // homogeneous coordinate
		Eigen::MatrixXf xyz_trans(3,1); //homogeneous coordinates after transformation


		float* px;
		unsigned int px_size = 0;
		px_size = readToArray("/home/fangwen/masThesis/point_splatting/px.txt", &px, 1);

		float* py;
		unsigned int py_size = 0;
		py_size = readToArray("/home/fangwen/masThesis/point_splatting/py.txt", &py, 1);

		float* pdepth;
		unsigned int pdepth_size = 0;
		pdepth_size = readToArray("/home/fangwen/masThesis/point_splatting/depth.txt", &pdepth, 1);


		std::cout << "\n==>splatting radius (kxk Gaussian): " << k_guss_ << std::endl;


		for(unsigned int c = 0; c < big_cloud_.size(); c++)
		{
			for(unsigned int i = 0; i < big_cloud_[c]->points.size(); i++)
			{
				// xyz(0,0) = big_cloud_[c]->points[i].x;
				// xyz(1,0) = big_cloud_[c]->points[i].y;
				// xyz(2,0) = big_cloud_[c]->points[i].z;
				// xyz(3,0) = 1.0;

				// xyz_trans = transform * xyz;
	 

				// if(abs(xyz_trans(2,0)) > 0.05f && mask[c][i] == 1)
				// {	     
					// proj_point[0] = xyz_trans(0,0) / xyz_trans(2,0); // Normalization of pixel points
					// proj_point[1] = xyz_trans(1,0) / xyz_trans(2,0);

					proj_point[0] = px[i];
					proj_point[1] = py[i];

					// float depth = xyz_trans(2,0);

					float depth = abs(pdepth[i]);
					//float depth = 0.0;



					// float r = big_cloud_[c]->points[i].r;
					// float g = big_cloud_[c]->points[i].g;
					// float b = big_cloud_[c]->points[i].b;

					// float label = big_cloud_[c]->points[i].a;

					// if ((int)label == 6)
					// {
					// 	depth = abs(pdepth[i]) + 50.0;
					// }
					// else 
					// 	depth = abs(pdepth[i]);

					float conf_val = conf_vec_.size()==0 ? 1.0f: conf_vec_[c][i];

					addPoint(proj_point, depth, POINT_CLOUD_TO_DEPTH_GAUSS, i, conf_val);


				// } 

				pts_cnt++; 
			}
			//get ready for new cloud
			// cloud_id_++;  // it will not change since we only have 1 cloud
		}

		delete[] px;
		delete[] py;
		delete[] pdepth;

	}

	void PointCloudToDepthBase::addPoint(float* pixel, float depth,unsigned int method, unsigned int id, float conf)
	{
	    //std::cout<<"addPoint\n";

	    // switch(method)
	    // {
	        // case POINT_CLOUD_TO_DEPTH_NEAREST:
	        //     addPointNearest(pixel, depth,id, conf);
	        //     break;
	        // case POINT_CLOUD_TO_DEPTH_BILINEAR:
	        //     addPointBilinear(pixel, depth,id, conf);
	        //     break;
			// case POINT_CLOUD_TO_DEPTH_GAUSS: 
			 	addPointGauss(pixel, depth,id, conf);
	            // break;

	        // default:
	        //     std::cout<<"PointCloudToDepthBase::addPoint: requested add point method not implemented. Use nearest instead\n";
	        //     addPointNearest(pixel, depth,id, conf);
	    // }
	}

	// void PointCloudToDepthBase::addPointNearest(float* pixel, float depth, unsigned int id, float conf)
	// {
	// 	int pixel_x = int(pixel[0] + 0.5f);
	// 	int pixel_y = int(pixel[1] + 0.5f);

	// 	addPointInteger(pixel, pixel_x, pixel_y, depth, id, conf);
	// }

	//  void PointCloudToDepthBase::addPointInteger(float* pixel, int pixel_x,int pixel_y, float depth, unsigned int id, float conf)
	// {
	//     if(pixel_x >= 0 && pixel_x < cols_ && pixel_y >= 0 && pixel_y < rows_)
	//     {
	//         float dist_x = pixel_x-pixel[0];
	//         float dist_y = pixel_y-pixel[1];
	//         float dist = std::sqrt(dist_x*dist_x+dist_y*dist_y);
	//         float w = conf*(1.0f-dist);

	// 		int ind = pixel_y * cols_+ pixel_x;

	// 		DepthImagePoint p(depth,w, cloud_id_, id);
	        
	//         depth_image_points_[ind].push_back(p);
	//     }
	// }

	// void PointCloudToDepthBase::addPointBilinear(float* pixel, float depth, unsigned int id, float conf)
	// {
	// 	int pixel_x = int(pixel[0] + 0.5f);
	// 	int pixel_y = int(pixel[1] + 0.5f);

	// 	addPointInteger(pixel, pixel_x, pixel_y, depth, id, conf);
	// 	addPointInteger(pixel, pixel_x+1, pixel_y, depth, id, conf);
	// 	addPointInteger(pixel, pixel_x, pixel_y+1, depth, id, conf);
	// 	addPointInteger(pixel, pixel_x+1, pixel_y+1, depth, id, conf);
	// }


	void PointCloudToDepthBase::addPointGauss(float* pixel, float depth,unsigned int id, float conf)
	{
	  int pixel_x = int(pixel[0] + 0.5f);
	  int pixel_y = int(pixel[1] + 0.5f);

	  addPointIntegerGauss(pixel, pixel_x, pixel_y, depth, id, conf);
	  // 3x3
		// addPointIntegerGauss(pixel, pixel_x+1, pixel_y, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x, pixel_y+1, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x+1, pixel_y+1, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x-1, pixel_y+1, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x+1, pixel_y-1, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x, pixel_y-1, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x-1, pixel_y, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x-1, pixel_y-1, depth, id, conf);
		// 5x5
		// addPointIntegerGauss(pixel, pixel_x+2, pixel_y, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x+2, pixel_y-1, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x+2, pixel_y-2, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x+2, pixel_y+1, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x+2, pixel_y+2, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x, pixel_y-2, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x+1, pixel_y-2, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x-1, pixel_y-2, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x-2, pixel_y-2, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x, pixel_y+2, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x+1, pixel_y+2, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x-1, pixel_y+2, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x-2, pixel_y+2, depth+2, id, conf);
		// addPointIntegerGauss(pixel, pixel_x-2, pixel_y-1, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x-2, pixel_y, depth, id, conf);
		// addPointIntegerGauss(pixel, pixel_x-2, pixel_y+1, depth, id, conf);
		// 7x7
		addPointIntegerGauss(pixel, pixel_x+3, pixel_y, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+3, pixel_y-1, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+3, pixel_y-2, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+3, pixel_y-3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+3, pixel_y+1, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+3, pixel_y+2, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+3, pixel_y+3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x, pixel_y-3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+1, pixel_y-3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+2, pixel_y-3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-1, pixel_y-3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-2, pixel_y-3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-3, pixel_y-3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-3, pixel_y, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-3, pixel_y-1, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-3, pixel_y-2, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-3, pixel_y+1, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-3, pixel_y+2, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-3, pixel_y+3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x, pixel_y+3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+1, pixel_y+3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+2, pixel_y+3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-1, pixel_y+3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-2, pixel_y+3, depth, id, conf);
		// 9x9
		addPointIntegerGauss(pixel, pixel_x+4, pixel_y-1, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+4, pixel_y-2, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+4, pixel_y-3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+4, pixel_y-4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+4, pixel_y+1, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+4, pixel_y+2, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+4, pixel_y+3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+4, pixel_y+4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+4, pixel_y, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+1, pixel_y-4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+2, pixel_y-4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+3, pixel_y-4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x, pixel_y-4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-1, pixel_y-4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-2, pixel_y-4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-3, pixel_y-4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-4, pixel_y-4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-4, pixel_y-1, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-4, pixel_y-2, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-4, pixel_y-3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-4, pixel_y, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-4, pixel_y+1, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-4, pixel_y+2, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-4, pixel_y+3, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-4, pixel_y+4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+1, pixel_y+4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+2, pixel_y+4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x+3, pixel_y+4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x, pixel_y+4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-1, pixel_y+4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-2, pixel_y+4, depth, id, conf);
		addPointIntegerGauss(pixel, pixel_x-3, pixel_y+4, depth, id, conf);









	}


	void PointCloudToDepthBase::addPointIntegerGauss(float* pixel, int pixel_x,int pixel_y, float depth, unsigned int id, float conf)
	{

	    if(pixel_x >= 0 && pixel_x < cols_ && pixel_y >= 0 && pixel_y < rows_) 
	    {	
	        float dist_x = pixel_x-pixel[0];
	        float dist_y = pixel_y-pixel[1];

	        float dist_sqr = dist_x*dist_x+dist_y*dist_y;
	        float w = conf*1.0f/(2*M_PI*0.25f)*std::exp(-0.5f*dist_sqr/0.25f);

	        DepthImagePoint p(depth , w, cloud_id_, id);

	        int ind = pixel_y * cols_+ pixel_x;

	        depth_image_points_[ind].push_back(p);
	    }
	}

	/*****************************
		* Modified based on origin code

		* by Fangwen Shu
	*****************************/
	void PointCloudToDepthBase::processDepthImageMeanShift(float cluster_val_threshold, float lim, unsigned int num_iterations, float cluster_width)
	{	
		std::cout<<"postProcessDepthImageMeanShift2... \n";

		depth_image_ = new float[depth_image_points_.size()];
		R_image_ = new float[depth_image_points_.size()];
		G_image_ = new float[depth_image_points_.size()];
		B_image_ = new float[depth_image_points_.size()];
		label_image_ = new int[depth_image_points_.size()];
		cluster_val_image_ = new float[depth_image_points_.size()];
		num_sources_ = new unsigned int[depth_image_points_.size()];

	  	visible_points_.resize(depth_image_points_.size()); //depth_image_points_.size() = cols*rows

		float gaussian_var = 0.5f*cluster_width*0.5f*cluster_width;
		int max_num_points = 100;

		// std::cout << "depth_image_points_.size(): " << depth_image_points_.size() << std::endl;


		#pragma omp parallel for
		for(unsigned int i = 0; i < depth_image_points_.size(); i++)
		{
			depth_image_[i] = 0.0f;
			cluster_val_image_[i] = 0.0f;
			R_image_[i] = 0.0f;
			G_image_[i] = 0.0f;
			B_image_[i] = 0.0f;
			label_image_[i] = 0; //background = 0
			num_sources_[i] = 0;
	    
			if(depth_image_points_[i].size() > 0)
		  	{

				if(depth_image_points_[i].size() == 1)
				{
					depth_image_[i] = depth_image_points_[i][0].d;
					R_image_[i] = big_cloud_[depth_image_points_[i][0].cloud_id_]->points[depth_image_points_[i][0].pt_id_].r;
					G_image_[i] = big_cloud_[depth_image_points_[i][0].cloud_id_]->points[depth_image_points_[i][0].pt_id_].g;
					B_image_[i] = big_cloud_[depth_image_points_[i][0].cloud_id_]->points[depth_image_points_[i][0].pt_id_].b;
					label_image_[i] = big_cloud_[depth_image_points_[i][0].cloud_id_]->points[depth_image_points_[i][0].pt_id_].a;

	   				continue;
				}

	      		srand (time(NULL)); // random seed

			    unsigned char *rand_mask = new unsigned char[depth_image_points_[i].size()];
			    std::memset(rand_mask, 1, depth_image_points_[i].size());

	      		// if the pixel contains more that maximum number of points cluster wrt a random sub set of the points
				if(depth_image_points_[i].size() > max_num_points)
				{ 
					float tot_num_f = (float)(depth_image_points_[i].size());
					float max_num_points_f = (float)(max_num_points);

					float rand_thresh = (10000.0f * max_num_points_f)/tot_num_f;
					rand_thresh = rand_thresh < 30.0f ? 30.0f : rand_thresh;

					int mask_cnt = depth_image_points_[i].size();

					mask_cnt = 0;
					for(unsigned int ps = 0; ps < depth_image_points_[i].size(); ps++)
					{
						int r1 = rand()%10000 + 1;
						if((float)r1 > rand_thresh)
						{
							rand_mask[ps] = 0;
							mask_cnt++;
						}
					}

					if(depth_image_points_[i].size() - mask_cnt < 10)
					{
						for(unsigned int ps = 0; ps < max_num_points; ps++)
						{
							rand_mask[ps] = 1;
						}
					}
				}

				/*
				* Find good starting points for the mean shift clustering. This is for speeding up the calculations
				*/

		      	//sample kernel density at the positions of the points
				std::vector< DepthIdPair > depth_id_pair;
				std::vector< DepthImagePoint > pts = depth_image_points_[i];

				float* kernel_desity_eval = new float[pts.size()];
				
				for(unsigned int ps = 0; ps < pts.size(); ps++)
				{
					if(rand_mask[ps] == 1)
					{
						DepthIdPair my_pair(pts[ps].d, ps);
						depth_id_pair.push_back(my_pair);

						kernel_desity_eval[ps] = 0.0f;		

						for(unsigned int p = 0; p < pts.size(); p++)
						{
							if(rand_mask[p] == 1)
							{
								float dist = pts[p].d-pts[ps].d;
								dist*=dist;
								kernel_desity_eval[ps] += pts[p].w*exp(-0.5f*dist/gaussian_var); //use some bandwidth of gaussian kernel based on cluster_width
							}				  
						}
					}
				}	
			
		        //sort samples wrt kde
				std::sort(depth_id_pair.begin(), depth_id_pair.end());
				
				float min_depth = depth_id_pair[0].d;
				float max_depth = depth_id_pair[depth_id_pair.size()-1].d;

		        //find maximum kde for a set if intervall within the range of the depth distribution of the pixel
				int num_steps = (int)ceil((max_depth-min_depth)/(8*cluster_width)); 
				unsigned char *max_points = new unsigned char[pts.size()];

				std::memset(max_points, 0, pts.size());
				unsigned int ds = 0;

				max_points[depth_id_pair[0].id] = 1; // in case no steps

				for(int st = 1; st <= num_steps && ds < depth_id_pair.size(); st++)
				{
					float curent_lim = min_depth + st * 8 * cluster_width;//
					unsigned int current_max;
					//check if points exists in bin
					if(depth_id_pair[ds].d <curent_lim)
					{
						//initialize max in bin with first point (smallest depth)
						current_max = ds;
						max_points[depth_id_pair[current_max].id] = 1;
						
					}

					//loop through all points in bin
					while(depth_id_pair[ds].d < curent_lim && ds < depth_id_pair.size())
				    {
						if(kernel_desity_eval[depth_id_pair[ds].id] > kernel_desity_eval[depth_id_pair[current_max].id])
						{
							max_points[depth_id_pair[current_max].id] = 0;	
							current_max = ds;
							max_points[depth_id_pair[current_max].id] = 1;
						}

						ds++;
					}
				}

	        	unsigned int num_points_for_iter = pts.size() > 60 ? 60 : pts.size();
			
				//clustering
				std::vector<MeanShiftClusterColor*> clusters;
				for(unsigned int ps = 0; ps < pts.size(); ps++)
				{
					if(max_points[ps] == 1 && rand_mask[ps] == 1)
					{
						//meanshift
						float mx = pts[ps].d;

						float mr = big_cloud_[pts[ps].cloud_id_]->points[pts[ps].pt_id_].r;
						float mg = big_cloud_[pts[ps].cloud_id_]->points[pts[ps].pt_id_].g;
						float mb = big_cloud_[pts[ps].cloud_id_]->points[pts[ps].pt_id_].b;
						float diff = 1.0f;
						float mx_old;

						float cluster_val;
						unsigned int iter = 0;
						float sum_w = 0.0f;
						while(diff > lim && iter++ < num_iterations)
						{
							float sum_xw = 0.0f;
							sum_w = 0.0f;


							for(unsigned int p = 0; p < pts.size(); p++)
							{
								if(rand_mask[p] == 1)
								{
									float dist = pts[p].d-mx;
									dist *= dist;

									float w_K = pts[p].w * exp(-0.5f * dist / gaussian_var); //use some width of gaussian kernel based on cluster_width
									sum_xw += w_K*pts[p].d;
									sum_w += w_K;
								}
							}

							mx_old = mx;
							mx = sum_w > 0 ? sum_xw/sum_w: 0.0f;


							diff = std::abs(mx_old-mx);
							cluster_val = sum_w;
						}

						//generate cluster color
						if(sum_w > 0.0f)
						{
							float sum_xr = 0.0f;
							float sum_xg = 0.0f;
							float sum_xb = 0.0f;
							for(unsigned int p = 0; p < pts.size(); p++)
							{
								if(rand_mask[p] == 1)
								{
									float dist = pts[p].d-mx;
									dist *= dist;

									float w_K = pts[p].w * exp(-0.5f * dist / gaussian_var); //use some width of gaussian kernel based on cluster_width
									sum_xr += w_K*big_cloud_[pts[p].cloud_id_]->points[pts[p].pt_id_].r;
									sum_xg += w_K*big_cloud_[pts[p].cloud_id_]->points[pts[p].pt_id_].g;
									sum_xb += w_K*big_cloud_[pts[p].cloud_id_]->points[pts[p].pt_id_].b;
								}
								mr = sum_xr/sum_w;
								mg = sum_xg/sum_w;
								mb = sum_xb/sum_w;
							}
						}

						bool new_cluster_found = true;

						for(unsigned int c = 0; c < clusters.size(); c++)
						{	
							if(std::abs(clusters[c]->mean-mx) < cluster_width*0.5f)
							{
								new_cluster_found = false;
								clusters[c]->update(cluster_val, mx, mr, mg, mb);
								break;
							}

						}

						if(new_cluster_found)
						{
							MeanShiftClusterColor* cluster = new MeanShiftClusterColor(cluster_val, mx, mr, mg, mb, 1.0f);
							clusters.push_back(cluster);
						}
					}
				}

		        // sort cluster wrt the cmp-metric (kde and depth)
				std::sort(clusters.begin(), clusters.end(), &cmp);

				bool found = false;
				for(unsigned int c = 0; c < clusters.size(); c++)
				{
					if( clusters[c]->val > cluster_val_threshold)
					{
						if(!found)
						{
							depth_image_[i] = clusters[c]->mean;
							R_image_[i] = clusters[c]->r;
							G_image_[i] = clusters[c]->g;
							B_image_[i] = clusters[c]->b;
							cluster_val_image_[i] =  clusters[c]->val;

							for(unsigned int p = 0; p < pts.size(); p++)
							{
								float dist = pts[p].d-clusters[c]->mean;

								//add point to its cluster
								if(std::fabs(dist) < cluster_width && pts[p].w > 0.0f)
								{
									float w = pts[p].w * exp(-0.5f * dist * dist / gaussian_var);
									clusters[c]->addPoint(w, pts[p].cloud_id_, pts[p].pt_id_, int(big_cloud_[pts[p].cloud_id_]->points[pts[p].pt_id_].a + 0.5f));
								}
							}

							//decoding pixel to point correspondance
							num_sources_[i] = clusters[c]->pt_id_.size();
							visible_points_[i] = clusters[c]->pt_id_;

							label_image_[i] = clusters[c]->getLabel();

							found = true;
						}
					}

					if(found)
						break;
				}
		
				for(unsigned int c = 0; c < clusters.size(); c++)
				{
					delete clusters[c];
				}

				clusters.clear();
				depth_id_pair.clear();
				delete[] max_points;
				delete[] kernel_desity_eval;
		     	delete[] rand_mask;

				if(i % 100000 == 0)
					std::cout<<"mean shift pixel "<<i<<" depth_image_points_[i].size() =  "<<depth_image_points_[i].size()<<std::endl;
			}
		}
	}

void PointCloudToDepthBase::insertionSortn(float* array, int size, unsigned int* inds) 
{
  int lenSortedSublist = 1;
	inds = new unsigned int[size];
	inds[0] = 0;
  while(lenSortedSublist < size) 
	{
    int pivot = array[lenSortedSublist];
    int i;
    for(i = lenSortedSublist-1; i >= 0; --i) 
		{
      if(pivot < array[i]) 
			{
        array[i+1] = array[i];
				inds[i+1] = inds[i];
      } 
			else 
			{
        array[i+1] = pivot;
				inds[i+1] = lenSortedSublist;
        break;
      }
    }
    if(i < 0) 
		{
      array[0] = pivot;
			inds[0] = lenSortedSublist;
    }
    ++lenSortedSublist;
  }
}

uint32_t rgb_to_pixel_local(uint8_t red, uint8_t green, uint8_t blue)
{
  return (red << 16) | (green << 8) | blue;
}

unsigned int PointCloudToDepthBase::getRows()
{
	return rows_;
}
unsigned int PointCloudToDepthBase::getCols()
{
	return cols_;
}

}
