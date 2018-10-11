#include "point_cloud_to_depth.h"
#include <math.h>
#include <algorithm>    // std::sort
#include <Eigen/Dense>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

namespace pointCloudProjection
{

	bool cmp( const MeanShiftClusterColor* p1, const MeanShiftClusterColor* p2 )
	{
		return *p1 > *p2;
	}


	/*****************************
		* Modified based on origin code
			* re-arrange code

		* by Fangwen Shu
	*****************************/
	PointCloudToDepthBase::PointCloudToDepthBase(double* intrinsic_camera_matrix, double* distortion_coefficients, unsigned int in_rows, unsigned int in_cols)
	{
	    //allocate vector memory
	    for(unsigned int i = 0; i < in_rows*in_cols; i++)
	    {
	      std::vector<DepthImagePoint> tmp;
	      depth_image_points_.push_back(tmp);

	    }

	    rows_ = in_rows;
	    cols_ = in_cols;

	    fx_ = float(intrinsic_camera_matrix[0]);
	    fy_ = float(intrinsic_camera_matrix[4]);
	    cx_ = float(intrinsic_camera_matrix[2]);
	    cy_ = float(intrinsic_camera_matrix[5]);

		std::cout<<"Initializing point cloud to depth mapper\n";
		std::cout<<"focal length: \n"<<"fx = "<<fx_<<" fy = "<<fy_<<std::endl;
		std::cout<<"principle points: \n"<<" cx = "<<cx_<<" cy = "<<cy_<<std::endl;
		std::cout<<"size of image: "<<" rows = "<<rows_<<" cols = "<<cols_<<std::endl;
	    
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



	void PointCloudToDepthBase::addDepthImage(float* depth_image, unsigned int rows, unsigned int cols, unsigned int method, float* conf, float conf_thresh)
	{	
		if(rows != rows_ || cols != cols_)
		{
			std::cout<<"PointCloudToDepthBase::addDepthImage: invalid depth image size\n";
			return;
		}

		float pixel_coordinates[2];
		
		for(unsigned int i = 0; i < cols*rows; i++)
	  	{
			if(depth_image[i] > 0.0f)
			{
				unsigned int x = i % cols;
				unsigned int y = i / cols;
			    pixel_coordinates[0] = (float)x;
			    pixel_coordinates[1] = (float)y;
				float conf_val = conf==NULL ? 1.0f: conf[i];
		    	if(conf_val >= conf_thresh)
		    		addPoint(pixel_coordinates, depth_image[i],method, i, conf_val);
			}    
		}

		//get ready for new cloud
		cloud_id_++;
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
	  	visible_points_.resize(depth_image_points_.size()); //contains ids of
		float gaussian_var = 0.5f*cluster_width*0.5f*cluster_width;
		int max_num_points = 80;

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

	      		//if the pixel contains more that maximum number of points cluster wrt a random sub set of the points
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

				if(i % 10000 == 0)
					std::cout<<"mean shift pixel "<<i<<" depth_image_points_[i].size() =  "<<depth_image_points_[i].size()<<std::endl;
			}
		}
	}


	float* PointCloudToDepthBase::getDepthImageMeanShift(float cluster_val_threshold, float lim, unsigned int num_iterations, float cluster_width, Eigen::Matrix4f transform)
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


	void PointCloudToDepthBase::addPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
	{
		std::cout<<"adding point cloud..\n";
		big_cloud_.push_back(cloud);

		std::vector<unsigned char> tmp_mask(cloud->points.size(),1);
		cluster_point_mask_.push_back(tmp_mask);
	}


	Eigen::Matrix4f PointCloudToDepthBase::transformCloud(float alfa,float beta,float theta,float trans_x,float trans_y,float trans_z)
	{	Eigen::Matrix4f rot_X = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f rot_Y = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f rot_Z = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
	 	
		//rot around z
	  	rot_Z (0,0) = cos (theta);
	  	rot_Z (0,1) = -sin(theta);
	  	rot_Z (1,0) = sin (theta);
	  	rot_Z (1,1) = cos (theta);
		//rot around y
	  	rot_Y (0,0) = cos (beta);
	  	rot_Y (0,2) = sin (beta);
	  	rot_Y (2,0) = -sin(beta);
	  	rot_Y (2,2) = cos (beta);
		//rot around x  	
		rot_X (1,1) = cos (alfa);
	  	rot_X (1,2) = -sin(alfa);
	  	rot_X (2,1) = sin (alfa);
	  	rot_X (2,2) = cos (alfa);

		//tot rot
		transform=rot_Z*rot_Y*rot_X;

		//add translation
		transform(0,3)=trans_x;
		transform(1,3)=trans_y;
		transform(2,3)=trans_z;

		Eigen::Vector4f xyz;
		Eigen::Vector4f xyz_trans;

		for(unsigned int c = 0; c < big_cloud_.size();c++)
		for(unsigned int i = 0; i < big_cloud_[c]->points.size();i++)
		{
		
		  xyz(0)=big_cloud_[c]->points[i].x;
		  xyz(1)=big_cloud_[c]->points[i].y;
		  xyz(2)=big_cloud_[c]->points[i].z;
		  xyz(3)=1;
		  xyz_trans=transform*xyz;

		  big_cloud_[c]->points[i].x=xyz_trans(0);
		  big_cloud_[c]->points[i].y=xyz_trans(1);
		  big_cloud_[c]->points[i].z=xyz_trans(2);
		}
		return transform;
	}


	/*****************************
		* Modified based on origin code
			* modify to fit aerial image situation

		* by Fangwen Shu
	*****************************/
	Eigen::Matrix4f PointCloudToDepthBase::transform(float alfa, float beta, float theta, float trans_x, float trans_y, float trans_z)
	{	
		Eigen::Matrix4f rot_X = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f rot_Y = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f rot_Z = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

		//rot around z
		rot_Z (0,0) = cos (theta);
		rot_Z (0,1) = -sin(theta);
		rot_Z (1,0) = sin (theta);
		rot_Z (1,1) = cos (theta);
		//rot around y
		rot_Y (0,0) = cos (beta);
		rot_Y (0,2) = sin (beta);
		rot_Y (2,0) = -sin(beta);
		rot_Y (2,2) = cos (beta);
		//rot around x  	
		rot_X (1,1) = cos (alfa);
		rot_X (1,2) = -sin(alfa);
		rot_X (2,1) = sin (alfa);
		rot_X (2,2) = cos (alfa);

		//tot rot
		transform = rot_Z * rot_Y * rot_X;
		//add translation
		transform(0,3) = trans_x;
		transform(1,3) = trans_y;
		transform(2,3) = trans_z;

		return transform;
	}


void PointCloudToDepthBase::projectCloud(std::vector< std::vector< unsigned int > >& mask)
{	
  std::cout<<"projectCloud... \n";

  float proj_point[2];
  float distorted_point[2];
  float pixel_coordinates[2];
	unsigned int pts_cnt = 0;
	for(unsigned int c = 0; c < big_cloud_.size(); c++)
	{

		for(unsigned int i = 0; i < big_cloud_[c]->points.size(); i++)
		{

			if(big_cloud_[c]->points[i].z > 0.05f && mask[c][i] == 1)
			{
				proj_point[0] = big_cloud_[c]->points[i].x/big_cloud_[c]->points[i].z;
				proj_point[1] = big_cloud_[c]->points[i].y/big_cloud_[c]->points[i].z;
				float depth = big_cloud_[c]->points[i].z;
				float r = big_cloud_[c]->points[i].r;
				float g = big_cloud_[c]->points[i].g;
				float b = big_cloud_[c]->points[i].b;
				float label = big_cloud_[c]->points[i].a;
				opencvDistort(distorted_point, proj_point, dist_coeff_, length_dist_coeff_);
				pixel_coordinates[0] = fx_ * distorted_point[0] + cx_;
				pixel_coordinates[1] = fy_ * distorted_point[1] + cy_;
				float conf_val = conf_vec_.size()==0 ? 1.0f: conf_vec_[c][i];

				addPoint(pixel_coordinates, depth, POINT_CLOUD_TO_DEPTH_GAUSS, i, conf_val);

			}   

			pts_cnt++; 
		}
		//get ready for new cloud
		cloud_id_++;
	}
}

void PointCloudToDepthBase::projectCloud(std::vector< std::vector< unsigned int > >& mask, Eigen::Matrix4f transform)
{	
  std::cout<<"projectCloud... \n";

  float proj_point[2];
  float distorted_point[2];
  float pixel_coordinates[2];
	unsigned int pts_cnt = 0;
	Eigen::Vector4f xyz;
	Eigen::Vector4f xyz_trans;

	for(unsigned int c = 0; c < big_cloud_.size(); c++)
	{

		for(unsigned int i = 0; i < big_cloud_[c]->points.size(); i++)
		{
		  xyz(0)=big_cloud_[c]->points[i].x;
      xyz(1)=big_cloud_[c]->points[i].y;
      xyz(2)=big_cloud_[c]->points[i].z;
      xyz(3)=1;
      xyz_trans=transform*xyz;
 

			if(xyz_trans(2) > 0.05f && mask[c][i] == 1)
			{	     
				proj_point[0] = xyz_trans(0)/xyz_trans(2);
				proj_point[1] = xyz_trans(1)/xyz_trans(2);
				float depth = xyz_trans(2);
				float r = big_cloud_[c]->points[i].r;
				float g = big_cloud_[c]->points[i].g;
				float b = big_cloud_[c]->points[i].b;
				float label = big_cloud_[c]->points[i].a;
				opencvDistort(distorted_point, proj_point, dist_coeff_, length_dist_coeff_);
				pixel_coordinates[0] = fx_ * distorted_point[0] + cx_;
				pixel_coordinates[1] = fy_ * distorted_point[1] + cy_;
				float conf_val = conf_vec_.size()==0 ? 1.0f: conf_vec_[c][i];

				addPoint(pixel_coordinates, depth, POINT_CLOUD_TO_DEPTH_GAUSS, i, conf_val);

			}   

			pts_cnt++; 
		}
		//get ready for new cloud
		cloud_id_++;
	}
}

void PointCloudToDepthBase::addPoint(float* pixel, float depth,unsigned int method, unsigned int id, float conf)
{
    //std::cout<<"addPoint\n";
    switch(method)
    {
        case POINT_CLOUD_TO_DEPTH_NEAREST:
            addPointNearest(pixel, depth,id, conf);
            break;
        case POINT_CLOUD_TO_DEPTH_BILINEAR:
            addPointBilinear(pixel, depth,id, conf);
            break;
				case POINT_CLOUD_TO_DEPTH_GAUSS:
				  addPointGauss(pixel, depth,id, conf);
            break;
        default:
            std::cout<<"PointCloudToDepthBase::addPoint: requested add point method not implemented. Use nearest instead\n";
            addPointNearest(pixel, depth,id, conf);
    }
}

void PointCloudToDepthBase::addPointNearest(float* pixel, float depth, unsigned int id, float conf)
{
  int pixel_x = int(pixel[0] + 0.5f);
  int pixel_y = int(pixel[1] + 0.5f);

  addPointInteger(pixel, pixel_x, pixel_y, depth, id, conf);
}

 void PointCloudToDepthBase::addPointInteger(float* pixel, int pixel_x,int pixel_y, float depth, unsigned int id, float conf)
{
    if(pixel_x >= 0 && pixel_x < cols_ && pixel_y >= 0 && pixel_y < rows_)
    {
        float dist_x = pixel_x-pixel[0];
        float dist_y = pixel_y-pixel[1];
        float dist = std::sqrt(dist_x*dist_x+dist_y*dist_y);
        float w = conf*(1.0f-dist);
				int ind = pixel_y * cols_+ pixel_x;
				DepthImagePoint p(depth,w, cloud_id_, id);
        
        depth_image_points_[ind].push_back(p);
    }
}

void PointCloudToDepthBase::addPointBilinear(float* pixel, float depth, unsigned int id, float conf)
{
  int pixel_x = int(pixel[0] + 0.5f);
  int pixel_y = int(pixel[1] + 0.5f);

  addPointInteger(pixel, pixel_x, pixel_y, depth, id, conf);
  addPointInteger(pixel, pixel_x+1, pixel_y, depth, id, conf);
  addPointInteger(pixel, pixel_x, pixel_y+1, depth, id, conf);
  addPointInteger(pixel, pixel_x+1, pixel_y+1, depth, id, conf);
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


void PointCloudToDepthBase::addPointGauss(float* pixel, float depth,unsigned int id, float conf)
{
  int pixel_x = int(pixel[0] + 0.5f);
  int pixel_y = int(pixel[1] + 0.5f);
  //std::cout<<"pixel = "<<pixel[0]<<" "<<pixel[1]<<" pixel_int = "<<pixel_x<<" "<<pixel_y<<std::endl;
  addPointIntegerGauss(pixel, pixel_x, pixel_y, depth, id, conf);
  addPointIntegerGauss(pixel, pixel_x+1, pixel_y, depth, id, conf);
  addPointIntegerGauss(pixel, pixel_x, pixel_y+1, depth, id, conf);
  addPointIntegerGauss(pixel, pixel_x+1, pixel_y+1, depth, id, conf);
  addPointIntegerGauss(pixel, pixel_x-1, pixel_y+1, depth, id, conf);
  addPointIntegerGauss(pixel, pixel_x+1, pixel_y-1, depth, id, conf);
  addPointIntegerGauss(pixel, pixel_x, pixel_y-1, depth, id, conf);
  addPointIntegerGauss(pixel, pixel_x-1, pixel_y, depth, id, conf);
  addPointIntegerGauss(pixel, pixel_x-1, pixel_y-1, depth, id, conf);
}
  

//dist_coeff = [k1 k2 p1 p2 [k3 [k4 k5 k6]]] or [k1 k2 k3]
void PointCloudToDepthBase::opencvDistort(float* distPoint, float* projPoint, float* dist_coeff, unsigned int length_dist_coeff)
{
	if(length_dist_coeff == 0)
	{
		distPoint[0] = projPoint[0];
		distPoint[1] = projPoint[1];
		return;
	}

	float r = sqrt(projPoint[0]*projPoint[0]+projPoint[1]*projPoint[1]);

	//1+k1^2+k2^4+k3^6
	float nom = 1.0;
	int k_num=0;
	if(length_dist_coeff==3)
	{
		nom+=dist_coeff[0]*pow(r,2)+dist_coeff[1]*pow(r,4)+dist_coeff[2]*pow(r,6);
	}
	else
	{
		for(int i = 0; i < 5 && i<length_dist_coeff; i++)
		{
			if(i!=2 && i!=3)
			{
				k_num++;
				nom+=dist_coeff[i]*pow(r,k_num*2);
			}
		}
	}

	//1+k4^2+k5^4+k6^6
	float denom=1.0;
	k_num=0;
	for(int i = 5; i< 8 && i<length_dist_coeff; i++)
	{
		k_num++;
		denom+=dist_coeff[i]*pow(r,k_num*2);
	}
	float quota = nom/denom;

	float p1=0.0;
	float p2=0.0;
	if(length_dist_coeff>2 && length_dist_coeff!=3)
		p1=dist_coeff[2];
	if(length_dist_coeff>3 & length_dist_coeff!=3)
		p2=dist_coeff[3];

	float xprime = projPoint[0];
	float yprime = projPoint[1];

	distPoint[0] = xprime*quota+2*p1*xprime*yprime+p2*(r*r+2*xprime*xprime);
	distPoint[1] = yprime*quota+p1*(r*r+2*yprime*yprime)+2*p2*xprime*yprime;
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
