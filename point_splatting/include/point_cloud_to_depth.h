#ifndef POINT_CLOUD_TO_DEPTH_H
#define POINT_CLOUD_TO_DEPTH_H
#include <vector>
#include <utility> 
#include <pcl/registration/registration.h>

namespace pointCloudProjection
{
#define POINT_CLOUD_TO_DEPTH_NEAREST 0
#define POINT_CLOUD_TO_DEPTH_BILINEAR 1
#define POINT_CLOUD_TO_DEPTH_GAUSS 2

struct DepthImagePoint
{
public:
  DepthImagePoint(float d_in, float w_in,unsigned int cloud_id_in, unsigned int pt_id_in)
	{
		d = d_in;
		w = w_in;
		cloud_id_ = cloud_id_in;
		pt_id_ = pt_id_in;
	}
  float d,w;
	unsigned int cloud_id_, pt_id_;
};

class DepthIdPair
{
public:
  DepthIdPair(float d_in,unsigned int id_in)
	{
		d = d_in;
		id = id_in;
	}

	bool operator < (const DepthIdPair& other) const
	{
		return this->d < other.d;
	}
  float d;
	unsigned int id;
};

struct MeanShiftClusterColor
{
  MeanShiftClusterColor(float val_in, float mean_in, float r_in, float g_in, float b_in, unsigned int id, unsigned int pt)
	{
		val = val_in;
		mean = mean_in;
		ids_.push_back(id);
		cloud_ids_.push_back(id);
		pt_id_.push_back(pt);
		r = r_in;
		g = g_in;
		b = b_in;
		if(id == 0)
			has_view_zero = true;
		else
			has_view_zero = false;

	}
	MeanShiftClusterColor(float val_in, float mean_in, float r_in, float g_in, float b_in, float weight_in)
	{
		val = val_in;
		mean = mean_in;
		has_view_zero = false;
		weight = weight_in;
		r = r_in;
		g = g_in;
		b = b_in;
	}
	~MeanShiftClusterColor()
	{
		ids_.clear();
		cloud_ids_.clear();
		pt_id_.clear();
	}
	bool operator > (const MeanShiftClusterColor& other) const
	{
		bool asdf = (val+weight*20.0f/mean) > (other.val+weight*20.0f/other.mean);
		return asdf;
	}

	void update(float val_in, float mean_in, float r_in, float g_in, float b_in)
	{
		//update cluster if larger val
		if(val_in > val)
		{
			val = val_in;
			mean = mean_in;
			r = r_in;
		  g = g_in;
		  b = b_in;
		}
	}
	
	void addPoint(float w, unsigned int id, unsigned int pt, int label)
	{

		cloud_ids_.push_back(id);
		pt_id_.push_back(pt);

		bool label_found = false;
		//add label and increase label weight
		for(unsigned int i = 0; i < label_stats_.size(); i++)
		{
			if(label_stats_[i].first == label)
			{
				label_stats_[i].second += w;
				label_found = true;
				break;
			}
		}
	
		if(!label_found)
		{
			label_stats_.push_back(std::make_pair(label, w));
		}
	}
	
	int getLabel()
	{
		int label = 255; //unlabeled = 255
		float max_w = -1.0f;;
		for(unsigned int i = 0; i < label_stats_.size(); i++)
		{
			if(max_w < label_stats_[i].second && label_stats_[i].first != 0) //0 is no label
			{
				max_w = label_stats_[i].second;
				label = label_stats_[i].first;
			}
		}
		
		return label;
	}
	float val;
	float mean, r, g, b;
	float weight;
	std::vector<unsigned int> ids_, pt_id_, cloud_ids_;
	std::vector< std::pair<int, float> > label_stats_; //vector < <label, weight (significance) > >
	bool has_view_zero;
};

struct MeanShiftCluster
{
	MeanShiftCluster(float val_in, float mean_in, unsigned int id, unsigned int pt)
	{
		val = val_in;
		mean = mean_in;
		ids_.push_back(id);
		cloud_ids_.push_back(id);
		pt_id_.push_back(pt);

		if(id == 0)
			has_view_zero = true;
		else
			has_view_zero = false;

	}
	MeanShiftCluster(float val_in, float mean_in, float weight_in)
	{
		val = val_in;
		mean = mean_in;
		has_view_zero = false;
		weight = weight_in;
	}
	~MeanShiftCluster()
	{
		ids_.clear();
		cloud_ids_.clear();
		pt_id_.clear();
	}
	bool operator > (const MeanShiftCluster& other) const
	{
		bool asdf = (val+weight*20.0f/mean) > (other.val+weight*20.0f/other.mean);
		return asdf;
	}

	void update(float val_in, float mean_in)
	{
		//update cluster if larger val
		if(val_in > val)
		{
			val = val_in;
			mean = mean_in;
		}
	}
	
	void addPoint(float w, unsigned int id, unsigned int pt, int label)
	{

		cloud_ids_.push_back(id);
		pt_id_.push_back(pt);

		bool label_found = false;
		//add label and increase label weight
		for(unsigned int i = 0; i < label_stats_.size(); i++)
		{
			if(label_stats_[i].first == label)
			{
				label_stats_[i].second += w;
				label_found = true;
				break;
			}
		}
	
		if(!label_found)
		{
			label_stats_.push_back(std::make_pair(label, w));
		}
	}
	
	int getLabel()
	{
		int label = 255; //unlabeled = 255
		float max_w = -1.0f;;
		for(unsigned int i = 0; i < label_stats_.size(); i++)
		{
			if(max_w < label_stats_[i].second && label_stats_[i].first != 0) //0 is no label
			{
				max_w = label_stats_[i].second;
				label = label_stats_[i].first;
			}
		}
		
		return label;
	}
	float val;
	float mean;
	float weight;
	std::vector<unsigned int> ids_, pt_id_, cloud_ids_;
	std::vector< std::pair<int, float> > label_stats_; 
	bool has_view_zero;
};

class PointCloudToDepthBase
{
  public:
    PointCloudToDepthBase(float* intr, double* distortion_coefficients, unsigned int in_rows, unsigned int in_cols);
    ~PointCloudToDepthBase();

		void addDepthImage(float* depth_image, unsigned int rows, unsigned int cols, unsigned int method, float* conf = NULL, float conf_thresh = 0.0f);
    	float* getDepthImageNearestNeighbor(float cluster_dist);
		void addPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud);
		float* getDepthImageMeanShift(float cluster_val_threshold, float lim, unsigned int num_iterations, float cluster_width, Eigen::MatrixXf transform);
		float* getClusterValImage();
		float* getRimage();
		float* getGimage();
		float* getBimage();
		int* getLabelimage();	
		unsigned int* getNumSources();
		std::vector< std::vector< unsigned int > >* getVisablePoints();
		unsigned int getRows();
		unsigned int getCols();
		// Eigen::Matrix4f transformCloud(float alfa,float beta,float theta,float trans_x,float trans_y,float trans_z);
		// Eigen::MatrixXf transform(float alfa,float beta,float theta,float trans_x,float trans_y,float trans_z);
		Eigen::MatrixXf transform(float trans_x, float trans_y, float trans_z, float r11, float r12, float r13, float r21, float r22, float r23, float r31, float r32, float r33);

  private:
	  void addPoint(float* pixel, float depth,unsigned int method, unsigned int id, float conf=1.0f);
	  void addPointLabelInteger(float* pixel, int pixel_x,int pixel_y, float depth, int label, unsigned int id, float conf=1.0f);
	  void addPointNearest(float* pixel, float depth,unsigned int id, float conf=1.0f);
	  void addPointBilinear(float* pixel, float depth,unsigned int id, float conf=1.0f);
	  void addPointGauss(float* pixel, float depth,unsigned int id, float conf=1.0f);
	  void addPointIntegerGauss(float* pixel, int pixel_x,int pixel_y, float depth,unsigned int id, float conf=1.0f);
	  void addPointInteger(float* pixel, int pixel_x,int pixel_y, float depth, unsigned int id, float conf=1.0f);
	  void opencvDistort(float* distPoint, float* projPoint, float* dist_coeff, unsigned int length_dist_coeff);
	  void insertionSortn(float* array, int size, unsigned int* inds);
	  void processDepthImageMeanShift(float cluster_val_threshold, float lim, unsigned int num_iterations, float cluster_width);
   	  //void visualizeCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, float norm_factor, std::vector< unsigned int >& mask);
	  // void projectCloud(std::vector< std::vector< unsigned int > >& mask);
   	  void projectCloud(std::vector< std::vector< unsigned int > >& mask, Eigen::MatrixXf transform);
    
	  std::vector<std::vector<DepthImagePoint> > depth_image_points_;

	  unsigned int rows_, cols_, length_dist_coeff_ ,cloud_id_;
	  unsigned int* num_sources_;
	  int* label_image_;
	  std::vector< std::vector<unsigned char> > cluster_point_mask_;
	  float* dist_coeff_, *depth_image_, *cluster_val_image_,*R_image_,*G_image_,*B_image_;
	  float fx_,fy_,cx_,cy_;
	  std::vector< std::vector< unsigned int > > visible_points_;
	  std::vector<float*> conf_vec_;
	  std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr > big_cloud_;
};

}

#endif // POINT_CLOUD_TO_DEPTH_H
