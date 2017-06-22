/* myslam */
#include "myslam/common_include.h"
#include "myslam/config.h"
#include <myslam/camera.h>
#include "myslam/g2o_type.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/viz.hpp> 
#include <opencv2/core/eigen.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/core/eigen.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <boost/timer.hpp>
#include <boost/concept_check.hpp>

#include "myslam/vo.h"

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>


#include <stdio.h>
#include <fstream>


using namespace std;


int main( int argc, char** argv )
{
  if ( argc != 2 )
  {
      cout<<"usage: run_vo parameter_file"<<endl;
      return 1;
  }
  
  myslam::Config::setParameterFile(argv[1]);
  myslam::Camera::Ptr camera(new myslam::Camera);
  
  cv::Mat camera_matrix = camera->matrix_;
  cv::Mat camera_distor = camera->distor_;
  
  string dataset_dir = myslam::Config::get<string> ( "dataset_dir" );
  cout<<"dataset: "<<dataset_dir<<endl;
  ifstream fin ( dataset_dir+"/associate.txt" );
  if ( !fin )
  {
      cout<<"please generate the associate file called associate.txt!"<<endl;
      return 1;
  }

  vector<string> rgb_files, depth_files;
  vector<double> rgb_times, depth_times;
  while ( !fin.eof() )
  {
      string rgb_time, rgb_file, depth_time, depth_file;
      fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
      if ( fin.good() == false )
	  break;
      rgb_times.push_back ( atof ( rgb_time.c_str() ) );
      depth_times.push_back ( atof ( depth_time.c_str() ) );
      rgb_files.push_back ( dataset_dir+"/"+rgb_file );
      depth_files.push_back ( dataset_dir+"/"+depth_file );
  }
  
  // visualization
  cv::viz::Viz3d vis("Visual Odometry");
  cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(1.0);
  cv::Point3d cam_pos( 0, -1.0, -1.0 ), cam_focal_point(0,0,0), cam_y_dir(0,1,0);
  cv::Affine3d cam_pose = cv::viz::makeCameraPose( cam_pos, cam_focal_point, cam_y_dir );
  vis.setViewerPose( cam_pose );
  
  world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
  camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
  vis.showWidget( "World", world_coor );
  vis.showWidget( "Camera", camera_coor );
  
  //vector<cv::viz::WCoordinateSystem*> p_camera_i;
  
  myslam::VO::Ptr vo(new myslam::VO());
  
 
  for(int i=0; i < rgb_files.size();i++)
  {
    cv::Mat curr_img_color = cv::imread(rgb_files[i],CV_LOAD_IMAGE_COLOR);
    cv::Mat curr_img_grey;
    cv::cvtColor(curr_img_color, curr_img_grey, CV_BGR2GRAY);
    //cv::normalize(curr_img_grey, curr_img_grey, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::Mat img_depth = cv::imread(depth_files[i],CV_LOAD_IMAGE_UNCHANGED);
    
    boost::timer timer;
    
    myslam::Frame::Ptr frame = myslam::Frame::creatFrame();
    
    frame->camera_ = camera;
    frame->color_ = curr_img_color;
    frame->depth_ = img_depth;
    frame->time_stamp_ = rgb_times[i];
    
    static Eigen::Matrix4d T_cw_m = Eigen::Matrix4d::Identity();
	
    if(vo->addFrame(frame))
      T_cw_m = vo->T_c_w_estimated_.matrix();
    
    cout<<"time cost:"<<timer.elapsed()<<endl;

    cv::Mat img_show = curr_img_color.clone();
    
    for(auto p_3d:vo->matched_3d_points_)
    {
      cv::Mat p;
      cv::eigen2cv(vo->curr_->camera_->world2pixel(p_3d->pos_,vo->curr_->T_c_w_),p);
      cv::circle(img_show,cv::Point2d(p.at<double>(0,0),p.at<double>(1,0)),pow(p_3d->matched_ratio_,2)*6,cv::Scalar(0,255,0),1);
    }
   //  cv::imshow("Depth",img_depth);
    cv::imshow("Features",img_show);
   
  
      Eigen::Matrix4d T_show;
      
      T_show = T_cw_m.inverse();
      
      cv::Mat_<double> R_show(3,3);
      cv::Mat_<double> t_show(3,1);
      
      R_show.at<double>(0,0) = T_show(0,0); R_show.at<double>(0,1) = T_show(0,1); R_show.at<double>(0,2) = T_show(0,2);
      R_show.at<double>(1,0) = T_show(1,0); R_show.at<double>(1,1) = T_show(1,1); R_show.at<double>(1,2) = T_show(1,2);
      R_show.at<double>(2,0) = T_show(2,0); R_show.at<double>(2,1) = T_show(2,1); R_show.at<double>(2,2) = T_show(2,2);
      
      t_show.at<double>(0,0) = T_show(0,3);
      t_show.at<double>(1,0) = T_show(1,3);
      t_show.at<double>(2,0) = T_show(2,3);
      

           
      cv::Affine3d M(R_show,t_show);
      

      vis.setWidgetPose( "Camera", M);
      vis.spinOnce(1, false);
      cv::waitKey(1);
    }
    
  typedef pcl::PointXYZRGB PointT;
  typedef pcl::PointCloud<PointT> PointCloud;
  
  PointCloud::Ptr pointCloud(new PointCloud);
  
  for(auto iter=vo->map_->map_points_.begin();iter !=vo->map_->map_points_.end();)
  {
  	PointT p;
	p.x = iter->second->pos_(0,0);
	p.y = iter->second->pos_(1,0);
	p.z = iter->second->pos_(2,0);
	p.b = 0;
	p.g = 0;
	p.r = 255;
	pointCloud->points.push_back(p);
	iter++;
  }
  pointCloud->is_dense = false;
  cout<<pointCloud->size()<<" Points"<<endl;
  pcl::io::savePCDFileBinary("map.pcd",*pointCloud);
    
  return 0;
}