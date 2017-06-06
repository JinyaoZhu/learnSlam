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



#include <stdio.h>
#include <fstream>


using namespace std;

void pose_estimation_2d2d(const vector<cv::KeyPoint>keypoint_1,
			  const vector<cv::KeyPoint>keypoint_2,
			  const vector<cv::DMatch>matches,
			  cv::Mat camera_mat,
			  cv::Mat& R,cv::Mat& t);

void pose_estimation_3d2d(const vector<cv::Point2f>keypoints_1,
                          const vector<cv::Point2f>keypoints_2,
			  const cv::Mat camera_matrix,
			  const cv::Mat camera_distor,
			  const cv::Mat img_depth,
			  cv::Mat &R,
			  cv::Mat &t);

cv::Point2d pixel2cam ( const cv::Point2d& p, const cv::Mat& K );

void triangulation(const vector<cv::KeyPoint>& keypoints_1,const vector<cv::KeyPoint>& keypoints_2,
                   const vector<cv::DMatch>& matches, const cv::Mat &camera_matrix,
		   const cv::Mat& R, const cv::Mat& t,
		   vector<cv::Point3d>& points);


void feature_matching(const cv::Mat img_1,
		      const cv::Mat img_2,
		      vector<cv::KeyPoint> &keypoints_1,
		      vector<cv::KeyPoint> &keypoints_2,
		      vector<cv::DMatch> &good_matches);



inline Eigen::Vector3d project2Dto3D ( int x, int y, float d, float fx, float fy, float cx, float cy)
{
    float zz =  d ;
    float xx = zz* ( x-cx ) /fx;
    float yy = zz* ( y-cy ) /fy;
    return Eigen::Vector3d ( xx, yy, zz );
}

inline Eigen::Vector2d project3Dto2D ( float x, float y, float z,const cv::Mat& K)
{  
  float fx, fy, cx, cy;
  fx = K.at<double>(0,0);
  fy = K.at<double>(1,1);
  cx = K.at<double>(0,2);
  cy = K.at<double>(1,2);
  float u = fx*x/z+cx;
  float v = fy*y/z+cy;
  return Eigen::Vector2d ( u,v );
}




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
  
//  list<cv::Point2f> keypoints;
  cv::Mat curr_img_grey;
  cv::Mat last_img_rgb;
  vector<cv::Point2f> prev_keypoints;
  
  vector<Measurement> last_measurements;
  
  cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(10);
  //cv::Ptr<cv::ORB> detector = cv::ORB::create(500,1.1f,8,31,0,2,cv::ORB::HARRIS_SCORE,31,20);
  
  for(int i=0; i < rgb_files.size();i++)
  {
    cv::Mat curr_img_color = cv::imread(rgb_files[i],CV_LOAD_IMAGE_COLOR);
    cv::Mat curr_img_grey;
    cv::cvtColor(curr_img_color, curr_img_grey, CV_BGR2GRAY);
    //cv::normalize(curr_img_grey, curr_img_grey, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::Mat img_depth = cv::imread(depth_files[i],CV_LOAD_IMAGE_UNCHANGED);
    
   // cout << i <<"/"<<rgb_files.size()<< depth_files.size()  <<endl;
    
    //detect corners
    vector<cv::KeyPoint> corners;
    vector<cv::Point2f> keypoints;
    
    boost::timer timer;
         
    detector->detect(curr_img_grey,corners);
            
    for(int i=0; i<corners.size();i++)
      keypoints.push_back(corners[i].pt);
    
      
    //first frame
    if(i==0)
    {
      //update measurements
      last_measurements.clear();
      for ( auto kp:keypoints )
      {
	  // 去掉邻近边缘处的点
	  if ( kp.x < 20 || kp.y < 20 || ( kp.x+20 ) >curr_img_grey.cols || ( kp.y+20 ) >curr_img_grey.rows )
	      continue;
	  ushort d = img_depth.ptr<ushort> ( cvRound ( kp.y ) ) [ cvRound ( kp.x ) ];
	  if ( d==0 )
	      continue;
	  Eigen::Vector3d p3d = project2Dto3D ( kp.x, kp.y, (float)d/camera->depth_scale_,
						camera_matrix.at<double>(0,0),camera_matrix.at<double>(1,1),
						camera_matrix.at<double>(0,2),camera_matrix.at<double>(1,2));
	  float grayscale = float ( curr_img_grey.ptr<uchar> ( cvRound ( kp.y ) ) [ cvRound ( kp.x ) ] );
	  last_measurements.push_back ( Measurement ( p3d, grayscale ) );
      }
//       for ( int x=10; x<curr_img_grey.cols-10; x++ )
// 	  for ( int y=10; y<curr_img_grey.rows-10; y++ )
// 	  {
// 	      Eigen::Vector2d delta (
// 		  curr_img_grey.ptr<uchar>(y)[x+1] - curr_img_grey.ptr<uchar>(y)[x-1], 
// 		  curr_img_grey.ptr<uchar>(y+1)[x] - curr_img_grey.ptr<uchar>(y-1)[x]
// 	      );
// 	      if ( delta.norm() < 80 )
// 		  continue;
// 	      keypoints.push_back(cv::Point2f(x,y));
// 	      ushort d = img_depth.ptr<ushort> ( cvRound ( y ) ) [ cvRound ( x ) ];
// 	      if ( d==0 )
// 		  continue;
// 	      Eigen::Vector3d p3d = project2Dto3D ( x, y, (float)d/camera->depth_scale_,
// 					  camera_matrix.at<double>(0,0),camera_matrix.at<double>(1,1),
// 					  camera_matrix.at<double>(0,2),camera_matrix.at<double>(1,2));
// 	      float grayscale = float ( curr_img_grey.ptr<uchar> (y) [x] );
// 	      last_measurements.push_back ( Measurement ( p3d, grayscale ) );
// 	  }
      continue;
     }
    
    Eigen::Isometry3d T_cr = Eigen::Isometry3d::Identity();
    Eigen::Matrix3f K;
    cv::cv2eigen(camera_matrix,K);
    
    poseEstimationDirect (last_measurements, &curr_img_grey, K, T_cr);
    
    //update measurements
    last_measurements.clear();
    for ( auto kp:keypoints )
    {
	// 去掉邻近边缘处的点
	if ( kp.x < 20 || kp.y < 20 || ( kp.x+20 ) >curr_img_grey.cols || ( kp.y+20 ) >curr_img_grey.rows )
	    continue;
	ushort d = img_depth.ptr<ushort> ( cvRound ( kp.y ) ) [ cvRound ( kp.x ) ];
	if ( d==0 )
	    continue;
	Eigen::Vector3d p3d = project2Dto3D ( kp.x, kp.y, (float)d/camera->depth_scale_,
					      camera_matrix.at<double>(0,0),camera_matrix.at<double>(1,1),
					      camera_matrix.at<double>(0,2),camera_matrix.at<double>(1,2));
	float grayscale = float ( curr_img_grey.ptr<uchar> ( cvRound ( kp.y ) ) [ cvRound ( kp.x ) ] );
	last_measurements.push_back ( Measurement ( p3d, grayscale ) );
    }
//     for ( int x=10; x<curr_img_grey.cols-10; x+=1 )
// 	for ( int y=10; y<curr_img_grey.rows-10; y+=1 )
// 	{
// 	    Eigen::Vector2d delta (
// 		curr_img_grey.ptr<uchar>(y)[x+1] - curr_img_grey.ptr<uchar>(y)[x-1], 
// 		curr_img_grey.ptr<uchar>(y+1)[x] - curr_img_grey.ptr<uchar>(y-1)[x]
// 	    );
// 	    if ( delta.norm() < 80 )
// 		continue;
// 	   	    
// 	    keypoints.push_back(cv::Point2f(x,y));
// 	    ushort d = img_depth.ptr<ushort> ( cvRound ( y ) ) [ cvRound ( x ) ];
// 	    if ( d==0 )
// 		continue;
// 	    Eigen::Vector3d p3d = project2Dto3D ( x, y, (float)d/camera->depth_scale_,
// 					camera_matrix.at<double>(0,0),camera_matrix.at<double>(1,1),
// 					camera_matrix.at<double>(0,2),camera_matrix.at<double>(1,2));
// 	    float grayscale = float ( curr_img_grey.ptr<uchar> (y) [x] );
// 	    last_measurements.push_back ( Measurement ( p3d, grayscale ) );
// 	}       
    
    
    cout<<"time cost:"<<timer.elapsed()<<endl;
    
    cv::Mat R,t;
    
    Eigen::Matrix4d T_cr_m = T_cr.matrix();
    
    Eigen::Matrix<double,3,3>  R_;
      R_(0,0) = T_cr_m(0,0);R_(0,1) = T_cr_m(0,1);R_(0,2) = T_cr_m(0,2);
      R_(1,0) = T_cr_m(1,0);R_(1,1) = T_cr_m(1,1);R_(1,2) = T_cr_m(1,2);
      R_(2,0) = T_cr_m(2,0);R_(2,1) = T_cr_m(2,1);R_(2,2) = T_cr_m(2,2);
      
   Eigen::Matrix<double,3,1> t_;
   t_(0,0) = T_cr_m(0,3);
   t_(1,0) = T_cr_m(1,3);
   t_(2,0) = T_cr_m(2,3);
   
   cv::eigen2cv(R_,R);
   cv::eigen2cv(t_,t);
    
    
    
    cv::Mat img_show = curr_img_color.clone();
    
    for(auto kp:keypoints){
      cv::circle(img_show,kp,2,cv::Scalar(0,240,0),1);
    }
    
    cv::imshow("corners",img_show);
    
  
      Eigen::Matrix4d T_show = Eigen::Matrix4d::Identity();
      
      static Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
      
      T_cw = T_cr_m*T_cw;
      
      T_show = T_cw.inverse();
      
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
  
  cv::waitKey(0);
  return 0;
}

void feature_matching(const cv::Mat img_1,const cv::Mat img_2,
		      vector<cv::KeyPoint> &keypoints_1,
		      vector<cv::KeyPoint> &keypoints_2,
		      vector<cv::DMatch> &good_matches)
 		    
{
    cv::Mat descriptor_1,descriptor_2;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500,1.1f,8,31,0,2,cv::ORB::HARRIS_SCORE,31,20);
    
    boost::timer timer;
    orb->detect(img_1,keypoints_1);
    orb->detect(img_2,keypoints_2);
    
    orb->compute(img_1,keypoints_1,descriptor_1);
    orb->compute(img_2,keypoints_2,descriptor_2);
    
    vector<cv::DMatch> matches;
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    matcher.match(descriptor_1,descriptor_2,matches);
    
    double min_dist = 1000, max_dist = 0;
    for(auto m:matches)
    {
      double dist = m.distance;
      if(dist > max_dist) max_dist = dist;
      if(dist < min_dist) min_dist = dist;
    }
    
    for(auto m:matches)
    {
      if(m.distance <= max(2*min_dist,30.0))
      {
	good_matches.push_back(m);
      }
    }
}

void pose_estimation_2d2d(const vector<cv::KeyPoint>keypoint_1,
			  const vector<cv::KeyPoint>keypoint_2,
			  const vector<cv::DMatch>matches,
			  cv::Mat camera_mat,
			  cv::Mat& R,cv::Mat& t)
{
  vector<cv::Point2d> points1;
  vector<cv::Point2d> points2;
  
  for(int i=0;i<(int)matches.size();i++)
  {
    points1.push_back(keypoint_1[matches[i].queryIdx].pt);
    points2.push_back(keypoint_2[matches[i].trainIdx].pt);
  }
  
//   cv::Mat fundamental_matrix;
//   fundamental_matrix = cv::findFundamentalMat(points1,points2,CV_FM_RANSAC);
  
  cv::Mat essential_matrix;
  essential_matrix = cv::findEssentialMat(points1,points2,camera_mat,cv::RANSAC);
  
//   cv::Mat homography_matrix;
//   homography_matrix = cv::findHomography(points1,points2,cv::RANSAC);
  
  cv::recoverPose(essential_matrix,points1,points2,camera_mat,R,t);
}


void pose_estimation_3d2d(const vector<cv::Point2f>keypoints_1,
                          const vector<cv::Point2f>keypoints_2,
			  const cv::Mat camera_matrix,
			  const cv::Mat camera_distor,
			  const cv::Mat img_depth,
			  cv::Mat &R,
			  cv::Mat &t)
{
    vector<cv::Point3f> pts_3d;
    vector<cv::Point2f> pts_2d;
    cv::Mat r;
    for (int i=0; i<keypoints_1.size();i++)
    {
        ushort d = img_depth.ptr<unsigned short> (int ( keypoints_1[i].y )) [ int ( keypoints_1[i].x ) ];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/5000.0;
        cv::Point2d p1 = pixel2cam (keypoints_1[i], camera_matrix);
        pts_3d.push_back ( cv::Point3f ( p1.x*dd, p1.y*dd, dd ) );
        pts_2d.push_back (keypoints_2[i]);
    }
   
    //R = (cv::Mat_<double>(3,3)<<1,0,0,0,1,0,0,0,1);
    //t = (cv::Mat_<double>(3,1)<<0,0,0);
    
    //cv::solvePnP ( pts_3d, pts_2d, camera_matrix,camera_distor, r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    cv::solvePnPRansac( pts_3d, pts_2d, camera_matrix,camera_distor, r, t);
    cv::Rodrigues (r,R);
    
    bundleAdjustment(pts_3d,pts_2d,camera_matrix,R,t);
}

void triangulation(const vector<cv::KeyPoint>& keypoints_1,const vector<cv::KeyPoint>& keypoints_2,
                   const vector<cv::DMatch>& matches, const cv::Mat &camera_matrix,
		   const cv::Mat& R, const cv::Mat& t,
		   vector<cv::Point3d>& points)
{
  cv::Mat T1 = (cv::Mat_<double>(3,4) << 
                 1,0,0,0,
		 0,1,0,0,
		 0,0,1,0);
  cv::Mat T2 = (cv::Mat_<double>(3,4)<<
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0));
  
  vector<cv::Point2d> points_1,points_2; // points in camera coordinate(Z = 1)
  for(auto m:matches)
  {
    points_1.push_back(pixel2cam(keypoints_1[m.queryIdx].pt,camera_matrix));
    points_2.push_back(pixel2cam(keypoints_2[m.trainIdx].pt,camera_matrix));
  }
  
  cv::Mat points_4d;
  cv::triangulatePoints(T1,T2,points_1,points_2,points_4d);
  
  for(int i=0;i<points_4d.cols;i++)
  {
    cv::Mat x = points_4d.col(i);
    x /= x.at<double>(3,0);
    cv::Point3d p(x.at<double>(0,0),x.at<double>(1,0),x.at<double>(2,0));
    points.push_back(p);
  }
}


cv::Point2d pixel2cam ( const cv::Point2d& p, const cv::Mat& K )
{
    return cv::Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}
