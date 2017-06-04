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

void bundleAdjustment (
    const vector< cv::Point3f > points_3d,
    const vector< cv::Point2f > points_2d,
    const cv::Mat& K,
    cv::Mat& R, cv::Mat& t );



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
  cv::Mat curr_img_rgb;
  cv::Mat last_img_rgb;
  vector<cv::Point2f> prev_keypoints;
  
  //cv::Ptr<cv::ORB> detector = cv::ORB::create(500,1.1f,8,31,0,2,cv::ORB::HARRIS_SCORE,31,20);
  cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(40);
  
  for(int i=0; i < rgb_files.size();i++)
  {
  
    cv::Mat curr_img_rgb = cv::imread(rgb_files[i],CV_LOAD_IMAGE_COLOR);
    cv::Mat img_depth = cv::imread(depth_files[i],CV_LOAD_IMAGE_UNCHANGED);
    
   // cout << i <<"/"<<rgb_files.size()<< depth_files.size()  <<endl;
    
    //detect corners
    vector<cv::KeyPoint> corners;
    vector<cv::Point2f> keypoints;
    vector<cv::Point2f> next_keypoints;
    
    boost::timer timer;
         
    detector->detect(curr_img_rgb,corners);
            
    for(int i=0; i<corners.size();i++)
    {
      keypoints.push_back(corners[i].pt);
    }
      
    //first frame
    if(i==0)
    {
      last_img_rgb = curr_img_rgb;
      for(auto k:keypoints)
        prev_keypoints.push_back(k);
      continue;
    }
   
    vector<unsigned char> status;
    vector<float> error;
        
    cv::calcOpticalFlowPyrLK(last_img_rgb,curr_img_rgb,prev_keypoints,next_keypoints,status,error);
        
    vector<cv::Point2f> points_1,points_2;
    
    for(int i=0;i<prev_keypoints.size();i++)
    {
      if(status[i] != 0)
      {
	points_1.push_back(prev_keypoints[i]);
	points_2.push_back(next_keypoints[i]);
      }
    }
    
    last_img_rgb = curr_img_rgb;
    
    prev_keypoints.clear();
    for(auto k:keypoints)
      prev_keypoints.push_back(k);
   
    cout<<"tracked keypoints: "<< points_1.size()<<endl;
    
//    int index = 0;
    // erase the lost points
//     for(auto iter=keypoints.begin(); iter != keypoints.end(); index++)
//     {
//       if(status[index] == 0)
//       {
// 	iter = keypoints.erase(iter);
// 	continue;
//       }
//       *iter = next_keypoints[index];
//       iter++;
//     }
    
    //calculate center of mass of the keypoints
//     cv::Point2f c_p(0,0);
//     for(auto iter=keypoints.begin(); iter != keypoints.end(); iter++)
//     {
//       cv::Point2f p = *iter;
//       c_p.x +=  p.x;
//       c_p.y +=  p.y;
//     }
//     
//     c_p.x /= (float)keypoints.size();
//     c_p.y /= (float)keypoints.size();
    
    //cout << c_p<<endl;
//     double avg_r = 0;
//     
//     for(auto iter=keypoints.begin(); iter != keypoints.end(); iter++)
//     {
//       cv::Point2f p = *iter;
//       avg_r += sqrt((p.x - c_p.x)*(p.x - c_p.x) + (p.y - c_p.y)*(p.y - c_p.y));
//     }
//     
//     avg_r /= (float)keypoints.size();
        
    //cout << avg_r<<endl;
      
//     if((keypoints.size() <= 250 && keypoints.size() <= 500))
//     {
//       vector<cv::KeyPoint> keypoints_1;
//       
//       cout << "re-detect conner..."<<endl;
//   
//       orb->detect(curr_img_rgb,keypoints_1);
//             
//       for(int i=0; i<keypoints_1.size();i++)
//       {
// 	keypoints.push_back(keypoints_1[i].pt);
//       }
//       //break;
//     }
    
 
//    cout<<"cost time:"<<timer.elapsed()<<endl;
//    cv::Mat img_match;
//    cv::drawMatches(img1,keypoints_1,img2,keypoints_2,matches,img_match);
    
//    cv::imshow("match",img_match);
    
//     cout <<"Max distance:"<<max_dist<<endl;
//     cout << "Min distance:"<<min_dist<<endl;
//     cout<<"matches number:"<< matches.size() <<endl;
//    cout<<"good matches number:"<< matches.size() <<endl;
    
    cv::Mat R,t;
    
    //pose_estimation_2d2d(keypoints_1,keypoints_2,matches,camera_matrix,R,t);
     
    pose_estimation_3d2d(points_1,points_2,camera_matrix,camera_distor,img_depth,R,t);
    
    cout<<"time cost:"<<timer.elapsed()<<endl;
    
    cv::Mat img_show = curr_img_rgb.clone();
    
    for(auto kp:next_keypoints)
      cv::circle(img_show,kp,5,cv::Scalar(0,240,0),1);
    
    cv::imshow("corners",img_show);
    
    
//    cout<<"R="<<R<<endl;
//    cout<<"t="<<t<<endl;
    
//     //-- 验证E=t^R*scale
//       cv::Mat t_x = ( cv::Mat_<double> ( 3,3 ) <<
//                   0,                      -t.at<double> ( 2,0 ),     t.at<double> ( 1,0 ),
//                   t.at<double> ( 2,0 ),      0,                      -t.at<double> ( 0,0 ),
//                   -t.at<double> ( 1.0 ),     t.at<double> ( 0,0 ),      0 );
//   
//       cout<<"t^R="<<endl<<t_x*R<<endl;
//   
//       //-- 验证对极约束
//       cv::Mat K = camera_matrix;
//       double error_sum = 0;
//       for ( cv::DMatch m: matches )
//       {
//           cv::Point2d pt1 = pixel2cam ( keypoint_1[ m.queryIdx ].pt, K );
//           cv::Mat y1 = ( cv::Mat_<double> ( 3,1 ) << pt1.x, pt1.y, 1 );
//           cv::Point2d pt2 = pixel2cam ( keypoint_2[ m.trainIdx ].pt, K );
//           cv::Mat y2 = ( cv::Mat_<double> ( 3,1 ) << pt2.x, pt2.y, 1 );
//           cv::Mat d = y2.t() * t_x * R * y1;
//           //cout << "epipolar constraint = " << d << endl;
//   	printf("epipolar constraint = %8.5f\n",d.at<double>(0,0));
//   	error_sum += d.at<double>(0,0)*d.at<double>(0,0);
//       }
//     cout<<"error sum:"<<error_sum<<endl;
    
//    cv::Point3d points_3d;
// //   triangulation(keypoints_1,keypoints_2,matches,camera_matrix,R,t,points_3d);
//    
//    int index=0;
//    
//     for (auto m:matches )
//     {
//         ushort d = img_depth.ptr<unsigned short> (int ( keypoints_1[m.queryIdx].pt.y )) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
//         if ( d == 0 )   // bad depth
// 	{
// 	  index++;
//             continue;
// 	}
//         float dd = d/5000.0;
//         cv::Point2d p1 = pixel2cam (keypoints_1[m.queryIdx].pt, camera_matrix);
//         points_3d = ( cv::Point3f ( p1.x*dd, p1.y*dd, dd ) );
//     
//     
//       //-- 验证三角化点与特征点的重投影关系
// 
// 	  cv::Point2d pt1_cam = pixel2cam( keypoints_1[ matches[index].queryIdx ].pt,camera_matrix);
// 	  cv::Point2d pt1_cam_3d(
// 	      points_3d.x/points_3d.z, 
// 	      points_3d.y/points_3d.z 
// 	  );
// 	  
// 	  cout<<"point in the first camera frame: "<<pt1_cam<<endl;
// 	  cout<<"point projected from 3D in first camera frame "<<pt1_cam_3d<<", d="<<points_3d.z<<endl;
// 	  
// 	  // 第二个图
// 	  cv::Point2f pt2_cam = pixel2cam( keypoints_2[ matches[index].trainIdx ].pt, camera_matrix );
// 	  cv::Mat pt2_trans = R*( cv::Mat_<double>(3,1) << points_3d.x, points_3d.y, points_3d.z ) + t;
// 	  pt2_trans /= pt2_trans.at<double>(2,0);
// 	  cout<<"point in the second camera frame: "<<pt2_cam<<endl;
// 	  cout<<"point reprojected from second frame: "<<pt2_trans.t()<<endl;
// 	  cout<<endl;
// 	  
// 	  index++;
//       }
   
      // show the map and the camera pose 
      
      static cv::Mat R_c_w = (cv::Mat_<double>(3,3)<<1,0,0,0,1,0,0,0,1);
      static cv::Mat t_c_w = (cv::Mat_<double>(3,1)<<0,0,0);
      
      //Eigen::Matrix<double,3,3> R_ = Eigen::AngleAxis<double>(M_PI/4,Eigen::Vector3d(0,0,1)).toRotationMatrix();
      Eigen::Matrix<double,3,3>  R_;
      R_(0,0) = R.at<double>(0,0);R_(0,1) = R.at<double>(0,1);R_(0,2) = R.at<double>(0,2);
      R_(1,0) = R.at<double>(1,0);R_(1,1) = R.at<double>(1,1);R_(1,2) = R.at<double>(1,2);
      R_(2,0) = R.at<double>(2,0);R_(2,1) = R.at<double>(2,1);R_(2,2) = R.at<double>(2,2);
      //Eigen::Map<Matrix<double,3,3>> R_;
      
       Eigen::AngleAxis<double> angle_axis(R_);
       
     //  if(abs(angle_axis.angle()) > 0.05)
      // {
	// i++;
	// i++;
	 //cout << R <<endl;
	 //cv::waitKey(0);
     // }
      
      //else{
      
      R_c_w = R.t()*R_c_w;
      t_c_w = R.t()*t_c_w - t;
      //}
      
      cv::Affine3d M(R_c_w,t_c_w);
      

      vis.setWidgetPose( "Camera", M);
      vis.spinOnce(1, false);
         cv::waitKey(1);
    }
  

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
