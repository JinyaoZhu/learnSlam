#include "myslam/vo.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "myslam/g2o_type.h"

#include "boost/timer.hpp"

namespace myslam {
  
VO::VO():state_(INITIALIZE),map_(new Map()),ref_(nullptr),
curr_(nullptr), num_inliers_(0), num_lost_(0)
{
  detector_ = cv::ORB::create ( 600, 1.1f, 8 );
  matcher_ =  cv::DescriptorMatcher::create ( "BruteForce-Hamming" );;
}
  
  
bool VO::addFrame(Frame::Ptr frame)
{
  curr_ = frame;
  switch(state_)
  {
  case INITIALIZE:
  {
    extractKeyPoints();
    computeDescriptors();
    addKeyFrame();
    addMapPoints();
    state_ = OK;
    ref_ = curr_;
    break;
  }
  case OK:
  {
    curr_->T_c_w_ = ref_->T_c_w_;
    extractKeyPoints();
    computeDescriptors();
    featureMatching();
    poseEstimatePnP();
    curr_->T_c_w_ = T_c_w_estimated_;
    
    optimizeMap();
    ref_ = curr_;
    break;
  }
  case LOST:
  {
    break;
  }
}
}

  
  
void VO::extractKeyPoints()
{
  boost::timer timer;
  
  detector_->detect(curr_->color_,key_points_curr_);
//  cout << "key point size:"<<key_points_curr_.size()<<endl;
  
//  cout << "extractKeyPoints() cost time:"<<timer.elapsed()<<endl;
}

void VO::computeDescriptors()
{
  boost::timer timer;
  detector_->compute(curr_->color_,key_points_curr_,descriptor_curr_);
//  cout << "computeDescriptors() cost time:"<<timer.elapsed()<<endl;
}

void VO::featureMatching()
{
  boost::timer timer;
  vector<cv::DMatch> matches;
  // select candidates in map
  cv::Mat candidates_descriptor;
  vector<MapPoint::Ptr> map_candidates;
  for(auto & allpoints:map_->map_points_)
  {
    MapPoint::Ptr & p = allpoints.second;
    if(curr_->isInFrame(p->pos_))
    {
      //add candidates
      p->visible_times_++;
      map_candidates.push_back(p);
      candidates_descriptor.push_back(p->descriptor_);
    }
  }
  cout<<"candidates size:"<<map_candidates.size()<<endl;
  matcher_->match(candidates_descriptor,descriptor_curr_,matches);
  
  //select good matches
  float min_dis = std::min_element (
                        matches.begin(), matches.end(),
                        [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
    {return m1.distance < m2.distance;} )->distance;
    
    matched_3d_points_.clear();
    matched_2d_kp_index_.clear();
    for(cv::DMatch &m:matches)
    {
      if(m.distance < max<float>(min_dis*2,20))
      {
	matched_3d_points_.push_back(map_candidates[m.queryIdx]);
	matched_2d_kp_index_.push_back(m.trainIdx);
      }
    }
//    cout << "good matches: "<< matched_3d_points_.size()<<endl;
    
//    cout << "featureMatching() cost time:"<<timer.elapsed()<<endl;
}

void VO::poseEstimatePnP()
{
  boost::timer timer;
  vector<cv::Point3f> pts_3d;
  vector<cv::Point2f> pts_2d;
  
  for(int index:matched_2d_kp_index_)
    pts_2d.push_back(key_points_curr_[index].pt);
  
  for(MapPoint::Ptr pt:matched_3d_points_)
    pts_3d.push_back(pt->getPositionCV());
  
  cv::Mat rvec,tvec,inliers;
  cv::solvePnPRansac(pts_3d,pts_2d,curr_->camera_->matrix_,
   curr_->camera_->distor_,rvec,tvec,false,100,4.0,0.99,inliers);
  
  num_inliers_ = inliers.rows;
  
  for(int i=0; i<inliers.rows; i++)
  {
     int index = inliers.at<int> ( i,0 );
     matched_3d_points_[index]->matched_times_ ++;
  }
  
//  cout<<"inliers size:"<<num_inliers_<<endl;
  
  T_c_w_estimated_ = Sophus::SE3 (Sophus::SO3( rvec.at<double> ( 0,0 ), rvec.at<double> ( 1,0 ), rvec.at<double> ( 2,0 ) ),
                           Eigen::Vector3d ( tvec.at<double> ( 0,0 ), tvec.at<double> ( 1,0 ), tvec.at<double> ( 2,0 ) ) );
  
// using bundle adjustment to optimize the pose
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block ( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
        T_c_w_estimated_.rotation_matrix(), T_c_w_estimated_.translation()
    ));
    optimizer.addVertex ( pose );

    // edges
    for ( int i=0; i<inliers.rows; i++ )
    {
        int index = inliers.at<int> ( i,0 );
        // 3D -> 2D projection
        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId ( i );
        edge->setVertex ( 0, pose );
        edge->camera_ = curr_->camera_.get();
        edge->point_ = Eigen::Vector3d ( pts_3d[index].x, pts_3d[index].y, pts_3d[index].z );
        edge->setMeasurement ( Eigen::Vector2d ( pts_2d[index].x, pts_2d[index].y ) );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
	g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(1.0);
        edge->setRobustKernel(rk);
        optimizer.addEdge ( edge );
    }

    optimizer.initializeOptimization();
    optimizer.optimize ( 10 );

    T_c_w_estimated_ = Sophus::SE3 (
        pose->estimate().rotation(),
        pose->estimate().translation()
    );
    
//    cout << "poseEstimatePnP() cost time:"<<timer.elapsed()<<endl;
}

void VO::addKeyFrame()
{
  map_->insertKeyFrame(curr_);
  ref_ = curr_;
}

void VO::addMapPoints()
{
  vector<bool> matched(key_points_curr_.size(),false);
  // mark map points the already exist 
  for(int index:matched_2d_kp_index_)
    matched[index] = true;
  for(int i=0; i< key_points_curr_.size(); i++)
  {
    if(matched[i] == true)
      continue;
    double d = curr_->findDepth(key_points_curr_[i].pt);
    if(d < 0.0)
      continue;
    Eigen::Vector3d p_world = curr_->camera_->pixel2world(
      Eigen::Vector2d(key_points_curr_[i].pt.x,key_points_curr_[i].pt.y),curr_->T_c_w_,d);
    
    MapPoint::Ptr map_point = MapPoint::createMapPoint(p_world, curr_.get(), descriptor_curr_.row(i).clone());
    
    map_->insertMapPoint(map_point);
  }
  cout << "map points size:"<<map_->map_points_.size()<<endl;
}

void VO::optimizeMap()
{
  boost::timer timer;
  
  for(auto iter=map_->map_points_.begin();iter != map_->map_points_.end();)
  {
    MapPoint::Ptr mp = iter->second;
    float match_ratio = (float)mp->matched_times_/mp->visible_times_; 
    if(match_ratio < 0.3)
    {
      iter = map_->map_points_.erase(iter);
      continue;
    }
    
//     if(!curr_->isInFrame(mp->pos_))
//     {
//        iter = map_->map_points_.erase(iter);
//        continue;
//     }
    
    iter++;
  }
  
  if(map_->map_points_.size() < 5000)
    addMapPoints();
  
//  cout << "optimizeMap() cost time:"<<timer.elapsed()<<endl;
}


}