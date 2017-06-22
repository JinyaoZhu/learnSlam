#include "myslam/vo.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "myslam/g2o_type.h"
#include "myslam/config.h"

#include "boost/timer.hpp"
#include <sys/stat.h>

namespace myslam {
  
VO::VO():state_(INITIALIZE),map_(new Map()),ref_(nullptr),
curr_(nullptr), num_inliers_(0), num_lost_(0) //,matcher_(new cv::flann::LshIndexParams ( 5,10,2 ))
{  
  num_of_features_ = Config::get<int>("num_of_features");
  orb_sacle_factor_ = Config::get<double>("orb_sacle_factor");
  orb_level_pyramid_ = Config::get<int>("orb_level_pyramid");
  min_match_ratio_ = Config::get<double>("min_match_ratio");
  max_num_lost_ = Config::get<int>("max_num_lost");
  min_inliers_rate_ = Config::get<double>("min_inliers_rate");
  max_map_points_ = Config::get<int>("max_map_points");
  
  frame_max_rot_ = Config::get<int>("frame_max_rot");
  frame_max_trans_ = Config::get<int>("frame_max_trans");
  
  
  detector_ = cv::ORB::create ( num_of_features_,orb_sacle_factor_,orb_level_pyramid_);
  matcher_ =  cv::DescriptorMatcher::create ( "BruteForce-Hamming" );
}
  
  
bool VO::addFrame(Frame::Ptr frame)
{
  num_loop_++;
  curr_ = frame;
  cout << "map points size:"<<map_->map_points_.size()<<endl;
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
    if(checkEstimatedPose() == true)
    {
      curr_->T_c_w_ = T_c_w_estimated_;
      optimizeMap();
      ref_ = curr_;
      num_lost_ = 0;
    }
    else
    {
      //cv::waitKey(0);
      num_lost_++;
      if(num_lost_ > max_num_lost_)
	state_ = LOST;
    }
    break;
  }
  case LOST:
  {
    cout<<"I am lost!"<<endl;
    extractKeyPoints();
    computeDescriptors();
    featureMatching();
    poseEstimatePnP();
    if(checkEstimatedPose() == true)
    {
      state_ = OK;
      curr_->T_c_w_ = T_c_w_estimated_;
      ref_ = curr_;
    }
    break;
  }
}
if(state_ != LOST)
return true;
else
  return false;
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
    if(curr_->isInFrame(p->pos_) && state_!= LOST)
    {
      //add candidates
      p->visible_times_++;
      map_candidates.push_back(p);
      candidates_descriptor.push_back(p->descriptor_);
    }
    else
    {
      map_candidates.push_back(p);
      candidates_descriptor.push_back(p->descriptor_);
    }
    
  }
//  cout<<"candidates size:"<<map_candidates.size()<<endl;
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
      if(m.distance < max<float>(min_dis*2,30))
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
   curr_->camera_->distor_,rvec,tvec,false,100,6.0,0.99,inliers);
    
  num_inliers_ = inliers.rows;
  
//  cout<<"inliers rate:"<<(double)num_inliers_/matched_3d_points_.size()<<endl;
  
  // too few inliners
  if((double)num_inliers_/matched_3d_points_.size() < min_inliers_rate_)
    return;
  
  for(int i=0; i<inliers.rows; i++)
  {
     int index = inliers.at<int> ( i,0 );
     matched_3d_points_[index]->matched_times_ ++;
     matched_3d_points_[index]->matched_ratio_ = (float) matched_3d_points_[index]->matched_times_/ matched_3d_points_[index]->visible_times_; 
  }
  
  
  Sophus::SE3 T_c_w_estimated(Sophus::SO3( rvec.at<double> ( 0,0 ), rvec.at<double> ( 1,0 ), rvec.at<double> ( 2,0 ) ),
                           Eigen::Vector3d ( tvec.at<double> ( 0,0 ), tvec.at<double> ( 1,0 ), tvec.at<double> ( 2,0 ) ));
  
     // using bundle adjustment to optimize the pose
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block ( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
//    g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );
    optimizer.setVerbose ( true );

    // vertex pose
    VertexProjectXYZ2UVPose* pose = new VertexProjectXYZ2UVPose();
    pose->setId ( 0 );
    pose->setEstimate (g2o::SE3Quat (T_c_w_estimated.rotation_matrix(), T_c_w_estimated.translation()).log());
    optimizer.addVertex ( pose );
    
    // landmarks
    for ( int i=0; i<inliers.rows; i++ )
    {
      int index = inliers.at<int>( i,0 );
      VertexProjectXYZ2UVPoint* point = new VertexProjectXYZ2UVPoint();
      point->setId ( i + 1 );
      point->setEstimate ( Eigen::Vector3d ( pts_3d[index].x, pts_3d[index].y, pts_3d[index].z ) );
      point->setMarginalized ( true ); 
      optimizer.addVertex (point);
    }

    // edges
    for ( int i=0; i<inliers.rows; i++ )
    {
        int index = inliers.at<int>( i,0 );
        // 3D -> 2D projection
        EdgeProjectXYZ2UV* edge = new EdgeProjectXYZ2UV();
        edge->setId ( i );
        edge->setVertex ( 0, optimizer.vertex(0) );
	edge->setVertex ( 1, optimizer.vertex(i+1));
        edge->camera_ = curr_->camera_.get();
        edge->setMeasurement ( Eigen::Vector2d ( pts_2d[index].x, pts_2d[index].y ) );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
	g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        rk->setDelta(10.0);
        edge->setRobustKernel(rk);
        optimizer.addEdge ( edge );
    }

    
    optimizer.initializeOptimization();
    optimizer.optimize ( 1000 );
    
     T_c_w_estimated_ = Sophus::SE3 (g2o::SE3Quat::exp(pose->estimate()).rotation(),
                           g2o::SE3Quat::exp(pose->estimate()).translation() );
     
      for ( int i=0; i<inliers.rows; i++ )
      {
	int index = inliers.at<int>( i,0 );
	matched_3d_points_[index]->pos_ = dynamic_cast<VertexProjectXYZ2UVPoint*>(optimizer.vertex(i+1))->estimate();
      }
}

void VO::addKeyFrame()
{
  map_->insertKeyFrame(curr_);
}

void VO::addMapPoints()
{
  int add_size = 0;
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
    add_size++;
  }
  cout << "add "<<add_size<<" map points"<<endl;
//  cout << "map points size:"<<map_->map_points_.size()<<endl;
}

void VO::optimizeMap()
{
  boost::timer timer;
  static float min_match_ratio = min_match_ratio_;
  for(auto iter=map_->map_points_.begin();iter != map_->map_points_.end();)
  {
    MapPoint::Ptr mp = iter->second;
    
    if(mp->matched_ratio_ < min_match_ratio)
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
  
  if(map_->map_points_.size() > max_map_points_)
  {
      min_match_ratio += 0.01;
  }
  else
    min_match_ratio = min_match_ratio_;
  
  addMapPoints();
  
//  cout << "map points size:"<<map_->map_points_.size()<<endl;
//  cout << "optimizeMap() cost time:"<<timer.elapsed()<<endl;
}

bool VO::checkEstimatedPose()
{
//   if(num_inliers_ < min_inliers_)
//     return false;
  
  if((double)num_inliers_/matched_3d_points_.size() < min_inliers_rate_)
    return false;
  
  return true;
}

 unsigned long VO::num_loop_ = 0;
}