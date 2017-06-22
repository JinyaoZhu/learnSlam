#ifndef __VO_H__
#define __VO_H__

#include "myslam/common_include.h"
#include <opencv2/features2d/features2d.hpp>
#include "myslam/map.h"


namespace myslam
{
class VO
{
public:
  typedef std::shared_ptr<VO> Ptr;
  enum VOState{INITIALIZE = -1, OK =0, LOST};
  
  VOState state_;
  Map::Ptr map_;
  Frame::Ptr ref_;
  Frame::Ptr curr_;
  
  cv::Ptr<cv::ORB> detector_;
  vector<cv::KeyPoint> key_points_curr_;
  cv::Mat descriptor_curr_;
  
  cv::Ptr<cv::DescriptorMatcher> matcher_;
  vector<MapPoint::Ptr> matched_3d_points_;
  vector<int> matched_2d_kp_index_;
  
  Sophus::SE3 T_c_w_estimated_;
  int num_inliers_;
  int num_lost_;
  static unsigned long num_loop_;
  
public:
  VO();
  ~VO(){};
  bool addFrame(Frame::Ptr frame);
  
  
private:
   int num_of_features_;
  double orb_sacle_factor_;
   int orb_level_pyramid_;
  double min_match_ratio_;
   int max_num_lost_;
   double min_inliers_rate_;
   int max_map_points_;
   double frame_max_rot_;
   double frame_max_trans_;
  
  
protected:
  void extractKeyPoints();
  void computeDescriptors();
  void featureMatching();
  void poseEstimatePnP();
  void optimizeMap();
  
  void addKeyFrame();
  void addMapPoints();
  
  bool checkEstimatedPose();

};
}



#endif /* __VO_H__ */