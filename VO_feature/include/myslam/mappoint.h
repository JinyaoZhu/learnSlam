#ifndef __MAP_POINT_H__
#define __MAP_POINT_H__

#include "myslam/common_include.h"


namespace myslam
{
class Frame;
class MapPoint
{
public:
  typedef shared_ptr<MapPoint> Ptr;
  static unsigned long factory_id_;
  unsigned long id_;
  bool is_good_; // whether the map point is is_good
  Eigen::Vector3d pos_; // position of the map point in world coordinate
  cv::Mat descriptor_; // discriptor for machting
  list<Frame*> observed_frames_; // Frames that can observe this map point
  
  int matched_times_;
  int visible_times_;
  double matched_ratio_;
  
  MapPoint();
  MapPoint(unsigned long id,const Eigen::Vector3d &pos,Frame* frame = nullptr,const cv::Mat &descriptor = cv::Mat());
  
  static MapPoint::Ptr createMapPoint();
  static MapPoint::Ptr createMapPoint(const Eigen::Vector3d &pos,Frame* frame,const cv::Mat &descriptor);
  
  inline cv::Point3d getPositionCV()const
  { 
    return cv::Point3d(pos_(0,0),pos_(1,0),pos_(2,0)); 
  }
};
  
}



#endif /* __MAP_POINT_H__ */