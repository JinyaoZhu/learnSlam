#include "myslam/mappoint.h"

namespace myslam 
{
  
MapPoint::MapPoint():id_(0),is_good_(true),pos_(Eigen::Vector3d(0,0,0)),
                    visible_times_(0),matched_times_(0),matched_ratio_(0)
{}
  
MapPoint::MapPoint(long unsigned int id, const Eigen::Vector3d& pos, Frame* frame, const cv::Mat& descriptor):
id_(id),pos_(pos),is_good_(true),descriptor_(descriptor),visible_times_(1),matched_times_(1),matched_ratio_(1)
{
  observed_frames_.push_back(frame);
}

MapPoint::Ptr MapPoint::createMapPoint()
{
  return MapPoint::Ptr(new MapPoint(factory_id_++,Eigen::Vector3d(0,0,0)));
}

MapPoint::Ptr MapPoint::createMapPoint(const Eigen::Vector3d& pos, Frame* frame, const cv::Mat& descriptor)
{
  return MapPoint::Ptr(new MapPoint(factory_id_++,pos,frame,descriptor));
}


unsigned long MapPoint::factory_id_ = 0;

  
}