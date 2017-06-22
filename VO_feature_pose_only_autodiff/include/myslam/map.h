#ifndef __MAP_H__
#define __MAP_H__

#include "myslam/common_include.h"

#include "myslam/mappoint.h"
#include "myslam/frame.h"

namespace myslam
{
class Map
{
public:
  typedef std::shared_ptr<Map> Ptr;
  unordered_map<unsigned long,MapPoint::Ptr> map_points_; // all landmark
  unordered_map<unsigned long,Frame::Ptr> key_frames_;    // all key key_frames
  Map(){}
  void insertKeyFrame(Frame::Ptr frame);
  void insertMapPoint(MapPoint::Ptr map_point);
};
}



#endif /* __MAP_H__ */