#ifndef __FRAME_H__
#define __FRAME_H__

#include "myslam/common_include.h"
#include "camera.h"

namespace myslam
{
class Frame
{
public:
  typedef std::shared_ptr<Frame> Ptr;
  unsigned long   id_;  //id of the frame
  double          time_stamp_; //time stamp of the frame
  Sophus::SE3             T_c_w_; // camera pose of the frame
  Camera::Ptr     camera_; //camera modell
  cv::Mat         color_;
  cv::Mat         depth_;
  bool            is_key_frame_;
public:
  Frame();
  ~Frame(){}
  Frame(unsigned long id,double time_stamp=0,Sophus::SE3 T_c_w=Sophus::SE3(),Camera::Ptr camera = nullptr,
        cv::Mat color = cv::Mat(),cv::Mat depth = cv::Mat());
  
  static Frame::Ptr creatFrame();
  
  double findDepth(cv::Point2f p);
  
  Eigen::Vector3d getCameraCenter()const;
  
  void setPost(const Sophus::SE3 &T_c_w);
  
  bool isInFrame(const Eigen::Vector3d &p_world);
  
};
}




#endif /* __FRAME_H__ */