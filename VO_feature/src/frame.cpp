#include "myslam/frame.h"
#include <boost/concept_check.hpp>

namespace myslam 
{
  
Frame::Frame():id_(-1),time_stamp_(-1),camera_(nullptr),is_key_frame_(false)
{}

  
Frame::Frame(long unsigned int id, double time_stamp, Sophus::SE3 T_c_w, Camera::Ptr camera, cv::Mat color, cv::Mat depth):
id_(id),time_stamp_(time_stamp),T_c_w_(T_c_w),camera_(camera),color_(color),depth_(depth)
{}

Frame::Ptr Frame::creatFrame()
{
  static unsigned long fatory_id = 0;
  
  return Frame::Ptr(new Frame(fatory_id++));
}

double Frame::findDepth(cv::Point2f p)
{
  ushort d = depth_.ptr<ushort>(cvRound(p.y))[cvRound(p.x)];
  double dd;
  if( d!= 0)
    dd = (double)d/camera_->depth_scale_;
  else
    dd = -1.0;
  return dd;
}

Eigen::Vector3d Frame::getCameraCenter() const
{
  return T_c_w_.inverse().translation();
}

bool Frame::isInFrame(const Eigen::Vector3d& p_world)
{
  Eigen::Vector3d p_camera;
  p_camera = camera_->world2camera(p_world,T_c_w_);
  if(p_camera(2,0) < 0)
    return false;
  
  Eigen::Vector2d p_pixel;
  p_pixel = camera_->camera2pixel(p_camera);
  
  if(p_pixel(0,0) < 10 || p_pixel(1,0) < 10 || p_pixel(0,0)>color_.cols-11 || p_pixel(1,0) > color_.rows-11)
    return false;
  
  return true;
}

void Frame::setPost(const Sophus::SE3& T_c_w)
{
  T_c_w_ = T_c_w;
}

}