#include "myslam/camera.h"
#include "myslam/config.h"

namespace myslam 
{
Camera::Camera()
{
    fx_ = Config::get<double>("camera.fx");
    fy_ = Config::get<double>("camera.fy");
    cx_ = Config::get<double>("camera.cx");
    cy_ = Config::get<double>("camera.cy");
    depth_scale_ = Config::get<double>("camera.depth_scale");
    matrix_= ((cv::Mat_<double>(3,3))<<fx_,0,cx_,
              0,fy_,cy_,
	      0,0,1.0);
    k1_ =  Config::get<double>("camera.k1");
    k2_ =  Config::get<double>("camera.k2");
    p1_ =  Config::get<double>("camera.p1");
    p2_ =  Config::get<double>("camera.p2");
    k3_ =  Config::get<double>("camera.k3");
    distor_ = ((cv::Mat_<double>(5,1))<<k1_,k2_,p1_,p2_,k3_);
}
Eigen::Vector3d Camera::world2camera(const Eigen::Vector3d& p_w, const Sophus::SE3& T_c_w)
{
  return T_c_w*p_w;
}

Eigen::Vector3d Camera::camera2world(const Eigen::Vector3d& p_c, const Sophus::SE3& T_c_w)
{
  return T_c_w.inverse()*p_c;
}

Eigen::Vector2d Camera::camera2pixel(const Eigen::Vector3d& p_c)
{
  Eigen::Vector2d camera_n(p_c(0)/p_c(2),p_c(1)/p_c(2));
  return Eigen::Vector2d(fx_*camera_n(0) + cx_,fy_*camera_n(1) + cy_);
}

Eigen::Vector3d Camera::pixel2camera(const Eigen::Vector2d& p_p, double depth)
{
  Eigen::Vector2d camera_n((p_p(0)-cx_)/fx_,(p_p(1)-cy_)/fy_);
  
  return Eigen::Vector3d(camera_n(0)*depth,camera_n(1)*depth,depth);
}

Eigen::Vector2d Camera::world2pixel(const Eigen::Vector3d& p_w, const Sophus::SE3& T_c_w)
{
  return camera2pixel(world2camera(p_w,T_c_w));
}

Eigen::Vector3d Camera::pixel2world(const Eigen::Vector2d& p_p, const Sophus::SE3& T_c_w, double depth)
{
  return camera2world(pixel2camera(p_p,depth),T_c_w);
}

}