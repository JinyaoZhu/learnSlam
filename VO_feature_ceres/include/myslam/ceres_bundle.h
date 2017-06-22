#ifndef CERES_BUNDLE_H
#define CERES_BUNDLE_H

#include "myslam/common_include.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>


class ceresCostFunction
{
public:
  ceresCostFunction(Eigen::Vector3d p_world,Eigen::Vector2d observe, Eigen::Matrix3d K):
  p_world_(p_world),observe_(observe),K_(K) {}
  
  template <typename T>
  bool operator()(const T* const camera,T* residual)const
  {
    Eigen::Matrix<T,3,1> p1(T(p_world_(0)),T(p_world_(1)),T(p_world_(2)));
    Eigen::Matrix<T,3,1> p2_est;
    
    T so3[3] = {camera[0],camera[1],camera[2]};
    T R[9];
    
    ceres::AngleAxisToRotationMatrix(so3,R);
    
    Eigen::Matrix<T,3,3> Rot;
    
    Rot = Eigen::Map<Eigen::Matrix<T,3,3>>(R);
    
    p2_est = Rot*p1;
    
    //ceres::AngleAxisRotatePoint<T>(camera, p1, p2_est);
    
    p2_est(0) += camera[3];
    p2_est(1) += camera[4];
    p2_est(2) += camera[5];
    
    p2_est(0) = p2_est(0)*T(K_(0,0))/p2_est(2) + T(K_(0,2));
    p2_est(1) = p2_est(1)*T(K_(1,1))/p2_est(2) + T(K_(1,2));
    
    residual[0] = p2_est(0) - T(observe_(0));
    residual[1] = p2_est(1) - T(observe_(1));
    
    return true;
  }
  
  static ceres::CostFunction* Create(Eigen::Vector3d p_world,Eigen::Vector2d observe, Eigen::Matrix3d K)
  {
    return new ceres::AutoDiffCostFunction<ceresCostFunction,2,6>(new ceresCostFunction(p_world,observe,K));
  }
private:
  Eigen::Vector3d p_world_;
  Eigen::Vector2d observe_;
  Eigen::Matrix3d K_;
};



void ceresBundle(vector<Eigen::Vector3d> &p3d,vector<Eigen::Vector2d> &p2d,Eigen::Matrix3d &K,Eigen::Matrix<double,6,1> &se3);







#endif //CERES_BUNDLE_H