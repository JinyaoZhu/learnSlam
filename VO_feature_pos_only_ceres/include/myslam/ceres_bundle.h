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
  
    template<typename T>
    bool operator ()(const T* const x,T* residual)const
    {
      Eigen::Matrix<T,3,1> omega(x[0],x[1],x[2]);
      Eigen::Matrix<T,3,1> upsilon(x[3],x[4],x[5]);
      
      Eigen::Matrix<T,3,1> p_world(T(p_world_(0)),T(p_world_(1)),T(p_world_(2))); 
      
      T theta = omega.norm();
      
      Eigen::Matrix<T,3,1> omega_n = omega/theta;
      
      Eigen::Matrix<T,3,3> omega_n_hat;
      
      omega_n_hat(0,0) = T(0.0);      omega_n_hat(0,1) = -omega_n(2);  omega_n_hat(0,2) = omega_n(1);
      omega_n_hat(1,0) = omega_n(2);  omega_n_hat(1,1) = T(0.0);       omega_n_hat(1,2) = -omega_n(0);
      omega_n_hat(2,0) = -omega_n(1); omega_n_hat(2,1) = omega_n(0);   omega_n_hat(2,2) = T(0.0);
      
      Eigen::AngleAxis<T> angle_axis(theta,omega_n);
      
      Eigen::Matrix<T,3,3>V;
       
      if(theta < T(1e-10))
      {
	V = angle_axis.toRotationMatrix();
      }
      else
      {
	V = Eigen::Matrix<T,3,3>::Identity()*sin(theta)/theta
	    + (1.0 - sin(theta)/theta)*omega_n*omega_n.transpose() 
	    + (1.0-cos(theta))/theta*omega_n_hat;
      }
	      
      Eigen::Matrix<T,3,1> p_cam = angle_axis*p_world + V*upsilon;

      p_cam(0) /= p_cam(2);
      p_cam(1) /= p_cam(2);
          
      Eigen::Matrix<T,2,1> p_pixel;
      
      p_pixel(0) = p_cam(0)*T(K_(0,0)) + T(K_(0,2));
      p_pixel(1) = p_cam(1)*T(K_(1,1)) + T(K_(1,2));
	      
      residual[0] = T(observe_(0)) - p_pixel(0);
      residual[1] = T(observe_(1)) - p_pixel(1);
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



void ceresBundle(vector<Eigen::Vector3d> &p3d,vector<Eigen::Vector2d> &p2d,
		 const Eigen::Matrix3d &K,Eigen::Matrix<double,6,1> &se3);





#endif //CERES_BUNDLE_H