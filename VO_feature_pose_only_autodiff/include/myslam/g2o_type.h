/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef MYSLAM_G2O_TYPE_H
#define MYSLAM_G2O_TYPE_H

#include "myslam/common_include.h"
#include "camera.h"

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>

#include "Eigen/Core"
#include "Eigen/Geometry"

#include "ceres/rotation.h"
#include <ceres/internal/autodiff.h>

namespace myslam
{
class EdgeProjectXYZRGBD : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    virtual void computeError();
    virtual void linearizeOplus();
    virtual bool read( std::istream& in ){}
    virtual bool write( std::ostream& out) const {}
    
};

// only to optimize the pose, no point
class EdgeProjectXYZRGBDPoseOnly: public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap >
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Error: measure = R*point+t
    virtual void computeError();
    virtual void linearizeOplus();
    
    virtual bool read( std::istream& in ){}
    virtual bool write( std::ostream& out) const {}
    
    Eigen::Vector3d point_;
};

class VertexProjectXYZ2UVPoseOnly : public g2o::BaseVertex<6,Eigen::Matrix<double,6,1>>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexProjectXYZ2UVPoseOnly(){}

  virtual bool read(istream& in){}
  virtual bool write(ostream& out)const{};

  virtual void setToOriginImpl() {
       _estimate.setZero();
  }

  virtual void oplusImpl(const double* update_)  {
    
    Eigen::Matrix<double,6,1>::ConstMapType update(update_);
    
//       _estimate = (g2o::SE3Quat::exp(update)*g2o::SE3Quat::exp(estimate())).log();
       _estimate += update;
  }  
};



class EdgeProjectXYZ2UVPoseOnly: public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexProjectXYZ2UVPoseOnly>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    void computeError()
    {
 	const VertexProjectXYZ2UVPoseOnly* pose = static_cast<const VertexProjectXYZ2UVPoseOnly*> ( _vertices[0] );

	(*this)(pose->estimate().data(),_error.data());
    }
    
    template<typename T>
    bool operator ()(const T* const x,T* residual)const
    {
      Eigen::Matrix<T,3,1> omega(x[0],x[1],x[2]);
      Eigen::Matrix<T,3,1> upsilon(x[3],x[4],x[5]);
      
      T theta = omega.norm();
      
      Eigen::Matrix<T,3,1> omega_n = omega/theta;
      
      Eigen::Matrix<T,3,3> omega_n_hat;
      
      omega_n_hat(0,0) = T(0.0);      omega_n_hat(0,1) = -omega_n(2);  omega_n_hat(0,2) = omega_n(1);
      omega_n_hat(1,0) = omega_n(2);  omega_n_hat(1,1) = T(0.0);       omega_n_hat(1,2) = -omega_n(0);
      omega_n_hat(2,0) = -omega_n(1); omega_n_hat(2,1) = omega_n(0);   omega_n_hat(2,2) = T(0.0);
      
      Eigen::AngleAxis<T> angle_axis(theta,omega_n);
      
      Eigen::Matrix<T,3,3>V;
       
      V = Eigen::Matrix<T,3,3>::Identity()*sin(theta)/theta
	  + (1.0 - sin(theta)/theta)*omega_n*omega_n.transpose() 
	  + (1.0-cos(theta))/theta*omega_n_hat;
	  
      Eigen::Matrix<T,3,1> p_world(T(point_(0)),T(point_(1)),T(point_(2))); 
	      
      Eigen::Matrix<T,3,1> p_cam = angle_axis*p_world + V*upsilon;

      Eigen::Matrix<T,2,1> p_pixel;

      p_pixel(0) = p_cam(0)*T(camera_->fx_)/p_cam(2) + T(camera_->cx_);
      p_pixel(1) = p_cam(1)*T(camera_->fy_)/p_cam(2) + T(camera_->cy_);
	      
      residual[0] = T(_measurement(0)) - p_pixel(0);
      residual[1] = T(_measurement(1)) - p_pixel(1);
      return true;
    }

    void linearizeOplus()
    {
	const VertexProjectXYZ2UVPoseOnly* pose = static_cast<const VertexProjectXYZ2UVPoseOnly*> ( _vertices[0] );
	
// 	Eigen::Matrix<double,6,1> se3 = pose->estimate();
// 	
// 	Eigen::Vector3d omega(se3(0),se3(1),se3(2));
// 	Eigen::Vector3d upsilon(se3(3),se3(4),se3(5));
//  	
//  	double theta = omega.norm();
// 	
// 	Eigen::Vector3d omega_n = omega/theta;
// 	
// 	Eigen::Matrix<double,3,3> omega_n_hat;
// 	
// 	omega_n_hat(0,0) = 0;           omega_n_hat(0,1) = -omega_n(2); omega_n_hat(0,2) = omega_n(1);
// 	omega_n_hat(1,0) = omega_n(2);  omega_n_hat(1,1) = 0;           omega_n_hat(1,2) = -omega_n(0);
// 	omega_n_hat(2,0) = -omega_n(1); omega_n_hat(2,1) = omega_n(0);  omega_n_hat(2,2) = 0;
//  	
//  	Eigen::AngleAxis<double> angle_axis(theta,omega_n);
// 	
// 	Eigen::Matrix<double,3,3>V;
// 	
// 	V = Eigen::Matrix<double,3,3>::Identity()*sin(theta)/theta
// 	    + (1 - sin(theta)/theta)*omega_n*omega_n.transpose() 
// 	    + (1-cos(theta))/theta*omega_n_hat;
//  		
//  	Eigen::Vector3d p_cam = angle_axis*point_ + V*upsilon;
// 	
// 	double x = p_cam(0);
// 	double y = p_cam(1);
// 	double z = p_cam(2);
// 	double z_2 = z*z;
// 
// 	_jacobianOplusXi ( 0,0 ) =  x*y/z_2 *camera_->fx_;
// 	_jacobianOplusXi ( 0,1 ) = -( 1+ ( x*x/z_2 ) ) *camera_->fx_;
// 	_jacobianOplusXi ( 0,2 ) = y/z * camera_->fx_;
// 	
// 	_jacobianOplusXi ( 0,3 ) = -1./z * camera_->fx_;
// 	_jacobianOplusXi ( 0,4 ) = 0;
// 	_jacobianOplusXi ( 0,5 ) = x/z_2 * camera_->fx_;
// 
// 	_jacobianOplusXi ( 1,0 ) = ( 1+y*y/z_2 ) *camera_->fy_;
// 	_jacobianOplusXi ( 1,1 ) = -x*y/z_2 *camera_->fy_;
// 	_jacobianOplusXi ( 1,2 ) = -x/z *camera_->fy_;
// 	
// 	_jacobianOplusXi ( 1,3 ) = 0;
// 	_jacobianOplusXi ( 1,4 ) = -1./z *camera_->fy_;
// 	_jacobianOplusXi ( 1,5 ) = y/z_2 *camera_->fy_;

	double se3[6] = {pose->estimate()(0),pose->estimate()(1),pose->estimate()(2),
	  pose->estimate()(3),pose->estimate()(4),pose->estimate()(5)};
	  
	Eigen::Matrix<double,2, 6,Eigen::RowMajor> J_error;
	double *parameters[] = {se3};
	double *jacobians[] = {J_error.data()};
	double value[2];
	
	typedef ceres::internal::AutoDiff<EdgeProjectXYZ2UVPoseOnly,double,6> AutoDiffType;
	
	bool diffState = AutoDiffType::Differentiate( *this, parameters, 2, value, jacobians);
	
	if(diffState)
	{
	  _jacobianOplusXi = J_error;
	}
	else
	{
	  assert ( 0 && "Error while differentiating" );
	  _jacobianOplusXi.setZero();
	  _jacobianOplusXi.setZero();
	}
    }
    
    virtual bool read( std::istream& in ){}
    virtual bool write(std::ostream& os) const {};
    
    Eigen::Vector3d point_;
    Camera* camera_;
};

}


#endif // MYSLAM_G2O_TYPE_H
