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

class VertexProjectXYZ2UVPoint : public g2o::BaseVertex<3,Eigen::Matrix<double,3,1>>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexProjectXYZ2UVPoint(){}

  virtual bool read(istream& in){}
  virtual bool write(ostream& out)const{};

  virtual void setToOriginImpl() {
       _estimate.setZero();
  }

  virtual void oplusImpl(const double* update_)  {
    
    Eigen::Matrix<double,3,1>::ConstMapType update(update_);
   
    _estimate += update;
  }  
};

class VertexProjectXYZ2UVPose : public g2o::BaseVertex<6,Eigen::Matrix<double,6,1>>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexProjectXYZ2UVPose(){}

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



class EdgeProjectXYZ2UV: public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexProjectXYZ2UVPose,VertexProjectXYZ2UVPoint>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    void computeError()
    {
 	const VertexProjectXYZ2UVPose* pose = static_cast<const VertexProjectXYZ2UVPose*> (vertex(0));
        const VertexProjectXYZ2UVPoint* point = static_cast<const VertexProjectXYZ2UVPoint*> (vertex(1));
	
	(*this)(pose->estimate().data(),point->estimate().data(),_error.data());
    }
    
    template<typename T>
    bool operator ()(const T* const x,const T*const p,T* residual)const
    {
      Eigen::Matrix<T,3,1> omega(x[0],x[1],x[2]);
      Eigen::Matrix<T,3,1> upsilon(x[3],x[4],x[5]);
      
      Eigen::Matrix<T,3,1> land_mark(p[0],p[1],p[2]); 
      
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
	      
      Eigen::Matrix<T,3,1> p_cam = angle_axis*land_mark + V*upsilon;

      Eigen::Matrix<T,2,1> p_pixel;

      p_pixel(0) = p_cam(0)*T(camera_->fx_)/p_cam(2) + T(camera_->cx_);
      p_pixel(1) = p_cam(1)*T(camera_->fy_)/p_cam(2) + T(camera_->cy_);
	      
      residual[0] = T(_measurement(0)) - p_pixel(0);
      residual[1] = T(_measurement(1)) - p_pixel(1);
      return true;
    }

    void linearizeOplus()
    {
	const VertexProjectXYZ2UVPose* pose = static_cast<const VertexProjectXYZ2UVPose*> ( vertex(0) );
	const VertexProjectXYZ2UVPoint* point = static_cast<const VertexProjectXYZ2UVPoint*> (vertex(1) );
		  
	Eigen::Matrix<double,2, 6,Eigen::RowMajor> J_pose;
	Eigen::Matrix<double,2, 3,Eigen::RowMajor> J_point;
	double *parameters[] = {const_cast<double*> (pose->estimate().data()),const_cast<double*> (point->estimate().data())};
	double *jacobians[] = {J_pose.data(),J_point.data()};
	double value[2];
	
	typedef ceres::internal::AutoDiff<EdgeProjectXYZ2UV,double,6,3> AutoDiffType;
	
	bool diffState = AutoDiffType::Differentiate( *this, parameters, 2, value, jacobians);
	
	if(diffState)
	{
	  _jacobianOplusXi = J_pose;
	  _jacobianOplusXj = J_point;
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
    
    Camera* camera_;
};

}


#endif // MYSLAM_G2O_TYPE_H
