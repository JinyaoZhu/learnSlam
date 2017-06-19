#include <iostream>
#include <ceres/ceres.h>
#include <boost/timer.hpp>
#include <boost/concept_check.hpp>

#include <opencv2/core/core.hpp>

#include "Eigen/Core"
#include "Eigen/Geometry"

#include "sophus/so3.h"

using namespace std;

#include <algorithm>
#include <cmath>
#include <limits>

//////////////////////////////////////////////////////////////////
// math functions needed for rotation conversion. 

// dot and cross production 

template<typename T> 
inline T DotProduct(const T x[3], const T y[3]) {
  return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
}

template<typename T>
inline void CrossProduct(const T x[3], const T y[3], T result[3]){
  result[0] = x[1] * y[2] - x[2] * y[1];
  result[1] = x[2] * y[0] - x[0] * y[2];
  result[2] = x[0] * y[1] - x[1] * y[0];
}

template<typename T>
inline void AngleAxisRotatePoint(const T angle_axis[3], const T pt[3], T result[3]) {
  const T theta2 = DotProduct(angle_axis, angle_axis);
  if (theta2 > T(std::numeric_limits<double>::epsilon())) {
    // Away from zero, use the rodriguez formula
    //
    //   result = pt costheta +
    //            (w x pt) * sintheta +
    //            w (w . pt) (1 - costheta)
    //
    // We want to be careful to only evaluate the square root if the
    // norm of the angle_axis vector is greater than zero. Otherwise
    // we get a division by zero.
    //
    const T theta = sqrt(theta2);
    const T costheta = cos(theta);
    const T sintheta = sin(theta);
    const T theta_inverse = 1.0 / theta;

    const T w[3] = { angle_axis[0] * theta_inverse,
                     angle_axis[1] * theta_inverse,
                     angle_axis[2] * theta_inverse };

    // Explicitly inlined evaluation of the cross product for
    // performance reasons.
    /*const T w_cross_pt[3] = { w[1] * pt[2] - w[2] * pt[1],
                              w[2] * pt[0] - w[0] * pt[2],
                              w[0] * pt[1] - w[1] * pt[0] };*/
    T w_cross_pt[3];
    CrossProduct(w, pt, w_cross_pt);                          


    const T tmp = DotProduct(w, pt) * (T(1.0) - costheta);
    //    (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (T(1.0) - costheta);

    result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
    result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
    result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
  } else {
    // Near zero, the first order Taylor approximation of the rotation
    // matrix R corresponding to a vector w and angle w is
    //
    //   R = I + hat(w) * sin(theta)
    //
    // But sintheta ~ theta and theta * w = angle_axis, which gives us
    //
    //  R = I + hat(w)
    //
    // and actually performing multiplication with the point pt, gives us
    // R * pt = pt + w x pt.
    //
    // Switching to the Taylor expansion near zero provides meaningful
    // derivatives when evaluated using Jets.
    //
    // Explicitly inlined evaluation of the cross product for
    // performance reasons.
    /*const T w_cross_pt[3] = { angle_axis[1] * pt[2] - angle_axis[2] * pt[1],
                              angle_axis[2] * pt[0] - angle_axis[0] * pt[2],
                              angle_axis[0] * pt[1] - angle_axis[1] * pt[0] };*/
    T w_cross_pt[3];
    CrossProduct(angle_axis, pt, w_cross_pt); 

    result[0] = pt[0] + w_cross_pt[0];
    result[1] = pt[1] + w_cross_pt[1];
    result[2] = pt[2] + w_cross_pt[2];
  }
}


class CostFunction
{
public:
  CostFunction(const Eigen::Vector3d p1, const Eigen::Vector3d p2):_p1(p1),_p2(p2){}
  
  template<typename T>
  bool operator()( const T * const a, T* residual)const
  {
    T p1[3] = {T(_p1(0)),T(_p1(1)),T(_p1(2))};
    T p2_est[3];
    AngleAxisRotatePoint(a,p1, p2_est);
    residual[0] = _p2(0) - p2_est[0];
    residual[1] = _p2(1) - p2_est[1];
    residual[2] = _p2(2) - p2_est[2];
    return true;
  }
   
  
  static ceres::CostFunction* Create(const Eigen::Vector3d p1,const Eigen::Vector3d p2)
  {
    return new ceres::AutoDiffCostFunction<CostFunction,3,3>(new CostFunction(p1,p2));
  }
  
private:
  Eigen::Vector3d _p1;
  Eigen::Vector3d _p2;
  
};

int main(int argc, char **argv)
{
  Eigen::Vector3d p1(1,0,0);
  Eigen::Vector3d p2(0,1,0);
  
  Eigen::Matrix3d R = Eigen::Quaterniond(1,1,0,0).toRotationMatrix();
  
  Sophus::SO3 SO3_R(R);  
  
  Eigen::Vector3d so3 = SO3_R.log();
  
  p2 = R*p1;
  
  double so3_d[3] = {0,0,0};
   
  cout << so3 <<endl;
  
  
   
  ceres::Problem problem;
  
  problem.AddResidualBlock(CostFunction::Create(p1,p2),nullptr,so3_d);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  
  ceres::Solver::Summary summary;
  
  boost::timer timer;
  ceres::Solve(options,&problem,&summary);
  cout << timer.elapsed() << endl;
  
  cout << summary.BriefReport() <<endl;


  cout << Eigen::Vector3d(so3_d) <<endl;
    
    return 0;
}
