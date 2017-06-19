#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <boost/timer.hpp>
#include <boost/concept_check.hpp>

#include <opencv2/core/core.hpp>

#include "Eigen/Core"
#include "Eigen/Geometry"

#include "sophus/so3.h"

using namespace std;

/*
 * given two 3D vectors, using ceres to estimate the rotation(AngleAxis)
 * between the vectors
 */


class CostFunction
{
public:
  CostFunction(const Eigen::Vector3d p1, const Eigen::Vector3d p2):_p1(p1),_p2(p2){}
  
  template<typename T>
  bool operator()( const T * const a, T* residual)const
  {
//     Eigen::Matrix<T,3,1> p1(T(_p1(0)),T(_p1(1)),T(_p1(2)));
//     Eigen::Matrix<T,3,1> p2_est;
//     
//     T a_theta = ceres::sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
//     T a_n[3] = {a[0]/a_theta,a[1]/a_theta,a[2]/a_theta};
// 
//     Eigen::AngleAxis<T> so3(a_theta,Eigen::Matrix<T,3,1>(a_n));
//     Eigen::Matrix<T,3,3> SO3 = so3.toRotationMatrix();
//     p2_est = SO3*p1;
    T p1[3] = {T(_p1(0)),T(_p1(1)),T(_p1(2))};
    T p2_est[3];
    ceres::AngleAxisRotatePoint(a,p1,p2_est);
    
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
  vector<Eigen::Vector3d> p1;
  vector<Eigen::Vector3d> p2;
  
  cv::RNG rng;
  
  int N = 100;
  double sigma_w = 0.1;
  
  double theta(M_PI*0.5);
  Eigen::Vector3d axis(0.5,1.2,1);
    
  Eigen::AngleAxis<double> so3(theta,axis.normalized());
  
  //generate data
  for(int i=0; i<N; i++)
  {
    Eigen::Vector3d p((double)i/N*10.0,-(double)i/N*5.0,(double)i/N*10.0);
    p1.push_back(p);
    p2.push_back(so3.toRotationMatrix()*p + Eigen::Vector3d(rng.gaussian(sigma_w),rng.gaussian(sigma_w),rng.gaussian(sigma_w)));
  }
  
  double so3_est[3];
  so3_est[0] = so3.angle()*so3.axis()(0) + rng.gaussian(0.1);  
  so3_est[1] = so3.angle()*so3.axis()(1) + rng.gaussian(0.1);  
  so3_est[2] = so3.angle()*so3.axis()(2) + rng.gaussian(0.1);  
   
  ceres::Problem problem;
  
  for(int i=0; i<N; i++)
    problem.AddResidualBlock(CostFunction::Create(p1[i],p2[i]),nullptr,so3_est);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  
  ceres::Solver::Summary summary;
  
  boost::timer timer;
  ceres::Solve(options,&problem,&summary);
  cout << timer.elapsed() << endl;
  
  cout << summary.BriefReport() <<endl;


  cout<< "so3:"<< (so3.angle()*so3.axis()).transpose() <<endl;
  cout<< "so3 estimate:"<< Eigen::Vector3d(so3_est).transpose() <<endl;
  cout <<"error:" <<(so3.angle()*so3.axis() - Eigen::Vector3d(so3_est)).transpose() <<endl;
    
    return 0;
}
