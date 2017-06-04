#include <iostream>
#include <ceres/ceres.h>
#include <boost/timer.hpp>
#include <boost/concept_check.hpp>

#include <opencv2/core/core.hpp>

#include "Eigen/Core"
#include "Eigen/Geometry"

#include "sophus/so3.hpp"

using namespace std;

struct COST_FUNCTION
{
  COST_FUNCTION(const Eigen::Vector3d p1, const Eigen::Vector3d p2):_p1(p1),_p2(p2){}
  
  template<typename T>
  bool operator()(const T * const a,T* residual)const
  {
    //Sophus::SO3<T> SO3;
    //Sophus::Vector<T,3> so3(a[0],a[1],a[2]);
    //Sophus::SO3<T> SO3;
    
    //Sophus::Vector<T,3> so3(0,0,0);
    //so3_d = dynamic_cast<Sophus::Vector<double,3>*>(&so3);
   // SO3 = Sophus::SO3<T>::exp(so3);
    return true;
  }
   
  
  static ceres::CostFunction* Create(const Eigen::Vector3d p1,const Eigen::Vector3d p2)
  {
    return new ceres::AutoDiffCostFunction<COST_FUNCTION,3,3>(new COST_FUNCTION(p1,p2));
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
  
  Sophus::SO3<double> SO3_R(R);  
  
  Sophus::Vector<double,3> so3 = SO3_R.log();
  
  double so3_d[3];
  
  so3_d[0] = so3(0,0);
  so3_d[1] = so3(1,0);
  so3_d[2] = so3(2,0);
  
  Sophus::SO3<double> SO3;
  
  Sophus::Vector<double,3> update(1e-1,0,0);
  
  SO3 = Sophus::SO3<double>::exp(update);
  
  so3 = SO3.log();
  
  cout << so3 <<endl;
  
  
   
//   ceres::Problem problem;
//   
//   problem.AddResidualBlock(COST_FUNCTION::Create(p1,p2),NULL,so3_d);
// 
//   ceres::Solver::Options options;
//   options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
//   options.minimizer_progress_to_stdout = true;
//   
//   ceres::Solver::Summary summary;
//   
//   boost::timer timer;
//   ceres::Solve(options,&problem,&summary);
//   cout << timer.elapsed() << endl;
//   
//   cout << summary.BriefReport() <<endl;


  cout << endl;
    
    return 0;
}
