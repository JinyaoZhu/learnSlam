#include <iostream>
#include <ceres/ceres.h>
#include <boost/timer.hpp>

#include <opencv2/core/core.hpp>

#include "Eigen/Core"
#include "Eigen/Geometry"


using namespace std;


// class CostFunction : public ceres::FirstOrderFunction
// {
// public:
//    virtual ~CostFunction() {}
// 
//     virtual bool Evaluate(const double* parameters,double* cost, double* gradient) const 
//     {
//       const double x = parameters[0];
//       const double y = parameters[1];
//       cost[0] = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
//       if (gradient != NULL) {
// 	gradient[0] = -2.0 * (1.0 - x) - 200.0 * (y - x * x) * 2.0 * x;
// 	gradient[1] = 200.0 * (y - x * x);
//       }
// //      cost[0] = x*x + y*y;
// //       if(gradient != NULL)
// //       {
// // 	gradient[0] = 2*x;
// // 	gradient[1] = 2*y;
// //       }
//       return true;
//     }
//    
//    virtual int NumParameters() const { return 2; }
// };
// 
// int main(int argc, char **argv)
// {
//   double xy[2] = {-0,-0};
//    
//   ceres::GradientProblem problem (new CostFunction());
//   
//   ceres::GradientProblemSolver::Options options;
//   options.minimizer_progress_to_stdout = true;
//   options.max_num_iterations = 1000;
//   
//   ceres::GradientProblemSolver::Summary summary;
//   
//   ceres::Solve(options,problem,xy,&summary);
// 
//   cout << summary.FullReport() <<endl;
// 
//   cout << "x = "<<xy[0] << " y = "<< xy[1]<<endl;
//     
//     return 0;
// }


//Powell's Funtion
//   F = 1/2 (f1^2 + f2^2 + f3^2 + f4^2)
//   f1 = x1 + 10*x2;
//   f2 = sqrt(5) * (x3 - x4)
//   f3 = (x2 - 2*x3)^2
//   f4 = sqrt(10) * (x1 - x4)^2

// class CostFunction
// {
// public:
//     CostFunction(){}
//     template <typename T>
//     bool operator()(const T* const x,T* residual)const
//     {
//       residual[0] = x[0] + T(10.0)*x[1];
//       residual[1] = T(sqrt(5.0))*(x[2]-x[3]);
//       residual[2] = pow(x[1] - T(2.0)*x[2],2);
//       residual[3] = T(sqrt(10.0))*pow(x[0] - x[3],2);
//       return true;
//     }
//     
//     static ceres::CostFunction* Create(){
//       return new ceres::AutoDiffCostFunction<CostFunction,4,4>(new CostFunction());
//     }
//   
// };


// class CostFunction : public ceres::SizedCostFunction<1,2> 
// {
// public:    
//    virtual ~CostFunction() {}
//    
//    virtual bool Evaluate(double const* const* parameters, double* residual,double** jacobians) const 
//    {
//       double x = parameters[0][0];
//       double y = parameters[0][1];
//       residual[0] = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
//    
//    
//     if (jacobians != NULL && jacobians[0] != NULL) 
//     {
//       jacobians[0][0] = -2*(1.0-x) - 400.0*(y-x*x)*x;
//       jacobians[0][1] = 200*(y - x*x);
//     }
//     
//     return true;
//    }
//     
// };

class CostFunction
{
public:
    CostFunction(){}
    template <typename T>
    bool operator()(const T* const p,T* residual)const
    {
      T x = p[0];
      T y = p[1];
      residual[0] = (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
      //residual[0] = (x*x+y*y);
      return true;
    }
    
    static ceres::CostFunction* Create(){
      return new ceres::AutoDiffCostFunction<CostFunction,1,2>(new CostFunction());
    }
  
};

int main(int argc,char** argv)
{
  double x[2] = {2,0};
  
  ceres::Problem problem ;
  
  problem.AddResidualBlock(CostFunction::Create(),nullptr,x);
  
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 50000;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  //options.gradient_tolerance = 1e-20;
  //options.parameter_tolerance = 1e-20;
  
  ceres::Solver::Summary summary;
  
  ceres::Solve(options,&problem,&summary);

  cout << summary.BriefReport() <<endl;

  cout << Eigen::Vector2d(x) << endl;
  return 0;
}