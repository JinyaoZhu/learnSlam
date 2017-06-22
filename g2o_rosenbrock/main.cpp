#include <iostream>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/robust_kernel.h>

#include "ceres/internal/autodiff.h"

using namespace std;

class Vertex: public g2o::BaseVertex<5,Eigen::Matrix<double,5,1>>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  virtual void setToOriginImpl()
  {
    _estimate.setZero();
  }
  
  virtual void oplusImpl(const double* update)
  { 
    Eigen::Matrix<double,5,1>::ConstMapType v(update);
    _estimate += v;
  }
  
  virtual bool read(istream& in){}
  virtual bool write(ostream& out)const{};
};


class Edge: public g2o::BaseUnaryEdge<4,Eigen::Matrix<double,4,1>,Vertex>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Edge(){}
  
  void computeError()
  {
    const Vertex * param = static_cast<const Vertex*>(vertex(0));
    
    double x1 = param->estimate()(0);
    double x2 = param->estimate()(1);
    double x3 = param->estimate()(2);
    double x4 = param->estimate()(3);
    double x5 = param->estimate()(4);
    
    Eigen::Matrix<double,4,1> error;
    
    error(0) = (1.0-x1)*(1.0-x1)+ 100.0*(x2 - x1*x1)*(x2 - x1*x1); // RosenBrock function
    error(1) = (1.0-x2)*(1.0-x2)+ 100.0*(x3 - x2*x2)*(x3 - x2*x2);
    error(2) = (1.0-x3)*(1.0-x3)+ 100.0*(x4 - x3*x3)*(x4 - x3*x3);
    error(3) = (1.0-x4)*(1.0-x4)+ 100.0*(x5 - x4*x4)*(x5 - x4*x4);
    
    //_error(0) = error(0)+error(1)+error(2)+error(3);
    _error = error;
  }
  
  template<typename T>
  bool operator() ( const T* const parameters, T* residuals ) const
  {
    T x1 = parameters[0];
    T x2 = parameters[1];
    T x3 = parameters[2];
    T x4 = parameters[3];
    T x5 = parameters[4];
    
    T error[4];
    
    error[0] = (1.0-x1)*(1.0-x1)+ 100.0*(x2 - x1*x1)*(x2 - x1*x1); // RosenBrock function
    error[1] = (1.0-x2)*(1.0-x2)+ 100.0*(x3 - x2*x2)*(x3 - x2*x2);
    error[2] = (1.0-x3)*(1.0-x3)+ 100.0*(x4 - x3*x3)*(x4 - x3*x3);
    error[3] = (1.0-x4)*(1.0-x4)+ 100.0*(x5 - x4*x4)*(x5 - x4*x4);
    
    //residuals[0] = error[0]+error[1]+error[2]+error[3];
    residuals[0] = error[0];
    residuals[1] = error[1];
    residuals[2] = error[2];
    residuals[3] = error[3];
    return true;
  }
  
  void linearizeOplus()
  {
    const Vertex * v = static_cast<const Vertex*>(vertex(0));
    
    double x1 = v->estimate()(0);
    double x2 = v->estimate()(1);
    double x3 = v->estimate()(2);
    double x4 = v->estimate()(3);
    double x5 = v->estimate()(4);
    
    double x[] = {x1,x2,x3,x4,x5};
    Eigen::Matrix<double,4, 5,Eigen::RowMajor> J_error;
    double *parameters[] = {x};
    double *jacobians[] = {J_error.data()};
    double value[4];
    
    typedef ceres::internal::AutoDiff<Edge,double,5> AutoDiffType;
    
    bool diffState = AutoDiffType::Differentiate( *this, parameters, 4, value, jacobians);
    
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
    
//     _jacobianOplusXi(0,0) = -2.0*(1.0 - x) - 400.0*(y - x*x)*x;
//     _jacobianOplusXi(0,1) = 200*(y - x*x);
  }
  
  virtual bool read(istream& in){}
  virtual bool write(ostream& out)const{}
};


int main(int argc, char** argv)
{
  // some handy typedefs
  typedef g2o::BlockSolver< g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >  MyBlockSolver;
  typedef g2o::LinearSolverDense<MyBlockSolver::PoseMatrixType> MyLinearSolver;

  // setup the solver
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);
  MyLinearSolver* linearSolver = new MyLinearSolver();
  MyBlockSolver* solver_ptr = new MyBlockSolver(linearSolver);
  //g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
  //g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
  g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg(solver_ptr);
  optimizer.setAlgorithm(solver);

  // 1. add the parameter vertex
  double p[5] = {10,-15,10,-1,-2};
  Vertex* v = new Vertex();
  v->setId(0);
  v->setEstimate(Eigen::Matrix<double,5,1>(p));
  optimizer.addVertex(v);
  
  cout <<"initial:" <<endl<<v->estimate() << endl;
  
  // 2. add edge
  Edge* e = new Edge;
  e->setInformation(Eigen::Matrix<double, 4, 4>::Identity());
  e->setVertex(0, v);
  g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
  rk->setDelta(1.0);
  e->setRobustKernel(rk);
  optimizer.addEdge(e);

  // perform the optimization
  optimizer.initializeOptimization();
  optimizer.setVerbose(true);
  optimizer.optimize(100000);
  
  cout <<"final:" <<endl<<v->estimate() << endl;
  
  return 0;
}