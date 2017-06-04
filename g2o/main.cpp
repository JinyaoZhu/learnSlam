#include <iostream>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>

#include <cmath>
#include <boost/timer.hpp>
#include <boost/iterator/iterator_concepts.hpp>

#include <sophus/so3.hpp>

using namespace std;

class CurveFittingVertex: public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  virtual void setToOriginImpl()
  {
    _estimate << 0,0,0;
  }
  
  virtual void oplusImpl(const double* update)
  {    
    _estimate += Eigen::Matrix<double,3,1>(update);
  }
  
  virtual bool read(istream& in){}
  virtual bool write(ostream& out)const{};
};

class CurveFittingEdge: public g2o::BaseUnaryEdge<3,Eigen::Matrix<double,3,1> ,CurveFittingVertex>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CurveFittingEdge( Eigen::Matrix<double,3,1>  p1): BaseUnaryEdge(), _p1(p1) {}
  
  void computeError()
  {
    const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]);
    const Eigen::Vector3d so3 = v->estimate();
    Eigen::Matrix<double,3,1> mes(_measurement);
    Eigen::Matrix<double,3,1> p2_est;
    p2_est = Sophus::SO3<double>::exp(so3)*_p1;
    Eigen::Matrix<double,3,1> error;
    error = mes - p2_est;
    _error = error;
  }
  
  virtual bool read(istream& in){}
  virtual bool write(ostream& out)const{}
public:
  Eigen::Matrix<double,3,1>  _p1;  
};


int main(int argc, char **argv) {

    int N = 100;
    double w_sigma = 0.05;
    cv::RNG rng;

    Eigen::Matrix<double,3,3> R = Eigen::AngleAxis<double>(M_PI/12,Eigen::Vector3d(1,0.5,1)).toRotationMatrix();
    Eigen::Matrix<double,3,1> so3;
    
    so3 = Sophus::SO3<double>::log(R);
    
    vector< Eigen::Matrix<double,3,1> > p1_data,p2_data;
    
    cout << "generating data..."<<endl;
    for(int i=0;i<N;i++)
    {
      Eigen::Matrix<double,3,1>  p1;
      Eigen::Matrix<double,3,1>  p2;
      Eigen::Matrix<double,3,1> noise;
      p1 << i/10.0,i/10.0,i/20.0;
      noise << rng.gaussian(w_sigma),rng.gaussian(w_sigma),rng.gaussian(w_sigma);
      p2 = Sophus::SO3<double>::exp(so3)*p1 + noise;
      
      p1_data.push_back(p1);
      p2_data.push_back(p2);
    }
    
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,3>> Block;
    
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block(linearSolver);
    
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);
    
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate(so3 + Eigen::Vector3d(1e-2,1e-2,1e-2));
    v->setId(0);
    optimizer.addVertex(v);
    
    for(int i=0;i<N;i++)
    {
      CurveFittingEdge* edge = new CurveFittingEdge(p1_data[i]);
      edge->setId(0);
      edge->setVertex(0,v);
      edge->setMeasurement(p2_data[i]);
      edge->setInformation(Eigen::Matrix<double,3,3>::Identity()*1/(w_sigma*w_sigma));
      optimizer.addEdge(edge);
    }

    
    cout<<"start optimization..."<<endl;
    boost::timer timer;
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    cout<<"solve time cost: "<<timer.elapsed()<<" seconds"<<endl;
    
    Eigen::Vector3d abc_estimated = v->estimate();
    cout << "real model:" <<so3.transpose()<<endl;
    cout << "estimated model: "<<abc_estimated.transpose()<<endl;
   
    return 0;
}
