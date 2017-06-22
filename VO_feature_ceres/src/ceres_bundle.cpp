#include "myslam/ceres_bundle.h"

void ceresBundle(vector<Eigen::Vector3d> &p3d,vector<Eigen::Vector2d> &p2d,Eigen::Matrix3d &K,Eigen::Matrix<double,6,1> &se3)
{
  double est_se3[6];
  
  est_se3[0] = se3(0); 
  est_se3[1] = se3(1); 
  est_se3[2] = se3(2); 
  est_se3[3] = se3(3); 
  est_se3[4] = se3(4); 
  est_se3[5] = se3(5); 
  
  ceres::Problem problem;
  
  for (int i = 0; i < p3d.size(); ++i) 
    problem.AddResidualBlock(ceresCostFunction::Create(p3d[i],p2d[i],K), new ceres::HuberLoss(10.0),est_se3);
  
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  cout << summary.BriefReport() << "\n";
  
  se3 = Eigen::Map<Eigen::Matrix<double,6,1>>(est_se3);
//   se3(0) = est_se3[0];
//   se3(1) = est_se3[1];
//   se3(2) = est_se3[2];
//   se3(3) = est_se3[3];
//   se3(4) = est_se3[4];
//   se3(5) = est_se3[5];
}