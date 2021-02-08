#include <eris/simultaneous_solver.hpp>

namespace eris::hand_eye_calibration2
{
auto Solver::AddResidualBlock(const Eigen::Vector4d& qi1, const Eigen::Vector3d& ti1, const Eigen::Vector3d& pi1, 
                                const Eigen::Vector4d& qj1, const Eigen::Vector3d& tj1, const Eigen::Vector3d& pj1, 
                                const Eigen::Vector4d& qi2, const Eigen::Vector3d& ti2, const Eigen::Vector3d& pi2, 
                                const Eigen::Vector4d& qj2, const Eigen::Vector3d& tj2, const Eigen::Vector3d& pj2) -> bool
{
  ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 3, 4, 3>(new CostFunctor(qi1, ti1, pi1, qj1, tj1, pj1, qi2, ti2, pi2, qj2, tj2, pj2));
  problem_.AddResidualBlock(cost_function, NULL, qx_opt_.data(), tx_opt_.data(), qy_opt_.data(), ty_opt_.data());
  return true;
}

auto Solver::Solve() -> std::tuple<Eigen::Vector4d, Eigen::Vector3d, , Eigen::Vector4d, Eigen::Vector3d>
{
  if (!local_parameterization_is_set_)
  {
    problem_.SetParameterization(qx_opt_.data(), qy_opt_.data() , new ceres::QuaternionParameterization());
    local_parameterization_is_set_ = true;
  }

  options_.linear_solver_type = ceres::DENSE_QR;
  // options_.num_threads = 12;

  ceres::Solve(options_, &problem_, &summary_);
  return std::make_tuple<Eigen::Vector4d, Eigen::Vector3d, Eigen::Vector4d, Eigen::Vector3d>(std::move(qx_opt_), std::move(tx_opt_), std::move(qy_opt_), std::move(ty_opt_));
};

auto Solver::Options() -> ceres::Solver::Options
{
  return options_;
}

auto Solver::Summary() -> ceres::Solver::Summary
{
  return summary_;
}

}