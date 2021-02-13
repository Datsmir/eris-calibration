#include <eris/solver2.hpp>

namespace eris::hand_eye_calibration2
{
auto Solver2::AddResidualBlock(const Eigen::Vector4d& qi, const Eigen::Vector3d& ti, const Eigen::Vector3d& pi, const Eigen::Vector3d& pj) -> bool
{
  ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 3, 4, 3>(new CostFunctor(qi, ti, pi, pj));
  problem_.AddResidualBlock(cost_function, NULL, q_opt_.data(), t_opt_.data());
  return true;
}

auto Solver2::Solve() -> std::tuple<Eigen::Vector4d, Eigen::Vector3d>
{
  if (!local_parameterization_is_set_)
  {
    problem_.SetParameterization(q_opt_.data(), new ceres::QuaternionParameterization());
    local_parameterization_is_set_ = true;
  }

  options_.linear_solver_type = ceres::DENSE_QR;
  // options_.num_threads = 12;
  options_.max_num_iterations = 200;

  ceres::Solve(options_, &problem_, &summary_);
  return std::make_tuple<Eigen::Vector4d, Eigen::Vector3d>(std::move(q_opt_), std::move(t_opt_));
};

auto Solver2::Options() -> ceres::Solver::Options
{
  return options_;
}

auto Solver2::Summary() -> ceres::Solver::Summary
{
  return summary_;
}

}