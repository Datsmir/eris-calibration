#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>

#include <glog/logging.h>
#include <iostream>

namespace eris::hand_eye_calibration2
{
class CostFunctor
{
public:
   // qi = pose orientation, quaternion , ti = pose position, pi = observed point , pj = true point in base 
  CostFunctor(const Eigen::Vector4d& qi, const Eigen::Vector3d& ti, const Eigen::Vector3d& pi, const Eigen::Vector3d& pj)
    : qi_(qi), ti_(ti), pi_(pi), pj_(pj)
  {
  }

  template <typename T>
  auto operator()(const T* const qx, const T* const tx, T* residual) const -> bool
  {
    Eigen::Matrix<T, 4, 1> qi = qi_.cast<T>();
    Eigen::Matrix<T, 3, 1> ti = ti_.cast<T>();
    Eigen::Matrix<T, 3, 1> pi = pi_.cast<T>();
    Eigen::Matrix<T, 3, 1> pj = pj_.cast<T>();

    Eigen::Matrix<T, 3, 1> ppi;
    Eigen::Matrix<T, 3, 1> pppi;
    ceres::QuaternionRotatePoint(qx, pi.data(), ppi.data());
    ppi(0) += tx[0];
    ppi(1) += tx[1];
    ppi(2) += tx[2];
    ceres::QuaternionRotatePoint(qi.data(), ppi.data(), pppi.data());
    pppi(0) += ti[0];
    pppi(1) += ti[1];
    pppi(2) += ti[2];

    residual[0] = pj(0) - pppi(0);
    residual[1] = pj(1) - pppi(1);
    residual[2] = pj(2) - pppi(2);
    return true;
  }

private:
  const Eigen::Vector4d qi_;
  const Eigen::Vector3d ti_;
  const Eigen::Vector3d pi_;
  const Eigen::Vector3d pj_;
};

class Solver
{
public:
  Solver(const Eigen::Vector4d& q_init, const Eigen::Vector3d t_init) : q_opt_(q_init), t_opt_(t_init)
  {
  }

  auto AddResidualBlock(const Eigen::Vector4d&, const Eigen::Vector3d&, const Eigen::Vector3d&, const Eigen::Vector3d&) -> bool;

  auto Solve() -> std::tuple<Eigen::Vector4d, Eigen::Vector3d>;

  auto Summary() -> ceres::Solver::Summary;

  auto Options() -> ceres::Solver::Options;

private:
  ceres::Problem problem_;
  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;

  bool local_parameterization_is_set_ = false;

  Eigen::Vector4d q_opt_;
  Eigen::Vector3d t_opt_;
};
}