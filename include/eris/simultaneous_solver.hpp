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
  CostFunctor(
      const Eigen::Vector4d& qi1, const Eigen::Vector3d& ti1, const Eigen::Vector3d& pi1, 
      const Eigen::Vector4d& qj1, const Eigen::Vector3d& tj1, const Eigen::Vector3d& pj1,
      const Eigen::Vector4d& qi2, const Eigen::Vector3d& ti2, const Eigen::Vector3d& pi2,
      const Eigen::Vector4d& qj2, const Eigen::Vector3d& tj2, const Eigen::Vector3d& pj2
      )
    : qi1_(qi1), ti1_(ti1), pi1_(pi1), qj1_(qj1), tj1_(tj1), pj1_(pj1), qi2_(qi2), ti2_(ti2), pi2_(pi2), qj2_(qj2), tj2_(tj2), pj2_(pj2)
  {
  }

  template <typename T>
  auto operator()(const T* const qx, 
                  const T* const tx,
                  const T* const qy,
                  const T* const ty, 
                  T* residual) const -> bool
  {
    Eigen::Matrix<T, 4, 1> qi1 = qi1_.cast<T>();
    Eigen::Matrix<T, 3, 1> ti1 = ti1_.cast<T>();
    Eigen::Matrix<T, 3, 1> pi1 = pi1_.cast<T>();
    Eigen::Matrix<T, 4, 1> qj1 = qj1_.cast<T>();
    Eigen::Matrix<T, 3, 1> tj1 = tj1_.cast<T>();
    Eigen::Matrix<T, 3, 1> pj1 = pj1_.cast<T>();

    Eigen::Matrix<T, 4, 1> qi2 = qi2_.cast<T>();
    Eigen::Matrix<T, 3, 1> ti2 = ti2_.cast<T>();
    Eigen::Matrix<T, 3, 1> pi2 = pi2_.cast<T>();
    Eigen::Matrix<T, 4, 1> qj2 = qj2_.cast<T>();
    Eigen::Matrix<T, 3, 1> tj2 = tj2_.cast<T>();
    Eigen::Matrix<T, 3, 1> pj2 = pj2_.cast<T>();

    Eigen::Matrix<T, 3, 1> ppi1;
    Eigen::Matrix<T, 3, 1> pppi1;
    ceres::QuaternionRotatePoint(qx, pi1.data(), ppi1.data());
    ppi1(0) += tx[0];
    ppi1(1) += tx[1];
    ppi1(2) += tx[2];
    ceres::QuaternionRotatePoint(qi1.data(), ppi1.data(), pppi1.data());
    pppi1(0) += ti1[0];
    pppi1(1) += ti1[1];
    pppi1(2) += ti1[2];

    Eigen::Matrix<T, 3, 1> ppj1;
    Eigen::Matrix<T, 3, 1> pppj1;
    ceres::QuaternionRotatePoint(qx, pj1.data(), ppj1.data());
    ppj1(0) += tx[0];
    ppj1(1) += tx[1];
    ppj1(2) += tx[2];
    ceres::QuaternionRotatePoint(qj1.data(), ppj1.data(), pppj1.data());
    pppj1(0) += tj1[0];
    pppj1(1) += tj1[1];
    pppj1(2) += tj1[2];

    Eigen::Matrix<T, 3, 1> ppi2;
    Eigen::Matrix<T, 3, 1> pppi2;
    ceres::QuaternionRotatePoint(qy, pi2.data(), ppi2.data());
    ppi2(0) += ty[0];
    ppi2(1) += ty[1];
    ppi2(2) += ty[2];
    ceres::QuaternionRotatePoint(qi2.data(), ppi2.data(), pppi2.data());
    pppi2(0) += ti2[0];
    pppi2(1) += ti2[1];
    pppi2(2) += ti2[2];

    Eigen::Matrix<T, 3, 1> ppj2;
    Eigen::Matrix<T, 3, 1> pppj2;
    ceres::QuaternionRotatePoint(qy, pj2.data(), ppj2.data());
    ppj2(0) += ty[0];
    ppj2(1) += ty[1];
    ppj2(2) += ty[2];
    ceres::QuaternionRotatePoint(qj2.data(), ppj2.data(), pppj2.data());
    pppj2(0) += tj2[0];
    pppj2(1) += tj2[1];
    pppj2(2) += tj2[2];

    residual[0] = pppi1(0) + pppj1(0) + pppi2(0) + pppj2(0) ;
    residual[1] = pppi1(1) + pppj1(1) + pppi2(1) + pppj2(1) ;
    residual[2] = pppi1(2) + pppj1(2) + pppi2(2) + pppj2(2) ;
    return true;
  }

private:
  const Eigen::Vector4d qi1_;
  const Eigen::Vector3d ti1_;
  const Eigen::Vector3d pi1_;
  const Eigen::Vector4d qj1_;
  const Eigen::Vector3d tj1_;
  const Eigen::Vector3d pj1_;

  const Eigen::Vector4d qi2_;
  const Eigen::Vector3d ti2_;
  const Eigen::Vector3d pi2_;
  const Eigen::Vector4d qj2_;
  const Eigen::Vector3d tj2_;
  const Eigen::Vector3d pj2_;
};

class Solver
{
public:
  Solver(const Eigen::Vector4d& qx_init, const Eigen::Vector3d tx_init, 
         const Eigen::Vector4d& qy_init, const Eigen::Vector3d ty_init) 
    : qx_opt_(qx_init), tx_opt_(tx_init),qy_opt_(qy_init), ty_opt_(ty_init)
  {
  }

  auto AddResidualBlock(const Eigen::Vector4d&, const Eigen::Vector3d&, const Eigen::Vector3d&, 
                        const Eigen::Vector4d&, const Eigen::Vector3d&, const Eigen::Vector3d&, 
                        const Eigen::Vector4d&, const Eigen::Vector3d&, const Eigen::Vector3d&, 
                        const Eigen::Vector4d&, const Eigen::Vector3d&, const Eigen::Vector3d&) -> bool;

  auto Solve() -> std::tuple<Eigen::Vector4d, Eigen::Vector3d, Eigen::Vector4d, Eigen::Vector3d>;

  auto Summary() -> ceres::Solver::Summary;

  auto Options() -> ceres::Solver::Options;

private:
  ceres::Problem problem_;
  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;

  bool local_parameterization_is_set_ = true;

  Eigen::Vector4d qx_opt_;
  Eigen::Vector3d tx_opt_;
  Eigen::Vector4d qy_opt_;
  Eigen::Vector3d ty_opt_;
};
}  // namespace eris::hand_eye_calibration2
