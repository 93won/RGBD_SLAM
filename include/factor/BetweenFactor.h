#ifndef RGBD_SLAM_BETWEEN_FACTOR_H
#define RGBD_SLAM_BETWEEN_FACTOR_H

#include <Eigen/Core>
#include "types/Common.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace RGBDSLAM
{

    // Standard bundle adjustment cost function for variable
    // camera pose and calibration and point parameters.

    class BetweenFactor
    {
    public:
        explicit BetweenFactor(const Vec4 &qvec_odom, const Vec3 &tvec_odom, const double w) : qvec_odom_(qvec_odom),
                                                                                               tvec_odom_(tvec_odom),
                                                                                               w_(w) {}

        static ceres::CostFunction *Create(const Vec4 &qvec_odom, const Vec3 &tvec_odom, const double w)
        {
            return (new ceres::AutoDiffCostFunction<BetweenFactor, 7, 4, 3, 4, 3>(new BetweenFactor(qvec_odom, tvec_odom, w)));
        }

        template <typename T>
        bool operator()(const T *const qvec_i, const T *const tvec_i, const T *const qvec_j, const T *const tvec_j, T *residuals) const
        {

            // Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(tvec_i);
            // Eigen::Map<const Eigen::Quaternion<T>> q_a(qvec_i);

            // Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_b(tvec_j);
            // Eigen::Map<const Eigen::Quaternion<T>> q_b(qvec_j);

            // T q_ab_hat_vec[4];
            // q_ab_hat_vec[0] = T(qvec_odom_[0]);
            // q_ab_hat_vec[1] = T(qvec_odom_[1]);
            // q_ab_hat_vec[2] = T(qvec_odom_[2]);
            // q_ab_hat_vec[3] = T(qvec_odom_[3]);

            // T t_ab_hat_vec[3];
            // t_ab_hat_vec[0] = T(tvec_odom_[0]);
            // t_ab_hat_vec[1] = T(tvec_odom_[1]);
            // t_ab_hat_vec[2] = T(tvec_odom_[2]);

            // Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_ab_hat(t_ab_hat_vec);
            // Eigen::Map<const Eigen::Quaternion<T>> q_ab_hat(q_ab_hat_vec);

            // // Compute the relative transformation between the two frames.
            // Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
            // Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;

            // // Represent the displacement between the two frames in the A frame.
            // Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);
            // // Compute the error between the two orientation estimates.
            // Eigen::Quaternion<T> delta_q = q_ab_hat.template cast<T>() * q_ab_estimated.conjugate();

            // // Compute the residuals.
            // // [ position         ]   [ delta_p          ]
            // // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
            // Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
            // residuals.template block<3, 1>(0, 0) = p_ab_estimated - p_ab_hat.template cast<T>();
            // residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();

            // Eigen::MatrixXd sqrt_information_ = Eigen::MatrixXd::Identity(6, 6);

            // residuals.applyOnTheLeft(sqrt_information_.template cast<T>());  // information matrix

            // Rotate and translate.
            Eigen::Matrix<T, 3, 3> R_i, R_j, R_ij, R_o;
            Eigen::Matrix<T, 4, 4> T_i, T_j, T_ij;

            // Eigen::Quaternion<T> q_i(qvec_i[0], qvec_i[1], qvec_i[2], qvec_i[3]);
            // Eigen::Quaternion<T> q_j(qvec_j[0], qvec_j[1], qvec_j[2], qvec_j[3]);

            // R_i = q_i.toRotationMatrix();
            // R_j = q_j.toRotationMatrix();

            // Q to R --> Edit ceres function like bellow:
            R_i(0, 0) = T(2) * (qvec_i[0] * qvec_i[0] + qvec_i[1] * qvec_i[1]) - T(1);
            R_i(0, 1) = T(2) * (qvec_i[1] * qvec_i[2] - qvec_i[0] * qvec_i[3]);
            R_i(0, 2) = T(2) * (qvec_i[1] * qvec_i[3] + qvec_i[0] * qvec_i[2]);
            R_i(1, 0) = T(2) * (qvec_i[1] * qvec_i[2] + qvec_i[0] * qvec_i[3]);
            R_i(1, 1) = T(2) * (qvec_i[0] * qvec_i[0] + qvec_i[2] * qvec_i[2]) - T(1);
            R_i(1, 2) = T(2) * (qvec_i[2] * qvec_i[3] - qvec_i[0] * qvec_i[1]);
            R_i(2, 0) = T(2) * (qvec_i[1] * qvec_i[3] - qvec_i[0] * qvec_i[2]);
            R_i(2, 1) = T(2) * (qvec_i[2] * qvec_i[3] + qvec_i[0] * qvec_i[1]);
            R_i(2, 2) = T(2) * (qvec_i[0] * qvec_i[0] + qvec_i[3] * qvec_i[3]) - T(1);

            R_j(0, 0) = T(2) * (qvec_j[0] * qvec_j[0] + qvec_j[1] * qvec_j[1]) - T(1);
            R_j(0, 1) = T(2) * (qvec_j[1] * qvec_j[2] - qvec_j[0] * qvec_j[3]);
            R_j(0, 2) = T(2) * (qvec_j[1] * qvec_j[3] + qvec_j[0] * qvec_j[2]);
            R_j(1, 0) = T(2) * (qvec_j[1] * qvec_j[2] + qvec_j[0] * qvec_j[3]);
            R_j(1, 1) = T(2) * (qvec_j[0] * qvec_j[0] + qvec_j[2] * qvec_j[2]) - T(1);
            R_j(1, 2) = T(2) * (qvec_j[2] * qvec_j[3] - qvec_j[0] * qvec_j[1]);
            R_j(2, 0) = T(2) * (qvec_j[1] * qvec_j[3] - qvec_j[0] * qvec_j[2]);
            R_j(2, 1) = T(2) * (qvec_j[2] * qvec_j[3] + qvec_j[0] * qvec_j[1]);
            R_j(2, 2) = T(2) * (qvec_j[0] * qvec_j[0] + qvec_j[3] * qvec_j[3]) - T(1);

            T_i.topLeftCorner(3, 3) = R_i;
            T_j.topLeftCorner(3, 3) = R_j;

            T_i(3, 3) = T(1);
            T_j(3, 3) = T(1);

            T_i(0, 3) = tvec_i[0];
            T_i(1, 3) = tvec_i[1];
            T_i(2, 3) = tvec_i[2];

            T_j(0, 3) = tvec_j[0];
            T_j(1, 3) = tvec_j[1];
            T_j(2, 3) = tvec_j[2];

            T_ij = T_j * T_i.inverse();
            R_ij = T_ij.topLeftCorner(3, 3);

            T t_ij[3];
            t_ij[0] = T_ij(0, 3);
            t_ij[1] = T_ij(1, 3);
            t_ij[2] = T_ij(2, 3);

            T q_ij[4];
            ceres::RotationMatrixToQuaternion(R_ij.data(), q_ij);

            residuals[0] = q_ij[0] - T(qvec_odom_[0]) * T(w_);
            residuals[1] = q_ij[1] - T(qvec_odom_[1]) * T(w_);
            residuals[2] = q_ij[2] - T(qvec_odom_[2]) * T(w_);
            residuals[3] = q_ij[3] - T(qvec_odom_[3]) * T(w_);
            residuals[4] = t_ij[0] - T(tvec_odom_[0]) * T(w_);
            residuals[5] = t_ij[1] - T(tvec_odom_[1]) * T(w_);
            residuals[6] = t_ij[2] - T(tvec_odom_[2]) * T(w_);

            return true;
        }

    private:
        const Vec4 qvec_odom_;
        const Vec3 tvec_odom_;
        const double w_;
    };
}
#endif