#ifndef RGBD_SLAM_ODOMETRY_FACTOR_H
#define RGBD_SLAM_ODOMETRY_FACTOR_H

#include <Eigen/Core>
#include "types/Common.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace RGBDSLAM
{

    // Standard bundle adjustment cost function for variable
    // camera pose and calibration and point parameters.

    class OdometryFactor
    {
    public:
        explicit OdometryFactor(const Vec4 &qvec_odom, const Vec3 &tvec_odom) : qvec_odom_(qvec_odom),
                                                                                tvec_odom_(tvec_odom) {}

        static ceres::CostFunction *Create(const Vec4 &qvec_odom, const Vec3 &tvec_odom)
        {
            return (new ceres::AutoDiffCostFunction<OdometryFactor, 7, 4, 3, 4, 3>(new OdometryFactor(qvec_odom, tvec_odom)));
        }

        template <typename T>
        bool operator()(const T *const qvec_i, const T *const tvec_i, const T *const qvec_j, const T *const tvec_j, T *residuals) const
        {

            // Rotate and translate.
            Eigen::Matrix<T, 3, 3> R_i, R_j, R_ij, R_o;
            Eigen::Matrix<T, 4, 4> T_i, T_j, T_ij;

            // Q to R --> Edit ceres function like bellow:
            R_i(0, 0) = T(2)*(qvec_i[0]*qvec_i[0] + qvec_i[1]*qvec_i[1]) - T(1);
            R_i(0, 1) = T(2)*(qvec_i[1]*qvec_i[2] - qvec_i[0]*qvec_i[3]);
            R_i(0, 2) = T(2)*(qvec_i[1]*qvec_i[3] + qvec_i[0]*qvec_i[2]);
            R_i(1, 0) = T(2)*(qvec_i[1]*qvec_i[2] + qvec_i[0]*qvec_i[3]);
            R_i(1, 1) = T(2)*(qvec_i[0]*qvec_i[0] + qvec_i[2]*qvec_i[2]) - T(1);
            R_i(1, 2) = T(2)*(qvec_i[2]*qvec_i[3] - qvec_i[0]*qvec_i[1]);
            R_i(2, 0) = T(2)*(qvec_i[1]*qvec_i[3] - qvec_i[0]*qvec_i[2]);
            R_i(2, 1) = T(2)*(qvec_i[2]*qvec_i[3] + qvec_i[0]*qvec_i[1]);
            R_i(2, 2) = T(2)*(qvec_i[0]*qvec_i[0] + qvec_i[3]*qvec_i[3]) - T(1);

            R_j(0, 0) = T(2)*(qvec_j[0]*qvec_j[0] + qvec_j[1]*qvec_j[1]) - T(1);
            R_j(0, 1) = T(2)*(qvec_j[1]*qvec_j[2] - qvec_j[0]*qvec_j[3]);
            R_j(0, 2) = T(2)*(qvec_j[1]*qvec_j[3] + qvec_j[0]*qvec_j[2]);
            R_j(1, 0) = T(2)*(qvec_j[1]*qvec_j[2] + qvec_j[0]*qvec_j[3]);
            R_j(1, 1) = T(2)*(qvec_j[0]*qvec_j[0] + qvec_j[2]*qvec_j[2]) - T(1);
            R_j(1, 2) = T(2)*(qvec_j[2]*qvec_j[3] - qvec_j[0]*qvec_j[1]);
            R_j(2, 0) = T(2)*(qvec_j[1]*qvec_j[3] - qvec_j[0]*qvec_j[2]);
            R_j(2, 1) = T(2)*(qvec_j[2]*qvec_j[3] + qvec_j[0]*qvec_j[1]);
            R_j(2, 2) = T(2)*(qvec_j[0]*qvec_j[0] + qvec_j[3]*qvec_j[3]) - T(1);

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

            residuals[0] = q_ij[0] - T(qvec_odom_[0]);
            residuals[1] = q_ij[1] - T(qvec_odom_[1]);
            residuals[2] = q_ij[2] - T(qvec_odom_[2]);
            residuals[3] = q_ij[3] - T(qvec_odom_[3]);
            residuals[4] = t_ij[0] - T(tvec_odom_[0]);
            residuals[5] = t_ij[1] - T(tvec_odom_[1]);
            residuals[6] = t_ij[2] - T(tvec_odom_[2]);

            // LOG(INFO) << "Residual Check: "
            //           << T(residuals[0]) << " "
            //           << T(residuals[1]) << " "
            //           << T(residuals[2]) << " "
            //           << T(residuals[3]) << " "
            //           << T(residuals[4]) << " "
            //           << T(residuals[5]) << " "
            //           << T(residuals[6]) << " ";

            return true;
        }

    private:
        const Vec4 qvec_odom_;
        const Vec3 tvec_odom_;
    };

    class LoopClosureFactor
    {
    public:
        explicit LoopClosureFactor(const Vec4 &qvec_odom, const Vec3 &tvec_odom) : qvec_odom_(qvec_odom),
                                                                                tvec_odom_(tvec_odom) {}

        static ceres::CostFunction *Create(const Vec4 &qvec_odom, const Vec3 &tvec_odom)
        {
            return (new ceres::AutoDiffCostFunction<LoopClosureFactor, 7, 4, 3, 4, 3>(new LoopClosureFactor(qvec_odom, tvec_odom)));
        }

        template <typename T>
        bool operator()(const T *const qvec_i, const T *const tvec_i, const T *const qvec_j, const T *const tvec_j, T *residuals) const
        {

            // Rotate and translate.
            Eigen::Matrix<T, 3, 3> R_i, R_j, R_ij, R_o;
            Eigen::Matrix<T, 4, 4> T_i, T_j, T_ij;

            // Q to R --> Edit ceres function like bellow:
            R_i(0, 0) = T(2)*(qvec_i[0]*qvec_i[0] + qvec_i[1]*qvec_i[1]) - T(1);
            R_i(0, 1) = T(2)*(qvec_i[1]*qvec_i[2] - qvec_i[0]*qvec_i[3]);
            R_i(0, 2) = T(2)*(qvec_i[1]*qvec_i[3] + qvec_i[0]*qvec_i[2]);
            R_i(1, 0) = T(2)*(qvec_i[1]*qvec_i[2] + qvec_i[0]*qvec_i[3]);
            R_i(1, 1) = T(2)*(qvec_i[0]*qvec_i[0] + qvec_i[2]*qvec_i[2]) - T(1);
            R_i(1, 2) = T(2)*(qvec_i[2]*qvec_i[3] - qvec_i[0]*qvec_i[1]);
            R_i(2, 0) = T(2)*(qvec_i[1]*qvec_i[3] - qvec_i[0]*qvec_i[2]);
            R_i(2, 1) = T(2)*(qvec_i[2]*qvec_i[3] + qvec_i[0]*qvec_i[1]);
            R_i(2, 2) = T(2)*(qvec_i[0]*qvec_i[0] + qvec_i[3]*qvec_i[3]) - T(1);

            R_j(0, 0) = T(2)*(qvec_j[0]*qvec_j[0] + qvec_j[1]*qvec_j[1]) - T(1);
            R_j(0, 1) = T(2)*(qvec_j[1]*qvec_j[2] - qvec_j[0]*qvec_j[3]);
            R_j(0, 2) = T(2)*(qvec_j[1]*qvec_j[3] + qvec_j[0]*qvec_j[2]);
            R_j(1, 0) = T(2)*(qvec_j[1]*qvec_j[2] + qvec_j[0]*qvec_j[3]);
            R_j(1, 1) = T(2)*(qvec_j[0]*qvec_j[0] + qvec_j[2]*qvec_j[2]) - T(1);
            R_j(1, 2) = T(2)*(qvec_j[2]*qvec_j[3] - qvec_j[0]*qvec_j[1]);
            R_j(2, 0) = T(2)*(qvec_j[1]*qvec_j[3] - qvec_j[0]*qvec_j[2]);
            R_j(2, 1) = T(2)*(qvec_j[2]*qvec_j[3] + qvec_j[0]*qvec_j[1]);
            R_j(2, 2) = T(2)*(qvec_j[0]*qvec_j[0] + qvec_j[3]*qvec_j[3]) - T(1);

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

            T w = T(10.0);
            residuals[0] = (q_ij[0] - T(qvec_odom_[0]))*w;
            residuals[1] = (q_ij[1] - T(qvec_odom_[1]))*w;
            residuals[2] = (q_ij[2] - T(qvec_odom_[2]))*w;
            residuals[3] = (q_ij[3] - T(qvec_odom_[3]))*w;
            residuals[4] = (t_ij[0] - T(tvec_odom_[0]))*w;
            residuals[5] = (t_ij[1] - T(tvec_odom_[1]))*w;
            residuals[6] = (t_ij[2] - T(tvec_odom_[2]))*w;

            // LOG(INFO) << "Residual Check: "
            //           << T(residuals[0]) << " "
            //           << T(residuals[1]) << " "
            //           << T(residuals[2]) << " "
            //           << T(residuals[3]) << " "
            //           << T(residuals[4]) << " "
            //           << T(residuals[5]) << " "
            //           << T(residuals[6]) << " ";

            return true;
        }

    private:
        const Vec4 qvec_odom_;
        const Vec3 tvec_odom_;
    };

    class OdometryFactorPlane
    {
    public:
        explicit OdometryFactorPlane(const Vec4 &qvec_odom, const Vec2 &tvec_odom) : qvec_odom_(qvec_odom),
                                                                                     tvec_odom_(tvec_odom) {}

        static ceres::CostFunction *Create(const Vec4 &qvec_odom, const Vec2 &tvec_odom)
        {
            return (new ceres::AutoDiffCostFunction<OdometryFactorPlane, 6, 4, 2, 4, 2>(new OdometryFactorPlane(qvec_odom, tvec_odom)));
        }

        template <typename T>
        bool operator()(const T *const qvec_i, const T *const tvec_i, const T *const qvec_j, const T *const tvec_j, T *residuals) const
        {

            // Rotate and translate.
            Eigen::Matrix<T, 3, 3> R_i, R_j, R_ij, R_o;
            Eigen::Matrix<T, 4, 4> T_i, T_j, T_ij;
            // qvec_i = [w, y] (1, 3 ==> 0)
            // teve_i = [x, z]

            // // Q to R --> Edit ceres function like bellow:
            // R_i(0, 0) = T(2) * (qvec_i[0] * qvec_i[0]) - T(1);
            // R_i(0, 1) = T(0);
            // R_i(0, 2) = T(2) * (qvec_i[0] * qvec_i[1]);
            // R_i(1, 0) = T(0);
            // R_i(1, 1) = T(2) * (qvec_i[0] * qvec_i[0] + qvec_i[1] * qvec_i[1]) - T(1);
            // R_i(1, 2) = T(0);
            // R_i(2, 0) = T(2) * (-qvec_i[0] * qvec_i[1]);
            // R_i(2, 1) = T(0);
            // R_i(2, 2) = T(2) * (qvec_i[0] * qvec_i[0]) - T(1);

            // R_j(0, 0) = T(2) * (qvec_j[0] * qvec_j[0]) - T(1);
            // R_j(0, 1) = T(0);
            // R_j(0, 2) = T(2) * (qvec_j[0] * qvec_j[1]);
            // R_j(1, 0) = T(0);
            // R_j(1, 1) = T(2) * (qvec_j[0] * qvec_j[0] + qvec_j[1] * qvec_j[1]) - T(1);
            // R_j(1, 2) = T(0);
            // R_j(2, 0) = T(2) * (-qvec_j[0] * qvec_j[1]);
            // R_j(2, 1) = T(0);
            // R_j(2, 2) = T(2) * (qvec_j[0] * qvec_j[0]) - T(1);

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
            T_i(1, 3) = T(0);
            T_i(2, 3) = tvec_i[1];

            T_j(0, 3) = tvec_j[0];
            T_j(1, 3) = T(0);
            T_j(2, 3) = tvec_j[1];
            T_ij = T_j * T_i.inverse();
            R_ij = T_ij.topLeftCorner(3, 3);
            T t_ij[3];
            t_ij[0] = T_ij(0, 3);
            t_ij[1] = T_ij(1, 3);
            t_ij[2] = T_ij(2, 3);
            T q_ij[4];
            ceres::RotationMatrixToQuaternion(R_ij.data(), q_ij);
            residuals[0] = q_ij[0] - T(qvec_odom_[0]); // w
            residuals[1] = q_ij[1] - T(qvec_odom_[1]); // y
            residuals[2] = q_ij[2] - T(qvec_odom_[2]); // w
            residuals[3] = q_ij[3] - T(qvec_odom_[3]); // y
            residuals[4] = t_ij[0] - T(tvec_odom_[0]); // x
            residuals[5] = t_ij[2] - T(tvec_odom_[1]); // z
            return true;
        }

    private:
        const Vec4 qvec_odom_;
        const Vec2 tvec_odom_;
    };

    class LoopClosureFactorPlane
    {
    public:
        explicit LoopClosureFactorPlane(const Vec4 &qvec_odom, const Vec2 &tvec_odom) : qvec_odom_(qvec_odom),
                                                                                     tvec_odom_(tvec_odom) {}

        static ceres::CostFunction *Create(const Vec4 &qvec_odom, const Vec2 &tvec_odom)
        {
            return (new ceres::AutoDiffCostFunction<LoopClosureFactorPlane, 6, 4, 2, 4, 2>(new LoopClosureFactorPlane(qvec_odom, tvec_odom)));
        }

        template <typename T>
        bool operator()(const T *const qvec_i, const T *const tvec_i, const T *const qvec_j, const T *const tvec_j, T *residuals) const
        {

            // Rotate and translate.
            Eigen::Matrix<T, 3, 3> R_i, R_j, R_ij, R_o;
            Eigen::Matrix<T, 4, 4> T_i, T_j, T_ij;
            // qvec_i = [w, y] (1, 3 ==> 0)
            // teve_i = [x, z]

            // // Q to R --> Edit ceres function like bellow:
            // R_i(0, 0) = T(2) * (qvec_i[0] * qvec_i[0]) - T(1);
            // R_i(0, 1) = T(0);
            // R_i(0, 2) = T(2) * (qvec_i[0] * qvec_i[1]);
            // R_i(1, 0) = T(0);
            // R_i(1, 1) = T(2) * (qvec_i[0] * qvec_i[0] + qvec_i[1] * qvec_i[1]) - T(1);
            // R_i(1, 2) = T(0);
            // R_i(2, 0) = T(2) * (-qvec_i[0] * qvec_i[1]);
            // R_i(2, 1) = T(0);
            // R_i(2, 2) = T(2) * (qvec_i[0] * qvec_i[0]) - T(1);

            // R_j(0, 0) = T(2) * (qvec_j[0] * qvec_j[0]) - T(1);
            // R_j(0, 1) = T(0);
            // R_j(0, 2) = T(2) * (qvec_j[0] * qvec_j[1]);
            // R_j(1, 0) = T(0);
            // R_j(1, 1) = T(2) * (qvec_j[0] * qvec_j[0] + qvec_j[1] * qvec_j[1]) - T(1);
            // R_j(1, 2) = T(0);
            // R_j(2, 0) = T(2) * (-qvec_j[0] * qvec_j[1]);
            // R_j(2, 1) = T(0);
            // R_j(2, 2) = T(2) * (qvec_j[0] * qvec_j[0]) - T(1);

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
            T_i(1, 3) = T(0);
            T_i(2, 3) = tvec_i[1];

            T_j(0, 3) = tvec_j[0];
            T_j(1, 3) = T(0);
            T_j(2, 3) = tvec_j[1];
            T_ij = T_j * T_i.inverse();
            R_ij = T_ij.topLeftCorner(3, 3);
            T t_ij[3];
            t_ij[0] = T_ij(0, 3);
            t_ij[1] = T_ij(1, 3);
            t_ij[2] = T_ij(2, 3);
            T q_ij[4];

            T w = T(1.0);

            ceres::RotationMatrixToQuaternion(R_ij.data(), q_ij);
            residuals[0] = (q_ij[0] - T(qvec_odom_[0]))*w; // w
            residuals[1] = (q_ij[1] - T(qvec_odom_[1]))*w; // y
            residuals[2] = (q_ij[2] - T(qvec_odom_[2]))*w; // w
            residuals[3] = (q_ij[3] - T(qvec_odom_[3]))*w; // y
            residuals[4] = (t_ij[0] - T(tvec_odom_[0]))*w; // x
            residuals[5] = (t_ij[2] - T(tvec_odom_[1]))*w; // z
            return true;
        }

    private:
        const Vec4 qvec_odom_;
        const Vec2 tvec_odom_;
    };

    
}
#endif