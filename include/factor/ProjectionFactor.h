#ifndef RGBD_SLAM_REPROJECTION_FACTOR_H
#define RGBD_SLAM_REPROJECTION_FACTOR_H

#include <Eigen/Core>
#include "types/Common.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace RGBDSLAM
{

    // Standard bundle adjustment cost function for variable
    // camera pose and calibration and point parameters.

    class ProjectionFactor
    {
    public:
        explicit ProjectionFactor(const Vec2 &point2D, const Mat33 &intrinsic) : observed_x_(point2D(0)),
                                                                                 observed_y_(point2D(1)),
                                                                                 intrinsic_(intrinsic) {}

        static ceres::CostFunction *Create(const Vec2 &point2D, const Mat33 &intrinsic)
        {
            return (new ceres::AutoDiffCostFunction<ProjectionFactor, 2, 4, 3, 3>(
                new ProjectionFactor(point2D, intrinsic)));
        }

        template <typename T>
        bool operator()(const T *const qvec, const T *const tvec, const T *const point3D, T *residuals) const
        {
            // Rotate and translate.
            T projection[3];

            //// R * P + t / w x y z
            ceres::QuaternionRotatePoint(qvec, point3D, projection);
            projection[0] += tvec[0];
            projection[1] += tvec[1];
            projection[2] += tvec[2];

            // Project to image plane.
            projection[0] /= projection[2];
            projection[1] /= projection[2];

            // Distort and transform to pixel space.
            // World To Image

            T fx = T(intrinsic_(0, 0));
            T fy = T(intrinsic_(1, 1));
            T cx = T(intrinsic_(0, 2));
            T cy = T(intrinsic_(1, 2));

            // No distortion
            residuals[0] = (fx * projection[0] + cx) - T(observed_x_);
            residuals[1] = (fy * projection[1] + cy) - T(observed_y_);
            // residuals[2] = qvec[1]*T(10000.0);
            // residuals[3] = qvec[3]*T(10000.0);
            // residuals[4] = tvec[1]*T(10000.0);

            // std::cerr<<"Check Residual: "<<residuals[0]<<" / "<<residuals[1]<<std::endl;

            return true;
        }

    private:
        const double observed_x_;
        const double observed_y_;
        const Mat33 intrinsic_;
    };
}
#endif