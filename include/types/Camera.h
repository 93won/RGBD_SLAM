#pragma once

#ifndef RGBD_SLAM_CAMERA_H
#define RGBD_SLAM_CAMERA_H

#include "types/Common.h"
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <ceres/ceres.h>

namespace RGBDSLAM
{

    // Pinhole Camera
    class Camera
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Camera> Ptr;

        double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;

        Camera();

        Camera(double fx, double fy, double cx, double cy);


        // return intrinsic matrix
        Mat33 K();
       
    };

}
#endif