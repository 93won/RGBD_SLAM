#pragma once

#ifndef RGBD_SLAM_ESTIMATOR_H
#define RGBD_SLAM_ESTIMATOR_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "types/Common.h"
#include "types/Frame.h"
#include "types/Map.h"
#include "types/Camera.h"

#include "factor/ProjectionFactor.h"
#include "factor/OdometryFactor.h"

namespace RGBDSLAM
{
    class Estimator
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Estimator> Ptr;

        Estimator(std::string config_file_path);

        void SetMap(Map::Ptr &map) { map_ = map; }
        void SetCamera(Camera::Ptr &camera) { camera_ = camera; }

        void LocalBundleAdjustment(Frame::Ptr& frame);
        SE3 EstimateRelPose(std::unordered_map<int, int> &frame2param);
        void PoseGraphOptimization(std::vector<std::vector<int>> &loopInfoIdx, std::vector<SE3> &loopInfoRelPose);

        SE3 EstimateRelPose2D(std::unordered_map<int, int> &frame2param);
        void LocalBundleAdjustment2D(Frame::Ptr& frame);
        void PoseGraphOptimization2D(std::vector<std::vector<int>> &loopInfoIdx, std::vector<SE3> &loopInfoRelPose);

        Camera::Ptr camera_;
        Map::Ptr map_;
        int window_size_ = 5;
    };
}

#endif