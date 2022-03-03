#pragma once

#ifndef RGBD_SLAM_POSE_GRAPH_OPTIMIZER_H
#define RGBD_SLAM_POSE_GRAPH_OPTIMIZER_H

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
    class PoseGraphOptimizer
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<PoseGraphOptimizer> Ptr;

        PoseGraphOptimizer(std::string config_file_path);

        void SetMap(Map::Ptr &map) { map_ = map; }
        void SetCamera(Camera::Ptr &camera) { camera_ = camera; }

        void PoseGraphOptimization(std::vector<std::vector<int>> &loopInfoIdx, std::vector<SE3> &loopInfoRelPose);
        void PoseGraphOptimization2D(std::vector<std::vector<int>> &loopInfoIdx, std::vector<SE3> &loopInfoRelPose);

        ceres::Solver::Options options;
        ceres::Solver::Summary summary;

        Camera::Ptr camera_;
        Map::Ptr map_;
    };
}

#endif