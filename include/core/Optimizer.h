#pragma once

#ifndef RGBD_SLAM_OPTIMIZER_H
#define RGBD_SLAM_OPTIMIZER_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "types/Common.h"
#include "types/Frame.h"
#include "types/Map.h"
#include "types/Camera.h"

#include "factor/ProjectionFactor.h"
#include "factor/BetweenFactor.h"

#include <Eigen/Geometry> 


namespace RGBDSLAM
{
    class Optimizer
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Optimizer> Ptr;

        Optimizer(std::string config_file_path);

        void SetMap(Map::Ptr &map) { map_ = map; }
        void SetCamera(Camera::Ptr &camera) { camera_ = camera; }

        SE3 LocalBundleAdjustment(Frame::Ptr& frame, std::unordered_map<int, int> &frame2param, bool loop = false);
        void PoseGraphOptimization(std::vector<std::vector<int>> &loopInfoIdx, std::vector<SE3> &loopInfoRelPose);

        
        Camera::Ptr camera_;
        Map::Ptr map_;
        int window_size_ = 5;
    };
}

#endif