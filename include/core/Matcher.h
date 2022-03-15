#pragma once

#ifndef RGBD_SLAM_MATCHER_H
#define RGBD_SLAM_MATCHER_H

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "types/Frame.h"
#include "types/Map.h"
#include "Config.h"

namespace RGBDSLAM
{
    class Matcher
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Matcher> Ptr;

        Matcher(std::string config_file_path);

        int MatchTwoFrames(Frame::Ptr &frame_i, Frame::Ptr &frame_j, Map::Ptr &map_);
        int MatchTwoFrames(Frame::Ptr &frame_i, Frame::Ptr &frame_j, Map::Ptr &map_, std::unordered_map<int, int>& frame2param);

        cv::FlannBasedMatcher matcher_;
        double knn_ratio_ = 0.7;
        int match_threshold_tracking_ = 30;
    };
}

#endif