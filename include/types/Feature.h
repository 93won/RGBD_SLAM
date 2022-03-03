#pragma once
#ifndef RGBD_SLAM_FEATURE_H
#define RGBD_SLAM_FEATURE_H

#include <memory>
#include <opencv2/features2d.hpp>
#include "types/Common.h"

namespace RGBDSLAM
{
    struct Frame;
    struct MapPoint;

    struct Feature
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        typedef std::shared_ptr<Feature> Ptr;
        std::weak_ptr<Frame> frame_;
        cv::KeyPoint position_;
        std::vector<double> rgb_;
        double depth_;
        std::weak_ptr<MapPoint> map_point_;
        bool is_outlier_ = false;

        Feature() {}
        Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp, double depth);
        Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp, double depth, std::vector<double> rgb);

    };

}

#endif
