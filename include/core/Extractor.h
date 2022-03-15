#pragma once

#ifndef RGBD_SLAM_EXTRACTOR_H
#define RGBD_SLAM_EXTRACTOR_H

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "types/Frame.h"
#include "Config.h"

namespace RGBDSLAM
{
    class Extractor
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Extractor> Ptr;

        Extractor(std::string config_file_path);

        int DetectFeatures(Frame::Ptr &frame);

        double depth_min_ = 0.5;
        double depth_max_ = 10.0;
        cv::Ptr<cv::FeatureDetector> detector_;       // feature detector in opencv
        cv::Ptr<cv::DescriptorExtractor> descriptor_; // feature descriptor extractor in opencv
    };
}

#endif