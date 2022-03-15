#pragma once

#ifndef RGBD_SLAM_FRAME_H
#define RGBD_SLAM_FRAME_H

#include "types/Common.h"
#include "types/Camera.h"
#include "types/Feature.h"
#include "types/MapPoint.h"
#include "DBoW3/DBoW3.h"
#include <opencv2/features2d.hpp>

namespace RGBDSLAM
{
    // forward declare
    // struct MapPoint;
    // struct Feature;

    struct Frame
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Frame> Ptr;

        unsigned long id_ = 0;          // id of this frame
        unsigned long keyframe_id_ = 0; // id of key frame
        bool is_keyframe_ = false;      // is this frame keyframe?
        double time_stamp_;             // time stamp
        SE3 pose_;                      // Tcw Pose
        std::mutex pose_mutex_;         // Pose lock
        cv::Mat rgb_;                   // RGB image
        cv::Mat gray_;                  // Gray image
        cv::Mat depth_;                 // Depth image
        Mat33 K_;                       // Intrinsic
        cv::Mat des_;                   // Descriptor
        // DBoW2::BowVector BoW_vec;       // Bag of words vector
        // DBoW2::FeatureVector Feat_vec;  // Feature vector

        std::string stamp_;

        DBoW3::BowVector BoW_vec;

        std::vector<std::shared_ptr<Feature>> features_;

        // triangle pathces

    public: // data members
        Frame() {}
        Frame(long id, double time_stamp, const SE3 &pose, const cv::Mat &rgb, const cv::Mat &gray, const cv::Mat &depth, const Mat33 &K, const std::string& stamp);

        // set and get pose, thread safe
        SE3 Pose();

        void SetPose(const SE3 &pose);

        // Set up keyframes, allocate and keyframe id
        void SetKeyFrame();
        void SetDescriptor(const cv::Mat &descriptor);

        // factory function
        static std::shared_ptr<Frame> CreateFrame();

        // coordinate transform: world, camera, pixel
        Vec3 world2camera(const Vec3 &p_w);
        Vec3 camera2world(const Vec3 &p_c);
        Vec2 camera2pixel(const Vec3 &p_c);
        Vec3 pixel2camera(const Vec2 &p_p, double depth = 1);
        Vec3 pixel2world(const Vec2 &p_p, double depth = 1);
        Vec2 world2pixel(const Vec3 &p_w);
    };

}

#endif