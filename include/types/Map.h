#pragma once
#ifndef RGBD_SLAM_MAP_H
#define RGBD_SLAM_MAP_H

#include "types/Frame.h"
#include "types/MapPoint.h"
#include "Config.h"

namespace RGBDSLAM
{
    typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType; // id and class (hash)
    typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;    // id and class (hash)

    class Map
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Map> Ptr;

        Map() { min_dis_th = Config::Get<double>("keyframe_min_dist_threshold"); }

        LandmarksType GetAllMapPoints();
        KeyframesType GetAllKeyFrames();
        LandmarksType GetActiveMapPoints();
        KeyframesType GetActiveKeyFrames();
        void SetWindowSize(int window_size);

        void InsertKeyFrame(Frame::Ptr frame);
        void InsertMapPoint(MapPoint::Ptr map_point);
        void UpdateActiveMapPoint(int current_frame_id, int window_size);
        void RemoveOldKeyframe();

        std::mutex data_mutex_;
        LandmarksType landmarks_;        // all landmarks
        LandmarksType active_landmarks_; // active landmarks
        KeyframesType keyframes_;        // all keyframes
        KeyframesType active_keyframes_; // all keyframes
        Frame::Ptr current_keyframe_;    // current keyframe
        Frame::Ptr current_frame_ = nullptr;

        std::vector<int> keyframe_id_;

        // settings --> add to configuration file
        int num_active_keyframes_ = 10;
        double min_dis_th = 0.1;
    };
}

#endif
