#pragma once
#ifndef RGBD_SLAM_TRACKER_H
#define RGBD_SLAM_TRACKER_H

#include "DBoW3/DBoW3.h"

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "factor/ProjectionFactor.h"
#include "factor/BetweenFactor.h"

#include <ctime>

#include "Config.h"

#include "core/Matcher.h"
#include "core/Extractor.h"
#include "core/Optimizer.h"

#include "utils/Viewer.h"

#include "types/Feature.h"
#include "types/Common.h"
#include "types/Frame.h"
#include "types/Map.h"

namespace RGBDSLAM
{

    class Backend;
    class Viewer;

    enum class TrackerStatus
    {
        INITING,
        Tracker_GOOD,
        Tracker_BAD,
        LOST
    };

    class Tracker
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Tracker> Ptr;

        Tracker(std::string config_file_path);

        bool AddFrame(Frame::Ptr frame);

        void SetMap(Map::Ptr map)
        {
            map_ = map;
            map_->SetWindowSize(window_size_);
            optimizer_->SetMap(map_);
            // pose_graph_optimizer_->SetMap(map_);
        }

        void SetViewer(std::shared_ptr<Viewer> viewer)
        {
            viewer_ = viewer;
            viewer_->Initialize();
        }

        void SetGT(std::vector<Vec4> &q, std::vector<Vec3> &t)
        {
            gt_q = q;
            gt_t = t;
        }

        TrackerStatus GetStatus() const { return status_; }

        void SetCamera(Camera::Ptr camera)
        {
            camera_ = camera;
            optimizer_->SetCamera(camera);
            // pose_graph_optimizer_->SetCamera(camera);
        }

        bool RGBDInit();

        bool Track();
        bool InsertKeyframe();
        void SetObservationsForKeyFrame();
        void ComputeCurrentBoW();
        void DetectLoop(std::vector<std::vector<int>> &loopInfoIdx, std::vector<SE3> &loopInfoRelPos);

        // data
        TrackerStatus status_ = TrackerStatus::INITING;

        Frame::Ptr current_frame_ = nullptr;
        Camera::Ptr camera_ = nullptr;

        Map::Ptr map_ = nullptr;
        std::shared_ptr<Backend> backend_ = nullptr;
        std::shared_ptr<Viewer> viewer_ = nullptr;

        SE3 relative_motion_; // relative transform fron former frame to current frame ( inv(T_(t-1)) * T_(t) )

        int Tracker_inliers_ = 0; // inliers, used for testing new keyframes

        // params
        int num_features_tracking_threshold_ = 50;
        int loop_frame_th_ = 100;
        double rel_pose_thresh_ = 0.5;

        Matcher::Ptr matcher_;
        Extractor::Ptr extractor_;
        Optimizer::Ptr optimizer_;
        // PoseGraphOptimizer::Ptr pose_graph_optimizer_;

        int window_size_ = 5;

        std::vector<Frame::Ptr> frames_;
        std::vector<Frame::Ptr> keyframes_order_;

        DBoW3::Vocabulary vocab_;
        DBoW3::Database db_;

        bool loop_detect_;
        double score_threshold_;

        // gt
        std::vector<Vec4> gt_q;
        std::vector<Vec3> gt_t;
    };

}

#endif