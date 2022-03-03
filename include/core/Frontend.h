#pragma once
#ifndef RGBD_SLAM_FRONTEND_H
#define RGBD_SLAM_FRONTEND_H

#include "DBoW3/DBoW3.h"

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "factor/ProjectionFactor.h"
#include "factor/OdometryFactor.h"

#include <ctime>

#include "Config.h"

#include "core/Matcher.h"
#include "core/Extractor.h"
#include "core/Estimator.h"

#include "utils/PointCloudUtils.h"
#include "utils/Viewer.h"

#include "types/Feature.h"
#include "types/Common.h"
#include "types/Frame.h"
#include "types/Map.h"

namespace RGBDSLAM
{

    class Backend;
    class Viewer;

    enum class FrontendStatus
    {
        INITING,
        TRACKING_GOOD,
        TRACKING_BAD,
        LOST
    };

    class Frontend
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Frontend> Ptr;

        Frontend(std::string config_file_path);

        bool AddFrame(Frame::Ptr frame);

        void SetMap(Map::Ptr map)
        {
            map_ = map;
            map_->SetWindowSize(window_size_);
            estimator_->SetMap(map_);
        }

        void SetBackend(std::shared_ptr<Backend> backend) { backend_ = backend; }

        void SetViewer(std::shared_ptr<Viewer> viewer)
        {
            viewer_ = viewer;
            viewer_->Initialize();
        }

        void SetGT(std::vector<std::vector<double>> &q, std::vector<std::vector<double>> &t)
        {
            gt_q = q;
            gt_t = t;
        }

        FrontendStatus GetStatus() const { return status_; }

        void SetCamera(Camera::Ptr camera)
        {
            camera_ = camera;
            estimator_->SetCamera(camera);
        }

        bool RGBDInit();

        bool Track();
        bool InsertKeyframe();
        void SetObservationsForKeyFrame();
        void ComputeCurrentBoW();
        void DetectLoop(std::vector<std::vector<int>> &loopInfoIdx, std::vector<SE3> &loopInfoRelPos);

        // data
        FrontendStatus status_ = FrontendStatus::INITING;

        Frame::Ptr current_frame_ = nullptr;
        Camera::Ptr camera_ = nullptr;

        Map::Ptr map_ = nullptr;
        std::shared_ptr<Backend> backend_ = nullptr;
        std::shared_ptr<Viewer> viewer_ = nullptr;

        SE3 relative_motion_; // relative transform fron former frame to current frame ( inv(T_(t-1)) * T_(t) )

        int tracking_inliers_ = 0; // inliers, used for testing new keyframes

        // params

        int num_features_tracking_threshold = 50;

        Matcher::Ptr matcher_;
        Extractor::Ptr extractor_;
        Estimator::Ptr estimator_;

        int window_size_ = 5;

        std::vector<Frame::Ptr> frames_;

        DBoW3::Vocabulary vocab_;
        DBoW3::Database db_;

        bool loop_detect_;
        double score_threshold_;

        // gt
        std::vector<std::vector<double>> gt_q;
        std::vector<std::vector<double>> gt_t;
    };

}

#endif