#include "core/Tracker.h"

namespace RGBDSLAM
{

    Tracker::Tracker(std::string config_file_path)
    {
        if (!Config::SetParameterFile(config_file_path))
            LOG(INFO) << "No configuration file loaded.";

        matcher_ = Matcher::Ptr(new Matcher(config_file_path)); // cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        extractor_ = Extractor::Ptr(new Extractor(config_file_path));
        optimizer_ = Optimizer::Ptr(new Optimizer(config_file_path));
        // pose_graph_optimizer_ = PoseGraphOptimizer::Ptr(new PoseGraphOptimizer(config_file_path));

        window_size_ = Config::Get<int>("window_size");
        num_features_tracking_threshold_ = Config::Get<int>("num_features_tracking_threshold");

        loop_detect_ = (bool)(Config::Get<int>("loop_detect"));
        score_threshold_ = Config::Get<double>("score_threshold");
        loop_frame_th_ = Config::Get<int>("loop_frame_th");
        rel_pose_thresh_ = Config::Get<double>("rel_pose_thresh");

        std::string voc_dir = Config::Get<std::string>("voc_dir");
        DBoW3::Vocabulary vocab(voc_dir);
        vocab_ = vocab;
        DBoW3::Database db(vocab_, false, 0);
        db_ = db;
    }

    bool Tracker::AddFrame(RGBDSLAM::Frame::Ptr frame)
    {
        current_frame_ = frame;
        frames_.emplace_back(current_frame_);

        switch (status_)
        {
        case TrackerStatus::INITING:
            RGBDInit();
            break;
        case TrackerStatus::Tracker_GOOD:
        case TrackerStatus::Tracker_BAD:
            Track();
            break;
        }

        return true;
    }

    bool Tracker::RGBDInit()
    {
        int num_features = extractor_->DetectFeatures(current_frame_);
        status_ = TrackerStatus::Tracker_GOOD;
        InsertKeyframe();
        return true;
    }

    bool Tracker::Track()
    {

        std::unordered_map<int, int> frame2param;
        frame2param.insert(std::make_pair(current_frame_->id_, 0));
        frame2param.insert(std::make_pair(map_->current_keyframe_->id_, 1));
        int num_features = extractor_->DetectFeatures(current_frame_); // current frame feature detection
        int num_matches = matcher_->MatchTwoFrames(map_->current_keyframe_, current_frame_, map_, frame2param);

        current_frame_->SetPose(optimizer_->LocalBundleAdjustment(current_frame_, frame2param, false));


        if (num_matches < num_features_tracking_threshold_)
        {

            std::vector<std::vector<int>> loopInfoIdx;
            std::vector<SE3> loopInfoRelPose;

            InsertKeyframe();

            if (loop_detect_)
            {
                DetectLoop(loopInfoIdx, loopInfoRelPose);

                if (loopInfoIdx.size() > 0)
                {

                    optimizer_->PoseGraphOptimization(loopInfoIdx, loopInfoRelPose);
                    // pose_graph_optimizer_->PoseGraphOptimization(loopInfoIdx, loopInfoRelPose);

                    // update map point

                    for (auto &lm_map : map_->landmarks_)
                    {
                        auto lm = lm_map.second;
                        int frame_id = lm->id_frame_; // first observation frame
                        auto obs = lm->GetObs();

                        for (auto &ob_weak : obs)
                        {
                            Feature::Ptr ob = ob_weak.lock();
                            if (ob)
                            {
                                Frame::Ptr frame = ob->frame_.lock();

                                if (frame)
                                {
                                    if (frame->id_ == frame_id)
                                    {
                                        Vec2 p_c(ob->position_.pt.x, ob->position_.pt.y);
                                        Vec3 p_w = frame->pixel2world(p_c, ob->depth_);
                                        lm->SetPos(p_w);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (viewer_)
        {
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->SpinOnce();
        }

        // clear
        cv::Mat nullMat;
        current_frame_->rgb_ = nullMat;
        current_frame_->gray_ = nullMat;
        current_frame_->depth_ = nullMat;

        return true;
    }

    void Tracker::ComputeCurrentBoW()
    {
        vocab_.transform(current_frame_->des_, current_frame_->BoW_vec);
    }

    void Tracker::DetectLoop(std::vector<std::vector<int>> &loopInfoIdx, std::vector<SE3> &loopInfoRelPose)
    {
        DBoW3::QueryResults ret;

        db_.query(current_frame_->des_, ret, 10);

        int kf_id_current = (int)(map_->keyframe_id_.size()) - 1;

        int cnt = 0;
        double score_best;
        for (auto &r : ret)
        {
            if (cnt == 0)
            {
                score_best = r.Score;
                cnt += 1;
            }

            int kf_id = r.Id;

            // LOG(INFO) << "LOOP BETWEEN: " << kf_id_current << " and " << kf_id <<" kf num: "<<map_->keyframes_.size() <<" score: "<<r.Score;
            if (kf_id_current - kf_id > loop_frame_th_ && r.Score > score_threshold_)
            {

                int frame_id = map_->keyframe_id_[kf_id];
                double score = r.Score;

                auto item = map_->keyframes_.find(frame_id);
                if (item != map_->keyframes_.end())
                {
                    Frame::Ptr frame_ref = item->second;

                    std::unordered_map<int, int> frame2param; // frame_id -> parameter_id

                    frame2param.insert(std::make_pair((int)current_frame_->id_, 0));
                    frame2param.insert(std::make_pair((int)frame_ref->id_, 1));

                    int nb_match = matcher_->MatchTwoFrames(frame_ref, current_frame_, map_, frame2param);

                    SE3 RelPose;

                    if (nb_match < 20)
                        RelPose = current_frame_->Pose() * frame_ref->Pose().inverse();
                    else
                        RelPose = optimizer_->LocalBundleAdjustment(current_frame_, frame2param, true) * frame_ref->Pose().inverse();

                    if (RelPose.translation().norm() < rel_pose_thresh_)
                    {
                        LOG(INFO) << "######## Loop Candidate Result ########";
                        LOG(INFO) << "Frame between: " << frame_ref->id_ << " and " << current_frame_->id_;

                        LOG(INFO) << "Dist: " << RelPose.translation().norm();
                        LOG(INFO) << "Score: " << score;
                        LOG(INFO) << "RELPOSE: " << RelPose.translation()[0] << " / " << RelPose.translation()[1] << " / " << RelPose.translation()[2];

                        std::vector<int> loopIdxs = {kf_id, kf_id_current};
                        loopInfoIdx.emplace_back(loopIdxs);
                        loopInfoRelPose.emplace_back(RelPose);
                    }
                }
            }
        }
    }

    bool Tracker::InsertKeyframe()
    {
        // current frame is a new keyframe
        current_frame_->SetKeyFrame();

        // compute BoW
        db_.add(current_frame_->des_);

        // insert current frame as keyframe to map
        map_->InsertKeyFrame(current_frame_);
        SetObservationsForKeyFrame();

        return true;
    }

    void Tracker::SetObservationsForKeyFrame()
    {
        for (auto &feat : current_frame_->features_)
        {
            auto mp = feat->map_point_.lock();
            if (mp)
                mp->AddObservation(feat);
        }
    }
}
