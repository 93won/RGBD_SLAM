#include "core/Matcher.h"

namespace RGBDSLAM
{

    Matcher::Matcher(std::string config_file_path)
    {
        if (!Config::SetParameterFile(config_file_path))
            LOG(INFO) << "No configuration file loaded.";

        matcher_ = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        knn_ratio_ = Config::Get<double>("knn_ratio");
        match_threshold_tracking_ = Config::Get<int>("match_threshold_tracking");
    }


    int Matcher::MatchTwoFrames(Frame::Ptr &frame_i, Frame::Ptr &frame_j, Map::Ptr &map_, std::unordered_map<int, int> &frame2param)
    {
        // frame_i: ref frame
        // frame_j: current frame

        std::vector<cv::DMatch> matches;

        std::vector<cv::KeyPoint> kps1, kps2;
        std::vector<double> d1, d2;
        cv::Mat des_1, des_2;
        for (auto &feature : frame_i->features_)
        {
            kps1.emplace_back(feature->position_);
            d1.emplace_back(feature->depth_);
        }

        for (auto &feature : frame_j->features_)
        {
            kps2.emplace_back(feature->position_);
            d2.emplace_back(feature->depth_);
        }

        std::vector<std::vector<cv::DMatch>> matches_temp;

        matcher_.knnMatch(frame_i->des_, frame_j->des_, matches_temp, 2);

        for (auto &match : matches_temp)
        {
            if (match.size() == 2)
            {
                if (match[0].distance < match[1].distance * knn_ratio_)
                {
                    matches.emplace_back(match[0]);
                }
            }
        }

        // filtering using ransac

        std::vector<cv::DMatch> matches_good;

        if (matches.size() < match_threshold_tracking_)
            return 0;

        std::vector<cv::Point2f> kps_1_pt;
        std::vector<cv::Point2f> kps_2_pt;
        for (size_t i = 0; i < matches.size(); i++)
        {
            //-- Get the keypoints from the good matches
            kps_1_pt.emplace_back(kps1[matches[i].queryIdx].pt);
            kps_2_pt.emplace_back(kps2[matches[i].trainIdx].pt);
        }

        std::vector<int> mask;

        cv::Mat H = cv::findHomography(kps_1_pt, kps_2_pt, cv::RANSAC, 50, mask, 10000, 0.9995);

        for (size_t i = 0; i < mask.size(); i++)
        {
            if (mask[i] == 1)
            {
                matches_good.emplace_back(matches[i]);
            }
        }
        matches = matches_good;
        int ii = (int)frame2param.size();
        for (size_t i = 0; i < matches.size(); i++)
        {
            auto m = matches[i];

            // link map point
            auto mp = frame_i->features_[m.queryIdx]->map_point_.lock();
            if (mp)
            {

                // existing map point
                frame_j->features_[m.trainIdx]->map_point_ = frame_i->features_[m.queryIdx]->map_point_;
                mp->AddObservation(frame_j->features_[m.trainIdx]);
                map_->InsertMapPoint(mp);

                auto item = frame2param.find((int)mp->id_frame_);

                if (item == frame2param.end())
                {

                    int frame_id = (int)mp->id_frame_;
                    frame2param.insert(std::make_pair(frame_id, ii++));
                }
            }
            else
            {
                // new map point
                Vec2 p_c_last(frame_i->features_[m.queryIdx]->position_.pt.x, frame_i->features_[m.queryIdx]->position_.pt.y);
                Vec3 p_w_last = frame_i->pixel2world(p_c_last, frame_i->features_[m.queryIdx]->depth_);
                auto new_map_point = MapPoint::CreateNewMappoint();
                new_map_point->SetPos(p_w_last);
                new_map_point->rgb_ = frame_i->features_[m.queryIdx]->rgb_;
                new_map_point->AddObservation(frame_i->features_[m.queryIdx]);
                new_map_point->AddObservation(frame_j->features_[m.trainIdx]);
                new_map_point->id_frame_ = frame_i->id_;

                frame_i->features_[m.queryIdx]->map_point_ = new_map_point;
                frame_j->features_[m.trainIdx]->map_point_ = new_map_point;

                map_->InsertMapPoint(new_map_point);
            }
        }

        return (int)matches.size();
    }

}