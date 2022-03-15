#include "types/Map.h"
#include "types/Feature.h"

namespace RGBDSLAM
{

    LandmarksType Map::GetAllMapPoints()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return landmarks_;
    }
    KeyframesType Map::GetAllKeyFrames()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return keyframes_;
    }

    LandmarksType Map::GetActiveMapPoints()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_landmarks_;
    }

    KeyframesType Map::GetActiveKeyFrames()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_keyframes_;
    }

    void Map::SetWindowSize(int window_size)
    {
        num_active_keyframes_ = window_size;
    }

    void Map::InsertKeyFrame(Frame::Ptr frame)
    {
        current_frame_ = frame;
        current_keyframe_ = frame;

        keyframe_id_.emplace_back(frame->id_);

        keyframes_.insert(make_pair(frame->id_, frame));
        active_keyframes_.insert(make_pair(frame->id_, frame));

        if (active_keyframes_.size() > num_active_keyframes_)
        {
            // LOG(INFO) << "Remove old keyframe!!!";
            RemoveOldKeyframe();
        }
    }

    void Map::InsertMapPoint(MapPoint::Ptr map_point)
    {

        if (landmarks_.find(map_point->id_) == landmarks_.end())
        {
            // New map point

            landmarks_.insert(make_pair(map_point->id_, map_point));
            active_landmarks_.insert(make_pair(map_point->id_, map_point));
        }
        else
        {
            // If exist, make it active!
            if (active_landmarks_.find(map_point->id_) == active_landmarks_.end())
            {
                active_landmarks_.insert(make_pair(map_point->id_, map_point));
            }
            else
            {
                // Nothing
            }
        }
    }

    void Map::UpdateActiveMapPoint(int current_frame_id, int window_size)
    {
        int mm = 0;
        // LOG(INFO) << "Active landmark size: " << active_landmarks_.size();
        std::vector<unsigned long> deleted_idx;
        for (auto &mp : active_landmarks_)
        {
            if ((int)mp.second->id_frame_ <= (int)current_frame_id - (int)window_size)
            {
                deleted_idx.emplace_back(mp.first);
                mm += 1;
            }
        }

        for (auto &ii : deleted_idx)
        {
            active_landmarks_.erase(ii);
        }

        // LOG(INFO) << "Deleted active landmarks: " << mm;
    }

    void Map::RemoveOldKeyframe()
    {
        if (current_frame_ == nullptr)
            return;

        // Find two frames closest to the current frame
        double max_dis = 0, min_dis = 9999;
        double max_kf_id = 0, min_kf_id = 0;
        auto Twc = current_frame_->Pose().inverse(); // inverse for relative transform

        for (auto &kf : active_keyframes_)
        {
            if (kf.second == current_frame_)
                continue;
            
            Vec3 translation = (kf.second->Pose() * Twc).translation();
            //auto dis = (kf.second->Pose() * Twc).log().norm();
            double dis = translation.norm();
            if (dis > max_dis)
            {
                max_dis = dis;
                max_kf_id = kf.first;
            }
            if (dis < min_dis)
            {
                min_dis = dis;
                min_kf_id = kf.first;
            }
        }

        Frame::Ptr frame_to_remove = nullptr;
        
        if (min_dis < min_dis_th)
        {
            // if distance between current keyframe and nearest one is close enough,
            // delete the nearest keyframe
            frame_to_remove = keyframes_.at(min_kf_id);
        }
        else
        {
            // delete the furthest keyframe
            frame_to_remove = keyframes_.at(max_kf_id);
        }

        // LOG(INFO) << "remove keyframe " << frame_to_remove->id_;

        // remove keyframe and related landmark observation
        active_keyframes_.erase(frame_to_remove->id_);

        // Map Clean

        for (auto feat : frame_to_remove->features_)
        {
            auto mp = feat->map_point_.lock();
            if (mp)
            {
                active_landmarks_.erase(mp->id_);
            }
        }

        // LOG(INFO) << "The number of active map pts : " << active_landmarks_.size();
        // LOG(INFO) << "The number of total map pts : " << landmarks_.size();
        //CleanMap();
    }
}