#pragma once
#ifndef RGBD_SLAM_MAPPOINT_H
#define RGBD_SLAM_MAPPOINT_H

#include "types/Common.h"

namespace RGBDSLAM
{

    struct Frame;
    struct Feature;

    struct MapPoint
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<MapPoint> Ptr;
        unsigned long id_ = 0; // ID
        bool is_outlier_ = false;
        Vec3 pos_ = Vec3::Zero(); // Position in world
        std::vector<double> rgb_;
        std::mutex data_mutex_;
        int observed_times_ = 0; // being observed by feature matching algo.
        std::list<std::weak_ptr<Feature>> observations_;

        unsigned long id_frame_ = 0; // first observation frame id

        MapPoint() {}
        MapPoint(long id, Vec3 position);

        Vec3 Pos();
        void SetPos(const Vec3 &pos);
        void AddObservation(std::shared_ptr<Feature> feature);
        std::list<std::weak_ptr<Feature>> GetObs();
 
        // factory function
        static MapPoint::Ptr CreateNewMappoint();
    };
}
#endif
