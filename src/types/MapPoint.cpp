
#include "types/Map.h"
#include "types/Feature.h"

namespace RGBDSLAM
{

    MapPoint::MapPoint(long id, Vec3 position) : id_(id), pos_(position) {}

    Vec3 MapPoint::Pos()
    {
        // share same mutex for protecting data
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pos_;
    }

    void MapPoint::SetPos(const Vec3 &pos)
    {
        // share same mutex for protecting data
        std::unique_lock<std::mutex> lck(data_mutex_);
        pos_ = pos;
    };

    void MapPoint::AddObservation(std::shared_ptr<Feature> feature)
    {
        // share same mutex for protecting data
        std::unique_lock<std::mutex> lck(data_mutex_);
        observations_.push_back(feature);
        observed_times_++;
    }

    std::list<std::weak_ptr<Feature>> MapPoint::GetObs()
    {
        // share same mutex for protecting data
        std::unique_lock<std::mutex> lck(data_mutex_);
        return observations_;
    }

    MapPoint::Ptr MapPoint::CreateNewMappoint()
    {
        static long factory_id = 0;
        MapPoint::Ptr new_mappoint(new MapPoint);
        new_mappoint->id_ = factory_id++;
        return new_mappoint;
    }

}
