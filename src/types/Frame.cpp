#include "types/Frame.h"

namespace RGBDSLAM
{

    Frame::Frame(long id, double time_stamp, const SE3 &pose, const cv::Mat &rgb, const cv::Mat &gray, const cv::Mat &depth, const Mat33 &K, const std::string& stamp)
        : id_(id), time_stamp_(time_stamp), pose_(pose), rgb_(rgb), gray_(gray), depth_(depth), K_(K), stamp_(stamp) {}

    Frame::Ptr Frame::CreateFrame()
    {
        static long factory_id = 0;
        Frame::Ptr new_frame(new Frame);
        new_frame->id_ = factory_id++;
        return new_frame;
    }

    SE3 Frame::Pose()
    {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        return pose_;
    }

    void Frame::SetPose(const SE3 &pose)
    {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        pose_ = pose;
    }

    void Frame::SetDescriptor(const cv::Mat& descriptor)
    {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        des_ = descriptor;
    }

    void Frame::SetKeyFrame()
    {
        static long keyframe_factory_id = 0;
        is_keyframe_ = true;
        keyframe_id_ = keyframe_factory_id++;
    }

    

    Vec3 Frame::world2camera(const Vec3 &p_w)
    {
        // Rwc = pose_.rotationMatrix()
        // Twc = pose_.translation()
        // return Rwc*(p_w - Twc);
        return pose_ * p_w;
    }

    Vec3 Frame::camera2world(const Vec3 &p_c)
    {
        return pose_.inverse() * p_c;
    }

    Vec2 Frame::camera2pixel(const Vec3 &p_c)
    {
        return Vec2(
            K_(0, 0) * p_c(0) / p_c(2) + K_(0, 2),
            K_(1, 1) * p_c(1) / p_c(2) + K_(1, 2));
    }

    Vec3 Frame::pixel2camera(const Vec2 &p_p, double depth)
    {
        // from image plane pixel to camera frame coordinates
        return Vec3(
            (p_p(0) - K_(0, 2)) * depth / K_(0, 0),
            (p_p(1) - K_(1, 2)) * depth / K_(1, 1),
            depth);
    }

    Vec2 Frame::world2pixel(const Vec3 &p_w)
    {
        return camera2pixel(world2camera(p_w));
    }

    Vec3 Frame::pixel2world(const Vec2 &p_p, double depth)
    {
        return camera2world(pixel2camera(p_p, depth));
    }
}