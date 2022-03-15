#include "core/Extractor.h"

namespace RGBDSLAM
{

    Extractor::Extractor(std::string config_file_path)
    {
        if (!Config::SetParameterFile(config_file_path))
            LOG(INFO) << "No configuration file loaded.";

        if (Config::Get<std::string>("feature_type") == "ORB")
        {
            detector_ = cv::ORB::create(Config::Get<int>("num_features"));
            descriptor_ = cv::ORB::create(Config::Get<int>("num_features"));
        }

        depth_min_ = Config::Get<double>("depth_min");
        depth_max_ = Config::Get<double>("depth_max");
    }

    int Extractor::DetectFeatures(Frame::Ptr &frame)
    {

        // detect features
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        std::vector<cv::KeyPoint> keypoints_temp;

        detector_->detect(frame->gray_, keypoints_temp);

        int cnt_detected = 0;
        for (auto &kp : keypoints_temp)
        {

            int u = (int)kp.pt.y;
            int v = (int)kp.pt.x;
            double depth = (double)(frame->depth_.at<float>(kp.pt.y, kp.pt.x));

            std::vector<double> rgb_{frame->rgb_.at<cv::Vec3b>(kp.pt.y, kp.pt.x)[2] / 1.0,
                                     frame->rgb_.at<cv::Vec3b>(kp.pt.y, kp.pt.x)[1] / 1.0,
                                     frame->rgb_.at<cv::Vec3b>(kp.pt.y, kp.pt.x)[0] / 1.0};

            if (depth > depth_min_ && depth <= depth_max_)
            {
                frame->features_.push_back(Feature::Ptr(new Feature(frame, kp, depth, rgb_)));
                keypoints.emplace_back(kp);
            }

            cnt_detected++;
        }

        descriptor_->compute(frame->gray_, keypoints, descriptors);
        frame->SetDescriptor(descriptors);
        return cnt_detected;
    }

}
