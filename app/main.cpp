
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream> // ifstream header
#include <string>  // getline header
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <pangolin/pangolin.h>
#include "utils/Viewer.h"
#include "core/Tracker.h"
#include "Config.h"

#include <pangolin/pangolin.h>

using namespace RGBDSLAM;

std::vector<std::string> split(std::string input, char delimiter)
{
    std::vector<std::string> answer;
    std::stringstream ss(input);
    std::string temp;

    while (getline(ss, temp, delimiter))
    {
        answer.push_back(temp);
    }

    return answer;
}

int main(int argc, char **argv)
{

    std::string config_file_path_ = "../config/f2_desk.yaml";

    // Initialize detector, descriptor extractor, the number of features to extract

    Tracker::Ptr tracker = Tracker::Ptr(new Tracker(config_file_path_));
    Map::Ptr map = Map::Ptr(new Map);

    double fx = Config::Get<double>("camera.fx");
    double fy = Config::Get<double>("camera.fy");
    double cx = Config::Get<double>("camera.cx");
    double cy = Config::Get<double>("camera.cy");
    double k1 = Config::Get<double>("camera.k1");
    double k2 = Config::Get<double>("camera.k2");
    double p1 = Config::Get<double>("camera.p1");
    double p2 = Config::Get<double>("camera.p2");
    double k3 = Config::Get<double>("camera.k3");

    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64FC1);
    cv::Mat distCoeffs = cv::Mat::zeros(1, 5, CV_64FC1);

    cameraMatrix = (cv::Mat1d(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    distCoeffs = (cv::Mat1d(1, 5) << k1, k2, p1, p2, k3);

    cv::Mat map1, map2;
    cv::Size imageSize = cv::Size(640, 480);
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix, imageSize, CV_32FC1, map1, map2);

    int max = 0; // Config::Get<int>("nb_img");
    double PIXEL_TO_METER_SCALEFACTOR = 1.0 / Config::Get<double>("pixel_to_meter_scalefactor");

    std::string gt_dir = Config::Get<std::string>("gt_dir");
    std::string est_dir = Config::Get<std::string>("est_dir");
    std::string data_dir = Config::Get<std::string>("data_dir");
    std::string association_dir = Config::Get<std::string>("association_dir");
    int stride = Config::Get<int>("stride");

    std::vector<std::string> rgb_list;
    std::vector<std::string> depth_list;
    std::vector<std::string> stamp;

    LOG(INFO) << "Read Association...";
    // read association
    if (association_dir != "None")
    {
        std::ifstream file(association_dir);

        if (true == file.is_open())
        {
            std::string s;
            while (file)
            {
                getline(file, s);

                std::vector<std::string> association = split(s, ' ');
                if (association.size() == 0)
                    break;

                max += 1;
                depth_list.emplace_back(association[3]);
                rgb_list.emplace_back(association[1]);
                stamp.emplace_back(association[0]);
            }

            file.close();
        }
        else
        {
            std::cout << "file open fail" << std::endl;
        }
    }

    Mat33 K;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    LOG(INFO) << "Intrinsic (fx, fy, cx, cy) : " << fx << ", " << fy << ", " << cx << ", " << cy;

    Vec3 t_1(0, 0, 0);
    Camera::Ptr camera = Camera::Ptr(new Camera(fx, fy, cx, cy));
    Viewer::Ptr viewer = Viewer::Ptr(new Viewer);

    tracker->SetCamera(camera);
    tracker->SetMap(map);

    viewer->SetMap(map);
    tracker->SetViewer(viewer);

    std::vector<SE3> poses;

    for (int i = 0; i < max; i++)
    {

        if (i % stride == 0)
        {
            // LOG(INFO) << i << "-th image processing...";

            cv::Mat img = cv::imread(data_dir + rgb_list[i], 1);
            cv::Mat gray = cv::imread(data_dir + rgb_list[i], 0);
            cv::Mat depth = cv::imread(data_dir + depth_list[i], cv::IMREAD_UNCHANGED);

            cv::remap(img, img, map1, map2, cv::INTER_LINEAR);
            cv::remap(gray, gray, map1, map2, cv::INTER_LINEAR);
            cv::remap(depth, depth, map1, map2, cv::INTER_LINEAR);

            // cv::imwrite("/home/cadit/Data/snu_lib_rect/debug/" + std::to_string(i) + ".png", img);
            depth.convertTo(depth, CV_32F, PIXEL_TO_METER_SCALEFACTOR);

            if (i == 0)
            {
                Vec3 T_0(0, 0, 0);
                SE3 Pose_0(SO3(), T_0);
                Frame::Ptr frame(new Frame(i, 0, Pose_0, img, gray, depth, K, stamp[i]));

                tracker->AddFrame(frame);
            }
            else
            {
                Frame::Ptr frame(new Frame(i, 0, tracker->current_frame_->Pose(), img, gray, depth, K, stamp[i]));
                tracker->AddFrame(frame);
            }

            SE3 pose = tracker->current_frame_->Pose();

            Vec3 trans = pose.translation();
            Eigen::Quaterniond rotation(pose.rotationMatrix());
        }
    }

    // Write estimation result for evaluation

    std::ofstream ofile(est_dir);

    if (ofile.is_open())
    {
        for (int i = 0; i < (int)map->keyframe_id_.size(); i++)
        {
            int id = map->keyframe_id_[i]; // frame id
            auto item = map->keyframes_.find(id);
            if (item != map->keyframes_.end())
            {
                // SE3 Twc = item->second->Pose().inverse();

                SE3 pose = item->second->Pose().inverse();

                Eigen::Quaterniond rotation(pose.rotationMatrix());
                Vec3 translation = pose.translation();
                std::string data = item->second->stamp_ + " " +
                                   std::to_string(translation[0]) + " " +
                                   std::to_string(translation[1]) + " " +
                                   std::to_string(translation[2]) + " " +
                                   std::to_string(rotation.x()) + " " +
                                   std::to_string(rotation.y()) + " " +
                                   std::to_string(rotation.z()) + " " +
                                   std::to_string(rotation.w()) + "\n";

                ofile << data;

                // Vec3 Translation = Twc.translation();
                // traj.emplace_back(Vec3(Translation[0], Translation[1], Translation[2]));
            }
        }
        // for (int i = 0; i < (int)tracker->frames_.size(); i++)
        // {
        //     if (tracker->frames_[i]->is_keyframe_)
        //     {
        //         SE3 pose = tracker->frames_[i]->Pose().inverse();
        //         Eigen::Quaterniond rotation(pose.rotationMatrix());
        //         Vec3 translation = pose.translation();
        //         std::string data = stamp[i] + " " +
        //                            std::to_string(translation[0]) + " " +
        //                            std::to_string(translation[1]) + " " +
        //                            std::to_string(translation[2]) + " " +
        //                            std::to_string(rotation.x()) + " " +
        //                            std::to_string(rotation.y()) + " " +
        //                            std::to_string(rotation.z()) + " " +
        //                            std::to_string(rotation.w()) + "\n";

        //         ofile << data;
        //     }
        // }
    }

    ofile.close();

    while (1)
    {
        tracker->viewer_->ShowResult();
    }

    return 0;
}
