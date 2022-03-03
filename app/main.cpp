
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
#include <pcl/io/pcd_io.h>
#include "utils/PointCloudUtils.h"
#include "utils/Viewer.h"
#include "core/Frontend.h"
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

    std::string config_file_path_ = "../config/snu_lib_rect.yaml";

    // Initialize detector, descriptor extractor, the number of features to extract

    Frontend::Ptr frontend = Frontend::Ptr(new Frontend(config_file_path_));
    // Backend::Ptr backend = Backend::Ptr(new Backend(config_file_path_));
    Map::Ptr map = Map::Ptr(new Map);

    double fx = Config::Get<double>("camera.fx");
    double fy = Config::Get<double>("camera.fy");
    double cx = Config::Get<double>("camera.cx");
    double cy = Config::Get<double>("camera.cy");
    int max = 0; // Config::Get<int>("nb_img");
    double PIXEL_TO_METER_SCALEFACTOR = Config::Get<double>("pixel_to_meter_scalefactor");

    std::string gt_dir = Config::Get<std::string>("gt_dir");
    std::string est_dir = Config::Get<std::string>("est_dir");
    std::string data_dir = Config::Get<std::string>("data_dir");
    std::string association_dir = Config::Get<std::string>("association_dir");
    int stride = Config::Get<int>("stride");

    std::vector<std::string> rgb_list;
    std::vector<std::string> depth_list;

    std::vector<std::vector<double>> qvec;
    std::vector<std::vector<double>> tvec;

    // read gt path
    LOG(INFO) << "Read GT...";
    if (gt_dir != "None")
    {
        std::ifstream file(gt_dir);

        if (true == file.is_open())
        {
            std::string s;
            while (file)
            {
                getline(file, s);

                std::vector<std::string> pose = split(s, ' ');
                if (pose.size() == 0)
                    break;
                std::vector<double> q{std::stod(pose[7]), std::stod(pose[4]), std::stod(pose[5]), std::stod(pose[6])};
                std::vector<double> t{std::stod(pose[1]), std::stod(pose[2]), std::stod(pose[3]) + 2.5};
                qvec.emplace_back(q);
                tvec.emplace_back(t);
            }

            file.close();
        }
        else
        {
            std::cout << "file open fail" << std::endl;
        }
    }

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
                depth_list.emplace_back(association[1]);
                rgb_list.emplace_back(association[3]);
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

    frontend->SetCamera(camera);
    frontend->SetMap(map);
    viewer->SetMap(map);
    viewer->SetGT(qvec, tvec);
    frontend->SetGT(qvec, tvec);
    frontend->SetViewer(viewer);

    std::vector<SE3> poses;

    for (int i = 0; i < max; i++)
    {

        if (i % stride == 0)
        {
            // LOG(INFO) << i << "-th image processing...";

            cv::Mat img = cv::imread(data_dir + rgb_list[i], 1);
            cv::Mat gray = cv::imread(data_dir + rgb_list[i], 0);
            cv::Mat depth = cv::imread(data_dir + depth_list[i], cv::IMREAD_UNCHANGED);

            cv::imwrite("/home/cadit/Data/snu_lib_rect/debug/" + std::to_string(i) + ".png", img);
            depth.convertTo(depth, CV_32F, PIXEL_TO_METER_SCALEFACTOR);

            if (i == 0)
            {
                Vec3 T_0(0, 0, 0);
                SE3 Pose_0(SO3(), T_0);
                Frame::Ptr frame(new Frame(i, 0, Pose_0, img, gray, depth, K));

                frontend->AddFrame(frame);
            }
            else
            {
                Frame::Ptr frame(new Frame(i, 0, frontend->current_frame_->Pose(), img, gray, depth, K));
                frontend->AddFrame(frame);
            }

            SE3 pose = frontend->current_frame_->Pose();

            Vec3 trans = pose.translation();
            Eigen::Quaterniond rotation(pose.rotationMatrix());

            // LOG(INFO) << "DEBUG TRANSLATION: " << trans[0] << " / "
            //           << trans[1] << " / "
            //           << trans[2] << " / "
            //           << rotation.w() << " / "
            //           << rotation.x() << " / "
            //           << rotation.y() << " / "
            //           << rotation.z();
        }
    }

    // Write estimation result for evaluation

    std::ofstream ofile(est_dir);

    if (ofile.is_open())
    {
        for (int i = 0; i < (int)frontend->frames_.size(); i++)
        {
            if (frontend->frames_[i]->is_keyframe_)
            {
                SE3 pose = frontend->frames_[i]->Pose().inverse();
                Eigen::Quaterniond rotation(pose.rotationMatrix());
                Vec3 translation = pose.translation();
                std::string data = std::to_string(i) + " " +
                                   std::to_string(translation[0]) + " " +
                                   std::to_string(translation[1]) + " " +
                                   std::to_string(translation[2]) + " " +
                                   std::to_string(rotation.x()) + " " +
                                   std::to_string(rotation.y()) + " " +
                                   std::to_string(rotation.z()) + " " +
                                   std::to_string(rotation.w()) + "\n";

                ofile << data;
            }
        }
    }

    ofile.close();

    while (1)
    {
        frontend->viewer_->ShowResult();
    }

    return 0;
}
