#include "utils/PointCloudUtils.h"

namespace RGBDSLAM
{
    void depth2PCD(const Frame::Ptr frame, const Mat33 intrinsic, PointCloud::Ptr &pcd)
    {
        int h = frame->rgb_.rows;
        int w = frame->rgb_.cols;
        Eigen::MatrixXd pts;

        // pts = (N, 3)
        double fx = intrinsic(0, 0);
        double fy = intrinsic(1, 1);

        double cx = intrinsic(0, 2);
        double cy = intrinsic(1, 2);

        for (int v = 0; v < h; v++)
        {
            for (int u = 0; u < w; u++)
            {
                double r = (double)frame->rgb_.at<cv::Vec3b>(v, u)[2];
                double g = (double)frame->rgb_.at<cv::Vec3b>(v, u)[1];
                double b = (double)frame->rgb_.at<cv::Vec3b>(v, u)[0];
                double z = (double)frame->depth_.at<float>(v, u);

                if (z != 0 && z <= 5)
                {
                    double x = (u - cx) * z / fx;
                    double y = (v - cy) * z / fy;

                    Vec3 p(x, y, z);
                    Vec3 p_w = frame->pose_.inverse() * p;

                    PointT P;
                    P.x = p_w(0, 0);
                    P.y = p_w(1, 0);
                    P.z = p_w(2, 0);
                    P.r = r;
                    P.g = g;
                    P.b = b;
                    pcd->points.push_back(P);
                }
            }
        }
    }

    void depth2PCDfromMap(const Map::Ptr &map, PointCloud::Ptr &pcd)
    {
        for (auto &mp_ : map->landmarks_)
        {
            auto mp = mp_.second;
            Vec3 pos_ = mp->Pos();
            double x = pos_(0, 0);
            double y = pos_(1, 0);
            double z = pos_(2, 0);
            double r = mp->rgb_[0];
            double g = mp->rgb_[1];
            double b = mp->rgb_[2];
            PointT P;
            P.x = x;
            P.y = y;
            P.z = z;
            P.r = r;
            P.g = g;
            P.b = b;
            pcd->points.push_back(P);
        }
    }
}