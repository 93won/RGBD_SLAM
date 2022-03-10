//
// Created by gaoxiang on 19-5-4.
//

#ifndef MYSLAM_VIEWER_H
#define MYSLAM_VIEWER_H

#include <thread>
#include <pangolin/pangolin.h>

#include "types/Common.h"
#include "types/Frame.h"
#include "types/Map.h"

namespace RGBDSLAM
{

    class Viewer
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Viewer> Ptr;

        Viewer();

        void SetMap(Map::Ptr map) { map_ = map; }
        void SetGT(std::vector<Vec4> &q, std::vector<Vec3> &t)
        {
            gt_q = q;
            gt_t = t;
        }

        void AddCurrentFrame(Frame::Ptr current_frame);

        // void UpdateMap();

        void SpinOnce();
        void ShowResult();
        void Initialize();

        void DrawFrame(Frame::Ptr frame, const float *color);

        void DrawTrajectory();
        void DrawMapPoints();

        void FollowCurrentFrame(pangolin::OpenGlRenderState &vis_camera);

        /// plot the features in current frame into an image
        cv::Mat PlotFrameImage();

        Frame::Ptr current_frame_ = nullptr;
        Map::Ptr map_ = nullptr;

        std::mutex viewer_data_mutex_;
        pangolin::View vis_display;
        pangolin::OpenGlRenderState vis_camera;

        double fx, fy, cx, cy;
        int width, height;

        std::vector<Vec4> gt_q;
        std::vector<Vec3> gt_t;
    };
}

#endif
