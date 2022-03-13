#include "core/Estimator.h"

namespace RGBDSLAM
{

    Estimator::Estimator(std::string config_file_path)
    {
        window_size_ = Config::Get<int>("window_size");

        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = false;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.num_threads = 4;
        options.max_num_iterations = 100;
        options.max_solver_time_in_seconds = 0.04;
    }

    SE3 Estimator::EstimateCurrentPose(Frame::Ptr &frame, std::unordered_map<int, int> &frame2param)
    {
        Mat33 K = camera_->K();

        ceres::Problem problem;

        Vec4 qvec_param[frame2param.size()];
        Vec3 tvec_param[frame2param.size()];
        Vec3 lm_param[map_->active_landmarks_.size()];

        // Pose parameters
        std::unordered_map<int, int>::iterator it;

        // Current pose parameterization
        SE3 pose = frame->Pose();
        Eigen::Quaterniond q_eigen(pose.rotationMatrix());
        qvec_param[0] = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
        tvec_param[0] = pose.translation();
        problem.AddParameterBlock(qvec_param[0].data(), 4);
        problem.AddParameterBlock(tvec_param[0].data(), 3);

        for (it = frame2param.begin(); it != frame2param.end(); it++)
        {
            int frame_id = it->first;
            int param_id = it->second;

            if (param_id > 0)
            {
                auto item = map_->keyframes_.find(frame_id);

                Frame::Ptr keyframe = item->second;
                SE3 pose = keyframe->Pose();
                Eigen::Quaterniond q_eigen(pose.rotationMatrix());
                qvec_param[param_id] = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
                tvec_param[param_id] = pose.translation();
                problem.AddParameterBlock(qvec_param[param_id].data(), 4);
                problem.AddParameterBlock(tvec_param[param_id].data(), 3);
            }
        }

        // Landmark parameters
        std::unordered_map<int, int> frame2param_lm;

        int jj = 0;
        for (auto &lm : map_->active_landmarks_)
        {
            frame2param_lm.insert(std::make_pair(lm.second->id_, jj));
            lm_param[jj] = lm.second->Pos();
            problem.AddParameterBlock(lm_param[jj].data(), 3);
            jj += 1;
        }

        // Add Reprojection factors
        std::vector<int> covisible_mp_idx;

        int max_obs_count = 0;

        double nb_landmarks = (double)(map_->active_landmarks_.size());
        double mean_obs = 0;
        double std_obs = 0;

        for (auto &mp : map_->active_landmarks_)
        {
            int obs_count = mp.second->observed_times_;
            if (obs_count > max_obs_count)
                max_obs_count = obs_count;
            mean_obs += (double)(obs_count) / nb_landmarks;
        }

        for (auto &mp : map_->active_landmarks_)
        {
            int obs_count = mp.second->observed_times_;
            std_obs += ((double)(obs_count)-mean_obs) * ((double)(obs_count)-mean_obs) / (nb_landmarks - 1);
        }

        std_obs = sqrt(std_obs);

        // LOG(INFO) << "OBSERVATION STATISTICS @@@@@@@@@@@@@@@@@@@@@";
        // LOG(INFO) << "MEAN OBS: " << mean_obs;
        // LOG(INFO) << "STD OBS: " << std_obs;

        double ratio = 0.1;

        for (auto &mp : map_->active_landmarks_)
        {
            int obs_count = mp.second->observed_times_;
            if ((double)obs_count > ((double)max_obs_count) * ratio)
            {
                // double weight = (double)mp.second->observed_times_;
                int lm_id = frame2param_lm.find(mp.second->id_)->second;
                for (auto &ob : mp.second->observations_)
                {
                    auto feature = ob.lock();
                    auto item = frame2param.find(feature->frame_.lock()->id_);
                    if (item != frame2param.end())
                    {
                        int frame_id = item->second;
                        Vec2 obs_src(ob.lock()->position_.pt.x, ob.lock()->position_.pt.y);
                        ceres::CostFunction *cost_function = ProjectionFactor::Create(obs_src, K);
                        problem.AddResidualBlock(cost_function,
                                                 new ceres::CauchyLoss(0.5),
                                                 qvec_param[frame_id].data(),
                                                 tvec_param[frame_id].data(),
                                                 lm_param[lm_id].data());
                    }
                }
            }
        }

        for (int i = 1; i < (int)frame2param.size(); i++)
        {
            problem.SetParameterBlockConstant(qvec_param[i].data());
            problem.SetParameterBlockConstant(tvec_param[i].data());
        }

        for (int i = 0; i < (int)map_->active_landmarks_.size(); i++)
        {
            problem.SetParameterBlockConstant(lm_param[i].data());
        }

        ceres::Solve(options, &problem, &summary);

        Eigen::Quaterniond Q(qvec_param[0][0], qvec_param[0][1], qvec_param[0][2], qvec_param[0][3]);
        // frame->SetPose(SE3(Q, tvec_param[0]));
        return SE3(Q, tvec_param[0]);
    }

    void Estimator::LocalBundleAdjustment(Frame::Ptr &frame)
    {
        Mat33 K = camera_->K();
        int window_size_temp = std::min(window_size_, (int)(map_->active_keyframes_.size()));

        // check co-visible frames

        Vec4 qvec_param[window_size_temp + 1];
        Vec3 tvec_param[window_size_temp + 1];
        Vec3 lm_param[map_->active_landmarks_.size()];
        std::unordered_map<int, int> frame2param;

        ceres::Problem problem;

        // Pose parameters
        int ii = 0;
        for (auto &kfs : map_->active_keyframes_)
        {
            frame2param.insert(std::make_pair(kfs.second->id_, ii));
            SE3 Twc = kfs.second->Pose();
            Mat33 R_ = Twc.rotationMatrix();
            Vec3 t_ = Twc.translation();
            Eigen::Quaterniond q_eigen(R_);
            qvec_param[ii] = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
            tvec_param[ii] = t_;
            problem.AddParameterBlock(qvec_param[ii].data(), 4);
            problem.AddParameterBlock(tvec_param[ii].data(), 3);
            ii += 1;
        }

        frame2param.insert(std::make_pair(frame->id_, ii));
        SE3 Twc = frame->Pose();
        Mat33 R_ = Twc.rotationMatrix();
        Vec3 t_ = Twc.translation();
        Eigen::Quaterniond q_eigen(R_);
        qvec_param[ii] = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
        tvec_param[ii] = t_;
        problem.AddParameterBlock(qvec_param[ii].data(), 4);
        problem.AddParameterBlock(tvec_param[ii].data(), 3);

        // Landmark parameters
        // Vec3 lm_param[map_->active_landmarks_.size()];
        std::unordered_map<int, int> frame2param_lm;
        int jj = 0;
        for (auto &lm : map_->active_landmarks_)
        {
            frame2param_lm.insert(std::make_pair(lm.second->id_, jj));
            lm_param[jj] = lm.second->Pos();
            problem.AddParameterBlock(lm_param[jj].data(), 3);
            jj += 1;
        }

        // Add Reprojection factors
        for (auto &mp : map_->active_landmarks_)
        {
            int lm_id = frame2param_lm.find(mp.second->id_)->second;
            for (auto &ob : mp.second->observations_)
            {
                auto feature = ob.lock();
                auto item = frame2param.find(feature->frame_.lock()->id_);
                if (item != frame2param.end())
                {
                    int frame_id = item->second;
                    Vec2 obs_src(ob.lock()->position_.pt.x, ob.lock()->position_.pt.y);
                    ceres::CostFunction *cost_function = ProjectionFactor::Create(obs_src, K);
                    problem.AddResidualBlock(cost_function,
                                             new ceres::CauchyLoss(0.5),
                                             qvec_param[frame_id].data(),
                                             tvec_param[frame_id].data(),
                                             lm_param[lm_id].data());
                }
            }
        }

        for (int i = 0; i < window_size_temp; i++)
        {
            problem.SetParameterBlockConstant(qvec_param[i].data());
            problem.SetParameterBlockConstant(tvec_param[i].data());
        }

        for (int i = 0; i < (int)map_->active_landmarks_.size(); i++)
        {
            problem.SetParameterBlockConstant(lm_param[i].data());
        }

        ceres::Solve(options, &problem, &summary);

        Eigen::Quaterniond Q(qvec_param[window_size_temp][0],
                             qvec_param[window_size_temp][1],
                             qvec_param[window_size_temp][2],
                             qvec_param[window_size_temp][3]);

        // This is just temporary solution ...
        SE3 pose_inv = SE3(Q, tvec_param[window_size_temp]).inverse();
        Mat33 R_inv = pose_inv.rotationMatrix();
        Vec3 t_inv = pose_inv.translation();
        Eigen::Quaterniond q_inv(R_inv);

        //// If 2D
        // t_inv[1] = 0.0;
        // q_inv.x() = 0.0;
        // q_inv.z() = 0.0;
        // q_inv.normalize();

        SE3 pose__ = SE3(q_inv, t_inv).inverse();

        frame->pose_ = pose__; // SE3(Q, tvec_param[window_size_temp]);

        // LOG(INFO) << pose_inv.translation()[0] << " " << pose_inv.translation()[1] << " " << pose_inv.translation()[2];
        // LOG(INFO) << q_inv.w() << " "
        //           << q_inv.x() << " "
        //           << q_inv.y() << " "
        //           << q_inv.z();
    }

    SE3 Estimator::EstimateRelPose(std::unordered_map<int, int> &frame2param)
    {
        Mat33 K = camera_->K();

        ceres::Problem problem;

        Vec4 qvec_param[frame2param.size()];
        Vec3 tvec_param[frame2param.size()];
        Vec3 lm_param[map_->active_landmarks_.size()];

        // Pose parameters
        std::unordered_map<int, int>::iterator it;

        for (it = frame2param.begin(); it != frame2param.end(); it++)
        {
            int frame_id = it->first;
            int param_id = it->second;

            LOG(INFO) << frame_id << " -> " << param_id;

            auto item = map_->keyframes_.find(frame_id);

            Frame::Ptr frame = item->second;
            SE3 pose = frame->Pose();
            Eigen::Quaterniond q_eigen(pose.rotationMatrix());
            qvec_param[param_id] = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
            tvec_param[param_id] = pose.translation();
            problem.AddParameterBlock(qvec_param[param_id].data(), 4);
            problem.AddParameterBlock(tvec_param[param_id].data(), 3);
        }

        // Landmark parameters
        std::unordered_map<int, int> frame2param_lm;

        int jj = 0;
        for (auto &lm : map_->active_landmarks_)
        {
            frame2param_lm.insert(std::make_pair(lm.second->id_, jj));
            lm_param[jj] = lm.second->Pos();
            problem.AddParameterBlock(lm_param[jj].data(), 3);
        }

        // Add Reprojection factors

        std::vector<int> covisible_mp_idx;
        for (auto &mp : map_->active_landmarks_)
        {
            int lm_id = frame2param_lm.find(mp.second->id_)->second;
            for (auto &ob : mp.second->observations_)
            {
                auto feature = ob.lock();
                auto item = frame2param.find(feature->frame_.lock()->id_);
                if (item != frame2param.end())
                {
                    int frame_id = item->second;
                    Vec2 obs_src(ob.lock()->position_.pt.x, ob.lock()->position_.pt.y);
                    ceres::CostFunction *cost_function = ProjectionFactor::Create(obs_src, K);
                    problem.AddResidualBlock(cost_function,
                                             new ceres::CauchyLoss(0.5),
                                             qvec_param[frame_id].data(),
                                             tvec_param[frame_id].data(),
                                             lm_param[lm_id].data());
                }
            }
        }

        for (int i = 0; i < (int)frame2param.size(); i++)
        {
            if (i != 1)
            {
                problem.SetParameterBlockConstant(qvec_param[i].data());
                problem.SetParameterBlockConstant(tvec_param[i].data());
            }
        }

        ceres::Solve(options, &problem, &summary);

        Eigen::Quaterniond Q_i(qvec_param[0][0], qvec_param[0][1], qvec_param[0][2], qvec_param[0][3]);
        Eigen::Quaterniond Q_j(qvec_param[1][0], qvec_param[1][1], qvec_param[1][2], qvec_param[1][3]);
        SE3 Pose_i(Q_i, tvec_param[0]);
        SE3 Pose_j(Q_j, tvec_param[1]);

        SE3 relativePose = Pose_j * Pose_i.inverse();

        // // This is just temporary solution ...
        // SE3 pose_inv = relativePose.inverse();
        // Mat33 R_inv = pose_inv.rotationMatrix();
        // Vec3 t_inv = pose_inv.translation();
        // Eigen::Quaterniond q_inv(R_inv);

        // //// IF 2D
        // // t_inv[1] = 0.0;
        // // q_inv.x() = 0.0;
        // // q_inv.z() = 0.0;
        // // q_inv.normalize();

        // SE3 pose__ = SE3(q_inv, t_inv).inverse();
        // relativePose = pose__;
        return relativePose;
    }

}