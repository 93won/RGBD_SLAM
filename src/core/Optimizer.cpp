#include "core/Optimizer.h"

namespace RGBDSLAM
{

    Optimizer::Optimizer(std::string config_file_path)
    {
    }

    SE3 Optimizer::LocalBundleAdjustment(Frame::Ptr &frame, std::unordered_map<int, int> &frame2param, bool loop)
    {
        Mat33 K = camera_->K();

        ceres::Problem problem;
        ceres::Solver::Options options;
        ceres::Solver::Summary summary;

        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = false;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.num_threads = 4;
        options.max_num_iterations = 10;
        options.max_solver_time_in_seconds = 0.04;

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

        if (!loop)
        {
            for (int i = 0; i < (int)map_->active_landmarks_.size(); i++)
            {
                problem.SetParameterBlockConstant(lm_param[i].data());
            }
        }

        ceres::Solve(options, &problem, &summary);

        Eigen::Quaterniond Q(qvec_param[0][0], qvec_param[0][1], qvec_param[0][2], qvec_param[0][3]);
        // frame->SetPose(SE3(Q, tvec_param[0]));
        return SE3(Q, tvec_param[0]);
    }

    void Optimizer::PoseGraphOptimization(std::vector<std::vector<int>> &loopInfoIdx, std::vector<SE3> &loopInfoRelPose)
    {

        ceres::Solver::Options options;
        ceres::Solver::Summary summary;

        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = false;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.num_threads = 10;
        options.max_num_iterations = 100;
        options.max_solver_time_in_seconds = 1.0;

        Mat33 K = camera_->K();

        int nb_poses = map_->keyframes_.size();
        Vec4 qvec_param[nb_poses];
        Vec3 tvec_param[nb_poses];

        ceres::Problem problem;

        // Pose parameters

        for (int i = 0; i < nb_poses; i++)
        {
            unsigned long frame_id = (unsigned long)map_->keyframe_id_[i];
            auto frame = map_->keyframes_.find(frame_id)->second;
            SE3 pose = frame->Pose();
            Eigen::Quaterniond q_eigen(pose.rotationMatrix());
            qvec_param[i] = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
            tvec_param[i] = pose.translation();

            problem.AddParameterBlock(qvec_param[i].data(), 4);
            problem.AddParameterBlock(tvec_param[i].data(), 3);
        }

        for (int i = 0; i < nb_poses - 1; i++)
        {
            unsigned long frame_id_i = (unsigned long)map_->keyframe_id_[i];
            unsigned long frame_id_j = (unsigned long)map_->keyframe_id_[i + 1];

            auto frame_i = map_->keyframes_.find(frame_id_i)->second;
            auto frame_j = map_->keyframes_.find(frame_id_j)->second;

            SE3 odometry = frame_j->Pose() * frame_i->Pose().inverse();
            Eigen::Quaterniond q_eigen(odometry.rotationMatrix());

            Vec4 qvec(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
            Vec3 tvec = odometry.translation();

            const double w = 1.0;

            ceres::CostFunction *cost_function = BetweenFactor::Create(qvec, tvec, w);
            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5),
                                     qvec_param[i].data(),
                                     tvec_param[i].data(),
                                     qvec_param[i + 1].data(),
                                     tvec_param[i + 1].data());
        }

        // Loop closure factor
        int nb_loop = (int)loopInfoIdx.size();

        for (int i = 0; i < nb_loop; i++)
        {
            int idx_i = loopInfoIdx[i][0]; // from
            int idx_j = loopInfoIdx[i][1]; // to
            SE3 relPose = loopInfoRelPose[i];

            Eigen::Quaterniond q_eigen(relPose.rotationMatrix());

            Vec4 qvec(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
            Vec3 tvec = relPose.translation();

            unsigned long frame_id_i = (unsigned long)map_->keyframe_id_[idx_i]; // global frame id of i
            unsigned long frame_id_j = (unsigned long)map_->keyframe_id_[idx_j]; // global frame id of j

            const double w = 1.0;

            ceres::CostFunction *cost_function = BetweenFactor::Create(qvec, tvec, w);
            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5),
                                     qvec_param[idx_i].data(),
                                     tvec_param[idx_i].data(),
                                     qvec_param[idx_j].data(),
                                     tvec_param[idx_j].data());
        }

        for (int i = 0; i < 1; i++)
        {
            problem.SetParameterBlockConstant(qvec_param[i].data());
            problem.SetParameterBlockConstant(tvec_param[i].data());
        }

        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;

        for (int i = 0; i < nb_poses; i++)
        {
            // keyframe_id : from keyframe_id to global frame_id
            unsigned long frame_id = (unsigned long)map_->keyframe_id_[i];
            auto frame = map_->keyframes_.find(frame_id)->second;

            Vec4 qvec_(qvec_param[i][0], qvec_param[i][1], qvec_param[i][2], qvec_param[i][3]);
            qvec_ /= qvec_.norm();

            // LOG(INFO) <<"CHECK QVEC NORM: "<<qvec_.norm();

            // Update Keyframe Pose
            Eigen::Quaterniond Q(qvec_[0], qvec_[1], qvec_[2], qvec_[3]);
            frame->SetPose(SE3(Q, tvec_param[i]));
        }
    }
}