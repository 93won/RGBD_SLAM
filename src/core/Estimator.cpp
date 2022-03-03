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
        options.max_num_iterations = 10;
        options.max_solver_time_in_seconds = 0.04;
    }

    void Estimator::LocalBundleAdjustment(Frame::Ptr &frame)
    {
        Mat33 K = camera_->K();
        int window_size_temp = std::min(window_size_, (int)(map_->active_keyframes_.size()));
        assert(window_size_temp == map_->active_keyframes_.size());

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
            SE3 pose_ = kfs.second->Pose();
            Eigen::Quaterniond q_eigen(pose_.rotationMatrix());
            qvec_param[ii] = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
            tvec_param[ii] = pose_.translation();
            problem.AddParameterBlock(qvec_param[ii].data(), 4);
            problem.AddParameterBlock(tvec_param[ii].data(), 3);
            ii += 1;
        }

        frame2param.insert(std::make_pair(frame->id_, ii));
        SE3 pose_ = frame->Pose();
        Eigen::Quaterniond q_eigen(pose_.rotationMatrix());
        qvec_param[ii] = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
        tvec_param[ii] = pose_.translation();
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
                    ceres::CostFunction *cost_function = ProjectionFactorSimplePinholeConstantIntrinsic::Create(obs_src, K);
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
        frame->pose_ = SE3(Q, tvec_param[window_size_temp]);
    }

    void Estimator::LocalBundleAdjustment2D(Frame::Ptr &frame)
    {

        // qvec = [w, y]
        // teve = [x, z]

        Mat33 K = camera_->K();
        int window_size_temp = std::min(window_size_, (int)(map_->active_keyframes_.size()));
        assert(window_size_temp == map_->active_keyframes_.size());

        Vec4 qvec_param[window_size_temp + 1];
        Vec2 tvec_param[window_size_temp + 1];
        Vec3 lm_param[map_->active_landmarks_.size()];
        std::unordered_map<int, int> frame2param;

        ceres::Problem problem;

        // Pose parameters
        int ii = 0;
        for (auto &kfs : map_->active_keyframes_)
        {
            frame2param.insert(std::make_pair(kfs.second->id_, ii));
            SE3 pose = kfs.second->Pose();

            Eigen::Quaterniond q_eigen(pose.rotationMatrix());
            qvec_param[ii] = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
            tvec_param[ii] = Vec2(pose.translation()[0], pose.translation()[2]);

            // std::pair<Vec4, Vec3> qt_vec = t2v(pose);
            // qvec_param[ii] = qt_vec.first;                             // Vec2(qt_vec.first[0], qt_vec.first[2]);   // [w, y]
            // tvec_param[ii] = Vec2(qt_vec.second[0], qt_vec.second[2]); // [x, z]
            problem.AddParameterBlock(qvec_param[ii].data(), 4);
            problem.AddParameterBlock(tvec_param[ii].data(), 2);
            ii += 1;
        }

        frame2param.insert(std::make_pair(frame->id_, ii));
        SE3 pose = frame->Pose();
        Eigen::Quaterniond q_eigen(pose.rotationMatrix());
        qvec_param[ii] = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
        tvec_param[ii] = Vec2(pose.translation()[0], pose.translation()[2]);
        problem.AddParameterBlock(qvec_param[ii].data(), 4);
        problem.AddParameterBlock(tvec_param[ii].data(), 2);

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
                    ceres::CostFunction *cost_function = ProjectionFactorSimplePinholeConstantIntrinsicPlane::Create(obs_src, K);
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
        Vec2 Trans = tvec_param[window_size_temp];

        frame->pose_ = SE3(Q, Vec3(Trans[0], 0.0, Trans[1]));
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
                    ceres::CostFunction *cost_function = ProjectionFactorSimplePinholeConstantIntrinsic::Create(obs_src, K);
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

        for (int i = 0; i < (int)map_->active_landmarks_.size(); i++)
        {
            problem.SetParameterBlockConstant(lm_param[i].data());
        }

        ceres::Solve(options, &problem, &summary);

        Eigen::Quaterniond Q_i(qvec_param[0][0], qvec_param[0][1], qvec_param[0][2], qvec_param[0][3]);
        Eigen::Quaterniond Q_j(qvec_param[1][0], qvec_param[1][1], qvec_param[1][2], qvec_param[1][3]);
        SE3 Pose_i(Q_i, tvec_param[0]);
        SE3 Pose_j(Q_j, tvec_param[1]);

        SE3 relativePose = Pose_j * Pose_i.inverse();

        return relativePose;
    }

    SE3 Estimator::EstimateRelPose2D(std::unordered_map<int, int> &frame2param)
    {

        Mat33 K = camera_->K();
        ceres::Problem problem;

        Vec4 qvec_param[frame2param.size()];
        Vec2 tvec_param[frame2param.size()];
        Vec3 lm_param[map_->active_landmarks_.size()];

        // Pose parameters
        std::unordered_map<int, int>::iterator it;
        for (it = frame2param.begin(); it != frame2param.end(); it++)
        {
            int frame_id = it->first;
            int param_id = it->second;

            auto item = map_->keyframes_.find(frame_id);

            Frame::Ptr frame = item->second;
            SE3 pose = frame->Pose();

            Eigen::Quaterniond q_eigen(pose.rotationMatrix());
            qvec_param[param_id] = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
            tvec_param[param_id] = Vec2(pose.translation()[0], pose.translation()[2]);

            problem.AddParameterBlock(qvec_param[param_id].data(), 4);
            problem.AddParameterBlock(tvec_param[param_id].data(), 2);
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
                    ceres::CostFunction *cost_function = ProjectionFactorSimplePinholeConstantIntrinsicPlane::Create(obs_src, K);
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
        
        SE3 Pose_i(Q_i, Vec3(tvec_param[0][0], 0.0, tvec_param[0][1]));
        SE3 Pose_j(Q_j, Vec3(tvec_param[1][0], 0.0, tvec_param[1][1]));

        SE3 relativePose = Pose_j * Pose_i.inverse();

        return relativePose;
    }
}