#include "core/PoseGraphOptimizer.h"

namespace RGBDSLAM
{

    PoseGraphOptimizer::PoseGraphOptimizer(std::string config_file_path)
    {
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = false;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.num_threads = 10;
        options.max_num_iterations = 100;
        options.max_solver_time_in_seconds = 1.0;
    }

    void PoseGraphOptimizer::PoseGraphOptimization(std::vector<std::vector<int>> &loopInfoIdx, std::vector<SE3> &loopInfoRelPose)
    {
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

            ceres::CostFunction *cost_function = OdometryFactor::Create(qvec, tvec);
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

            ceres::CostFunction *cost_function = LoopClosureFactor::Create(qvec, tvec);
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

    void PoseGraphOptimizer::PoseGraphOptimization2D(std::vector<std::vector<int>> &loopInfoIdx, std::vector<SE3> &loopInfoRelPose)
    {

        Mat33 K = camera_->K();

        int nb_poses = map_->keyframes_.size();
        Vec4 qvec_param[nb_poses];
        Vec2 tvec_param[nb_poses];

        ceres::Problem problem;

        // Pose parameters

        for (int i = 0; i < nb_poses; i++)
        {
            unsigned long frame_id = (unsigned long)map_->keyframe_id_[i];
            auto frame = map_->keyframes_.find(frame_id)->second;
            SE3 pose = frame->Pose();
            Eigen::Quaterniond q_eigen(pose.rotationMatrix());
            qvec_param[i] = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
            tvec_param[i] = Vec2(pose.translation()[0], pose.translation()[2]);
            problem.AddParameterBlock(qvec_param[i].data(), 4);
            problem.AddParameterBlock(tvec_param[i].data(), 2);
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
            Vec2 tvec(odometry.translation()[0], odometry.translation()[2]);

            ceres::CostFunction *cost_function = OdometryFactorPlane::Create(qvec, tvec);
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
            int idx_i = loopInfoIdx[i][0];     // pose_from
            int idx_j = loopInfoIdx[i][1];     // pose_to
            SE3 odometry = loopInfoRelPose[i]; // relative transform

            Eigen::Quaterniond q_eigen(odometry.rotationMatrix());
            Vec4 qvec(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
            Vec2 tvec(odometry.translation()[0], odometry.translation()[2]);

            unsigned long frame_id_i = (unsigned long)map_->keyframe_id_[idx_i]; // global frame id of i
            unsigned long frame_id_j = (unsigned long)map_->keyframe_id_[idx_j]; // global frame id of j

            ceres::CostFunction *cost_function = LoopClosureFactorPlane::Create(qvec, tvec);
            problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5),
                                     qvec_param[idx_i].data(),
                                     tvec_param[idx_i].data(),
                                     qvec_param[idx_j].data(),
                                     tvec_param[idx_j].data());
        }

        // fix initial pose (t=0)
        for (int i = 0; i < 100; i++)
        {
            problem.SetParameterBlockConstant(qvec_param[i].data());
            problem.SetParameterBlockConstant(tvec_param[i].data());
        }

        for (int i = 0; i < nb_poses; i++)
        {
            // keyframe_id : from keyframe_id to global frame_id
            unsigned long frame_id = (unsigned long)map_->keyframe_id_[i];
            auto frame = map_->keyframes_.find(frame_id)->second;
            Eigen::Quaterniond Q(qvec_param[i][0], qvec_param[i][1], qvec_param[i][2], qvec_param[i][3]);

            // update keyframe pose
            Vec3 tvec(tvec_param[i][0], 0.0, tvec_param[i][1]);
            frame->SetPose(SE3(Q, Vec3(tvec_param[i][0], 0.0, tvec_param[i][1])));
        }
    }

}