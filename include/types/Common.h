#pragma once
#ifndef RGBD_SLAM_COMMON_H
#define RGBD_SLAM_COMMON_H

// std
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <typeinfo>
#include <unordered_map>
#include <vector>

// define the commonly included file to avoid a long include list
#include <Eigen/Core>
#include <Eigen/Geometry>

// typedefs for eigen
typedef Eigen::Matrix<double, 3, 3> Mat33;
typedef Eigen::Matrix<double, 3, 4> Mat34;
typedef Eigen::Matrix<double, 4, 4> Mat44;

typedef Eigen::Vector2d Vec2;
typedef Eigen::Vector3d Vec3;
typedef Eigen::Vector4d Vec4;
typedef Eigen::Matrix<double, 5, 1> Vec5;
typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<double, 7, 1> Vec7;

// for Sophus
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

typedef Sophus::SE3d SE3;
typedef Sophus::SO3d SO3;

// for cv
#include <opencv2/core/core.hpp>

// glog
#include <glog/logging.h>

// converter

#endif