// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_CAMERA_HPP_
#define SURFACE_REFINEMENT_CAMERA_HPP_

#include <cnpy.h>
#include <glog/logging.h>

#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <regex>
#include <string>
#include <utility>

#include "memory_manager.hpp"

namespace surface_refinement {

struct CameraParams {
  double focal_length_x;
  double focal_length_y;
  double principal_point_x;
  double principal_point_y;
  Eigen::Matrix4d rectification_matrix;
  Eigen::Matrix4d transformation_matrix;
  Eigen::Matrix4d to_world_space_transformation_matrix;
  Eigen::Vector3d camera_normal;
};

class Camera {
 public:
  explicit Camera(boost::filesystem::path camera_parameters_path);
  ~Camera();

  CameraParams GetCameraParams();

  CameraParams* GetDeviceCameraParams();

 private:
  boost::filesystem::path camera_params_path_;

  CameraParams camera_params_;

  CameraParams* d_camera_params_;

  static bool ReadMatrix(const cv::FileStorage& fs, const std::string& key,
                         cv::Mat* matrix, int rows, int cols);

  void LoadCameraParameters();
};

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_CAMERA_HPP_
