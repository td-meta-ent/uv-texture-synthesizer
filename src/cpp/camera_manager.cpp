// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "camera_manager.hpp"

#include <open3d/Open3D.h>

namespace surface_refinement {

CameraManager::CameraManager(boost::filesystem::path camera_parameters_dir)
    : camera_params_dir_(std::move(camera_parameters_dir)) {
  LoadCameraParams();

  d_camera_params_ =
      MemoryManager<CameraParams>::AllocateArrayDevice(camera_params_);
}

CameraManager::~CameraManager() = default;

void CameraManager::LoadCameraParams() {
  std::vector<boost::filesystem::path> files;

  if (!boost::filesystem::exists(camera_params_dir_) ||
      !boost::filesystem::is_directory(camera_params_dir_)) {
    LOG(ERROR) << "Invalid camera parameters directory: "
               << camera_params_dir_.string();
    return;
  }

  // Iterate over the files in the directory and add them to the files vector.
  for (const auto& entry :
       boost::filesystem::directory_iterator(camera_params_dir_)) {
    if (boost::filesystem::is_regular_file(entry)) {
      files.push_back(entry.path());
      ++num_cameras_;
    }
  }

  // Sort the file paths alphabetically.
  std::sort(files.begin(), files.end());

  Eigen::Matrix4d t_0_0 =
      Camera(files[0]).GetCameraParams().transformation_matrix;

  // Assuming you have a std::vector<Eigen::Vector3d> camera_normals populated
  std::vector<Eigen::Vector3d> camera_normals;

  for (const auto& file : files) {
    Camera camera(file);
    CameraParams camera_params = camera.GetCameraParams();

    Eigen::Matrix4d camera_normal_tmp =
        t_0_0 * camera_params.transformation_matrix.inverse();

    camera_params.transformation_matrix =
        t_0_0 * camera_params.transformation_matrix.inverse() *
        camera_params.rectification_matrix.inverse();

    // Camera's viewing direction in camera space
    Eigen::Vector3d camera_view_direction(0.0, 0.0, 1.0);

    // Set camera_normal to point in the opposite direction and normalize
    camera_params.camera_normal =
        -(camera_normal_tmp.block<3, 3>(0, 0) * camera_view_direction)
             .normalized();
    camera_params.transformation_matrix =
        camera_params.transformation_matrix.inverse().eval();

    camera_params_.push_back(camera_params);

    camera_normals.push_back(camera_params.camera_normal);
  }
}

std::vector<CameraParams> CameraManager::GetCameraParams() const {
  return camera_params_;
}

CameraParams* CameraManager::GetDeviceCameraParams() const {
  return d_camera_params_;
}

}  // namespace surface_refinement
