// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_CAMERA_MANAGER_HPP_
#define SURFACE_REFINEMENT_CAMERA_MANAGER_HPP_

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <utility>
#include <vector>

#include "camera.hpp"
#include "memory_manager.hpp"

namespace surface_refinement {

class CameraManager {
 public:
  explicit CameraManager(boost::filesystem::path camera_parameters_dir);
  ~CameraManager();

  [[nodiscard]] std::vector<CameraParams> GetCameraParams() const;

  [[nodiscard]] CameraParams* GetDeviceCameraParams() const;

 private:
  void LoadCameraParams();

  boost::filesystem::path camera_params_dir_;
  int num_cameras_ = 0;
  std::vector<CameraParams> camera_params_;

  CameraParams* d_camera_params_ = nullptr;
};

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_CAMERA_MANAGER_HPP_
