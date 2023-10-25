// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "camera.hpp"

namespace surface_refinement {

Camera::Camera(boost::filesystem::path camera_parameters_path)
    : camera_params_path_(std::move(camera_parameters_path)) {
  LoadCameraParameters();
  AllocateDeviceVariables();
}

Camera::~Camera() {
  CUDA_ERROR_CHECK(cudaFree(device_focal_length_x_));
  CUDA_ERROR_CHECK(cudaFree(device_focal_length_y_));
  CUDA_ERROR_CHECK(cudaFree(device_principal_point_x_));
  CUDA_ERROR_CHECK(cudaFree(device_principal_point_y_));
}

double Camera::GetFocalLengthX() const { return focal_length_x_; }

double Camera::GetFocalLengthY() const { return focal_length_y_; }

double Camera::GetPrincipalPointX() const { return principal_point_x_; }

double Camera::GetPrincipalPointY() const { return principal_point_y_; }

double* Camera::GetDeviceFocalLengthX() const { return device_focal_length_x_; }

double* Camera::GetDeviceFocalLengthY() const { return device_focal_length_y_; }

double* Camera::GetDevicePrincipalPointX() const {
  return device_principal_point_x_;
}

double* Camera::GetDevicePrincipalPointY() const {
  return device_principal_point_y_;
}

void Camera::LoadCameraParameters() {
  LOG(INFO) << "Loading camera parameters from: "
            << camera_params_path_.string();

  if (!boost::filesystem::exists(camera_params_path_)) {
    LOG(ERROR) << "Camera parameters file not found: "
               << camera_params_path_.string();
    throw std::invalid_argument("Camera parameters file not found.");
  }

  cnpy::NpyArray camera_matrix = cnpy::npy_load(camera_params_path_.string());
  auto camera_matrix_data = camera_matrix.data<double>();

  if (camera_matrix.shape[0] != kExpectedMatrixRows ||
      camera_matrix.shape[1] != kExpectedMatrixCols) {
    LOG(ERROR) << "Unexpected matrix dimensions for camera parameters in: "
               << camera_params_path_.string();
    throw std::runtime_error(
        "Unexpected matrix dimensions for camera parameters.");
  }

  focal_length_x_ = GetMatrixElement(camera_matrix_data, 0, 0);
  focal_length_y_ = GetMatrixElement(camera_matrix_data, 1, 1);
  principal_point_x_ = GetMatrixElement(camera_matrix_data, 0, 2);
  principal_point_y_ = GetMatrixElement(camera_matrix_data, 1, 2);
}

void Camera::AllocateDeviceVariables() {
  device_focal_length_x_ =
      MemoryManager<double>::AllocateScalarDevice(focal_length_x_);
  device_focal_length_y_ =
      MemoryManager<double>::AllocateScalarDevice(focal_length_y_);
  device_principal_point_x_ =
      MemoryManager<double>::AllocateScalarDevice(principal_point_x_);
  device_principal_point_y_ =
      MemoryManager<double>::AllocateScalarDevice(principal_point_y_);
}

double Camera::GetMatrixElement(const double* data, int row, int col) {
  return data[row * kExpectedMatrixCols + col];
}

}  // namespace surface_refinement
