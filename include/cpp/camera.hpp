// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_CAMERA_HPP_
#define SURFACE_REFINEMENT_CAMERA_HPP_

#include <cnpy.h>
#include <glog/logging.h>

#include <boost/filesystem.hpp>
#include <utility>

#include "memory_manager.hpp"

namespace surface_refinement {

/**
 * @file camera.hpp
 * @class Camera
 * @brief Manages the camera parameters.
 * @details This class handles the camera parameters loaded from a .npy file and
 * provides accessors to them.
 */
class Camera {
 public:
  /**
   * @brief Constructs the Camera class and initializes camera parameters.
   *
   * @param camera_parameters_path Path to the camera parameters .npy file.
   */
  explicit Camera(boost::filesystem::path camera_parameters_path);
  ~Camera();

  /**
   * @brief Gets the x-coordinate of the focal length.
   * @return Focal length x-coordinate value.
   */
  [[nodiscard]] double GetFocalLengthX() const;

  /**
   * @brief Gets the y-coordinate of the focal length.
   * @return Focal length y-coordinate value.
   */
  [[nodiscard]] double GetFocalLengthY() const;

  /**
   * @brief Gets the x-coordinate of the principal point.
   * @return Principal point x-coordinate value.
   */
  [[nodiscard]] double GetPrincipalPointX() const;

  /**
   * @brief Gets the y-coordinate of the principal point.
   * @return Principal point y-coordinate value.
   */
  [[nodiscard]] double GetPrincipalPointY() const;

  /**
   * @brief Gets a pointer to the x-coordinate of the focal length stored in the
   * device memory.
   * @return Pointer to the device memory storing the focal length x-coordinate
   * value.
   */
  [[nodiscard]] double* GetDeviceFocalLengthX() const;

  /**
   * @brief Gets a pointer to the y-coordinate of the focal length stored in the
   * device memory.
   * @return Pointer to the device memory storing the focal length y-coordinate
   * value.
   */
  [[nodiscard]] double* GetDeviceFocalLengthY() const;

  /**
   * @brief Gets a pointer to the x-coordinate of the principal point stored in
   * the device memory.
   * @return Pointer to the device memory storing the principal point
   * x-coordinate value.
   */
  [[nodiscard]] double* GetDevicePrincipalPointX() const;

  /**
   * @brief Gets a pointer to the y-coordinate of the principal point stored in
   * the device memory.
   * @return Pointer to the device memory storing the principal point
   * y-coordinate value.
   */
  [[nodiscard]] double* GetDevicePrincipalPointY() const;

 private:
  /**
   * @brief Path to the camera parameters file (.npy format).
   */
  boost::filesystem::path camera_params_path_;

  /**
   * @brief X-coordinate of the camera's focal length.
   */
  double focal_length_x_{};

  /**
   * @brief Y-coordinate of the camera's focal length.
   */
  double focal_length_y_{};

  /**
   * @brief X-coordinate of the camera's principal point.
   */
  double principal_point_x_{};

  /**
   * @brief Y-coordinate of the camera's principal point.
   */
  double principal_point_y_{};

  /**
   * @brief Device pointer to the x-coordinate of the focal length.
   */
  double* device_focal_length_x_{nullptr};

  /**
   * @brief Device pointer to the y-coordinate of the focal length.
   */
  double* device_focal_length_y_{nullptr};

  /**
   * @brief Device pointer to the x-coordinate of the principal point.
   */
  double* device_principal_point_x_{nullptr};

  /**
   * @brief Device pointer to the y-coordinate of the principal point.
   */
  double* device_principal_point_y_{nullptr};

  /**
   * @brief Expected number of rows in the camera matrix.
   */
  static constexpr int kExpectedMatrixRows = 3;

  /**
   * @brief Expected number of columns in the camera matrix.
   */
  static constexpr int kExpectedMatrixCols = 4;

  /**
   * @brief Loads the camera parameters from a .npy file.
   *
   * The parameters include the focal length and the principal point
   * coordinates, which are stored in member variables for easy access.
   */
  void LoadCameraParameters();

  /**
   * @brief Allocates device memory for camera parameters.
   *
   * This method allocates memory on the device to store the focal length and
   * the principal point coordinates, facilitating their usage in CUDA kernels.
   */
  void AllocateDeviceVariables();

  /**
   * @brief Retrieves an element from a matrix.
   *
   * This method is used to access individual elements in the camera matrix
   * loaded from the .npy file.
   *
   * @param data Pointer to the matrix data.
   * @param row Row index.
   * @param col Column index.
   * @return Matrix element at the specified row and column.
   */
  static double GetMatrixElement(const double* data, int row, int col);
};

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_CAMERA_HPP_
