// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_SHIFT_VECTOR_HPP_
#define SURFACE_REFINEMENT_SHIFT_VECTOR_HPP_

#include <cnpy.h>
#include <glog/logging.h>

#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <utility>

#include "memory_manager.hpp"

namespace surface_refinement {

/**
 * @file shift_vector.hpp
 * @class ShiftVector
 * @brief Class to load a vector from a file and compute its magnitude and
 * distance value.
 * @details The ShiftVector class facilitates the loading of a vector from a
 * .npy file, and performs operations to calculate the magnitude of the vector
 * and a distance value derived from this magnitude.
 */
class ShiftVector {
 public:
  /**
   * @brief Constructor to initialize ShiftVector class with file path, image
   * width, and scale factor.
   *
   * @param npy_file_path Path to the .npy file which contains the vector data.
   * @param image_width The width of the image.
   * @param scale_factor Scale factor to be used in distance calculation.
   */
  explicit ShiftVector(boost::filesystem::path npy_file_path, int image_width,
                       double scale_factor);

  /**
   * @brief Destructor to free CUDA memory associated with the distance value.
   */
  ~ShiftVector();

  /**
   * @brief Get the calculated distance value.
   *
   * @return The calculated distance value.
   */
  [[nodiscard]] double GetDistance() const;

  /**
   * @brief Get the pointer to the device memory where the distance value is
   * stored.
   *
   * @return Pointer to the device memory where the distance value is stored.
   */
  [[nodiscard]] double* GetDeviceDistance() const;

 private:
  /**
   * @brief Load the vector data from a .npy file.
   */
  void LoadVectorFromNpy();

  /**
   * @brief Compute the magnitude of the loaded vector.
   */
  void ComputeVectorMagnitude();

  /**
   * @brief Calculate the distance value based on the vector magnitude.
   */
  void CalculateDistance();

  /**
   * @brief Load the calculated distance value to CUDA device memory.
   */
  void LoadDistanceToDevice();

  /**
   * @brief Path to the .npy file containing vector data.
   */
  boost::filesystem::path npy_file_path_;

  /**
   * @brief Width of the image.
   */
  int image_width_;

  /**
   * @brief Scale factor for distance calculation.
   */
  double scale_factor_;

  /**
   * @brief Loaded vector data.
   */
  Eigen::Vector3d vector_data_;

  /**
   * @brief Magnitude of the vector data.
   */
  double vector_magnitude_{0.0};

  /**
   * @brief Calculated distance value.
   */
  double distance_value_{0.0};

  /**
   * @brief Pointer to device memory for storing distance value.
   */
  double* d_distance_value_{nullptr};
};

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_SHIFT_VECTOR_HPP_
