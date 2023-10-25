// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_IMAGE_HPP_
#define SURFACE_REFINEMENT_IMAGE_HPP_

#include <glog/logging.h>

#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <utility>

#include "memory_manager.hpp"

namespace surface_refinement {

/**
 * @file image.hpp
 * @class Image
 * @brief Represents an image and provides associated utilities.
 * @details The Image class encapsulates image functionalities, bridging between
 * image formats and matrix representations suitable for numerical computations,
 * leveraging the Eigen library. It also handles CUDA device memory allocation
 * for the image matrix.
 */
class Image {
 public:
  /**
   * @brief Default height of the image.
   */
  static constexpr int kDefaultImageHeight = 3072;

  /**
   * @brief Default width of the image.
   */
  static constexpr int kDefaultImageWidth = 4096;

  /**
   * @brief Constructor to load an image from a given path.
   * @param image_path The path to the image file.
   */
  explicit Image(boost::filesystem::path image_path);

  /**
   * @brief Destructor that frees the CUDA device memory.
   */
  ~Image();

  /**
   * @brief Returns the image matrix.
   * @return Eigen::MatrixXd The image matrix.
   */
  [[nodiscard]] const Eigen::MatrixXd& GetImageMatrix() const;

  /**
   * @brief Returns a pointer to the CUDA device memory holding the image
   * matrix.
   * @return const double* Pointer to the device image matrix.
   */
  [[nodiscard]] const double* GetDeviceImageMatrix() const;

 private:
  /**
   * @brief Loads image from a path and converts to an Eigen Matrix.
   * @details Uses OpenCV for image loading and extracts the green channel to
   * populate the matrix.
   * @return Eigen::MatrixXd Loaded image matrix.
   */
  Eigen::MatrixXd LoadImageToEigenMatrix();

  /**
   * @brief Allocates CUDA device memory and copies the image matrix to the
   * device.
   */
  void LoadImageToDevice();

  /**
   * @brief The path to the image file.
   */
  boost::filesystem::path image_file_path_;

  /**
   * @brief Matrix representation of the image in host memory.
   */
  Eigen::MatrixXd image_matrix_;

  /**
   * @brief Pointer to the matrix representation of the image in CUDA device
   * memory.
   */
  double* d_image_matrix_{nullptr};
};

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_IMAGE_HPP_
