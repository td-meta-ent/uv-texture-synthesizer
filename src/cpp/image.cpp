// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "image.hpp"

namespace surface_refinement {

Image::Image(boost::filesystem::path image_path)
    : image_file_path_(std::move(image_path)) {
  image_matrix_ = LoadImageToEigenMatrix();
  LoadImageToDevice();
}

Image::~Image() { CUDA_ERROR_CHECK(cudaFree(d_image_matrix_)); }

const Eigen::MatrixXd& Image::GetImageMatrix() const { return image_matrix_; }

const double* Image::GetDeviceImageMatrix() const { return d_image_matrix_; }

Eigen::MatrixXd Image::LoadImageToEigenMatrix() {
  LOG(INFO) << "Attempting to load image from: " << image_file_path_.string();

  if (!boost::filesystem::exists(image_file_path_)) {
    LOG(ERROR) << "Image file not located at: " << image_file_path_.string();
    throw std::runtime_error("Image file not found");
  }

  cv::Mat opencv_image =
      cv::imread(image_file_path_.string(), cv::IMREAD_COLOR);

  if (opencv_image.empty() || opencv_image.rows != kDefaultImageHeight ||
      opencv_image.cols != kDefaultImageWidth || opencv_image.channels() != 3) {
    LOG(ERROR) << "Image at " << image_file_path_.string()
               << " has incorrect dimensions or failed to load.";
    throw std::runtime_error(
        "Image dimensions are incorrect or loading failed");
  }

  Eigen::MatrixXd eigen_image_matrix(kDefaultImageHeight, kDefaultImageWidth);

  for (int row = 0; row < kDefaultImageHeight; ++row) {
    for (int col = 0; col < kDefaultImageWidth; ++col) {
      // Extract the green channel (BGR format in OpenCV)
      eigen_image_matrix(row, col) =
          static_cast<double>(opencv_image.at<cv::Vec3b>(row, col)[1]);
    }
  }

  return eigen_image_matrix;
}

void Image::LoadImageToDevice() {
  d_image_matrix_ = MemoryManager<double>::AllocateMatrixDevice(
      kDefaultImageHeight, kDefaultImageWidth, image_matrix_);
}

}  // namespace surface_refinement
