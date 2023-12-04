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
#include <vector>

#include "memory_manager.hpp"

namespace surface_refinement {

class Image {
 public:
  static constexpr int imageHeight = 4096;

  static constexpr int rotatedImageHeight = 2798;
  //  static constexpr int rotatedImageHeight = 5596;

  static constexpr int imageWidth = 5596;

  static constexpr int rotatedImageWidth = 2048;
  //  static constexpr int rotatedImageWidth = 4096;

  explicit Image(boost::filesystem::path image_path);

  ~Image();

  [[nodiscard]] const cv::Mat& GetImage() const;

  [[nodiscard]] std::vector<double> GetImageGreenChannel() const;

 private:
  void LoadImageToCvMat();

  boost::filesystem::path image_file_path_;

  cv::Mat image_;

  std::vector<double> image_green_channel_;
};

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_IMAGE_HPP_
