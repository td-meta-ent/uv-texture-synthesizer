// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_IMAGE_MANAGER_HPP_
#define SURFACE_REFINEMENT_IMAGE_MANAGER_HPP_

#include <glog/logging.h>
#include <omp.h>

#include <algorithm>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

#include "image.hpp"

namespace surface_refinement {

struct ImageProperties {
  int rows = Image::rotatedImageHeight;
  int cols = Image::rotatedImageWidth;
  int num_images;
};

class ImageManager {
 public:
  explicit ImageManager(boost::filesystem::path image_dir,
                        const std::string& mode);
  ~ImageManager();

  [[nodiscard]] const std::vector<cv::Mat>& GetImages() const;

  [[nodiscard]] double* GetDeviceImages() const;

  [[nodiscard]] ImageProperties GetImageProperties() const;

  [[nodiscard]] ImageProperties* GetDeviceImageProperties() const;

 private:
  void LoadImages();

  [[nodiscard]] static bool IsImageFile(const std::string& extension);

  boost::filesystem::path image_dir_;

  int num_images_ = 0;

  std::vector<cv::Mat> images_;
  std::vector<double> images_double_;
  double* d_images_ = nullptr;

  ImageProperties image_properties_{};
  ImageProperties* d_image_properties_ = nullptr;
};

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_IMAGE_MANAGER_HPP_
