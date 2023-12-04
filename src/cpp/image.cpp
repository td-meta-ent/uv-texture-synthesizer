// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "image.hpp"

namespace surface_refinement {

Image::Image(boost::filesystem::path image_path)
    : image_file_path_(std::move(image_path)) {
  LoadImageToCvMat();
}

Image::~Image() = default;

const cv::Mat& Image::GetImage() const { return image_; }

std::vector<double> Image::GetImageGreenChannel() const {
  return image_green_channel_;
}

void Image::LoadImageToCvMat() {
  LOG(INFO) << "Loading image from: " << image_file_path_.string();

  if (!boost::filesystem::exists(image_file_path_)) {
    LOG(ERROR) << "Image file not located at: " << image_file_path_.string();
    throw std::runtime_error("Image file not found");
  }

  image_ = cv::imread(image_file_path_.string(), cv::IMREAD_COLOR);

  if (image_.empty() || image_.rows != imageHeight ||
      image_.cols != imageWidth || image_.channels() != 3) {
    LOG(ERROR) << "Image at " << image_file_path_.string()
               << " has incorrect dimensions or failed to load.";
    throw std::runtime_error(
        "Image dimensions are incorrect or loading failed");
  }

  cv::resize(image_, image_,
             cv::Size(image_.size().width / 2, image_.size().height / 2));
  cv::transpose(image_, image_);
  cv::flip(image_, image_, 0);

  if (image_.rows != rotatedImageHeight || image_.cols != rotatedImageWidth) {
    LOG(ERROR) << "Image at " << image_file_path_.string()
               << " has incorrect dimensions after resizing.";
    throw std::runtime_error("Image dimensions are incorrect after resizing");
  }

  std::vector<cv::Mat> channels(3);
  cv::split(image_, channels);

  cv::Mat image_green_channel;
  channels[1].convertTo(image_green_channel, CV_64F);
  image_green_channel_ = image_green_channel.reshape(1, 1);
}

}  // namespace surface_refinement
