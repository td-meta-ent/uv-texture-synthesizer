#include "image.hpp"

namespace uv_texture_synthesizer {

Image::Image(boost::filesystem::path image_path)
    : image_file_path_(std::move(image_path)) {
  LoadImageToCvMat();
}

Image::~Image() = default;

const cv::Mat& Image::GetImage() const { return image_; }

std::vector<Eigen::Vector3i> Image::GetImageRGBChannel() const {
  return image_rgb_channel_;
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

  // Reserve space for image_rgb_channel_
  image_rgb_channel_.reserve(image_.rows * image_.cols);

  // Iterate over each pixel and save RGB data
  for (int y = 0; y < image_.rows; ++y) {
    for (int x = 0; x < image_.cols; ++x) {
      cv::Vec3b color = image_.at<cv::Vec3b>(y, x);
      // BGR to RGB
      Eigen::Vector3i pixel(color[2], color[1], color[0]);
      image_rgb_channel_.push_back(pixel);
    }
  }
}

}  // namespace uv_texture_synthesizer
