#ifndef UV_TEXTURE_SYNTHESIZER_IMAGE_MANAGER_HPP_
#define UV_TEXTURE_SYNTHESIZER_IMAGE_MANAGER_HPP_

#include <glog/logging.h>
#include <omp.h>

#include <algorithm>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "image.hpp"

namespace uv_texture_synthesizer {

struct ImageProperties {
  int rows = Image::imageHeight;
  int cols = Image::imageWidth;
};

class ImageManager {
 public:
  ImageManager(boost::filesystem::path image_dir,
               const std::string& project_name, const std::string& date,
               const std::string& actor_name, const std::string& cut_number,
               const std::string& frame_number, const std::string& time_stamp);
  ~ImageManager();

  [[nodiscard]] const std::vector<cv::Mat>& GetImages() const;

  [[nodiscard]] Eigen::Vector3i* GetDeviceImages() const;

  [[nodiscard]] ImageProperties GetImageProperties() const;

  [[nodiscard]] ImageProperties* GetDeviceImageProperties() const;

  [[nodiscard]] int* GetDeviceImageHeight() const;

  [[nodiscard]] int* GetDeviceImageWidth() const;

 private:
  void LoadImages(const std::string& project_name, const std::string& date,
                  const std::string& actor_name, const std::string& cut_number,
                  const std::string& frame_number,
                  const std::string& time_stamp);

  boost::filesystem::path image_dir_;

  int* d_image_height_ = {nullptr};
  int* d_image_width_ = {nullptr};

  std::vector<cv::Mat> images_;
  std::vector<Eigen::Vector3i> images_eigen_;
  Eigen::Vector3i* d_images_ = {nullptr};

  ImageProperties image_properties_{};
  ImageProperties* d_image_properties_ = {nullptr};
};

}  // namespace uv_texture_synthesizer

#endif  // UV_TEXTURE_SYNTHESIZER_IMAGE_MANAGER_HPP_
