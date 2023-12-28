#ifndef UV_TEXTURE_SYNTHESIZER_IMAGE_HPP_
#define UV_TEXTURE_SYNTHESIZER_IMAGE_HPP_

#include <glog/logging.h>

#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

#include "memory_manager.hpp"

namespace uv_texture_synthesizer {

class Image {
 public:
  static constexpr int imageHeight = 5120;
  static constexpr int imageWidth = 5120;

  explicit Image(boost::filesystem::path image_path);
  ~Image();

  [[nodiscard]] const cv::Mat& GetImage() const;
  [[nodiscard]] std::vector<Eigen::Vector3i> GetImageRGBChannel() const;

 private:
  void LoadImageToCvMat();
  boost::filesystem::path image_file_path_;
  cv::Mat image_;
  std::vector<Eigen::Vector3i> image_rgb_channel_;
};

}  // namespace uv_texture_synthesizer

#endif  // UV_TEXTURE_SYNTHESIZER_IMAGE_HPP_
