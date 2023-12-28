#ifndef UV_TEXTURE_SYNTHESIZER_TEXTURE_HPP_
#define UV_TEXTURE_SYNTHESIZER_TEXTURE_HPP_

#include <glog/logging.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <highfive/H5Easy.hpp>
#include <string>

#include "memory_manager.hpp"

namespace uv_texture_synthesizer {

struct Triangle2D {
  Eigen::Vector2d vertex_a;
  Eigen::Vector2d vertex_b;
  Eigen::Vector2d vertex_c;
};

struct TexturePixelParams {
  int triangle_index;
  Eigen::Vector3d barycentric_coordinates;
};

class Texture {
 public:
  explicit Texture(boost::filesystem::path texture_pixel_info_path,
                   int num_cameras);

  static constexpr int imageHeight = 4096;
  static constexpr int imageWidth = 4096;

  ~Texture();

  [[nodiscard]] int* GetDeviceTextureImageHeight() const;

  [[nodiscard]] int* GetDeviceTextureImageWidth() const;

  [[nodiscard]] TexturePixelParams* GetDeviceTexturePixelParams() const;

  [[nodiscard]] Eigen::Vector3i* GetDeviceTextureImages() const;

  [[nodiscard]] double* GetDeviceCosineImages() const;

  [[nodiscard]] Eigen::Vector3i* GetDeviceInterpolatedTextureImage() const;

 private:
  int num_cameras_;

  void LoadTexturePixelInfo();

  void AllocateDeviceVariables();

  boost::filesystem::path texture_pixel_info_file_path_;

  int* d_texture_image_height_ = {nullptr};
  int* d_texture_image_width_ = {nullptr};

  std::vector<TexturePixelParams> texture_pixel_params_;
  TexturePixelParams* d_texture_pixel_params_ = {nullptr};

  std::vector<Eigen::Vector3i> texture_images_;
  Eigen::Vector3i* d_texture_images_ = {nullptr};

  std::vector<double> cosine_images_;
  double* d_cosine_images_ = {nullptr};

  std::vector<Eigen::Vector3i> interpolated_texture_image_;
  Eigen::Vector3i* d_interpolated_texture_image_ = {nullptr};

  void FreeDeviceMemory() const;
};

}  // namespace uv_texture_synthesizer

#endif  // UV_TEXTURE_SYNTHESIZER_TEXTURE_HPP_
