#include "texture.hpp"

namespace uv_texture_synthesizer {

Texture::Texture(boost::filesystem::path texture_pixel_info_path,
                 const int num_cameras)
    : texture_pixel_info_file_path_(std::move(texture_pixel_info_path)),
      num_cameras_(num_cameras) {
  LoadTexturePixelInfo();
  AllocateDeviceVariables();
}

Texture::~Texture() { FreeDeviceMemory(); }

void Texture::FreeDeviceMemory() const {
  CUDA_ERROR_CHECK(cudaFree(d_texture_image_height_))
  CUDA_ERROR_CHECK(cudaFree(d_texture_image_width_))
  CUDA_ERROR_CHECK(cudaFree(d_texture_pixel_params_))
  CUDA_ERROR_CHECK(cudaFree(d_texture_images_))
  CUDA_ERROR_CHECK(cudaFree(d_cosine_images_))
  CUDA_ERROR_CHECK(cudaFree(d_interpolated_texture_image_))
}

void Texture::LoadTexturePixelInfo() {
  LOG(INFO) << "Loading texture pixel information from "
            << texture_pixel_info_file_path_.string() << "...";

  H5Easy::File file(texture_pixel_info_file_path_.string(),
                    H5Easy::File::ReadOnly);

  // Load triangle indices as a 2D array
  LOG(INFO) << "Loading triangle indices...";
  auto triangle_indices =
      H5Easy::load<std::vector<std::vector<int>>>(file, "triangle_index");

  // Load barycentric coordinates
  LOG(INFO) << "Loading barycentric coordinates...";
  auto barycentric_coordinates_raw =
      H5Easy::load<std::vector<std::vector<std::vector<double>>>>(
          file, "barycentric_coordinates");

  LOG(INFO) << "Populating texture pixel parameters...";
  texture_pixel_params_.reserve(triangle_indices.size() *
                                triangle_indices[0].size());
  for (size_t i = 0; i < triangle_indices.size(); ++i) {
    for (size_t j = 0; j < triangle_indices[i].size(); ++j) {
      TexturePixelParams params;
      params.triangle_index = triangle_indices[i][j];
      params.barycentric_coordinates =
          Eigen::Vector3d(barycentric_coordinates_raw[i][j][0],
                          barycentric_coordinates_raw[i][j][1],
                          barycentric_coordinates_raw[i][j][2]);
      texture_pixel_params_.push_back(params);
    }
  }

  LOG(INFO) << "Initializing texture and cosine images...";
  size_t imageSize = imageHeight * imageWidth * num_cameras_;
  texture_images_.resize(imageSize, Eigen::Vector3i(255, 255, 255));
  cosine_images_.resize(imageSize, 0.0);
  interpolated_texture_image_.resize(imageHeight * imageWidth,
                                     Eigen::Vector3i(255, 255, 255));

  LOG(INFO) << "Texture pixel information loaded successfully.";
}

void Texture::AllocateDeviceVariables() {
  d_texture_image_height_ =
      MemoryManager<int>::AllocateScalarDevice(imageHeight);
  d_texture_image_width_ = MemoryManager<int>::AllocateScalarDevice(imageWidth);

  d_texture_pixel_params_ =
      MemoryManager<TexturePixelParams>::AllocateArrayDevice(
          texture_pixel_params_);
  d_texture_images_ =
      MemoryManager<Eigen::Vector3i>::AllocateArrayDevice(texture_images_);
  d_cosine_images_ = MemoryManager<double>::AllocateArrayDevice(cosine_images_);
  d_interpolated_texture_image_ =
      MemoryManager<Eigen::Vector3i>::AllocateArrayDevice(
          interpolated_texture_image_);
}

int* Texture::GetDeviceTextureImageHeight() const {
  return d_texture_image_height_;
}

int* Texture::GetDeviceTextureImageWidth() const {
  return d_texture_image_width_;
}

TexturePixelParams* Texture::GetDeviceTexturePixelParams() const {
  return d_texture_pixel_params_;
}

Eigen::Vector3i* Texture::GetDeviceTextureImages() const {
  return d_texture_images_;
}

double* Texture::GetDeviceCosineImages() const { return d_cosine_images_; }

Eigen::Vector3i* Texture::GetDeviceInterpolatedTextureImage() const {
  return d_interpolated_texture_image_;
}

}  // namespace uv_texture_synthesizer
