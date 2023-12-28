#include "synthesizer.hpp"

namespace uv_texture_synthesizer {

Synthesizer::Synthesizer(const Mesh &mesh, const CameraManager &camera_manager,
                         const ImageManager &image_manager,
                         const Texture &texture, const int num_cameras)
    : mesh_(mesh),
      camera_manager_(camera_manager),
      image_manager_(image_manager),
      texture_(texture),
      num_cameras_(num_cameras) {
  d_num_cameras_ = MemoryManager<int>::AllocateScalarDevice(num_cameras_);
}

Synthesizer::~Synthesizer() {
  // Cleanup if needed
}

std::vector<Eigen::Vector3d> Synthesizer::LaunchSynthesis() {
  LOG(INFO) << "Starting synthesis process...";

  std::vector<Eigen::Vector3d> synthesized_data;

  int block_size;
  int grid_size;

  block_size = 1024 / 2;
  grid_size = (uv_texture_synthesizer::Texture::imageHeight *
                   uv_texture_synthesizer::Texture::imageWidth * num_cameras_ +
               block_size - 1) /
              block_size;

  LaunchComputeOneRingFilterProperties(
      d_num_cameras_, mesh_.GetDeviceVertices(), mesh_.GetDeviceVertexNormals(),
      mesh_.GetDeviceNumVertices(), camera_manager_.GetDeviceCameraParams(),
      mesh_.GetDeviceCentroid(), mesh_.GetDeviceOneRingProperties(), grid_size,
      block_size);

  block_size = 1024 / 2;
  grid_size = (uv_texture_synthesizer::Texture::imageHeight *
                   uv_texture_synthesizer::Texture::imageWidth * num_cameras_ +
               block_size - 1) /
              block_size;

  LaunchGenerateTextureImages(
      texture_.GetDeviceTexturePixelParams(), d_num_cameras_,
      mesh_.GetDeviceVertices(), mesh_.GetDeviceNumVertices(),
      mesh_.GetDeviceTriangles(), camera_manager_.GetDeviceCameraParams(),
      image_manager_.GetDeviceImages(), texture_.GetDeviceTextureImages(),
      texture_.GetDeviceCosineImages(), mesh_.GetDeviceOneRingProperties(),
      texture_.GetDeviceTextureImageHeight(),
      texture_.GetDeviceTextureImageWidth(),
      image_manager_.GetDeviceImageHeight(),
      image_manager_.GetDeviceImageWidth(), grid_size, block_size);

  const int texture_width = 4096;
  const int texture_height = 4096;
  const int num_images = 10;

  // Allocate memory for all images
  auto *h_texture_images =
      new Eigen::Vector3i[texture_width * texture_height * num_images];
  CUDA_ERROR_CHECK(cudaMemcpy(
      h_texture_images, texture_.GetDeviceTextureImages(),
      sizeof(Eigen::Vector3i) * texture_width * texture_height * num_images,
      cudaMemcpyDeviceToHost));

  for (int img = 0; img < num_images; img++) {
    cv::Mat image(texture_height, texture_width, CV_8UC3);
    for (int i = 0; i < texture_height; i++) {
      for (int j = 0; j < texture_width; j++) {
        // Calculate index considering the current image
        int index =
            img * texture_width * texture_height + i * texture_width + j;
        image.at<cv::Vec3b>(i, j)[2] = h_texture_images[index][0];  // Red
        image.at<cv::Vec3b>(i, j)[1] = h_texture_images[index][1];  // Green
        image.at<cv::Vec3b>(i, j)[0] = h_texture_images[index][2];  // Blue
      }
    }

    // Construct the filename
    std::string filename =
        "/root/surface-refinement/data/output/v6.0.0/check_texture_" +
        std::to_string(img) + ".png";
    cv::imwrite(filename, image);
  }
  delete[] h_texture_images;

  block_size = 1024;
  grid_size = (uv_texture_synthesizer::Texture::imageHeight *
                   uv_texture_synthesizer::Texture::imageWidth +
               block_size - 1) /
              block_size;

  LaunchComputeInterpolatedTextureImage(
      d_num_cameras_, texture_.GetDeviceTextureImages(),
      texture_.GetDeviceCosineImages(),
      texture_.GetDeviceInterpolatedTextureImage(),
      texture_.GetDeviceTextureImageHeight(),
      texture_.GetDeviceTextureImageWidth(), grid_size, block_size);

  auto *h_interpolated_texture_image =
      new Eigen::Vector3i[texture_width * texture_height];
  CUDA_ERROR_CHECK(
      cudaMemcpy(h_interpolated_texture_image,
                 texture_.GetDeviceInterpolatedTextureImage(),
                 sizeof(Eigen::Vector3i) * texture_width * texture_height,
                 cudaMemcpyDeviceToHost));
  cv::Mat image(texture_height, texture_width, CV_8UC3);
  for (int i = 0; i < texture_height; i++) {
    for (int j = 0; j < texture_width; j++) {
      // Calculate index considering the current image
      int index = i * texture_width + j;
      image.at<cv::Vec3b>(i, j)[2] =
          h_interpolated_texture_image[index][0];  // Red
      image.at<cv::Vec3b>(i, j)[1] =
          h_interpolated_texture_image[index][1];  // Green
      image.at<cv::Vec3b>(i, j)[0] =
          h_interpolated_texture_image[index][2];  // Blue
    }
  }
  std::string filename =
      "/root/surface-refinement/data/output/v6.0.0/"
      "intertpolated_texture_image.png";
  cv::imwrite(filename, image);
  delete[] h_interpolated_texture_image;

  LOG(INFO) << "Synthesis process completed.";

  return synthesized_data;
}

}  // namespace uv_texture_synthesizer
