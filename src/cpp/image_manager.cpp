#include "image_manager.hpp"

namespace uv_texture_synthesizer {

ImageManager::ImageManager(boost::filesystem::path image_directory,
                           const std::string& project_name,
                           const std::string& date,
                           const std::string& actor_name,
                           const std::string& cut_number,
                           const std::string& frame_number,
                           const std::string& time_stamp)
    : image_dir_(std::move(image_directory)) {
  LoadImages(project_name, date, actor_name, cut_number, frame_number,
             time_stamp);

  d_images_ =
      MemoryManager<Eigen::Vector3i>::AllocateArrayDevice(images_eigen_);
  d_image_properties_ =
      MemoryManager<ImageProperties>::AllocateScalarDevice(image_properties_);
  d_image_height_ =
      MemoryManager<int>::AllocateScalarDevice(image_properties_.rows);
  d_image_width_ =
      MemoryManager<int>::AllocateScalarDevice(image_properties_.cols);
}

ImageManager::~ImageManager() {
  CUDA_ERROR_CHECK(cudaFree(d_images_))
  CUDA_ERROR_CHECK(cudaFree(d_image_properties_))
  CUDA_ERROR_CHECK(cudaFree(d_image_height_))
  CUDA_ERROR_CHECK(cudaFree(d_image_width_))
}

void ImageManager::LoadImages(const std::string& project_name,
                              const std::string& date,
                              const std::string& actor_name,
                              const std::string& cut_number,
                              const std::string& frame_number,
                              const std::string& time_stamp) {
  std::vector<boost::filesystem::path> files;

  if (!boost::filesystem::exists(image_dir_) ||
      !boost::filesystem::is_directory(image_dir_)) {
    LOG(ERROR) << "Invalid image directory: " << image_dir_.string();
    return;
  }

  for (const auto& entry : boost::filesystem::directory_iterator(image_dir_)) {
    if (!boost::filesystem::is_regular_file(entry)) continue;

    std::string pattern = "rectification_";
    pattern += project_name;
    pattern += "_";
    pattern += date;
    pattern += "_";
    pattern += actor_name;
    pattern += "_cut";
    pattern += cut_number;
    pattern += "_";
    pattern += "\\d"; // Wildcard for one digit
    pattern += "_";
    pattern += frame_number;
    pattern += "_";
    pattern += time_stamp;
    pattern += "\\.tiff"; // Escape dot to match it literally

    std::string filename = entry.path().filename().string();

    std::regex filename_regex(pattern);
    if (std::regex_match(filename, filename_regex)) {
      files.push_back(entry.path());
    }
  }

  // Sort the file paths alphabetically
  std::sort(files.begin(), files.end());

  // Reserve space in images_ and images_eigen_
  images_.reserve(files.size());
  images_eigen_.reserve(image_properties_.rows * image_properties_.cols * 3 *
                        files.size());

  // Temporary storage for parallel results
  std::vector<std::vector<Eigen::Vector3i>> temp_images_eigen(files.size());

#pragma omp parallel for default(none) shared(temp_images_eigen, files)
  for (int i = 0; i < files.size(); ++i) {
    Image image(files[i]);
    images_.push_back(image.GetImage());

    std::vector<Eigen::Vector3i> image_rgb_channel = image.GetImageRGBChannel();

    // Reserve space for the number of pixels
    std::vector<Eigen::Vector3i> image_eigen;
    image_eigen.reserve(image_rgb_channel.size());

    // Directly use the image_rgb_channel data
    image_eigen = std::move(image_rgb_channel);

    temp_images_eigen[i] = std::move(image_eigen);
  }

  // Assemble images_eigen_ in order
  images_eigen_.clear();
  for (const auto& img_eigen : temp_images_eigen) {
    images_eigen_.insert(images_eigen_.end(), img_eigen.begin(),
                         img_eigen.end());
  }
}

const std::vector<cv::Mat>& ImageManager::GetImages() const { return images_; }

Eigen::Vector3i* ImageManager::GetDeviceImages() const { return d_images_; }

ImageProperties ImageManager::GetImageProperties() const {
  return image_properties_;
}

ImageProperties* ImageManager::GetDeviceImageProperties() const {
  return d_image_properties_;
}

int* ImageManager::GetDeviceImageHeight() const { return d_image_height_; }

int* ImageManager::GetDeviceImageWidth() const { return d_image_width_; }

}  // namespace uv_texture_synthesizer
