// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "image_manager.hpp"

namespace surface_refinement {

ImageManager::ImageManager(boost::filesystem::path image_directory,
                           const std::string& mode)
    : image_dir_(std::move(image_directory)) {
  if (mode == "linear_combination") {
    LoadImages();

    d_images_ = MemoryManager<double>::AllocateArrayDevice(images_double_);
    d_image_properties_ =
        MemoryManager<ImageProperties>::AllocateScalarDevice(image_properties_);
  }
}

ImageManager::~ImageManager() = default;

bool ImageManager::IsImageFile(const std::string& extension) {
  std::string lower_extension = extension;
  std::transform(lower_extension.begin(), lower_extension.end(),
                 lower_extension.begin(), ::tolower);
  return lower_extension == ".png" || lower_extension == ".jpg" ||
         lower_extension == ".jpeg" || lower_extension == ".tiff";
}

void ImageManager::LoadImages() {
  std::vector<boost::filesystem::path> files;

  if (!boost::filesystem::exists(image_dir_) ||
      !boost::filesystem::is_directory(image_dir_)) {
    LOG(ERROR) << "Invalid image directory: " << image_dir_.string();
    return;
  }

  // Iterate over the files in the directory and add them to the files vector.
  for (const auto& entry : boost::filesystem::directory_iterator(image_dir_)) {
    if (boost::filesystem::is_regular_file(entry) &&
        IsImageFile(entry.path().extension().string())) {
      files.push_back(entry.path());
    }
  }
  num_images_ = static_cast<int>(files.size());
  image_properties_.num_images = num_images_;

  // Reserve space in images_ and images_double_
  images_.reserve(num_images_);
  images_double_.reserve(image_properties_.rows * image_properties_.cols *
                         num_images_);

  // Sort the file paths alphabetically.
  std::sort(files.begin(), files.end());

  // Temporary storage for parallel results
  std::vector<std::vector<double>> temp_images_double(num_images_);

#pragma omp parallel for default(none) \
    shared(temp_images_double, num_images_, files)
  for (int i = 0; i < num_images_; ++i) {
    Image image(files[i]);
    temp_images_double[i] = image.GetImageGreenChannel();
  }

  // Assemble images_double_ in order
  for (const auto& img_double : temp_images_double) {
    images_double_.insert(images_double_.end(), img_double.begin(),
                          img_double.end());
  }
}

const std::vector<cv::Mat>& ImageManager::GetImages() const { return images_; }

double* ImageManager::GetDeviceImages() const { return d_images_; }

ImageProperties ImageManager::GetImageProperties() const {
  return image_properties_;
}

ImageProperties* ImageManager::GetDeviceImageProperties() const {
  return d_image_properties_;
}

}  // namespace surface_refinement
