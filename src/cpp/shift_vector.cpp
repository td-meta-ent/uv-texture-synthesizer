// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "shift_vector.hpp"

namespace surface_refinement {

ShiftVector::ShiftVector(boost::filesystem::path npy_file_path, int image_width,
                         double scale_factor)
    : npy_file_path_(std::move(npy_file_path)),
      image_width_(image_width),
      scale_factor_(scale_factor) {
  LoadVectorFromNpy();
  ComputeVectorMagnitude();
  CalculateDistance();
  LoadDistanceToDevice();
}

ShiftVector::~ShiftVector() { CUDA_ERROR_CHECK(cudaFree(d_distance_value_)); }

double ShiftVector::GetDistance() const { return distance_value_; }

double* ShiftVector::GetDeviceDistance() const { return d_distance_value_; }

void ShiftVector::LoadVectorFromNpy() {
  LOG(INFO) << "Loading vector from " << npy_file_path_.string() << "...";

  cnpy::NpyArray loaded_data = cnpy::npy_load(npy_file_path_.string());

  if (loaded_data.word_size != sizeof(double) ||
      loaded_data.shape.size() != 2 || loaded_data.shape[0] != 3 ||
      loaded_data.shape[1] != 1) {
    LOG(ERROR) << "Incorrect vector data format in " << npy_file_path_.string()
               << ".";
    throw std::runtime_error("Incorrect vector data format.");
  }

  std::memcpy(vector_data_.data(), loaded_data.data<double>(),
              loaded_data.num_bytes());
  LOG(INFO) << "Vector loaded successfully.";
}

void ShiftVector::ComputeVectorMagnitude() {
  vector_magnitude_ = vector_data_.norm();
}

void ShiftVector::CalculateDistance() {
  distance_value_ =
      vector_magnitude_ * static_cast<double>(image_width_) / scale_factor_;
}

void ShiftVector::LoadDistanceToDevice() {
  d_distance_value_ =
      MemoryManager<double>::AllocateScalarDevice(distance_value_);
}

}  // namespace surface_refinement
