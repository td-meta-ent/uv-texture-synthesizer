// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "patch_metrics.cuh"

namespace surface_refinement {

__device__ float PatchMetrics::compute_normalized_cross_correlation(
    const Eigen::MatrixXf& patch1, const Eigen::MatrixXf& patch2) {
  float mean_patch1 = patch1.mean();
  float mean_patch2 = patch2.mean();

  Eigen::MatrixXf patch1_norm = patch1.array() - mean_patch1;
  Eigen::MatrixXf patch2_norm = patch2.array() - mean_patch2;

  float numerator = (patch1_norm.array() * patch2_norm.array()).sum();
  float denominator = std::sqrt((patch1_norm.array().square().sum()) *
                                (patch2_norm.array().square().sum()));

  return numerator / denominator;
}

__device__ float PatchMetrics::compute_photo_consistency_error(
    const Eigen::MatrixXf& patch_left, const Eigen::MatrixXf& patch_right) {
  float ncc = compute_normalized_cross_correlation(patch_left, patch_right);
  return (1.0f - ncc) / 2.0f;
}

__device__ float PatchMetrics::compute_delta_p(float error_minus_delta,
                                               float error_x,
                                               float error_plus_delta) {
  if (error_minus_delta < fmin(error_x, error_plus_delta)) {
    return -0.5f;
  } else if (error_x < fmin(error_minus_delta, error_plus_delta)) {
    return 0.5f * ((error_minus_delta - error_plus_delta) /
                   (error_minus_delta + error_plus_delta - 2.0f * error_x));
  } else if (error_plus_delta < fmin(error_minus_delta, error_x)) {
    return 0.5f;
  } else {
    return 0.0f;
  }
}

__device__ float PatchMetrics::compute_wp(float error_minus_delta,
                                          float error_x,
                                          float error_plus_delta) {
  if (error_minus_delta < fmin(error_x, error_plus_delta)) {
    return error_x - error_minus_delta;
  } else if (error_x < fmin(error_minus_delta, error_plus_delta)) {
    return 0.5f * (error_minus_delta + error_plus_delta - 2.0f * error_x);
  } else if (error_plus_delta < fmin(error_minus_delta, error_x)) {
    return error_x - error_plus_delta;
  } else {
    return 0.0f;
  }
}

}  // namespace surface_refinement
