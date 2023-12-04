// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "patch_metrics.cuh"

namespace surface_refinement {

__device__ double PatchMetrics::compute_photo_consistency_error(
    const double* patch1, const double* patch2) {
  const int patch_size = 3 * 3;

  // Compute means
  double mean_patch1 = 0.0;
  double mean_patch2 = 0.0;
  for (int i = 0; i < patch_size; ++i) {
    mean_patch1 += patch1[i];
    mean_patch2 += patch2[i];
  }
  mean_patch1 /= patch_size;
  mean_patch2 /= patch_size;

  // Compute normalized patches
  double patch1_norm[patch_size];
  double patch2_norm[patch_size];
  for (int i = 0; i < patch_size; ++i) {
    patch1_norm[i] = patch1[i] - mean_patch1;
    patch2_norm[i] = patch2[i] - mean_patch2;
  }

  // Compute numerator and denominator
  double numerator = 0.0;
  double sum_square_patch1 = 0.0;
  double sum_square_patch2 = 0.0;
  for (int i = 0; i < patch_size; ++i) {
    numerator += patch1_norm[i] * patch2_norm[i];
    sum_square_patch1 += patch1_norm[i] * patch1_norm[i];
    sum_square_patch2 += patch2_norm[i] * patch2_norm[i];
  }

  double denominator = sqrt(sum_square_patch1 * sum_square_patch2);

  double ncc;
  if (denominator == 0.0) {
    ncc = 0.0;
  } else {
    ncc = numerator / denominator;
  }

  return (1.0 - ncc) / 2.0;
}

__device__ double PatchMetrics::compute_photometric_delta(
    double error_minus_delta, double error_x, double error_plus_delta) {
  if (error_minus_delta < fmin(error_x, error_plus_delta)) {
    return -0.5;
  } else if (error_x < fmin(error_minus_delta, error_plus_delta)) {
    return 0.5 * ((error_minus_delta - error_plus_delta) /
                  (error_minus_delta + error_plus_delta - 2.0 * error_x));
  } else if (error_plus_delta < fmin(error_minus_delta, error_x)) {
    return 0.5;
  } else {
    return fmin(error_minus_delta, fmin(error_x, error_plus_delta));
  }
}

__device__ double PatchMetrics::compute_photometric_weight(
    double error_minus_delta, double error_x, double error_plus_delta) {
  if (error_minus_delta < fmin(error_x, error_plus_delta)) {
    return error_x - error_minus_delta;
  } else if (error_x < fmin(error_minus_delta, error_plus_delta)) {
    return 0.5 * (error_minus_delta + error_plus_delta - 2.0 * error_x);
  } else if (error_plus_delta < fmin(error_minus_delta, error_x)) {
    return error_x - error_plus_delta;
  } else {
    return fmin(error_minus_delta, fmin(error_x, error_plus_delta));
  }
}

}  // namespace surface_refinement
