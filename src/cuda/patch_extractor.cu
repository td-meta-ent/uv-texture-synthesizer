// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_PATCH_EXTRACTOR_H_
#define SURFACE_REFINEMENT_PATCH_EXTRACTOR_H_

#include <cuda_runtime.h>

#include <Eigen/Dense>

namespace surface_refinement {

class PatchExtractor {
 public:
  __device__ PatchExtractor(const Eigen::MatrixXf& image,
                            const Eigen::Vector2i& patch_dim)
      : image_(image), patch_dim_(patch_dim) {}

  __device__ Eigen::MatrixXf ExtractPatch(const Eigen::Vector2f& center_point);

 private:
  Eigen::MatrixXf image_;
  Eigen::Vector2i patch_dim_;
};

__device__ int d_errorFlag = 0;

__device__ Eigen::MatrixXf PatchExtractor::ExtractPatch(
    const Eigen::Vector2f& center_point) {
  int center_x = static_cast<int>(round(center_point(0)));
  int center_y = static_cast<int>(round(center_point(1)));

  float frac_x = center_point(0) - static_cast<float>(center_x);
  float frac_y = center_point(1) - static_cast<float>(center_y);

  int direction_x = frac_x < 0 ? -1 : 1;
  int direction_y = frac_y < 0 ? -1 : 1;

  frac_x = abs(frac_x);
  frac_y = abs(frac_y);

  float x = frac_x, y = 1 - frac_x, r = frac_y, s = 1 - frac_y;
  float weight_a = x * r, weight_b = y * r, weight_c = x * s, weight_d = y * s;
  float weight_sum = weight_a + weight_b + weight_c + weight_d;

  // Normalize weights
  weight_a /= weight_sum;
  weight_b /= weight_sum;
  weight_c /= weight_sum;
  weight_d /= weight_sum;

  int patch_width = patch_dim_(1);
  int patch_height = patch_dim_(0);

  // Compute patch boundaries
  int patch_left = center_x - patch_width / 2;
  int patch_right = patch_left + patch_width;
  int patch_top = center_y - patch_height / 2;
  int patch_bottom = patch_top + patch_height;

  Eigen::MatrixXf patch(patch_height, patch_width);

  for (int i = patch_left; i < patch_right; ++i) {
    for (int j = patch_top; j < patch_bottom; ++j) {
      int current_x = i;
      int current_y = j;

      // Check boundaries
      if (current_x < 0 || current_x >= image_.cols() || current_y < 0 ||
          current_y >= image_.rows()) {
        printf("Error: Invalid coordinates x=%d, y=%d | File: %s | Line: %d ",
               current_x, current_y, __FILE__, __LINE__);
        d_errorFlag = 1;
        return patch;  // Return an empty or undefined patch.
      }

      // Interpolation
      float color_a = image_(current_x, current_y);
      float color_b = image_(current_x + direction_x, current_y);
      float color_c = image_(current_x, current_y + direction_y);
      float color_d = image_(current_x + direction_x, current_y + direction_y);

      patch(j, i) = color_a * weight_a + color_b * weight_b +
                    color_c * weight_c + color_d * weight_d;
    }
  }

  return patch;
}

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_PATCH_EXTRACTOR_H_
