// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_IMAGE_PATCH_METRICS_CUH_
#define SURFACE_REFINEMENT_IMAGE_PATCH_METRICS_CUH_

#include <Eigen/Core>

namespace surface_refinement {

/**
 * @file patch_metrics.cuh
 * @class PatchMetrics
 * @brief Class that performs computations on image patches.
 * @details This class provides utility functions to perform various metric
 * computations on image patches. It supports operations such as summing matrix
 * elements, computing normalized cross-correlation (NCC) between two image
 * patches, and evaluating photo-consistency errors of a given point among other
 * utilities.
 */
class PatchMetrics {
 public:
  /**
   * @brief Computes the photo-consistency error of a given point.
   *
   * @param patch1 The left image patch.
   * @param patch2 The right image patch.
   * @return The computed photo-consistency error as (1 - NCC) / 2.
   */
  __device__ static double compute_photo_consistency_error(
      const double *patch1, const double *patch2);

  /**
   * @brief Computes the delta_p value based on given error values.
   *
   * @param error_minus_delta The error value for the point slightly shifted
   * backwards.
   * @param error_x The error value for the current point.
   * @param error_plus_delta The error value for the point slightly shifted
   * forwards.
   * @return The computed delta_p value.
   */
  __device__ static double compute_photometric_delta(double error_minus_delta,
                                                     double error_x,
                                                     double error_plus_delta);

  /**
   * @brief Computes the wp value based on given error values.
   *
   * @param error_minus_delta The error value for the point slightly shifted
   * backwards.
   * @param error_x The error value for the current point.
   * @param error_plus_delta The error value for the point slightly shifted
   * forwards.
   * @return The computed wp value.
   */
  __device__ static double compute_photometric_weight(double error_minus_delta,
                                                      double error_x,
                                                      double error_plus_delta);
};

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_IMAGE_PATCH_METRICS_CUH_
