// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "memory_manager.hpp"
#include "one_ring_kernel_wrappers.hpp"
#include "photo_kernel_wrappers.hpp"
#include "refiner_kernel_wrappers.hpp"
#include "triangle_kernel_wrappers.hpp"

namespace surface_refinement {

__global__ void UpdateCurvatureAdjustmentKernel(
    Eigen::Vector3d *d_vertices, const int *d_num_vertices, const char *d_mode,
    OneRingProperties *d_one_ring_triangle_properties,
    PhotometricProperties *d_photometric_properties, double *d_surface_weight,
    double *d_damping_value, const double *d_damping_factor) {
  int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);

  if (idx < *d_num_vertices) {
    // Curvature only
    if (*d_mode == 'c') {
      d_vertices[idx] +=
          d_one_ring_triangle_properties[idx].curvature_adjustment;
    }

    // Linear combination
    if (*d_mode == 'l') {
      d_vertices[idx] +=
          (*d_surface_weight *
               d_one_ring_triangle_properties[idx].curvature_adjustment +
           d_photometric_properties[idx].weight_p *
               d_photometric_properties[idx].photometric_adjustment *
               *d_damping_value) /
          (*d_surface_weight + d_photometric_properties[idx].weight_p);
    }

    *d_damping_value *= *d_damping_factor;
  }
}

void LaunchUpdateCurvatureAdjustmentKernel(
    Eigen::Vector3d *d_vertices, const int *d_num_vertices, char *d_mode,
    OneRingProperties *d_one_ring_properties,
    PhotometricProperties *d_photometric_properties, double *d_surface_weight,
    double *d_damping_value, const double *d_damping_factor, int grid_size,
    int block_size) {
  UpdateCurvatureAdjustmentKernel<<<grid_size, block_size>>>(
      d_vertices, d_num_vertices, d_mode, d_one_ring_properties,
      d_photometric_properties, d_surface_weight, d_damping_value,
      d_damping_factor);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize())
  CUDA_CHECK_LAST()
}

}  // namespace surface_refinement
