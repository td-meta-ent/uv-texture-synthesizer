// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_REFINER_KERNEL_WRAPPERS_HPP_
#define SURFACE_REFINEMENT_REFINER_KERNEL_WRAPPERS_HPP_

#include <cuda_runtime.h>

#include <Eigen/Core>

#include "one_ring_kernel_wrappers.hpp"
#include "photo_kernel_wrappers.hpp"
#include "triangle_kernel_wrappers.hpp"

namespace surface_refinement {

void LaunchUpdateCurvatureAdjustmentKernel(
    Eigen::Vector3d *d_vertices, const int *d_num_vertices, char *d_mode,
    OneRingProperties *d_one_ring_properties,
    PhotometricProperties *d_photometric_properties, double *d_surface_weight,
    double *d_damping_value, const double *d_damping_factor, int grid_size,
    int block_size);

__global__ void UpdateCurvatureAdjustmentKernel(
    Eigen::Vector3d *d_vertices, const int *d_num_vertices, const char *d_mode,
    OneRingProperties *d_one_ring_triangle_properties,
    PhotometricProperties *d_photometric_properties, double *d_surface_weight,
    double *d_damping_value, const double *d_damping_factor);

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_REFINER_KERNEL_WRAPPERS_HPP_
