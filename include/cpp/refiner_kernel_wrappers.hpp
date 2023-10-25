// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_REFINER_KERNEL_WRAPPERS_HPP_
#define SURFACE_REFINEMENT_REFINER_KERNEL_WRAPPERS_HPP_

#include <cuda_runtime.h>

#include <Eigen/Core>

#include "triangle_kernel_wrappers.hpp"

namespace surface_refinement {

struct OneRingProperties {
  double mixed_area;
  Eigen::Vector3d normal;
  Eigen::Vector3d curvature_normal_operator;
  double mean_curvature;
  Eigen::Vector3d curvature_adjustment;
};

void LaunchInitializeOneRingPropertiesKernel(
    OneRingProperties* d_one_ring_properties, int* d_num_vertices,
    int grid_size, int block_size);

void LaunchComputeOneRingPropertiesKernel(
    const Eigen::Vector3d* d_vertices, const int* d_num_vertices,
    const Eigen::Vector3i* d_triangles, const int* d_num_triangles,
    const int* d_one_ring_indices, const int* d_one_ring_indices_row_lengths,
    TriangleProperties* d_triangle_properties,
    OneRingProperties* d_one_ring_properties,
    const double* d_coefficient_curvature, int grid_size, int block_size);

void LaunchUpdateCurvatureAdjustmentKernel(
    Eigen::Vector3d* d_vertices, const int* d_num_vertices,
    OneRingProperties* d_one_ring_properties, double* d_damping_value,
    const double* d_damping_factor, int grid_size, int block_size);

__global__ void InitializeOneRingPropertiesKernel(
    OneRingProperties* d_one_ring_properties, const int* d_num_vertices);

__global__ void ComputeOneRingPropertiesKernel(
    const Eigen::Vector3d* d_vertices, const int* d_num_vertices,
    const Eigen::Vector3i* d_triangles, const int* d_one_ring_indices,
    const int* d_one_ring_indices_row_lengths,
    TriangleProperties* d_triangle_properties,
    OneRingProperties* d_one_ring_properties,
    const double* d_coefficient_curvature);

__global__ void UpdateCurvatureAdjustmentKernel(
    Eigen::Vector3d* d_vertices, const int* d_num_vertices,
    OneRingProperties* d_one_ring_triangle_properties, double* d_damping_value,
    const double* d_damping_factor);

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_REFINER_KERNEL_WRAPPERS_HPP_
