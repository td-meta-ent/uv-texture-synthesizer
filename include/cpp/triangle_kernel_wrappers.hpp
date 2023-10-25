// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_TRIANGLE_KERNEL_WRAPPERS_HPP_
#define SURFACE_REFINEMENT_TRIANGLE_KERNEL_WRAPPERS_HPP_

#include <cuda_runtime.h>

#include <Eigen/Core>

namespace surface_refinement {

struct TriangleProperties {
  Eigen::Vector3d normal;
  Eigen::Vector3d curvature_normal_tmp;
  double mixed_area;
};

void LaunchInitializeTrianglePropertiesKernel(
    TriangleProperties* d_triangle_properties, int* d_num_triangles,
    int grid_size, int block_size);

void LaunchComputeTrianglePropertiesKernel(
    const Eigen::Vector3d* d_vertices, const Eigen::Vector3i* d_triangles,
    const int* d_num_triangles, TriangleProperties* d_triangle_properties,
    int grid_size, int block_size);

__global__ void InitializeTrianglePropertiesKernel(
    TriangleProperties* d_triangle_properties, const int* d_num_triangles);

__global__ void ComputeTrianglePropertiesKernel(
    const Eigen::Vector3d* d_vertices, const Eigen::Vector3i* d_triangles,
    const int* d_num_triangles, TriangleProperties* d_triangle_properties);

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_TRIANGLE_KERNEL_WRAPPERS_HPP_
