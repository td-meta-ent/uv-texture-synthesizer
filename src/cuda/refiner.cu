// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "memory_manager.hpp"
#include "refiner_kernel_wrappers.hpp"
#include "triangle_kernel_wrappers.hpp"

namespace surface_refinement {

__global__ void InitializeOneRingPropertiesKernel(
    OneRingProperties* d_one_ring_properties, const int* d_num_vertices) {
  int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);

  if (idx < *d_num_vertices) {
    d_one_ring_properties[idx].normal = Eigen::Vector3d(0.0, 0.0, 0.0);
    d_one_ring_properties[idx].curvature_normal_operator =
        Eigen::Vector3d(0.0, 0.0, 0.0);
    d_one_ring_properties[idx].mean_curvature = 0.0;
    d_one_ring_properties[idx].curvature_normal_operator =
        Eigen::Vector3d(0.0, 0.0, 0.0);
  }
}

__global__ void ComputeOneRingPropertiesKernel(
    const Eigen::Vector3d* d_vertices, const int* d_num_vertices,
    const Eigen::Vector3i* d_triangles, const int* d_one_ring_indices,
    const int* d_one_ring_indices_row_lengths,
    TriangleProperties* d_triangle_properties,
    OneRingProperties* d_one_ring_properties,
    const double* d_coefficient_curvature) {
  int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);

  if (idx >= *d_num_vertices) return;

  int d_one_ring_triangle_indices_size = d_one_ring_indices_row_lengths[idx];
  Eigen::Vector3d d_vertex = d_vertices[idx];

  for (int i = 0; i < d_one_ring_triangle_indices_size; ++i) {
    int d_triangle_index = d_one_ring_indices[idx * 20 + i];
    Eigen::Vector3i d_triangle = d_triangles[d_triangle_index];

    int d_triangle_properties_index;
    if (d_vertices[idx][0] == d_vertices[d_triangle[0]][0] &&
        d_vertices[idx][1] == d_vertices[d_triangle[0]][1] &&
        d_vertices[idx][2] == d_vertices[d_triangle[0]][2]) {
      d_triangle_properties_index = 0;
    } else if (d_vertices[idx][0] == d_vertices[d_triangle[1]][0] &&
               d_vertices[idx][1] == d_vertices[d_triangle[1]][1] &&
               d_vertices[idx][2] == d_vertices[d_triangle[1]][2]) {
      d_triangle_properties_index = 1;
    } else if (d_vertices[idx][0] == d_vertices[d_triangle[2]][0] &&
               d_vertices[idx][1] == d_vertices[d_triangle[2]][1] &&
               d_vertices[idx][2] == d_vertices[d_triangle[2]][2]) {
      d_triangle_properties_index = 2;
    } else {
      printf("[CUDA Error] Vertex not found in triangle.\n");
      printf("[CUDA Debug] Vertex Index: %d | Coordinates: (%f, %f, %f)\n", idx,
             d_vertices[idx][0], d_vertices[idx][1], d_vertices[idx][2]);
      printf("[CUDA Debug] Triangle Indices: (%d, %d, %d)\n", d_triangle[0],
             d_triangle[1], d_triangle[2]);
    }

    d_one_ring_properties[idx].mixed_area +=
        d_triangle_properties[d_triangle_index * 3 +
                              d_triangle_properties_index]
            .mixed_area;
    d_one_ring_properties[idx].normal +=
        d_triangle_properties[d_triangle_index * 3 +
                              d_triangle_properties_index]
            .normal;
    d_one_ring_properties[idx].curvature_normal_operator +=
        d_triangle_properties[d_triangle_index * 3 +
                              d_triangle_properties_index]
            .curvature_normal_tmp;
  }
  if (d_one_ring_properties[idx].mixed_area == 0.0) {
    d_one_ring_properties[idx].mean_curvature = 0.0;
  } else {
    d_one_ring_properties[idx].normal.normalize();
    d_one_ring_properties[idx].curvature_normal_operator *=
        (1 / (2 * d_one_ring_properties[idx].mixed_area));
    d_one_ring_properties[idx].mean_curvature =
        0.5 * d_one_ring_properties[idx].curvature_normal_operator.dot(
                  d_one_ring_properties[idx].normal);
  }
  d_one_ring_properties[idx].curvature_adjustment =
      -d_one_ring_properties[idx].mean_curvature *
      d_one_ring_properties[idx].normal * *d_coefficient_curvature;
}

__global__ void UpdateCurvatureAdjustmentKernel(
    Eigen::Vector3d* d_vertices, const int* d_num_vertices,
    OneRingProperties* d_one_ring_triangle_properties, double* d_damping_value,
    const double* d_damping_factor) {
  int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);

  if (idx >= *d_num_vertices) return;

  d_vertices[idx] += *d_damping_value *
                     d_one_ring_triangle_properties[idx].curvature_adjustment;

  *d_damping_value *= *d_damping_factor;
}

void LaunchInitializeOneRingPropertiesKernel(
    OneRingProperties* d_one_ring_properties, int* d_num_vertices,
    int grid_size, int block_size) {
  InitializeOneRingPropertiesKernel<<<grid_size, block_size>>>(
      d_one_ring_properties, d_num_vertices);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK_LAST();
}

void LaunchComputeOneRingPropertiesKernel(
    const Eigen::Vector3d* d_vertices, const int* d_num_vertices,
    const Eigen::Vector3i* d_triangles, const int* d_num_triangles,
    const int* d_one_ring_indices, const int* d_one_ring_indices_row_lengths,
    TriangleProperties* d_triangle_properties,
    OneRingProperties* d_one_ring_properties,
    const double* d_coefficient_curvature, int grid_size, int block_size) {
  ComputeOneRingPropertiesKernel<<<grid_size, block_size>>>(
      d_vertices, d_num_vertices, d_triangles, d_one_ring_indices,
      d_one_ring_indices_row_lengths, d_triangle_properties,
      d_one_ring_properties, d_coefficient_curvature);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK_LAST();
}

void LaunchUpdateCurvatureAdjustmentKernel(
    Eigen::Vector3d* d_vertices, const int* d_num_vertices,
    OneRingProperties* d_one_ring_properties, double* d_damping_value,
    const double* d_damping_factor, int grid_size, int block_size) {
  UpdateCurvatureAdjustmentKernel<<<grid_size, block_size>>>(
      d_vertices, d_num_vertices, d_one_ring_properties, d_damping_value,
      d_damping_factor);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK_LAST();
}

}  // namespace surface_refinement
