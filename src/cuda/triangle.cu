// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "memory_manager.hpp"
#include "triangle.cuh"
#include "triangle_kernel_wrappers.hpp"

namespace surface_refinement {

__global__ void ComputeTrianglePropertiesKernel(
    const Eigen::Vector3d* d_vertices, const Eigen::Vector3i* d_triangles,
    const int* d_num_triangles, TriangleProperties* d_triangle_properties) {
  int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);

  if (idx >= *d_num_triangles * 3) return;

  int d_triangle_index = idx / 3;
  int d_vertex_index = d_triangles[d_triangle_index][idx % 3];
  Eigen::Vector3i current_triangle = d_triangles[d_triangle_index];
  int vertex_indices[3] = {current_triangle[0], current_triangle[1],
                           current_triangle[2]};

  Eigen::Vector3d vertex_positions[3] = {d_vertices[vertex_indices[0]],
                                         d_vertices[vertex_indices[1]],
                                         d_vertices[vertex_indices[2]]};

  Eigen::Vector3d vertex_center, vertex_neighbor0, vertex_neighbor1;
  if (vertex_indices[0] == d_vertex_index) {
    vertex_center = vertex_positions[0];
    vertex_neighbor0 = vertex_positions[1];
    vertex_neighbor1 = vertex_positions[2];
  } else if (vertex_indices[1] == d_vertex_index) {
    vertex_center = vertex_positions[1];
    vertex_neighbor0 = vertex_positions[2];
    vertex_neighbor1 = vertex_positions[0];
  } else {
    vertex_center = vertex_positions[2];
    vertex_neighbor0 = vertex_positions[0];
    vertex_neighbor1 = vertex_positions[1];
  }

  Triangle d_triangle(vertex_center, vertex_neighbor0, vertex_neighbor1);
  d_triangle_properties[idx].normal = d_triangle.getNormal();

  double n0_opposite_side_length_squared =
      pow(GeometricUtils::compute_distance(vertex_center, vertex_neighbor1), 2);
  double n1_opposite_side_length_squared =
      pow(GeometricUtils::compute_distance(vertex_center, vertex_neighbor0), 2);

  if (GeometricUtils::is_angle_obtuse(d_triangle.getAngleA())) {
    double half_triangle_area = d_triangle.getArea() / 2;
    d_triangle_properties[idx].curvature_normal_tmp =
        half_triangle_area * d_triangle.getVectorN1ToA() /
            n0_opposite_side_length_squared +
        half_triangle_area * d_triangle.getVectorN0ToA() /
            n1_opposite_side_length_squared;

    d_triangle_properties[idx].mixed_area = half_triangle_area;
  } else {
    if (GeometricUtils::is_angle_obtuse(d_triangle.getAngleN0()) ||
        GeometricUtils::is_angle_obtuse(d_triangle.getAngleN1())) {
      double quarter_triangle_area = d_triangle.getArea() / 4;
      d_triangle_properties[idx].curvature_normal_tmp =
          quarter_triangle_area * d_triangle.getVectorN1ToA() /
              n0_opposite_side_length_squared +
          quarter_triangle_area * d_triangle.getVectorN0ToA() /
              n1_opposite_side_length_squared;

      d_triangle_properties[idx].mixed_area = quarter_triangle_area;
    } else {
      d_triangle_properties[idx].curvature_normal_tmp =
          d_triangle.getCotangentN0() * d_triangle.getVectorN1ToA() +
          d_triangle.getCotangentN1() * d_triangle.getVectorN0ToA();

      d_triangle_properties[idx].mixed_area =
          0.125 *
          (d_triangle.getCotangentN0() * n0_opposite_side_length_squared +
           d_triangle.getCotangentN1() * n1_opposite_side_length_squared);
    }
  }
}

__global__ void InitializeTrianglePropertiesKernel(
    TriangleProperties* d_triangle_properties, const int* d_num_triangles) {
  int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);

  if (idx < *d_num_triangles * 3) {
    d_triangle_properties[idx].normal = Eigen::Vector3d(0.0, 0.0, 0.0);
    d_triangle_properties[idx].curvature_normal_tmp =
        Eigen::Vector3d(0.0, 0.0, 0.0);
    d_triangle_properties[idx].mixed_area = 0.0;
  }
}

void LaunchInitializeTrianglePropertiesKernel(
    TriangleProperties* d_triangle_properties, int* d_num_triangles,
    int grid_size, int block_size) {
  InitializeTrianglePropertiesKernel<<<grid_size, block_size>>>(
      d_triangle_properties, d_num_triangles);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK_LAST();
}

void LaunchComputeTrianglePropertiesKernel(
    const Eigen::Vector3d* d_vertices, const Eigen::Vector3i* d_triangles,
    const int* d_num_triangles, TriangleProperties* d_triangle_properties,
    int grid_size, int block_size) {
  ComputeTrianglePropertiesKernel<<<grid_size, block_size>>>(
      d_vertices, d_triangles, d_num_triangles, d_triangle_properties);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK_LAST();
}

}  // namespace surface_refinement
