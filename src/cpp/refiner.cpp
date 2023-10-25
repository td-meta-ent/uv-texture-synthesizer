// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "refiner.hpp"

#include "refiner_kernel_wrappers.hpp"
#include "triangle_kernel_wrappers.hpp"

namespace surface_refinement {

Refiner::Refiner(Eigen::Vector3d* d_vertices, int num_vertices,
                 int* d_num_vertices, Eigen::Vector3i* d_triangles,
                 int num_triangles, int* d_num_triangles,
                 TriangleProperties* d_triangle_properties,
                 OneRingProperties* d_one_ring_properties,
                 int* d_one_ring_indices, int* d_one_ring_indices_row_lengths,
                 const double* d_image_left, const double* d_image_right,
                 double* d_shift_distance, double damping_factor,
                 const std::string& mode, int num_iteration,
                 double refinement_resolution_delta,
                 double coefficient_curvature, double coefficient_photometric)
    : d_vertices_(d_vertices),
      num_vertices_(num_vertices),
      d_num_vertices_(d_num_vertices),
      d_triangles_(d_triangles),
      num_triangles_(num_triangles),
      d_num_triangles_(d_num_triangles),
      d_triangle_properties_(d_triangle_properties),
      d_one_ring_properties_(d_one_ring_properties),
      d_image_left_(d_image_left),
      d_image_right_(d_image_right),
      d_shift_distance_(d_shift_distance),
      d_one_ring_indices_(d_one_ring_indices),
      d_one_ring_indices_row_lengths_(d_one_ring_indices_row_lengths),

      num_iteration_(num_iteration),
      d_mode_(MemoryManager<char>::AllocateStringDevice(mode)),
      d_damping_value_(MemoryManager<double>::AllocateScalarDevice(1.0)),
      d_damping_factor_(
          MemoryManager<double>::AllocateScalarDevice(damping_factor)),
      d_refinement_resolution_delta_(
          MemoryManager<double>::AllocateScalarDevice(
              refinement_resolution_delta)),
      d_coefficient_curvature_(
          MemoryManager<double>::AllocateScalarDevice(coefficient_curvature)),
      d_coefficient_photometric_(MemoryManager<double>::AllocateScalarDevice(
          coefficient_photometric)) {}

Refiner::~Refiner() {
  CUDA_ERROR_CHECK(cudaFree(d_damping_value_));
  CUDA_ERROR_CHECK(cudaFree(d_damping_factor_));
  CUDA_ERROR_CHECK(cudaFree(d_mode_));
  CUDA_ERROR_CHECK(cudaFree(d_coefficient_curvature_));
  CUDA_ERROR_CHECK(cudaFree(d_coefficient_photometric_));
  CUDA_ERROR_CHECK(cudaFree(d_refinement_resolution_delta_));
}

std::vector<Eigen::Vector3d> Refiner::LaunchRefinement() {
  LOG(INFO) << "Start computing surface refinement adjustment...";

  size_t total_progress = num_iteration_;

  boost::timer::progress_display progress(total_progress);

  for (int it = 0; it < num_iteration_; ++it) {
    LOG(INFO) << "Iteration " << it + 1 << " / " << num_iteration_;
    int block_size = 1024;
    int grid_size = (num_triangles_ * 3 + block_size - 1) / block_size;
    LaunchInitializeTrianglePropertiesKernel(
        d_triangle_properties_, d_num_triangles_, grid_size, block_size);

    block_size = 1024 / 2;
    grid_size = (num_triangles_ * 3 + block_size - 1) / block_size;
    LaunchComputeTrianglePropertiesKernel(
        d_vertices_, d_triangles_, d_num_triangles_, d_triangle_properties_,
        grid_size, block_size);

    block_size = 1024;
    grid_size = (num_vertices_ + block_size - 1) / block_size;
    LaunchInitializeOneRingPropertiesKernel(
        d_one_ring_properties_, d_num_vertices_, grid_size, block_size);

    LaunchComputeOneRingPropertiesKernel(
        d_vertices_, d_num_vertices_, d_triangles_, d_num_triangles_,
        d_one_ring_indices_, d_one_ring_indices_row_lengths_,
        d_triangle_properties_, d_one_ring_properties_,
        d_coefficient_curvature_, grid_size, block_size);

    LaunchUpdateCurvatureAdjustmentKernel(
        d_vertices_, d_num_vertices_, d_one_ring_properties_, d_damping_value_,
        d_damping_factor_, grid_size, block_size);

    ++progress;
  }

  std::vector<Eigen::Vector3d> adjusted_vertices(num_vertices_);

  CUDA_ERROR_CHECK(cudaMemcpy(adjusted_vertices.data(), d_vertices_,
                              sizeof(Eigen::Vector3d) * num_vertices_,
                              cudaMemcpyDeviceToHost));

  LOG(INFO) << "Finished computing surface refinement adjustment!";

  return adjusted_vertices;
}

}  // namespace surface_refinement
