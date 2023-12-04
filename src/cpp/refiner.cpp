// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "refiner.hpp"

namespace surface_refinement {

Refiner::Refiner(const Mesh &mesh, const CameraManager &camera_manager,
                 const ImageManager &image_manager, double damping_factor,
                 const std::string &mode, int num_iteration,
                 double refinement_resolution_delta, double surface_weight,
                 double curvature_coefficient, double photometric_coefficient)
    : mesh_(mesh),
      d_vertices_(mesh.GetDeviceVertices()),
      num_vertices_(mesh.GetNumVertices()),
      d_num_vertices_(mesh.GetDeviceNumVertices()),
      d_triangles_(mesh.GetDeviceTriangles()),
      num_triangles_(mesh.GetNumTriangles()),
      d_num_triangles_(mesh.GetDeviceNumTriangles()),

      d_triangle_properties_(mesh.GetDeviceTriangleProperties()),
      d_one_ring_properties_(mesh.GetDeviceOneRingProperties()),
      d_photometric_properties_(mesh.GetDevicePhotometricProperties()),
      d_one_ring_indices_(mesh.GetDeviceOneRingIndices()),
      d_one_ring_indices_row_lengths_(mesh.GetDeviceOneRingIndicesRowLengths()),
      d_projected_pixel_indices_(mesh.GetDeviceProjectedPixelIndices()),
      d_patches_(mesh.GetDevicePatches()),
      d_delta_vertices_(mesh.GetDeviceDeltaVertices()),

      image_manager_(image_manager),

      camera_manager_(camera_manager),

      num_iteration_(num_iteration),
      mode_(mode),
      d_mode_(MemoryManager<char>::AllocateStringDevice(mode)),
      d_damping_value_(MemoryManager<double>::AllocateScalarDevice(1.0)),
      d_damping_factor_(
          MemoryManager<double>::AllocateScalarDevice(damping_factor)),
      d_refinement_resolution_delta_(
          MemoryManager<double>::AllocateScalarDevice(
              refinement_resolution_delta)),
      d_surface_weight_(
          MemoryManager<double>::AllocateScalarDevice(surface_weight)),
      d_curvature_coefficient_(
          MemoryManager<double>::AllocateScalarDevice(curvature_coefficient)),
      d_photometric_coefficient_(MemoryManager<double>::AllocateScalarDevice(
          photometric_coefficient)) {
  std::vector<double> vertex_camera_angles(
      num_vertices_ * image_manager_.GetImageProperties().num_images, M_PI);
  d_vertex_camera_angles_ =
      MemoryManager<double>::AllocateArrayDevice(vertex_camera_angles);
}

Refiner::~Refiner() {
  CUDA_ERROR_CHECK(cudaFree(d_vertex_camera_angles_));
  CUDA_ERROR_CHECK(cudaFree(d_damping_value_));
  CUDA_ERROR_CHECK(cudaFree(d_damping_factor_));
  CUDA_ERROR_CHECK(cudaFree(d_mode_));
  CUDA_ERROR_CHECK(cudaFree(d_surface_weight_));
  CUDA_ERROR_CHECK(cudaFree(d_curvature_coefficient_));
  CUDA_ERROR_CHECK(cudaFree(d_photometric_coefficient_));
  CUDA_ERROR_CHECK(cudaFree(d_refinement_resolution_delta_));
}

std::vector<Eigen::Vector3d> Refiner::LaunchRefinement() {
  LOG(INFO) << "Start computing surface refinement adjustment...";

  size_t total_progress = num_iteration_;

  boost::timer::progress_display progress(total_progress);

  int block_size;
  int grid_size;

  for (int it = 0; it < num_iteration_; ++it) {
    LOG(INFO) << "\nIteration " << it + 1 << " / " << num_iteration_;
    block_size = 1024;
    grid_size = (num_triangles_ * 3 + block_size - 1) / block_size;
    LaunchInitializeTrianglePropertiesKernel(
        d_triangle_properties_, d_num_triangles_, grid_size, block_size);

    block_size = 1024 / 2;
    grid_size = (num_triangles_ * 3 + block_size - 1) / block_size;
    LaunchComputeTrianglePropertiesKernel(
        d_vertices_, d_triangles_, d_num_triangles_, d_triangle_properties_,
        grid_size, block_size);

    //    if (it == 0) {
    //      LaunchStoreFirstNormalTrianglePropertiesKernel(
    //          d_vertices_, d_triangles_, d_num_triangles_,
    //          d_triangle_properties_,
    //          mesh_.GetDeviceTrianglePropertiesFixedNormal(), grid_size,
    //          block_size);
    //    } else {
    //      LaunchAssignFirstNormalTrianglePropertiesKernel(
    //          d_vertices_, d_triangles_, d_num_triangles_,
    //          d_triangle_properties_,
    //          mesh_.GetDeviceTrianglePropertiesFixedNormal(), grid_size,
    //          block_size);
    //    }

    block_size = 1024;
    grid_size = (num_vertices_ + block_size - 1) / block_size;
    LaunchInitializeOneRingPropertiesKernel(
        d_one_ring_properties_, d_num_vertices_, grid_size, block_size);

    block_size = 1024;
    grid_size = (num_vertices_ + block_size - 1) / block_size;
    LaunchComputeOneRingPropertiesKernel(
        d_vertices_, d_num_vertices_, d_triangles_, d_one_ring_indices_,
        d_one_ring_indices_row_lengths_, d_triangle_properties_,
        d_one_ring_properties_, d_curvature_coefficient_, grid_size,
        block_size);

    if (mode_ == "linear_combination") {
      block_size = 1024;
      grid_size =
          (num_vertices_ * image_manager_.GetImageProperties().num_images +
           block_size - 1) /
          block_size;

      LaunchComputeVertexCameraAnglesKernel(
          d_num_vertices_, image_manager_.GetDeviceImageProperties(),
          d_vertex_camera_angles_, d_one_ring_properties_,
          camera_manager_.GetDeviceCameraParams(), block_size, grid_size);

      block_size = 1024;
      grid_size = (num_vertices_ + block_size - 1) / block_size;

      LaunchComputeCameraPairIndices(
          d_num_vertices_, d_one_ring_properties_, d_vertex_camera_angles_,
          d_photometric_properties_, image_manager_.GetDeviceImageProperties(),
          block_size, grid_size);

      block_size = 1024 / 2;
      grid_size = (num_vertices_ * 6 + block_size - 1) / block_size;

      LaunchComputeProjectedPixelIndicesKernel(
          d_delta_vertices_, d_vertices_, d_num_vertices_,
          d_refinement_resolution_delta_, d_one_ring_properties_,
          d_photometric_properties_, d_projected_pixel_indices_,
          camera_manager_.GetDeviceCameraParams(), grid_size, block_size);

      LaunchExtractPatchKernel(
          d_num_vertices_, image_manager_.GetDeviceImages(),
          image_manager_.GetDeviceImageProperties(), d_photometric_properties_,
          d_projected_pixel_indices_, d_patches_, grid_size, block_size);

      block_size = 1024 / 2;
      grid_size = (num_vertices_ * 3 + block_size - 1) / block_size;

      LaunchComputePhotoConsistencyErrorKernel(
          d_num_vertices_, d_photometric_properties_, d_patches_, grid_size,
          block_size);

      block_size = 1024;
      grid_size = (num_vertices_ + block_size - 1) / block_size;

      LaunchComputePhotometricPropertiesKernel(
          d_num_vertices_, d_photometric_properties_,
          mesh_.GetDeviceOneRingProperties(), d_photometric_coefficient_,
          grid_size, block_size);
    }

    LaunchUpdateCurvatureAdjustmentKernel(
        d_vertices_, d_num_vertices_, d_mode_, d_one_ring_properties_,
        d_photometric_properties_, d_surface_weight_, d_damping_value_,
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
