#ifndef UV_TEXTURE_SYNTHESIZER_ONE_RING_KERNEL_WRAPPERS_HPP_
#define UV_TEXTURE_SYNTHESIZER_ONE_RING_KERNEL_WRAPPERS_HPP_

#include <cuda_runtime.h>
#include <Eigen/Core>

namespace uv_texture_synthesizer {

struct OneRingProperties {
  double vertex_camera_cosine;
  bool mask;
};

void LaunchInitializeOneRingPropertiesKernel(
    OneRingProperties* d_one_ring_properties, int* d_num_vertices,
    int grid_size, int block_size);

void LaunchComputeOneRingPropertiesKernel(
    const Eigen::Vector3d* d_vertices, const int* d_num_vertices,
    const Eigen::Vector3i* d_triangles, const int* d_one_ring_indices,
    const int* d_one_ring_indices_row_lengths,
    OneRingProperties* d_one_ring_properties,
    const double* d_coefficient_curvature, int grid_size, int block_size);

__global__ void InitializeOneRingPropertiesKernel(
    OneRingProperties* d_one_ring_properties, const int* d_num_vertices);

__global__ void ComputeOneRingPropertiesKernel(
    const Eigen::Vector3d* d_vertices, int* d_num_vertices,
    Eigen::Vector3i* d_triangles, int* d_one_ring_indices,
    int* d_one_ring_indices_row_lengths,
    OneRingProperties* d_one_ring_properties,
    const double* d_coefficient_curvature);

}  // namespace uv_texture_synthesizer

#endif  // UV_TEXTURE_SYNTHESIZER_ONE_RING_KERNEL_WRAPPERS_HPP_
