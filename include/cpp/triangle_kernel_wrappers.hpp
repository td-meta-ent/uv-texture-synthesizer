#ifndef UV_TEXTURE_SYNTHESIZER_TRIANGLE_KERNEL_WRAPPERS_HPP_
#define UV_TEXTURE_SYNTHESIZER_TRIANGLE_KERNEL_WRAPPERS_HPP_

#include <cuda_runtime.h>

#include <Eigen/Core>

namespace uv_texture_synthesizer {

struct TriangleProperties {
  Eigen::Vector3d normal;
};

void LaunchComputeTrianglePropertiesKernel(
    const Eigen::Vector3d* d_vertices, const Eigen::Vector3i* d_triangles,
    const int* d_num_triangles, TriangleProperties* d_triangle_properties,
    int grid_size, int block_size);

__global__ void ComputeTrianglePropertiesKernel(
    const Eigen::Vector3d* d_vertices, const Eigen::Vector3i* d_triangles,
    const int* d_num_triangles, TriangleProperties* d_triangle_properties);

}  // namespace uv_texture_synthesizer

#endif  // UV_TEXTURE_SYNTHESIZER_TRIANGLE_KERNEL_WRAPPERS_HPP_
