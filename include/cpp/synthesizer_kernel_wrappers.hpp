// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef UV_TEXTURE_SYNTHESIZER_REFINER_KERNEL_WRAPPERS_HPP_
#define UV_TEXTURE_SYNTHESIZER_REFINER_KERNEL_WRAPPERS_HPP_

#include <cuda_runtime.h>

#include <Eigen/Core>

#include "camera.hpp"
#include "geometric_utils.cuh"
#include "one_ring_kernel_wrappers.hpp"
#include "texture.hpp"

namespace uv_texture_synthesizer {

__global__ void generateTextureImages(
    TexturePixelParams *d_texture_pixels_params, const int *d_num_cameras,
    Eigen::Vector3d *d_vertices, const int *d_num_vertices,
    Eigen::Vector3i *d_triangles, CameraParams *d_camera_params,
    Eigen::Vector3i *d_images, Eigen::Vector3i *d_texture_images,
    double *d_cosine_images, OneRingProperties *d_one_ring_properties,
    const int *d_texture_image_height, const int *d_texture_image_width,
    const int *d_image_height, const int *d_image_width);

__global__ void computeOneRingFilterProperties(
    const int *d_num_cameras, Eigen::Vector3d *d_vertices,
    Eigen::Vector3d *d_vertex_normals, const int *d_num_vertices,
    CameraParams *d_camera_params, Eigen::Vector3d *d_centroid,
    OneRingProperties *d_one_ring_properties);

__global__ void computeInterpolatedTextureImage(
    const int *d_num_cameras, Eigen::Vector3i *d_texture_images,
    const double *d_cosine_images,
    Eigen::Vector3i *d_interpolated_texture_image,
    const int *d_texture_image_height, const int *d_texture_image_width);

void LaunchComputeOneRingFilterProperties(
    const int *d_num_cameras, Eigen::Vector3d *d_vertices,
    Eigen::Vector3d *d_vertex_normals, const int *d_num_vertices,
    CameraParams *d_camera_params, Eigen::Vector3d *d_centroid,
    OneRingProperties *d_one_ring_properties, int grid_size, int block_size);

void LaunchGenerateTextureImages(
    TexturePixelParams *d_texture_pixels_params, const int *d_num_cameras,
    Eigen::Vector3d *d_vertices, const int *d_num_vertices,
    Eigen::Vector3i *d_triangles, CameraParams *d_camera_params,
    Eigen::Vector3i *d_images, Eigen::Vector3i *d_texture_images,
    double *d_cosine_images, OneRingProperties *d_one_ring_properties,
    const int *d_texture_image_height, const int *d_texture_image_width,
    const int *d_image_height, const int *d_image_width, int grid_size,
    int block_size);

void LaunchComputeInterpolatedTextureImage(
    const int *d_num_cameras, Eigen::Vector3i *d_texture_images,
    const double *d_cosine_images,
    Eigen::Vector3i *d_interpolated_texture_image,
    const int *d_texture_image_height, const int *d_texture_image_width,
    int grid_size, int block_size);

}  // namespace uv_texture_synthesizer

#endif  // UV_TEXTURE_SYNTHESIZER_REFINER_KERNEL_WRAPPERS_HPP_
