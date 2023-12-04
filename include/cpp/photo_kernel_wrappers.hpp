// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_PHOTO_KERNEL_WRAPPERS_HPP_
#define SURFACE_REFINEMENT_PHOTO_KERNEL_WRAPPERS_HPP_

#include <Eigen/Core>

#include "camera.hpp"
#include "camera_manager.hpp"
#include "image_manager.hpp"
#include "one_ring_kernel_wrappers.hpp"

namespace surface_refinement {

// Add and subtract delta to generate two points based on the center vertex
struct DeltaVertex {
  Eigen::Vector3d vertex;
  double photo_consistency_error;
};

struct PhotometricProperties {
  int camera_pair_index;
  double error_minus_delta;
  double error_x;
  double error_plus_delta;
  double delta_p;
  double weight_p;
  Eigen::Vector3d photometric_adjustment;
};

void LaunchComputeCameraPairIndices(
    const int *d_num_vertices, OneRingProperties *d_one_ring_properties,
    double *d_vertex_camera_angles,
    PhotometricProperties *d_photometric_properties,
    ImageProperties *d_image_properties, int block_size, int grid_size);

void LaunchComputeVertexCameraAnglesKernel(
    const int *d_num_vertices, ImageProperties *d_image_properties,
    double *d_vertex_camera_angles, OneRingProperties *d_one_ring_properties,
    CameraParams *d_camera_params, int block_size, int grid_size);

void LaunchComputePhotometricPropertiesKernel(
    int *d_num_vertices, PhotometricProperties *d_photometric_properties,
    OneRingProperties *d_one_ring_properties, double *d_photometric_coefficient,
    int grid_size, int block_size);

void LaunchComputePhotoConsistencyErrorKernel(
    int *d_num_vertices, PhotometricProperties *d_photometric_properties,
    double *d_patches, int grid_size, int block_size);

void LaunchExtractPatchKernel(const int *d_num_vertices, double *d_images,
                              ImageProperties *d_image_properties,
                              PhotometricProperties *d_photometric_properties,
                              Eigen::Vector2d *d_projected_pixel_indices,
                              double *d_patches, int grid_size, int block_size);

void LaunchComputeProjectedPixelIndicesKernel(
    DeltaVertex *d_delta_vertices, Eigen::Vector3d *d_vertices,
    int *d_num_vertices, double *d_refinement_resolution_delta,
    OneRingProperties *d_one_ring_properties,
    PhotometricProperties *d_photometric_properties,
    Eigen::Vector2d *d_projected_pixel_indices, CameraParams *d_camera_params,
    int grid_size, int block_size);

__global__ void ComputePhotometricPropertiesKernel(
    const int *d_num_vertices, PhotometricProperties *d_photometric_properties,
    OneRingProperties *d_one_ring_properties,
    const double *d_photometric_coefficient);

__global__ void ComputePhotoConsistencyErrorKernel(
    const int *d_num_vertices, PhotometricProperties *d_photometric_properties,
    double *d_patches);

__global__ void ExtractPatchKernel(
    const int *d_num_vertices, double *d_images,
    ImageProperties *d_image_properties,
    PhotometricProperties *d_photometric_properties,
    Eigen::Vector2d *d_projected_pixel_indices, double *d_patches);

__global__ void ComputeProjectedPixelIndicesKernel(
    DeltaVertex *d_delta_vertices, Eigen::Vector3d *d_vertices,
    const int *d_num_vertices, double *d_refinement_resolution_delta,
    OneRingProperties *d_one_ring_properties,
    PhotometricProperties *d_photometric_properties,
    Eigen::Vector2d *d_projected_pixel_indices, CameraParams *d_camera_params);

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_PHOTO_KERNEL_WRAPPERS_HPP_
