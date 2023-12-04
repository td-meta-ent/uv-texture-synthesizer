// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "patch_metrics.cuh"
#include "photo_kernel_wrappers.hpp"

namespace surface_refinement {

__global__ void ComputeVertexCameraAnglesKernel(
    const int *d_num_vertices, ImageProperties *d_image_properties,
    double *d_vertex_camera_angles, OneRingProperties *d_one_ring_properties,
    CameraParams *d_camera_params) {
  int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);

  if (idx < *d_num_vertices * d_image_properties->num_images) {
    d_vertex_camera_angles[idx] = acos(
        d_one_ring_properties[idx / d_image_properties->num_images].normal.dot(
            d_camera_params[idx % d_image_properties->num_images]
                .camera_normal));
  }
}

__global__ void ComputePhotometricPropertiesKernel(
    const int *d_num_vertices, PhotometricProperties *d_photometric_properties,
    OneRingProperties *d_one_ring_properties,
    const double *d_photometric_coefficient) {
  int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);

  if (idx < *d_num_vertices) {
    d_photometric_properties[idx].delta_p =
        PatchMetrics::compute_photometric_delta(
            d_photometric_properties[idx].error_minus_delta,
            d_photometric_properties[idx].error_x,
            d_photometric_properties[idx].error_plus_delta);

    d_photometric_properties[idx].weight_p =
        PatchMetrics::compute_photometric_weight(
            d_photometric_properties[idx].error_minus_delta,
            d_photometric_properties[idx].error_x,
            d_photometric_properties[idx].error_plus_delta);

    d_photometric_properties[idx].photometric_adjustment =
        d_photometric_properties[idx].delta_p *
        d_one_ring_properties[idx].normal * *d_photometric_coefficient;
  }
}

__global__ void ComputePhotoConsistencyErrorKernel(
    const int *d_num_vertices, PhotometricProperties *d_photometric_properties,
    double *d_patches) {
  int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);

  if (idx < *d_num_vertices * 3) {
    int vertex_idx = idx / 3;
    int vertex_idx_offset = idx % 3;
    int first_patch_idx = vertex_idx * 6 + vertex_idx_offset;
    int second_patch_idx = vertex_idx * 6 + vertex_idx_offset + 3;
    double error = PatchMetrics::compute_photo_consistency_error(
        &d_patches[first_patch_idx], &d_patches[second_patch_idx]);

    if (vertex_idx_offset == 0) {
      d_photometric_properties[vertex_idx].error_minus_delta = error;
    } else if (vertex_idx_offset == 1) {
      d_photometric_properties[vertex_idx].error_x = error;
    } else if (vertex_idx_offset == 2) {
      d_photometric_properties[vertex_idx].error_plus_delta = error;
    }
  }
}

__global__ void ExtractPatchKernel(
    const int *d_num_vertices, double *d_images,
    ImageProperties *d_image_properties,
    PhotometricProperties *d_photometric_properties,
    Eigen::Vector2d *d_projected_pixel_indices, double *d_patches) {
  int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);

  if (idx < *d_num_vertices * 6) {
    double *d_image;

    int d_first_image_index = d_photometric_properties->camera_pair_index * 2;
    int d_second_image_index = d_first_image_index + 1;
    int one_image_size = d_image_properties->rows * d_image_properties->cols;
    if (idx % 6 < 3) {
      d_image = &d_images[d_first_image_index * one_image_size];
    } else {
      d_image = &d_images[d_second_image_index * one_image_size];
    }

    const int d_patch_width = 3;
    const int d_patch_height = 3;

    int d_center_x =
        static_cast<int>(round(d_projected_pixel_indices[idx](1) / 2));
    int d_center_y =
        static_cast<int>(round(d_projected_pixel_indices[idx](0) / 2));

    double d_frac_x =
        d_projected_pixel_indices[idx](1) - static_cast<double>(d_center_x);
    double d_frac_y =
        d_projected_pixel_indices[idx](0) - static_cast<double>(d_center_y);

    int d_direction_x = d_frac_x < 0 ? -1 : 1;
    int d_direction_y = d_frac_y < 0 ? -1 : 1;

    d_frac_x = abs(d_frac_x);
    d_frac_y = abs(d_frac_y);

    double d_x = d_frac_x, d_y = 1 - d_frac_x, d_r = d_frac_y,
           d_s = 1 - d_frac_y;
    double d_weight_a = d_x * d_r, d_weight_b = d_y * d_r,
           d_weight_c = d_x * d_s, d_weight_d = d_y * d_s;
    double d_weight_sum = d_weight_a + d_weight_b + d_weight_c + d_weight_d;

    // Normalize weights
    d_weight_a /= d_weight_sum;
    d_weight_b /= d_weight_sum;
    d_weight_c /= d_weight_sum;
    d_weight_d /= d_weight_sum;

    // Compute patch boundaries
    int d_patch_left = d_center_x - d_patch_width / 2;
    int d_patch_right = d_patch_left + d_patch_width;
    int d_patch_top = d_center_y - d_patch_height / 2;
    int d_patch_bottom = d_patch_top + d_patch_height;

    int d_patch_cursor = 0;
    for (int d_row = d_patch_top; d_row < d_patch_bottom; ++d_row) {
      for (int d_col = d_patch_left; d_col < d_patch_right; ++d_col) {
        if (d_row < 0 || d_row >= d_image_properties->rows || d_col < 0 ||
            d_col >= d_image_properties->cols) {
          printf(
              "[CUDA ERROR]: Invalid coordinates d_row=%d, d_col=%d | File: "
              "%s "
              "| "
              "Line: "
              "%d ",
              d_row, d_col, __FILE__, __LINE__);
          return;
        }

        // Interpolation
        double d_color_a = d_image[d_row * d_patch_width + d_col];
        double d_color_b =
            d_image[d_row * d_patch_width + (d_col + d_direction_x)];
        double d_color_c =
            d_image[(d_row + d_direction_y) * d_patch_width + d_row];
        double d_color_d = d_image[(d_col + d_direction_y) * d_patch_width +
                                   (d_row + d_direction_x)];

        d_patches[idx * d_patch_height * d_patch_width + d_patch_cursor++] =
            d_color_a * d_weight_a + d_color_b * d_weight_b +
            d_color_c * d_weight_c + d_color_d * d_weight_d;
      }
    }
  }
}

//__global__ void ExtractPatchKernel(const int *d_num_vertices,
//                                   double *d_image_left, double
//                                   *d_image_right, Eigen::Vector2d
//                                   *d_projected_pixel_indices, double
//                                   *d_patches) {
//  int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
//            static_cast<int>(threadIdx.x);
//
//  if (idx < *d_num_vertices * 6) {
//    double *d_image;
//    if (idx % 6 < 3) {
//      d_image = d_image_left;
//    } else {
//      d_image = d_image_right;
//    }
//
//    const int d_image_width = 4096;
//    const int d_image_height = 3072;
//    const int d_patch_width = 3;
//    const int d_patch_height = 3;
//
//    int d_center_x =
//    static_cast<int>(round(d_projected_pixel_indices[idx](1))); int
//    d_center_y = static_cast<int>(round(d_projected_pixel_indices[idx](0)));
//
//    // Compute patch boundaries
//    int d_patch_left = d_center_x - d_patch_width / 2;
//    int d_patch_right = d_patch_left + d_patch_width;
//    int d_patch_top = d_center_y - d_patch_height / 2;
//    int d_patch_bottom = d_patch_top + d_patch_height;
//
//    int d_patch_cursor = 0;
//    for (int row = d_patch_top; row < d_patch_bottom; ++row) {
//      for (int col = d_patch_left; col < d_patch_right; ++col) {
//        if (row < 0 || row >= d_image_height || col < 0 ||
//            col >= d_image_width) {
//          printf(
//              "[CUDA ERROR]: Invalid coordinates row=%d, col=%d | File: %s |
//              " "Line: "
//              "%d ",
//              row, col, __FILE__, __LINE__);
//          return;
//        }
//
//        d_patches[idx * d_patch_height * d_patch_width + d_patch_cursor++] =
//            d_image[row * d_image_width + col];
//      }
//    }
//  }
//}

__device__ Eigen::Vector2d project_vertex(Eigen::Vector3d vertex,
                                          const CameraParams *d_camera_params) {
  Eigen::Vector4d vertex_homogeneous(vertex(0) / 1000, vertex(1) / 1000,
                                     vertex(2) / 1000, 1.0);

  vertex_homogeneous =
      d_camera_params->transformation_matrix * vertex_homogeneous;
  vertex = vertex_homogeneous.head<3>();

  Eigen::Vector2d pixel_indices;
  pixel_indices(0) = d_camera_params->focal_length_y * vertex(1) / vertex(2) +
                     d_camera_params->principal_point_y;
  pixel_indices(1) = d_camera_params->focal_length_x * vertex(0) / vertex(2) +
                     d_camera_params->principal_point_x;
  return pixel_indices;
}

__global__ void ComputeProjectedPixelIndicesKernel(
    DeltaVertex *d_delta_vertices, Eigen::Vector3d *d_vertices,
    const int *d_num_vertices, double *d_refinement_resolution_delta,
    OneRingProperties *d_one_ring_properties,
    PhotometricProperties *d_photometric_properties,
    Eigen::Vector2d *d_projected_pixel_indices, CameraParams *d_camera_params) {
  int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);

  if (idx < *d_num_vertices * 6) {
    switch (idx % 3) {
      case 0:
        // For cases 0 and 3: Subtract the normal scaled by the delta.
        d_delta_vertices[idx].vertex =
            d_vertices[idx / 6] - *d_refinement_resolution_delta *
                                      d_one_ring_properties[idx / 6].normal;
        break;
      case 1:
        // For cases 1 and 4: No change to the vertex.
        d_delta_vertices[idx].vertex = d_vertices[idx / 6];
        break;
      case 2:
        // For cases 2 and 5: Add the normal scaled by the delta.
        d_delta_vertices[idx].vertex =
            d_vertices[idx / 6] + *d_refinement_resolution_delta *
                                      d_one_ring_properties[idx / 6].normal;
        break;
    }

    int d_camera_pair_index =
        d_photometric_properties[idx / 6].camera_pair_index;

    int d_camera_index;
    if (idx % 6 < 3) {
      d_camera_index = d_camera_pair_index * 2;
    } else {
      d_camera_index = d_camera_pair_index * 2 + 1;
    }

    // Compute projected pixel indices for each vertex
    d_projected_pixel_indices[idx] = project_vertex(
        d_delta_vertices[idx].vertex, &d_camera_params[d_camera_index]);
  }
}

__global__ void ComputeCameraPairIndices(
    const int *d_num_vertices, OneRingProperties *d_one_ring_properties,
    const double *d_vertex_camera_angles,
    PhotometricProperties *d_photometric_properties,
    ImageProperties *d_image_properties) {
  int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);

  if (idx < *d_num_vertices) {
    double min_angle = d_vertex_camera_angles[idx];
    int min_angle_idx = 0;
    for (int i = 1; i < d_image_properties->num_images; ++i) {
      if (d_vertex_camera_angles[idx * d_image_properties->num_images + i] <
          min_angle) {
        min_angle =
            d_vertex_camera_angles[idx * d_image_properties->num_images + i];
        min_angle_idx = i;
      }
    }

    d_photometric_properties[idx].camera_pair_index = min_angle_idx / 2;
  }
}

void LaunchComputeCameraPairIndices(
    const int *d_num_vertices, OneRingProperties *d_one_ring_properties,
    double *d_vertex_camera_angles,
    PhotometricProperties *d_photometric_properties,
    ImageProperties *d_image_properties, int block_size, int grid_size) {
  ComputeCameraPairIndices<<<grid_size, block_size>>>(
      d_num_vertices, d_one_ring_properties, d_vertex_camera_angles,
      d_photometric_properties, d_image_properties);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize())
  CUDA_CHECK_LAST()
}

void LaunchComputeVertexCameraAnglesKernel(
    const int *d_num_vertices, ImageProperties *d_image_properties,
    double *d_vertex_camera_angles, OneRingProperties *d_one_ring_properties,
    CameraParams *d_camera_params, int block_size, int grid_size) {
  ComputeVertexCameraAnglesKernel<<<grid_size, block_size>>>(
      d_num_vertices, d_image_properties, d_vertex_camera_angles,
      d_one_ring_properties, d_camera_params);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize())
  CUDA_CHECK_LAST()
}

void LaunchComputePhotometricPropertiesKernel(
    int *d_num_vertices, PhotometricProperties *d_photometric_properties,
    OneRingProperties *d_one_ring_properties, double *d_photometric_coefficient,
    int grid_size, int block_size) {
  ComputePhotometricPropertiesKernel<<<grid_size, block_size>>>(
      d_num_vertices, d_photometric_properties, d_one_ring_properties,
      d_photometric_coefficient);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize())
  CUDA_CHECK_LAST()
}

void LaunchComputePhotoConsistencyErrorKernel(
    int *d_num_vertices, PhotometricProperties *d_photometric_properties,
    double *d_patches, int grid_size, int block_size) {
  ComputePhotoConsistencyErrorKernel<<<grid_size, block_size>>>(
      d_num_vertices, d_photometric_properties, d_patches);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize())
  CUDA_CHECK_LAST()
}

void LaunchExtractPatchKernel(const int *d_num_vertices, double *d_images,
                              ImageProperties *d_image_properties,
                              PhotometricProperties *d_photometric_properties,
                              Eigen::Vector2d *d_projected_pixel_indices,
                              double *d_patches, int grid_size,
                              int block_size) {
  ExtractPatchKernel<<<grid_size, block_size>>>(
      d_num_vertices, d_images, d_image_properties, d_photometric_properties,
      d_projected_pixel_indices, d_patches);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize())
  CUDA_CHECK_LAST()
}

void LaunchComputeProjectedPixelIndicesKernel(
    DeltaVertex *d_delta_vertices, Eigen::Vector3d *d_vertices,
    int *d_num_vertices, double *d_refinement_resolution_delta,
    OneRingProperties *d_one_ring_properties,
    PhotometricProperties *d_photometric_properties,
    Eigen::Vector2d *d_projected_pixel_indices, CameraParams *d_camera_params,
    int grid_size, int block_size) {
  ComputeProjectedPixelIndicesKernel<<<grid_size, block_size>>>(
      d_delta_vertices, d_vertices, d_num_vertices,
      d_refinement_resolution_delta, d_one_ring_properties,
      d_photometric_properties, d_projected_pixel_indices, d_camera_params);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize())
  CUDA_CHECK_LAST()
}

}  // namespace surface_refinement
