// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "memory_manager.hpp"
#include "synthesizer_kernel_wrappers.hpp"

namespace uv_texture_synthesizer {

__global__ void computeOneRingFilterProperties(
    const int *d_num_cameras, Eigen::Vector3d *d_vertices,
    Eigen::Vector3d *d_vertex_normals, const int *d_num_vertices,
    CameraParams *d_camera_params, Eigen::Vector3d *d_centroid,
    OneRingProperties *d_one_ring_properties) {
  int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);

  if (idx < *d_num_vertices * *d_num_cameras) {
    int d_camera_index = idx / *d_num_vertices;
    int d_vertex_index = idx % *d_num_vertices;

    Eigen::Vector3d d_vertex = d_vertices[d_vertex_index];
    Eigen::Vector3d d_vertex_normal =
        //        d_camera_params[d_camera_index].transformation_matrix.block<3,
        //        3>(0, 0) *
        d_vertex_normals[d_vertex_index];
    Eigen::Vector3d d_camera_normal =
        d_camera_params[d_camera_index].camera_normal;

    double d_cosine = d_vertex_normal.dot(d_camera_normal);
    d_one_ring_properties[idx].vertex_camera_cosine = d_cosine;
    double d_angle = acos(d_cosine);

    if (d_camera_index == 4 or d_camera_index == 5) {
      *d_centroid = *d_centroid - 0.1 * d_camera_normal;
    } else {
      *d_centroid = *d_centroid + 0.007 * d_camera_normal;
    }

    double d_angle_filter =
        (d_camera_index == 4 || d_camera_index == 5) ? M_PI : M_PI / 2;
    bool d_angle_mask = d_angle > d_angle_filter;

    Eigen::Vector3d d_direction = d_vertices[idx] - *d_centroid;
    bool d_centroid_mask = d_direction.dot(d_camera_normal) < 0;

    d_one_ring_properties[idx].mask = d_angle_mask || d_centroid_mask;
//    d_one_ring_properties[idx].mask = d_angle_mask;
  }
}

__global__ void generateTextureImages(
    TexturePixelParams *d_texture_pixels_params, const int *d_num_cameras,
    Eigen::Vector3d *d_vertices, const int *d_num_vertices,
    Eigen::Vector3i *d_triangles, CameraParams *d_camera_params,
    Eigen::Vector3i *d_images, Eigen::Vector3i *d_texture_images,
    double *d_cosine_images, OneRingProperties *d_one_ring_properties,
    const int *d_texture_image_height, const int *d_texture_image_width,
    const int *d_image_height, const int *d_image_width) {
  int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);

  int d_texture_image_size = *d_texture_image_height * *d_texture_image_width;

  if (idx < d_texture_image_size * *d_num_cameras) {
    int d_texture_pixel_params_idx = idx % d_texture_image_size;
    int d_camera_index = idx / d_texture_image_size;

    TexturePixelParams d_texture_pixel_params =
        d_texture_pixels_params[d_texture_pixel_params_idx];

    if (d_texture_pixel_params.triangle_index == -1) {
      return;
    }

    Eigen::Vector3i d_triangle =
        d_triangles[d_texture_pixel_params.triangle_index];

    Eigen::Vector2i d_projected_triangle_vertices_indices[3];
    double d_texture_triangle_cosine[3];

    for (int i = 0; i < 3; ++i) {
      if (d_one_ring_properties[d_camera_index * *d_num_vertices +
                                d_triangle[i]]
              .mask) {
        return;
      }

      int d_vertex_index = d_triangle[i];

      d_texture_triangle_cosine[i] =
          d_one_ring_properties[d_camera_index * *d_num_vertices +
                                d_vertex_index]
              .vertex_camera_cosine;

      Eigen::Vector3d d_vertex = d_vertices[d_vertex_index];

      Eigen::Vector2d d_projection_indices = GeometricUtils::projectVertex3D(
          d_vertex, &d_camera_params[d_camera_index]);
      int d_projection_row_idx = int(d_projection_indices[0]);
      int d_projection_col_idx = int(d_projection_indices[1]);
      if (d_projection_row_idx < 0 || d_projection_row_idx >= *d_image_height ||
          d_projection_col_idx < 0 || d_projection_col_idx >= *d_image_width) {
        //        printf(
        //            "[CUDA ERROR]: Invalid coordinates d_row=%d, d_col=%d |
        //            File: "
        //            "%s "
        //            "| "
        //            "Line: "
        //            "%d ",
        //            d_projection_row_idx, d_projection_col_idx, __FILE__,
        //            __LINE__);
        continue;
      }
      Eigen::Vector2i d_projection_indices_int =
          Eigen::Vector2i(d_projection_col_idx, d_projection_row_idx);
      d_projected_triangle_vertices_indices[i] = d_projection_indices_int;
    }

    Eigen::Vector2d d_projected_point_indices =
        Eigen::Vector2d(d_texture_pixel_params.barycentric_coordinates[0] *
                                d_projected_triangle_vertices_indices[0][0] +
                            d_texture_pixel_params.barycentric_coordinates[1] *
                                d_projected_triangle_vertices_indices[1][0] +
                            d_texture_pixel_params.barycentric_coordinates[2] *
                                d_projected_triangle_vertices_indices[2][0],
                        d_texture_pixel_params.barycentric_coordinates[0] *
                                d_projected_triangle_vertices_indices[0][1] +
                            d_texture_pixel_params.barycentric_coordinates[1] *
                                d_projected_triangle_vertices_indices[1][1] +
                            d_texture_pixel_params.barycentric_coordinates[2] *
                                d_projected_triangle_vertices_indices[2][1]);

    //    d_texture_images[idx] =
    //        GeometricUtils::bilinearInterpolationAtPoint2DKernel(
    //            &d_images[d_camera_index], *d_image_width, *d_image_height,
    //            d_projected_point_indices);

    int d_x = static_cast<int>(floor(d_projected_point_indices.x()));
    int d_y = static_cast<int>(floor(d_projected_point_indices.y()));

    double d_frac_x = d_projected_point_indices.x() - d_x;
    double d_frac_y = d_projected_point_indices.y() - d_y;

    // Boundary check
    Eigen::Vector3i interpolated_color;
    if (d_x < 0 || d_x >= *d_image_width - 1 || d_y < 0 ||
        d_y >= *d_image_height - 1) {
      interpolated_color = Eigen::Vector3i(255, 255, 255);
    } else {
      // Get the four neighboring pixels
      Eigen::Vector3i d_color_tl =
          d_images[d_camera_index * *d_image_width * *d_image_height +
                   d_y * *d_image_width + d_x];  // Top left
      Eigen::Vector3i d_color_tr =
          d_images[d_camera_index * *d_image_width * *d_image_height +
                   d_y * *d_image_width + (d_x + 1)];  // Top right
      Eigen::Vector3i d_color_bl =
          d_images[d_camera_index * *d_image_width * *d_image_height +
                   (d_y + 1) * *d_image_width + d_x];  // Bottom left
      Eigen::Vector3i d_color_br =
          d_images[d_camera_index * *d_image_width * *d_image_height +
                   (d_y + 1) * *d_image_width + (d_x + 1)];  // Bottom right

      // Perform bilinear interpolation
      for (int i = 0; i < 3; ++i) {
        interpolated_color[i] =
            static_cast<int>((1 - d_frac_x) * (1 - d_frac_y) * d_color_tl[i] +
                             d_frac_x * (1 - d_frac_y) * d_color_tr[i] +
                             (1 - d_frac_x) * d_frac_y * d_color_bl[i] +
                             d_frac_x * d_frac_y * d_color_br[i]);
      }
    }

    d_texture_images[idx] = interpolated_color;

    //    d_texture_images[idx] =
    //        d_images[d_camera_index * *d_image_height * *d_image_width +
    //                 d_projected_point_indices[1] * *d_image_width +
    //                 d_projected_point_indices[0]];
    d_cosine_images[idx] = d_texture_pixel_params.barycentric_coordinates[0] *
                               d_texture_triangle_cosine[0] +
                           d_texture_pixel_params.barycentric_coordinates[1] *
                               d_texture_triangle_cosine[1] +
                           d_texture_pixel_params.barycentric_coordinates[2] *
                               d_texture_triangle_cosine[2];
  }
}

__global__ void computeInterpolatedTextureImage(
    const int *d_num_cameras, Eigen::Vector3i *d_texture_images,
    const double *d_cosine_images,
    Eigen::Vector3i *d_interpolated_texture_image,
    const int *d_texture_image_height, const int *d_texture_image_width) {
  int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);

  int d_texture_image_size = *d_texture_image_height * *d_texture_image_width;

  if (idx < d_texture_image_size) {
    double d_weight_sum = 0.0;
    bool all_weights_negative = true;

    // Check if all weights are negative
    for (int d_i = 0; d_i < *d_num_cameras; d_i++) {
      double weight = d_cosine_images[d_i * d_texture_image_size + idx];
      if (weight > 0) {
        all_weights_negative = false;
        break;
      }
    }

    // Compute the weighted pixel sum
    Eigen::Vector3d d_interpolated_pixel = Eigen::Vector3d(0.0, 0.0, 0.0);
    for (int d_i = 0; d_i < *d_num_cameras; d_i++) {
      double weight = d_cosine_images[d_i * d_texture_image_size + idx];

      // If not all weights are negative, ignore negative weights
      if (!all_weights_negative && weight < 0) continue;

      Eigen::Vector3d d_pixel = Eigen::Vector3d(
          d_texture_images[d_i * d_texture_image_size + idx][0],
          d_texture_images[d_i * d_texture_image_size + idx][1],
          d_texture_images[d_i * d_texture_image_size + idx][2]);

      d_interpolated_pixel += d_pixel * weight;
      d_weight_sum += weight;
    }

    if (d_weight_sum == 0.0) {
      d_interpolated_pixel = Eigen::Vector3d(255.0, 255.0, 255.0);
      d_weight_sum = 1.0;
    }

    d_interpolated_texture_image[idx] =
        Eigen::Vector3i(int(d_interpolated_pixel[0] / d_weight_sum),
                        int(d_interpolated_pixel[1] / d_weight_sum),
                        int(d_interpolated_pixel[2] / d_weight_sum));
  }
}

void LaunchComputeOneRingFilterProperties(
    const int *d_num_cameras, Eigen::Vector3d *d_vertices,
    Eigen::Vector3d *d_vertex_normals, const int *d_num_vertices,
    CameraParams *d_camera_params, Eigen::Vector3d *d_centroid,
    OneRingProperties *d_one_ring_properties, int grid_size, int block_size) {
  computeOneRingFilterProperties<<<grid_size, block_size>>>(
      d_num_cameras, d_vertices, d_vertex_normals, d_num_vertices,
      d_camera_params, d_centroid, d_one_ring_properties);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize())
  CUDA_CHECK_LAST()
}

void LaunchGenerateTextureImages(
    TexturePixelParams *d_texture_pixels_params, const int *d_num_cameras,
    Eigen::Vector3d *d_vertices, const int *d_num_vertices,
    Eigen::Vector3i *d_triangles, CameraParams *d_camera_params,
    Eigen::Vector3i *d_images, Eigen::Vector3i *d_texture_images,
    double *d_cosine_images, OneRingProperties *d_one_ring_properties,
    const int *d_texture_image_height, const int *d_texture_image_width,
    const int *d_image_height, const int *d_image_width, int grid_size,
    int block_size) {
  generateTextureImages<<<grid_size, block_size>>>(
      d_texture_pixels_params, d_num_cameras, d_vertices, d_num_vertices,
      d_triangles, d_camera_params, d_images, d_texture_images, d_cosine_images,
      d_one_ring_properties, d_texture_image_height, d_texture_image_width,
      d_image_height, d_image_width);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize())
  CUDA_CHECK_LAST()
}

void LaunchComputeInterpolatedTextureImage(
    const int *d_num_cameras, Eigen::Vector3i *d_texture_images,
    const double *d_cosine_images,
    Eigen::Vector3i *d_interpolated_texture_image,
    const int *d_texture_image_height, const int *d_texture_image_width,
    int grid_size, int block_size) {
  computeInterpolatedTextureImage<<<grid_size, block_size>>>(
      d_num_cameras, d_texture_images, d_cosine_images,
      d_interpolated_texture_image, d_texture_image_height,
      d_texture_image_width);

  CUDA_ERROR_CHECK(cudaDeviceSynchronize())
  CUDA_CHECK_LAST()
}

}  // namespace uv_texture_synthesizer
