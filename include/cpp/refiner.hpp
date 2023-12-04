// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_REFINER_HPP_
#define SURFACE_REFINEMENT_REFINER_HPP_

#include <glog/logging.h>

#include <Eigen/Core>
#include <boost/timer/progress_display.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "camera.hpp"
#include "camera_manager.hpp"
#include "image.hpp"
#include "image_manager.hpp"
#include "memory_manager.hpp"
#include "mesh.hpp"
#include "one_ring_kernel_wrappers.hpp"
#include "photo_kernel_wrappers.hpp"
#include "refiner_kernel_wrappers.hpp"
#include "triangle_kernel_wrappers.hpp"

namespace surface_refinement {

class Refiner {
 public:
  Refiner(const Mesh& mesh, const CameraManager& camera_manager,
          const ImageManager& image_manager, double damping_factor,
          const std::string& mode, int num_iteration,
          double refinement_resolution_delta, double surface_weight,
          double curvature_coefficient, double photometric_coefficient);
  ~Refiner();

  std::vector<Eigen::Vector3d> LaunchRefinement();

 private:
  Mesh mesh_;
  Eigen::Vector3d* d_vertices_{nullptr};
  int num_vertices_;
  int* d_num_vertices_{nullptr};

  Eigen::Vector3i* d_triangles_{nullptr};
  int num_triangles_;
  int* d_num_triangles_{nullptr};

  TriangleProperties* d_triangle_properties_{nullptr};
  OneRingProperties* d_one_ring_properties_{nullptr};
  double* d_vertex_camera_angles_{nullptr};
  PhotometricProperties* d_photometric_properties_{nullptr};

  ImageManager image_manager_;

  CameraManager camera_manager_;

  int* d_one_ring_indices_{nullptr};
  int* d_one_ring_indices_row_lengths_{nullptr};
  Eigen::Vector2d* d_projected_pixel_indices_{nullptr};
  double* d_patches_{nullptr};
  DeltaVertex* d_delta_vertices_{nullptr};

  int num_iteration_;
  double* d_damping_value_{nullptr};
  double* d_damping_factor_{nullptr};
  std::string mode_;
  char* d_mode_{nullptr};
  double* d_surface_weight_{nullptr};
  double* d_curvature_coefficient_{nullptr};
  double* d_photometric_coefficient_{nullptr};
  double* d_refinement_resolution_delta_{nullptr};
};

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_REFINER_HPP_
