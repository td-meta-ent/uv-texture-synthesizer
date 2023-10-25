// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_REFINER_HPP_
#define SURFACE_REFINEMENT_REFINER_HPP_

#include <glog/logging.h>

#include <Eigen/Core>
#include <boost/timer/progress_display.hpp>
#include <string>
#include <vector>

#include "memory_manager.hpp"
#include "refiner_kernel_wrappers.hpp"
#include "triangle_kernel_wrappers.hpp"

namespace surface_refinement {

class Refiner {
 public:
  Refiner(Eigen::Vector3d* d_vertices, int num_vertices, int* d_num_vertices,
          Eigen::Vector3i* d_triangles, int num_triangles, int* d_num_triangles,
          TriangleProperties* d_triangle_properties_,
          OneRingProperties* d_one_ring_properties, int* d_one_ring_indices,
          int* d_one_ring_indices_row_lengths, const double* d_image_left,
          const double* d_image_right, double* d_shift_distance,
          double damping_factor, const std::string& mode, int num_iteration,
          double refinement_resolution_delta, double coefficient_curvature,
          double coefficient_photometric);
  ~Refiner();

  std::vector<Eigen::Vector3d> LaunchRefinement();

 private:
  Eigen::Vector3d* d_vertices_;
  int* d_num_vertices_;
  int num_vertices_;
  Eigen::Vector3i* d_triangles_;
  int num_triangles_;
  int* d_num_triangles_;
  TriangleProperties* d_triangle_properties_;
  OneRingProperties* d_one_ring_properties_;
  const double* d_image_left_;
  const double* d_image_right_;
  double* d_shift_distance_;
  int* d_one_ring_indices_;
  int* d_one_ring_indices_row_lengths_;

  int num_iteration_;
  double* d_damping_value_;
  double* d_damping_factor_;
  char* d_mode_;
  double* d_coefficient_curvature_;
  double* d_coefficient_photometric_;
  double* d_refinement_resolution_delta_;
};

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_REFINER_HPP_
