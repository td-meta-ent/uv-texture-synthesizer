// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_GEOMETRIC_UTILS_CUH_
#define SURFACE_REFINEMENT_GEOMETRIC_UTILS_CUH_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace surface_refinement {

/**
 * @file GeometricUtils.cuh
 * @brief Utility class providing functions for common geometric calculations.
 * @details Provides utility functions such as computing distance between
 * points, checking for obtuse angles, computing triangle areas, etc.
 */
class GeometricUtils {
 public:
  __device__ static inline double compute_distance(
      const Eigen::Vector3d& point_a, const Eigen::Vector3d& point_b);

  __device__ static inline Eigen::Vector3d normalize_vector(
      const Eigen::Vector3d& input_vector);

  __device__ static inline double compute_angle_between_vectors(
      const Eigen::Vector3d& vector_a, const Eigen::Vector3d& vector_b);

  __device__ static inline bool is_angle_obtuse(double angle_in_radians);

  __device__ static inline double compute_cotangent(double angle_in_radians);

  __device__ static inline double compute_triangle_area(
      const Eigen::Vector3d& vertex_a, const Eigen::Vector3d& vertex_b,
      const Eigen::Vector3d& vertex_c);
};

// Definitions

__device__ inline double GeometricUtils::compute_distance(
    const Eigen::Vector3d& point_a, const Eigen::Vector3d& point_b) {
  return (point_a - point_b).norm();
}

__device__ inline Eigen::Vector3d GeometricUtils::normalize_vector(
    const Eigen::Vector3d& input_vector) {
  return input_vector.normalized();
}

__device__ inline double GeometricUtils::compute_angle_between_vectors(
    const Eigen::Vector3d& vector_a, const Eigen::Vector3d& vector_b) {
  Eigen::Vector3d normalized_vector_a = normalize_vector(vector_a);
  Eigen::Vector3d normalized_vector_b = normalize_vector(vector_b);
  return std::acos(normalized_vector_a.dot(normalized_vector_b));
}

__device__ inline bool GeometricUtils::is_angle_obtuse(
    double angle_in_radians) {
  return angle_in_radians > M_PI / 2.0 - 0.15;
}

__device__ inline double GeometricUtils::compute_cotangent(
    double angle_in_radians) {
  return 1.0 / std::tan(angle_in_radians);
}

__device__ inline double GeometricUtils::compute_triangle_area(
    const Eigen::Vector3d& vertex_a, const Eigen::Vector3d& vertex_b,
    const Eigen::Vector3d& vertex_c) {
  double side_length_a = compute_distance(vertex_b, vertex_c);
  double side_length_b = compute_distance(vertex_a, vertex_c);
  double side_length_c = compute_distance(vertex_a, vertex_b);
  double semi_perimeter = (side_length_a + side_length_b + side_length_c) / 2.0;

  return std::sqrt(semi_perimeter * (semi_perimeter - side_length_a) *
                   (semi_perimeter - side_length_b) *
                   (semi_perimeter - side_length_c));
}

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_GEOMETRIC_UTILS_CUH_
