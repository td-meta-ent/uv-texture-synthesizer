// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_TRIANGLE_CUH_
#define SURFACE_REFINEMENT_TRIANGLE_CUH_

#include <cuda_runtime.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <utility>

#include "geometric_utils.cuh"

namespace surface_refinement {

class Triangle {
 public:
  __device__ inline Triangle(Eigen::Vector3d vertex_a,
                             Eigen::Vector3d vertex_n0,
                             Eigen::Vector3d vertex_n1);

  __device__ inline const Eigen::Vector3d& getVectorAToN0();
  __device__ inline const Eigen::Vector3d& getVectorAToN1();
  __device__ inline const Eigen::Vector3d& getVectorN0ToA();
  __device__ inline const Eigen::Vector3d& getVectorN0ToN1();
  __device__ inline const Eigen::Vector3d& getVectorN1ToA();
  __device__ inline const Eigen::Vector3d& getVectorN1ToN0();

  __device__ inline double getAngleA();
  __device__ inline double getAngleN0();
  __device__ inline double getAngleN1();

  __device__ inline double getCotangentN0();
  __device__ inline double getCotangentN1();

  __device__ inline double getArea();

  __device__ inline const Eigen::Vector3d& getNormal();

 private:
  // Vertices
  Eigen::Vector3d vertex_a_;
  Eigen::Vector3d vertex_n0_;
  Eigen::Vector3d vertex_n1_;

  // Cached vectors
  Eigen::Vector3d vector_a_to_n0_;
  Eigen::Vector3d vector_a_to_n1_;
  Eigen::Vector3d vector_n0_to_a_;
  Eigen::Vector3d vector_n0_to_n1_;
  Eigen::Vector3d vector_n1_to_a_;
  Eigen::Vector3d vector_n1_to_n0_;

  // Flags indicating if vectors have been computed
  bool computed_vector_a_to_n0_ = false;
  bool computed_vector_a_to_n1_ = false;
  bool computed_vector_n0_to_a_ = false;
  bool computed_vector_n0_to_n1_ = false;
  bool computed_vector_n1_to_a_ = false;
  bool computed_vector_n1_to_n0_ = false;

  // Cached angles
  double angle_a_;
  double angle_n0_;
  double angle_n1_;

  // Flags indicating if angles have been computed
  bool computed_angle_a_ = false;
  bool computed_angle_n0_ = false;
  bool computed_angle_n1_ = false;

  // Cached cotangents
  double cotangent_n0_;
  double cotangent_n1_;

  // Flags indicating if cotangents have been computed
  bool computed_cotangent_n0_ = false;
  bool computed_cotangent_n1_ = false;

  // Cached area
  double area_;
  bool computed_area_ = false;

  // Cached normal
  Eigen::Vector3d normal_;
  bool computed_normal_ = false;
};

__device__ Triangle::Triangle(Eigen::Vector3d vertex_a,
                              Eigen::Vector3d vertex_n0,
                              Eigen::Vector3d vertex_n1)
    : vertex_a_(vertex_a), vertex_n0_(vertex_n0), vertex_n1_(vertex_n1) {}

__device__ const Eigen::Vector3d& Triangle::getVectorAToN0() {
  if (!computed_vector_a_to_n0_) {
    vector_a_to_n0_ = vertex_n0_ - vertex_a_;
    computed_vector_a_to_n0_ = true;
  }
  return vector_a_to_n0_;
}

__device__ const Eigen::Vector3d& Triangle::getVectorAToN1() {
  if (!computed_vector_a_to_n1_) {
    vector_a_to_n1_ = vertex_n1_ - vertex_a_;
    computed_vector_a_to_n1_ = true;
  }
  return vector_a_to_n1_;
}

__device__ const Eigen::Vector3d& Triangle::getVectorN0ToA() {
  if (!computed_vector_n0_to_a_) {
    vector_n0_to_a_ = vertex_a_ - vertex_n0_;
    computed_vector_n0_to_a_ = true;
  }
  return vector_n0_to_a_;
}

__device__ const Eigen::Vector3d& Triangle::getVectorN0ToN1() {
  if (!computed_vector_n0_to_n1_) {
    vector_n0_to_n1_ = vertex_n1_ - vertex_n0_;
    computed_vector_n0_to_n1_ = true;
  }
  return vector_n0_to_n1_;
}

__device__ const Eigen::Vector3d& Triangle::getVectorN1ToA() {
  if (!computed_vector_n1_to_a_) {
    vector_n1_to_a_ = vertex_a_ - vertex_n1_;
    computed_vector_n1_to_a_ = true;
  }
  return vector_n1_to_a_;
}

__device__ const Eigen::Vector3d& Triangle::getVectorN1ToN0() {
  if (!computed_vector_n1_to_n0_) {
    vector_n1_to_n0_ = vertex_n0_ - vertex_n1_;
    computed_vector_n1_to_n0_ = true;
  }
  return vector_n1_to_n0_;
}

__device__ double Triangle::getAngleA() {
  if (!computed_angle_a_) {
    angle_a_ = GeometricUtils::compute_angle_between_vectors(getVectorAToN0(),
                                                             getVectorAToN1());
    computed_angle_a_ = true;
  }
  return angle_a_;
}

__device__ double Triangle::getAngleN0() {
  if (!computed_angle_n0_) {
    angle_n0_ = GeometricUtils::compute_angle_between_vectors(
        getVectorN0ToA(), getVectorN0ToN1());
    computed_angle_n0_ = true;
  }
  return angle_n0_;
}

__device__ double Triangle::getAngleN1() {
  if (!computed_angle_n1_) {
    angle_n1_ = GeometricUtils::compute_angle_between_vectors(
        getVectorN1ToA(), getVectorN1ToN0());
    computed_angle_n1_ = true;
  }
  return angle_n1_;
}

__device__ double Triangle::getCotangentN0() {
  if (!computed_cotangent_n0_) {
    cotangent_n0_ = 1.0 / tan(getAngleN0());
    computed_cotangent_n0_ = true;
  }
  return cotangent_n0_;
}

__device__ double Triangle::getCotangentN1() {
  if (!computed_cotangent_n1_) {
    cotangent_n1_ = 1.0 / tan(getAngleN1());
    computed_cotangent_n1_ = true;
  }
  return cotangent_n1_;
}

__device__ double Triangle::getArea() {
  if (!computed_area_) {
    area_ = 0.5 * getVectorAToN0().cross(getVectorAToN1()).norm();
    computed_area_ = true;
  }
  return area_;
}

__device__ const Eigen::Vector3d& Triangle::getNormal() {
  if (!computed_normal_) {
    normal_ = getVectorAToN0().cross(getVectorAToN1()).normalized();
    computed_normal_ = true;
  }
  return normal_;
}

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_TRIANGLE_CUH_
