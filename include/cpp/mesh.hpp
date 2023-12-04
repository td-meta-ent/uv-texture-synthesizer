// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_MESH_HPP_
#define SURFACE_REFINEMENT_MESH_HPP_

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <open3d/Open3D.h>

#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <vector>

#include "camera_manager.hpp"
#include "memory_manager.hpp"
#include "one_ring_kernel_wrappers.hpp"
#include "photo_kernel_wrappers.hpp"
#include "refiner_kernel_wrappers.hpp"
#include "triangle_kernel_wrappers.hpp"

namespace surface_refinement {

/**
 * @file mesh.hpp
 * @class Mesh
 * @brief Represents a 3D mesh with operations for loading and saving.
 * @details Provides functionality to load a 3D mesh from a file, retrieve
 * vertices and triangles, and save back to a file.
 */
class Mesh {
 public:
  explicit Mesh(boost::filesystem::path mesh_file_path,
                const CameraManager& camera_manager);
  ~Mesh();

  void LoadMesh();
  void SaveMesh(std::vector<Eigen::Vector3d>* vertices,
                const boost::filesystem::path& output_file_path) const;

  std::vector<Eigen::Vector3d> GetVertices() const;
  Eigen::Vector3d* GetDeviceVertices() const;
  int GetNumVertices() const;
  int* GetDeviceNumVertices() const;
  Eigen::Vector3i* GetDeviceTriangles() const;
  int GetNumTriangles() const;
  int* GetDeviceNumTriangles() const;
  TriangleProperties* GetDeviceTriangleProperties() const;
  TriangleProperties* GetDeviceTrianglePropertiesFixedNormal() const;
  OneRingProperties* GetDeviceOneRingProperties() const;
  PhotometricProperties* GetDevicePhotometricProperties() const;
  int* GetDeviceOneRingIndices() const;
  int* GetDeviceOneRingIndicesRowLengths() const;
  DeltaVertex* GetDeviceDeltaVertices() const;
  Eigen::Vector2d* GetDeviceProjectedPixelIndices() const;
  double* GetDevicePatches() const;

 private:
  void AllocateDeviceVariables();

  boost::filesystem::path mesh_file_path_;
  mutable open3d::geometry::TriangleMesh open3d_mesh_;
  std::vector<Eigen::Vector3d> vertices_;
  std::vector<Eigen::Vector3i> triangles_;
  int num_vertices_{};
  int num_triangles_{};
  std::vector<std::vector<int>> one_ring_indices_{};

  Eigen::Vector3d* d_vertices_{nullptr};
  Eigen::Vector3i* d_triangles_{nullptr};
  int* d_num_vertices_{nullptr};
  int* d_num_triangles_{nullptr};
  TriangleProperties* d_triangle_properties_{nullptr};
  TriangleProperties* d_triangle_properties_fixed_normal_{nullptr};
  OneRingProperties* d_one_ring_properties_{nullptr};
  PhotometricProperties* d_photometric_properties_{nullptr};
  int* d_one_ring_indices_{nullptr};
  int* d_one_ring_indices_row_lengths_{nullptr};
  DeltaVertex* d_delta_vertices_{nullptr};
  Eigen::Vector2d* d_projected_pixel_indices_{nullptr};
  double* d_patches_{nullptr};
  std::vector<std::vector<int>> ComputeOneRingIndices() const;
  void ComputeDeviceOneRingIndices();

  CameraManager camera_manager_;

  void FreeDeviceMemory() const;
};

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_MESH_HPP_
