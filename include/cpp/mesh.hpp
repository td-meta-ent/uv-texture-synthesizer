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

#include "memory_manager.hpp"
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
  explicit Mesh(boost::filesystem::path mesh_file_path);
  ~Mesh();

  void LoadMesh();
  void SaveMesh(const std::vector<Eigen::Vector3d>& vertices,
                const boost::filesystem::path& output_file_path) const;

  int GetNumVertices() const;
  int GetNumTriangles() const;

  Eigen::Vector3d* GetDeviceVertices() const;
  Eigen::Vector3i* GetDeviceTriangles() const;
  TriangleProperties* GetDeviceTriangleProperties() const;
  OneRingProperties* GetDeviceOneRingProperties() const;
  int* GetDeviceNumVertices() const;
  int* GetDeviceNumTriangles() const;
  int* GetDeviceOneRingIndices() const;
  int* GetDeviceOneRingIndicesRowLengths() const;

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
  OneRingProperties* d_one_ring_properties_{nullptr};
  int* d_one_ring_indices_{nullptr};
  int* d_one_ring_indices_row_lengths_{nullptr};
  std::vector<std::vector<int>> ComputeOneRingIndices() const;
  void ComputeDeviceOneRingIndices();
};

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_MESH_HPP_
