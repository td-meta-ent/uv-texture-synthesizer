// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "mesh.hpp"

#include <cmath>
#include <iomanip>

namespace surface_refinement {

Mesh::Mesh(boost::filesystem::path mesh_file_path)
    : mesh_file_path_(std::move(mesh_file_path)) {
  LoadMesh();
  AllocateDeviceVariables();
  one_ring_indices_ = ComputeOneRingIndices();
  ComputeDeviceOneRingIndices();
}

Mesh::~Mesh() {
  CUDA_ERROR_CHECK(cudaFree(d_vertices_));
  CUDA_ERROR_CHECK(cudaFree(d_triangles_));
  CUDA_ERROR_CHECK(cudaFree(d_num_vertices_));
  CUDA_ERROR_CHECK(cudaFree(d_num_triangles_));
  CUDA_ERROR_CHECK(cudaFree(d_one_ring_indices_));
  CUDA_ERROR_CHECK(cudaFree(d_one_ring_indices_row_lengths_));
  CUDA_ERROR_CHECK(cudaFree(d_triangle_properties_));
  CUDA_ERROR_CHECK(cudaFree(d_one_ring_properties_));
}

void Mesh::LoadMesh() {
  LOG(INFO) << "Loading mesh from " << mesh_file_path_.string() << "...";

  if (!open3d::io::ReadTriangleMesh(mesh_file_path_.string(), open3d_mesh_)) {
    throw std::runtime_error("Failed to read mesh.");
  }

  vertices_ = open3d_mesh_.vertices_;
  num_vertices_ = static_cast<int>(vertices_.size());

  triangles_ = open3d_mesh_.triangles_;
  num_triangles_ = static_cast<int>(triangles_.size());

  LOG(INFO) << "Mesh loaded from " << mesh_file_path_.string() << ".";
}

void Mesh::AllocateDeviceVariables() {
  d_vertices_ = MemoryManager<Eigen::Vector3d>::AllocateArrayDevice(vertices_);
  d_triangles_ =
      MemoryManager<Eigen::Vector3i>::AllocateArrayDevice(triangles_);
  d_triangle_properties_ =
      MemoryManager<TriangleProperties>::AllocateArrayDevice(num_triangles_ *
                                                             3);
  d_one_ring_properties_ =
      MemoryManager<OneRingProperties>::AllocateArrayDevice(num_vertices_);
  d_num_vertices_ = MemoryManager<int>::AllocateScalarDevice(num_vertices_);
  d_num_triangles_ = MemoryManager<int>::AllocateScalarDevice(num_triangles_);
}

int Mesh::GetNumVertices() const { return num_vertices_; }

int Mesh::GetNumTriangles() const { return num_triangles_; }

Eigen::Vector3d* Mesh::GetDeviceVertices() const { return d_vertices_; }

Eigen::Vector3i* Mesh::GetDeviceTriangles() const { return d_triangles_; }

TriangleProperties* Mesh::GetDeviceTriangleProperties() const {
  return d_triangle_properties_;
}

OneRingProperties* Mesh::GetDeviceOneRingProperties() const {
  return d_one_ring_properties_;
}

int* Mesh::GetDeviceNumVertices() const { return d_num_vertices_; }

int* Mesh::GetDeviceNumTriangles() const { return d_num_triangles_; }

int* Mesh::GetDeviceOneRingIndices() const { return d_one_ring_indices_; }

int* Mesh::GetDeviceOneRingIndicesRowLengths() const {
  return d_one_ring_indices_row_lengths_;
}

std::vector<std::vector<int>> Mesh::ComputeOneRingIndices() const {
  std::vector<std::vector<int>> adjacent_triangles(vertices_.size());

  for (int i = 0; i < triangles_.size(); ++i) {
    for (int vertexIndex : triangles_[i]) {
      adjacent_triangles[vertexIndex].push_back(i);
    }
  }

  return adjacent_triangles;
}

void Mesh::ComputeDeviceOneRingIndices() {
  size_t num_rows = one_ring_indices_.size();
  size_t row_length = 20;
  size_t total_size = num_rows * row_length;

  std::vector<int> flattened_data(total_size, -1);
  std::vector<int> local_row_lengths(num_rows);

  for (std::size_t i = 0; i < num_rows; ++i) {
    for (std::size_t j = 0; j < one_ring_indices_[i].size() && j < row_length;
         ++j) {
      flattened_data[i * row_length + j] = one_ring_indices_[i][j];
    }
    local_row_lengths[i] = static_cast<int>(one_ring_indices_[i].size());
  }

  d_one_ring_indices_ = MemoryManager<int>::AllocateArrayDevice(flattened_data);
  d_one_ring_indices_row_lengths_ =
      MemoryManager<int>::AllocateArrayDevice(local_row_lengths);
}

void Mesh::SaveMesh(const std::vector<Eigen::Vector3d>& vertices,
                    const boost::filesystem::path& output_file_path) const {
  LOG(INFO) << "Saving mesh to " << output_file_path.string() << "...";

  open3d::geometry::TriangleMesh updated_mesh = open3d_mesh_;
  updated_mesh.vertices_ = vertices;

  open3d::io::WriteTriangleMesh(output_file_path.string(), updated_mesh);
  LOG(INFO) << "Mesh saved to " << output_file_path.string() << ".";
}

}  // namespace surface_refinement
