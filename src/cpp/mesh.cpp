// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "mesh.hpp"

namespace surface_refinement {

Mesh::Mesh(boost::filesystem::path mesh_file_path,
           const CameraManager& camera_manager)
    : mesh_file_path_(std::move(mesh_file_path)),
      camera_manager_(camera_manager) {
  LoadMesh();
  AllocateDeviceVariables();
  one_ring_indices_ = ComputeOneRingIndices();
  ComputeDeviceOneRingIndices();
}

Mesh::~Mesh() = default;

void Mesh::FreeDeviceMemory() const {
  CUDA_ERROR_CHECK(cudaFree(d_vertices_))
  CUDA_ERROR_CHECK(cudaFree(d_num_vertices_))
  CUDA_ERROR_CHECK(cudaFree(d_triangles_))
  CUDA_ERROR_CHECK(cudaFree(d_num_triangles_))
  CUDA_ERROR_CHECK(cudaFree(d_one_ring_indices_))
  CUDA_ERROR_CHECK(cudaFree(d_one_ring_indices_row_lengths_))
  CUDA_ERROR_CHECK(cudaFree(d_triangle_properties_))
  CUDA_ERROR_CHECK(cudaFree(d_triangle_properties_fixed_normal_))
  CUDA_ERROR_CHECK(cudaFree(d_one_ring_properties_))
  CUDA_ERROR_CHECK(cudaFree(d_photometric_properties_))
  CUDA_ERROR_CHECK(cudaFree(d_delta_vertices_))
  CUDA_ERROR_CHECK(cudaFree(d_projected_pixel_indices_))
  CUDA_ERROR_CHECK(cudaFree(d_patches_))
}

void Mesh::LoadMesh() {
  LOG(INFO) << "Loading mesh from " << mesh_file_path_.string() << "...";

  if (!open3d::io::ReadTriangleMesh(mesh_file_path_.string(), open3d_mesh_)) {
    throw std::runtime_error("Failed to read mesh.");
  }

  open3d_mesh_.RemoveDuplicatedVertices();
  open3d_mesh_.RemoveDuplicatedTriangles();
  open3d_mesh_.RemoveUnreferencedVertices();
  open3d_mesh_.RemoveDegenerateTriangles();
  open3d_mesh_.RemoveNonManifoldEdges();

  vertices_ = open3d_mesh_.vertices_;

  // Convert from meters to millimeters
  for (auto& vertex : vertices_) {
    vertex *= 1000;
  }

  for (Eigen::Vector3d& vertex : vertices_) {
    // Convert the 3D vertex into a 4D homogeneous coordinate
    Eigen::Vector4d vertex_homogeneous(vertex[0], vertex[1], vertex[2], 1.0);

    // Apply the transformation
    Eigen::Vector4d transformed_vertex =
        camera_manager_.GetCameraParams()[0].rectification_matrix.inverse() *
        vertex_homogeneous;

    // Convert back to 3D
    vertex = transformed_vertex.head<3>() / transformed_vertex.w();
  }

  num_vertices_ = static_cast<int>(vertices_.size());

  triangles_ = open3d_mesh_.triangles_;
  num_triangles_ = static_cast<int>(triangles_.size());

  LOG(INFO) << "Mesh loaded from " << mesh_file_path_.string() << ".";
}

void Mesh::AllocateDeviceVariables() {
  d_vertices_ = MemoryManager<Eigen::Vector3d>::AllocateArrayDevice(vertices_);
  d_num_vertices_ = MemoryManager<int>::AllocateScalarDevice(num_vertices_);
  d_triangles_ =
      MemoryManager<Eigen::Vector3i>::AllocateArrayDevice(triangles_);
  d_num_triangles_ = MemoryManager<int>::AllocateScalarDevice(num_triangles_);
  d_triangle_properties_ =
      MemoryManager<TriangleProperties>::AllocateArrayDevice(num_triangles_ *
                                                             3);
  d_triangle_properties_fixed_normal_ =
      MemoryManager<TriangleProperties>::AllocateArrayDevice(num_triangles_ *
                                                             3);
  d_one_ring_properties_ =
      MemoryManager<OneRingProperties>::AllocateArrayDevice(num_vertices_);

  d_photometric_properties_ =
      MemoryManager<PhotometricProperties>::AllocateArrayDevice(num_vertices_);
  int num_delta_vertices = num_vertices_ * 3 * 2;
  d_delta_vertices_ =
      MemoryManager<DeltaVertex>::AllocateArrayDevice(num_delta_vertices);
  int num_patches = num_delta_vertices;
  d_projected_pixel_indices_ =
      MemoryManager<Eigen::Vector2d>::AllocateArrayDevice(num_patches);
  int num_patch_pixel = num_patches * 3 * 3;
  d_patches_ = MemoryManager<double>::AllocateArrayDevice(num_patch_pixel);
}

int Mesh::GetNumVertices() const { return num_vertices_; }

int Mesh::GetNumTriangles() const { return num_triangles_; }

std::vector<Eigen::Vector3d> Mesh::GetVertices() const { return vertices_; }

Eigen::Vector3d* Mesh::GetDeviceVertices() const { return d_vertices_; }

int* Mesh::GetDeviceNumVertices() const { return d_num_vertices_; }

Eigen::Vector3i* Mesh::GetDeviceTriangles() const { return d_triangles_; }

int* Mesh::GetDeviceNumTriangles() const { return d_num_triangles_; }

TriangleProperties* Mesh::GetDeviceTriangleProperties() const {
  return d_triangle_properties_;
}

TriangleProperties* Mesh::GetDeviceTrianglePropertiesFixedNormal() const {
  return d_triangle_properties_fixed_normal_;
}

OneRingProperties* Mesh::GetDeviceOneRingProperties() const {
  return d_one_ring_properties_;
}

PhotometricProperties* Mesh::GetDevicePhotometricProperties() const {
  return d_photometric_properties_;
}

int* Mesh::GetDeviceOneRingIndices() const { return d_one_ring_indices_; }

int* Mesh::GetDeviceOneRingIndicesRowLengths() const {
  return d_one_ring_indices_row_lengths_;
}

DeltaVertex* Mesh::GetDeviceDeltaVertices() const { return d_delta_vertices_; }

Eigen::Vector2d* Mesh::GetDeviceProjectedPixelIndices() const {
  return d_projected_pixel_indices_;
}

double* Mesh::GetDevicePatches() const { return d_patches_; }

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

void Mesh::SaveMesh(std::vector<Eigen::Vector3d>* vertices,
                    const boost::filesystem::path& output_file_path) const {
  if (!vertices) {
    LOG(ERROR) << "Vertices pointer is null.";
    return;
  }

  LOG(INFO) << "Saving mesh to " << output_file_path.string() << "...";

  open3d::geometry::TriangleMesh updated_mesh = open3d_mesh_;

  for (auto& vertex : *vertices) {
    Eigen::Vector4d vertex_homogeneous(vertex[0], vertex[1], vertex[2], 1.0);
    Eigen::Vector4d transformed_vertex =
        camera_manager_.GetCameraParams()[0].rectification_matrix *
        vertex_homogeneous;
    vertex = transformed_vertex.head<3>() / transformed_vertex.w();
  }

  for (auto& vertex : *vertices) {
    vertex /= 1000;
  }

  updated_mesh.vertices_ = *vertices;

  updated_mesh.ComputeVertexNormals();

  open3d::io::WriteTriangleMesh(output_file_path.string(), updated_mesh);
  LOG(INFO) << "Mesh saved to " << output_file_path.string() << ".";

  FreeDeviceMemory();
}

}  // namespace surface_refinement
