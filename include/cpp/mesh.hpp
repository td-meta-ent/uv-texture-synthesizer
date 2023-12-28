#ifndef UV_TEXTURE_SYNTHESIZER_MESH_HPP_
#define UV_TEXTURE_SYNTHESIZER_MESH_HPP_

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <open3d/Open3D.h>

#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <vector>

#include "camera_manager.hpp"
#include "memory_manager.hpp"
#include "one_ring_kernel_wrappers.hpp"
#include "texture.hpp"

namespace uv_texture_synthesizer {

class Mesh {
 public:
  explicit Mesh(boost::filesystem::path mesh_file_path,
                const CameraManager& camera_manager, int num_cameras);
  ~Mesh();

  void LoadMesh();

  std::vector<Eigen::Vector3d> GetVertices() const;
  Eigen::Vector3d* GetDeviceVertices() const;
  int GetNumVertices() const;
  int* GetDeviceNumVertices() const;

  int GetNumTriangles() const;
  int* GetDeviceNumTriangles() const;

  Eigen::Vector3i* GetDeviceTriangles() const;

  Eigen::Vector3d* GetDeviceVertexNormals() const;

  Eigen::Vector2d* GetDeviceTriangleUVs() const;

  OneRingProperties* GetDeviceOneRingProperties() const;

  Eigen::Vector3d* GetDeviceCentroid() const;

 private:
  void AllocateDeviceVariables();

  int num_cameras_;

  boost::filesystem::path mesh_file_path_;
  mutable open3d::geometry::TriangleMesh open3d_mesh_;
  std::vector<Eigen::Vector3d> vertices_;
  std::vector<Eigen::Vector3d> vertex_normals_;
  std::vector<Eigen::Vector3i> triangles_;
  std::vector<Eigen::Vector2d> triangle_uvs_;
  Eigen::Vector3d centroid_{Eigen::Vector3d::Zero()};

  int num_vertices_{};
  int num_triangles_{};
  std::vector<std::vector<int>> one_ring_indices_{};

  Eigen::Vector3d* d_vertices_{nullptr};
  int* d_num_vertices_{nullptr};
  Eigen::Vector3d* d_vertex_normals_{nullptr};
  Eigen::Vector3i* d_triangles_{nullptr};
  int* d_num_triangles_{nullptr};
  Eigen::Vector2d* d_triangle_uvs_{nullptr};
  Eigen::Vector3d* d_centroid_{nullptr};
  OneRingProperties* d_one_ring_properties_{nullptr};

  CameraManager camera_manager_;

  void FreeDeviceMemory() const;
};

}  // namespace uv_texture_synthesizer

#endif  // UV_TEXTURE_SYNTHESIZER_MESH_HPP_
