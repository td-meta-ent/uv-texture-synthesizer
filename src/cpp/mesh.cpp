#include "mesh.hpp"

namespace uv_texture_synthesizer {

Mesh::Mesh(boost::filesystem::path mesh_file_path,
           const CameraManager& camera_manager, int num_cameras)
    : mesh_file_path_(std::move(mesh_file_path)),
      camera_manager_(camera_manager),
      num_cameras_(num_cameras) {
  LoadMesh();
  AllocateDeviceVariables();
}

Mesh::~Mesh() { FreeDeviceMemory(); }

void Mesh::FreeDeviceMemory() const {
  CUDA_ERROR_CHECK(cudaFree(d_vertices_))
  CUDA_ERROR_CHECK(cudaFree(d_num_vertices_))
  CUDA_ERROR_CHECK(cudaFree(d_vertex_normals_))
  CUDA_ERROR_CHECK(cudaFree(d_triangles_))
  CUDA_ERROR_CHECK(cudaFree(d_num_triangles_))
  CUDA_ERROR_CHECK(cudaFree(d_triangle_uvs_))
  CUDA_ERROR_CHECK(cudaFree(d_one_ring_properties_))
}

void Mesh::LoadMesh() {
  LOG(INFO) << "Loading mesh from " << mesh_file_path_.string() << "...";

  if (!open3d::io::ReadTriangleMesh(mesh_file_path_.string(), open3d_mesh_)) {
    throw std::runtime_error("Failed to read mesh.");
  }

  vertices_ = open3d_mesh_.vertices_;

  triangle_uvs_ = open3d_mesh_.triangle_uvs_;

  // Rotate mesh 90 degrees around z-axis
  Eigen::Matrix3d zAxisRotationMatrix;
  zAxisRotationMatrix << 0, -1, 0, 1, 0, 0, 0, 0, 1;

  for (Eigen::Vector3d& vertex : vertices_) {
    vertex = zAxisRotationMatrix * vertex;
  }

  open3d_mesh_.ComputeVertexNormals();

  vertex_normals_ = open3d_mesh_.vertex_normals_;

  num_vertices_ = static_cast<int>(vertices_.size());

  triangles_ = open3d_mesh_.triangles_;

  num_triangles_ = static_cast<int>(triangles_.size());

  // Compute centroid
  centroid_ = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d& vertex : vertices_) {
    centroid_ += vertex;
  }
  centroid_ /= double(num_vertices_);
  LOG(INFO) << "Centroid: " << centroid_;

  auto& normals = open3d_mesh_.vertex_normals_;
  double epsilon = 0.01;
  double front_epsilon = 0.1;

  // Calculate angles and masks
  std::vector<bool> angle_mask(normals.size());
  std::vector<bool> centroid_mask(normals.size());
  double angle_filter_radians = M_PI / 2;

  for (size_t i = 0; i < normals.size(); ++i) {
    double cos_angle =
        normals[i].dot(camera_manager_.GetCameraParams()[0].camera_normal);
    double angle = std::acos(cos_angle);
    angle_mask[i] = angle > angle_filter_radians;

    Eigen::Vector3d direction = open3d_mesh_.vertices_[i] - centroid_;
    centroid_mask[i] =
        direction.dot(camera_manager_.GetCameraParams()[0].camera_normal) < 0;
  }

  // Combine masks
  std::vector<bool> combined_mask(normals.size());
  for (size_t i = 0; i < normals.size(); ++i) {
    combined_mask[i] = angle_mask[i] || centroid_mask[i];
  }

  // Create a sphere at the centroid for visualization
  auto centroid_sphere = std::make_shared<open3d::geometry::TriangleMesh>(
      *open3d::geometry::TriangleMesh::CreateSphere(0.02));
  centroid_sphere->Translate(centroid_);
  centroid_sphere->PaintUniformColor(Eigen::Vector3d(1, 0, 0));

  // visualize mesh and sphere
  std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geometries;
  auto mesh_ptr =
      std::make_shared<open3d::geometry::TriangleMesh>(open3d_mesh_);
  geometries.push_back(centroid_sphere);
  geometries.push_back(mesh_ptr);

  open3d::visualization::DrawGeometries(geometries);

  LOG(INFO) << "Mesh loaded from " << mesh_file_path_.string() << ".";
}

void Mesh::AllocateDeviceVariables() {
  d_vertices_ = MemoryManager<Eigen::Vector3d>::AllocateArrayDevice(vertices_);
  d_num_vertices_ = MemoryManager<int>::AllocateScalarDevice(num_vertices_);
  d_triangles_ =
      MemoryManager<Eigen::Vector3i>::AllocateArrayDevice(triangles_);
  d_num_triangles_ = MemoryManager<int>::AllocateScalarDevice(num_triangles_);
  d_vertex_normals_ =
      MemoryManager<Eigen::Vector3d>::AllocateArrayDevice(vertex_normals_);
  d_triangle_uvs_ =
      MemoryManager<Eigen::Vector2d>::AllocateArrayDevice(triangle_uvs_);
  d_one_ring_properties_ =
      MemoryManager<OneRingProperties>::AllocateArrayDevice(num_vertices_ *
                                                            num_cameras_);
  d_centroid_ = MemoryManager<Eigen::Vector3d>::AllocateScalarDevice(centroid_);
}

int Mesh::GetNumVertices() const { return num_vertices_; }

std::vector<Eigen::Vector3d> Mesh::GetVertices() const { return vertices_; }

Eigen::Vector3d* Mesh::GetDeviceVertices() const { return d_vertices_; }

int* Mesh::GetDeviceNumVertices() const { return d_num_vertices_; }

int Mesh::GetNumTriangles() const { return num_triangles_; }

int* Mesh::GetDeviceNumTriangles() const { return d_num_triangles_; }

Eigen::Vector3i* Mesh::GetDeviceTriangles() const { return d_triangles_; }

Eigen::Vector3d* Mesh::GetDeviceVertexNormals() const {
  return d_vertex_normals_;
}

Eigen::Vector2d* Mesh::GetDeviceTriangleUVs() const { return d_triangle_uvs_; }

OneRingProperties* Mesh::GetDeviceOneRingProperties() const {
  return d_one_ring_properties_;
}

Eigen::Vector3d* Mesh::GetDeviceCentroid() const { return d_centroid_; }

}  // namespace uv_texture_synthesizer
