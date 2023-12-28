#include "camera_manager.hpp"

#include <open3d/Open3D.h>

namespace uv_texture_synthesizer {

CameraManager::CameraManager(boost::filesystem::path camera_parameters_dir,
                             const std::string& date,
                             const std::string& cut_number)
    : camera_params_dir_(std::move(camera_parameters_dir)) {
  LoadCameraParams(date, cut_number);
  d_camera_params_ =
      MemoryManager<CameraParams>::AllocateArrayDevice(camera_params_);
}

CameraManager::~CameraManager() = default;

void CameraManager::LoadCameraParams(const std::string& date,
                                     const std::string& cut_number) {
  std::vector<boost::filesystem::path> files;

  if (!boost::filesystem::exists(camera_params_dir_) ||
      !boost::filesystem::is_directory(camera_params_dir_)) {
    LOG(ERROR) << "Invalid camera parameters directory: "
               << camera_params_dir_.string();
    return;
  }

  for (const auto& entry :
       boost::filesystem::directory_iterator(camera_params_dir_)) {
    if (!boost::filesystem::is_regular_file(entry)) continue;

    std::string expected_prefix = "rectification_";
    expected_prefix += date;
    expected_prefix += "_test_";
    expected_prefix += cut_number;
    expected_prefix += "_";

    std::string filename = entry.path().filename().string();

    if (filename.find(expected_prefix) == 0 &&
        filename.substr(filename.find_last_of('.') + 1) == "yml") {
      files.push_back(entry.path());
    }
  }

  std::vector<Eigen::Vector3d> camera_normals;

  // Sort the files by file name
  std::sort(files.begin(), files.end());

  Camera cam_8 = Camera(files[8]);
  Eigen::Matrix4d t_0_8 = cam_8.GetCameraParams().rectification_matrix *
                          cam_8.GetCameraParams().transformation_matrix;

  for (const auto& file : files) {
    Camera camera(file);
    CameraParams camera_params = camera.GetCameraParams();

    camera_params.transformation_matrix =
        t_0_8 * camera_params.transformation_matrix.inverse() *
        camera_params.rectification_matrix.inverse();

    // Camera's viewing direction in camera space
    Eigen::Vector3d camera_view_direction(0.0, 0.0, 1.0);

    // Set camera_normal to point in the opposite direction and normalize
    camera_params.camera_normal =
        -(camera_params.transformation_matrix.block<3, 3>(0, 0) *
          camera_view_direction)
             .normalized();
//    camera_params.camera_normal = camera_params.transformation_matrix.block<3, 3>(0, 0) * camera_params.camera_normal;
//    camera_params.camera_normal = camera_params.transformation_matrix.block<3, 3>(0, 0) * camera_params.camera_normal;
//    camera_params.camera_normal = camera_params.transformation_matrix.block<3, 3>(0, 0).transpose() * camera_params.camera_normal;

    camera_params.transformation_matrix =
        camera_params.transformation_matrix.inverse().eval();

    camera_params_.push_back(camera_params);

    camera_normals.push_back(camera_params.camera_normal);
  }
}

std::vector<CameraParams> CameraManager::GetCameraParams() const {
  return camera_params_;
}

CameraParams* CameraManager::GetDeviceCameraParams() const {
  return d_camera_params_;
}

}  // namespace uv_texture_synthesizer
