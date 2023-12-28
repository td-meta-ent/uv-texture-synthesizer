#include "camera.hpp"

namespace uv_texture_synthesizer {

Camera::Camera(boost::filesystem::path camera_parameters_path)
    : camera_params_path_(std::move(camera_parameters_path)) {
  LoadCameraParameters();
  d_camera_params_ =
      MemoryManager<CameraParams>::AllocateScalarDevice(camera_params_);
}

Camera::~Camera() { CUDA_ERROR_CHECK(cudaFree(d_camera_params_)) }

bool Camera::ReadMatrix(const cv::FileStorage& fs, const std::string& key,
                        cv::Mat* matrix, int rows, int cols) {
  fs[key] >> *matrix;
  if (matrix->empty() || matrix->rows != rows || matrix->cols != cols) {
    LOG(ERROR) << "Matrix '" << key << "' not found or has invalid dimensions.";
    return false;
  }
  return true;
}

void Camera::ApplyCoordinateTransformation(Eigen::Matrix4d* spatialMatrixPtr) {
  // Define the rotation matrix for -90 degrees about the Z-axis
  Eigen::Matrix4d zAxisRotationMatrix;
  zAxisRotationMatrix << 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

  // Perform the coordinate transformation directly on the matrix
  *spatialMatrixPtr =
      zAxisRotationMatrix * (*spatialMatrixPtr) * zAxisRotationMatrix.inverse();
}

void Camera::LoadCameraParameters() {
  LOG(INFO) << "Loading camera parameters from: "
            << camera_params_path_.string();

  if (!boost::filesystem::exists(camera_params_path_)) {
    LOG(ERROR) << "Camera parameters file not found: "
               << camera_params_path_.string();
    throw std::invalid_argument("Camera parameters file not found.");
  }

  cv::FileStorage fs(camera_params_path_.string(), cv::FileStorage::READ);
  if (!fs.isOpened()) {
    LOG(ERROR) << "Failed to open " << camera_params_path_.string();
    return;
  }

  cv::Mat intrinsics_matrix, rectification_matrix;
  if (!ReadMatrix(fs, "P", &intrinsics_matrix, 3, 4) ||
      !ReadMatrix(fs, "R", &rectification_matrix, 3, 3))
    return;

  camera_params_.focal_length_x = intrinsics_matrix.at<double>(1, 1);
  camera_params_.focal_length_y = intrinsics_matrix.at<double>(0, 0);
  camera_params_.principal_point_x =
      Image::imageHeight - intrinsics_matrix.at<double>(1, 2);
  camera_params_.principal_point_y = intrinsics_matrix.at<double>(0, 2);

  cv::Mat rectification_matrix_extended =
      cv::Mat::eye(4, 4, rectification_matrix.type());
  rectification_matrix.copyTo(
      rectification_matrix_extended(cv::Rect(0, 0, 3, 3)));
  cv2eigen(rectification_matrix_extended, camera_params_.rectification_matrix);
  ApplyCoordinateTransformation(&camera_params_.rectification_matrix);

  // Search for translation matrix with pattern 'T_0_*'
  cv::FileNode translation_node;
  std::regex pattern("T_0_\\d+");
  for (const auto& node : fs.root()) {
    if (std::regex_match(node.name(), pattern)) {
      translation_node = node;
      break;
    }
  }

  if (translation_node.empty()) {
    LOG(ERROR) << "Translation matrix with pattern 'T_0_*' not found.";
    return;
  }

  cv::Mat T_0_x;
  if (!ReadMatrix(fs, translation_node.name(), &T_0_x, 4, 4)) return;
  cv2eigen(T_0_x, camera_params_.transformation_matrix);
  ApplyCoordinateTransformation(&camera_params_.transformation_matrix);
}

CameraParams Camera::GetCameraParams() { return camera_params_; }

CameraParams* Camera::GetDeviceCameraParams() { return d_camera_params_; }

}  // namespace uv_texture_synthesizer