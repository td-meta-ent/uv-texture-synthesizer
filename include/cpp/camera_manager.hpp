#ifndef UV_TEXTURE_SYNTHESIZER_CAMERA_MANAGER_HPP_
#define UV_TEXTURE_SYNTHESIZER_CAMERA_MANAGER_HPP_

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <string>
#include <vector>

#include "camera.hpp"
#include "memory_manager.hpp"

namespace uv_texture_synthesizer {

class CameraManager {
 public:
  CameraManager(boost::filesystem::path camera_parameters_dir,
                const std::string& date, const std::string& cut_number);
  ~CameraManager();

  [[nodiscard]] std::vector<CameraParams> GetCameraParams() const;
  [[nodiscard]] CameraParams* GetDeviceCameraParams() const;

 private:
  void LoadCameraParams(const std::string& date, const std::string& cut_number);

  boost::filesystem::path camera_params_dir_;

  std::vector<CameraParams> camera_params_;
  CameraParams* d_camera_params_ = {nullptr};
};

}  // namespace uv_texture_synthesizer

#endif  // UV_TEXTURE_SYNTHESIZER_CAMERA_MANAGER_HPP_
