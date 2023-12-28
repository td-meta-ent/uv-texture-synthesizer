#ifndef UV_TEXTURE_SYNTHESIZER_SYNTHESIZER_HPP_
#define UV_TEXTURE_SYNTHESIZER_SYNTHESIZER_HPP_

#include <glog/logging.h>

#include <Eigen/Core>
#include <vector>

#include "camera_manager.hpp"
#include "image_manager.hpp"
#include "memory_manager.hpp"
#include "mesh.hpp"
#include "synthesizer_kernel_wrappers.hpp"
#include "texture.hpp"

namespace uv_texture_synthesizer {

class Synthesizer {
 public:
  Synthesizer(const Mesh &mesh, const CameraManager &camera_manager,
              const ImageManager &image_manager, const Texture &texture,
              const int num_cameras);
  ~Synthesizer();

  std::vector<Eigen::Vector3d> LaunchSynthesis();

 private:
  Mesh mesh_;
  CameraManager camera_manager_;
  ImageManager image_manager_;
  Texture texture_;

  int num_cameras_;
  int *d_num_cameras_;
};

}  // namespace uv_texture_synthesizer

#endif  // UV_TEXTURE_SYNTHESIZER_SYNTHESIZER_HPP_
