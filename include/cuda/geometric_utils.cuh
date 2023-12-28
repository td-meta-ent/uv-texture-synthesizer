#ifndef UV_TEXTURE_SYNTHESIZER_GEOMETRIC_UTILS_CUH_
#define UV_TEXTURE_SYNTHESIZER_GEOMETRIC_UTILS_CUH_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

#include "camera.hpp"

namespace uv_texture_synthesizer {

class GeometricUtils {
 public:
  __device__ static inline bool isPointInsideOrOnEdgeOfTriangle2D(
      const Eigen::Vector2d &pt, const Eigen::Vector2d *tri);

  __device__ static inline double triangleArea2D(const Eigen::Vector2d &p1,
                                                 const Eigen::Vector2d &p2,
                                                 const Eigen::Vector2d &p3);

  __device__ static inline Eigen::Vector3d barycentricCoords2D(
      const Eigen::Vector2d &pt, const Eigen::Vector2d *tri);

  __device__ static inline void calculateBoundingBox2D(
      const Eigen::Vector2d *tri, double &min_x, double &max_x, double &min_y,
      double &max_y);

  __device__ static inline Eigen::Vector2d projectVertex3D(
      Eigen::Vector3d vertex, const CameraParams *d_camera_params);

  __device__ static inline Eigen::Vector3i bilinearInterpolationAtPoint2DKernel(
      Eigen::Vector3i *d_image, const int d_image_width,
      const int d_image_height, const Eigen::Vector2d &d_point);
};

__device__ inline bool GeometricUtils::isPointInsideOrOnEdgeOfTriangle2D(
    const Eigen::Vector2d &pt, const Eigen::Vector2d *tri) {
  auto sign = [](const Eigen::Vector2d &p1, const Eigen::Vector2d &p2,
                 const Eigen::Vector2d &p3) {
    return (p1.x() - p3.x()) * (p2.y() - p3.y()) -
           (p2.x() - p3.x()) * (p1.y() - p3.y());
  };

  bool b1 = sign(pt, tri[0], tri[1]) <= 0.0;
  bool b2 = sign(pt, tri[1], tri[2]) <= 0.0;
  bool b3 = sign(pt, tri[2], tri[0]) <= 0.0;

  return ((b1 == b2) && (b2 == b3));
}

__device__ inline double GeometricUtils::triangleArea2D(
    const Eigen::Vector2d &p1, const Eigen::Vector2d &p2,
    const Eigen::Vector2d &p3) {
  return 0.5 * abs((p1.x() * (p2.y() - p3.y()) + p2.x() * (p3.y() - p1.y()) +
                    p3.x() * (p1.y() - p2.y())));
}

__device__ inline Eigen::Vector3d GeometricUtils::barycentricCoords2D(
    const Eigen::Vector2d &pt, const Eigen::Vector2d *tri) {
  Eigen::Vector2d v0 = tri[0], v1 = tri[1], v2 = tri[2];

  double a0 = triangleArea2D(pt, v1, v2);
  double a1 = triangleArea2D(pt, v0, v2);
  double a2 = triangleArea2D(pt, v0, v1);
  double total_area = a0 + a1 + a2;

  if (total_area == 0) {
    return {0, 0, 0};
  }

  return {a0 / total_area, a1 / total_area, a2 / total_area};
}

__device__ inline void GeometricUtils::calculateBoundingBox2D(
    const Eigen::Vector2d *tri, double &min_x, double &max_x, double &min_y,
    double &max_y) {
  min_x = max_x = tri[0].x();
  min_y = max_y = tri[0].y();

  for (int i = 1; i < 3; ++i) {
    if (tri[i].x() < min_x) min_x = tri[i].x();
    if (tri[i].x() > max_x) max_x = tri[i].x();
    if (tri[i].y() < min_y) min_y = tri[i].y();
    if (tri[i].y() > max_y) max_y = tri[i].y();
  }
}

__device__ inline Eigen::Vector2d GeometricUtils::projectVertex3D(
    Eigen::Vector3d vertex, const CameraParams *d_camera_params) {
  Eigen::Vector4d vertex_homogeneous(vertex(0), vertex(1), vertex(2), 1.0);

  vertex_homogeneous =
      d_camera_params->transformation_matrix * vertex_homogeneous;
  vertex = vertex_homogeneous.head<3>();

  Eigen::Vector2d pixel_indices;
  pixel_indices(0) = d_camera_params->focal_length_y * vertex(1) / vertex(2) +
                     d_camera_params->principal_point_y;
  pixel_indices(1) = d_camera_params->focal_length_x * vertex(0) / vertex(2) +
                     d_camera_params->principal_point_x;
  return pixel_indices;
}

__device__ inline Eigen::Vector3i
GeometricUtils::bilinearInterpolationAtPoint2DKernel(
    Eigen::Vector3i *d_image, const int d_image_width, const int d_image_height,
    const Eigen::Vector2d &d_point) {
  int d_x = static_cast<int>(floor(d_point.x()));
  int d_y = static_cast<int>(floor(d_point.y()));

  double d_frac_x = d_point.x() - d_x;
  double d_frac_y = d_point.y() - d_y;

  // Boundary check
  if (d_x < 0 || d_x >= d_image_width - 1 || d_y < 0 ||
      d_y >= d_image_height - 1) {
    return {0, 0, 0};
  }

  // Get the four neighboring pixels
  Eigen::Vector3i d_color_tl = d_image[d_y * d_image_width + d_x];  // Top left
  Eigen::Vector3i d_color_tr =
      d_image[d_y * d_image_width + (d_x + 1)];  // Top right
  Eigen::Vector3i d_color_bl =
      d_image[(d_y + 1) * d_image_width + d_x];  // Bottom left
  Eigen::Vector3i d_color_br =
      d_image[(d_y + 1) * d_image_width + (d_x + 1)];  // Bottom right

  // Perform bilinear interpolation
  Eigen::Vector3i interpolated_color;
  for (int i = 0; i < 3; ++i) {
    interpolated_color[i] =
        static_cast<int>((1 - d_frac_x) * (1 - d_frac_y) * d_color_tl[i] +
                         d_frac_x * (1 - d_frac_y) * d_color_tr[i] +
                         (1 - d_frac_x) * d_frac_y * d_color_bl[i] +
                         d_frac_x * d_frac_y * d_color_br[i]);
  }

  return interpolated_color;
}

}  // namespace uv_texture_synthesizer

#endif  // UV_TEXTURE_SYNTHESIZER_GEOMETRIC_UTILS_CUH_
