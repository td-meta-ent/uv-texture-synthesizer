//#ifndef UV_TEXTURE_SYNTHESIZER_TRIANGLE_CUH_
//#define UV_TEXTURE_SYNTHESIZER_TRIANGLE_CUH_
//
//#include <cuda_runtime.h>
//
//#include <Eigen/Core>
//#include <Eigen/Geometry>
//#include <utility>
//
//#include "geometric_utils.cuh"
//
//namespace uv_texture_synthesizer {
//
//class Triangle {
// public:
//  __device__ inline Triangle(Eigen::Vector3d vertex_a,
//                             Eigen::Vector3d vertex_n0,
//                             Eigen::Vector3d vertex_n1);
//
//  __device__ inline const Eigen::Vector3d& getVectorAToN0();
//  __device__ inline const Eigen::Vector3d& getVectorAToN1();
//
//  __device__ inline double getArea();
//
//  __device__ inline const Eigen::Vector3d& getNormal();
//
// private:
//  // Vertices
//  Eigen::Vector3d vertex_a_;
//  Eigen::Vector3d vertex_n0_;
//  Eigen::Vector3d vertex_n1_;
//
//  // Cached vectors
//  Eigen::Vector3d vector_a_to_n0_;
//  Eigen::Vector3d vector_a_to_n1_;
//  Eigen::Vector3d vector_n0_to_a_;
//  Eigen::Vector3d vector_n0_to_n1_;
//  Eigen::Vector3d vector_n1_to_a_;
//  Eigen::Vector3d vector_n1_to_n0_;
//
//  // Flags indicating if vectors have been computed
//  bool computed_vector_a_to_n0_ = false;
//  bool computed_vector_a_to_n1_ = false;
//  bool computed_vector_n1_to_a_ = false;
//
//  // Cached area
//  double area_;
//  bool computed_area_ = false;
//
//  // Cached normal
//  Eigen::Vector3d normal_;
//  bool computed_normal_ = false;
//};
//
//__device__ Triangle::Triangle(Eigen::Vector3d vertex_a,
//                              Eigen::Vector3d vertex_n0,
//                              Eigen::Vector3d vertex_n1)
//    : vertex_a_(vertex_a), vertex_n0_(vertex_n0), vertex_n1_(vertex_n1) {}
//
//__device__ const Eigen::Vector3d& Triangle::getVectorAToN0() {
//  if (!computed_vector_a_to_n0_) {
//    vector_a_to_n0_ = vertex_n0_ - vertex_a_;
//    computed_vector_a_to_n0_ = true;
//  }
//  return vector_a_to_n0_;
//}
//
//__device__ const Eigen::Vector3d& Triangle::getVectorAToN1() {
//  if (!computed_vector_a_to_n1_) {
//    vector_a_to_n1_ = vertex_n1_ - vertex_a_;
//    computed_vector_a_to_n1_ = true;
//  }
//  return vector_a_to_n1_;
//}
//
//__device__ double Triangle::getArea() {
//  if (!computed_area_) {
//    area_ = 0.5 * getVectorAToN0().cross(getVectorAToN1()).norm();
//    computed_area_ = true;
//  }
//  return area_;
//}
//
//__device__ const Eigen::Vector3d& Triangle::getNormal() {
//  if (!computed_normal_) {
//    normal_ = getVectorAToN0().cross(getVectorAToN1()).normalized();
//    computed_normal_ = true;
//  }
//  return normal_;
//}
//
//}  // namespace uv_texture_synthesizer
//
//#endif  // UV_TEXTURE_SYNTHESIZER_TRIANGLE_CUH_