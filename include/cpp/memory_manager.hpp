// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_MEMORY_MANAGER_HPP_
#define SURFACE_REFINEMENT_MEMORY_MANAGER_HPP_

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <string>
#include <vector>

#define CUDA_CHECK_LAST()                                         \
  do {                                                            \
    cudaError_t error = cudaGetLastError();                       \
    if (error != cudaSuccess) {                                   \
      LOG(ERROR) << "[CUDA ERROR] " << cudaGetErrorString(error); \
      exit(EXIT_FAILURE);                                         \
    }                                                             \
  } while (0);

#define CUDA_ERROR_CHECK(call)                                             \
  {                                                                        \
    const cudaError_t error = call;                                        \
    if (error != cudaSuccess) {                                            \
      LOG(ERROR) << "[CUDA ERROR] " << cudaGetErrorString(error) << " at " \
                 << #call;                                                 \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  }

namespace surface_refinement {

/**
 * @file memory_manager.hpp
 * @class MemoryManager
 * @brief Manager for memory operations on the CUDA device.
 *
 * This class provides utilities to allocate memory for matrices and scalars on
 * the CUDA device.
 */
template <typename T>
class MemoryManager {
 public:
  /**
   * @brief Allocates memory on the device for a std::string and copies data
   * from host to device.
   *
   * @param host_string The string containing data to copy.
   *
   * @return Pointer to the allocated string data on the device.
   */
  static T *AllocateStringDevice(const std::basic_string<T> &host_string) {
    T *device_string_data;
    size_t string_size = sizeof(T) * host_string.size();
    CUDA_ERROR_CHECK(cudaMalloc(&device_string_data, string_size))
    CUDA_ERROR_CHECK(cudaMemcpy(device_string_data, host_string.c_str(),
                                string_size, cudaMemcpyHostToDevice))

    return device_string_data;
  }

  /**
   * @brief Allocates memory on the device for an array of type T and
   * initializes it to zero.
   *
   * @param num_elements The number of elements of type T to allocate on the
   * device.
   *
   * @return Pointer to the allocated array data on the device.
   */
  static T *AllocateArrayDevice(const size_t num_elements) {
    T *device_array_data;
    size_t array_size = sizeof(T) * num_elements;
    CUDA_ERROR_CHECK(cudaMalloc(&device_array_data, array_size))
    CUDA_ERROR_CHECK(cudaMemset(device_array_data, 0, array_size))

    return device_array_data;
  }

  /**
   * @brief Allocates memory on the device for an array of type T and copies
   * data from host to device.
   *
   * @param host_data The vector of T elements containing data to copy.
   *
   * @return Pointer to the allocated array data on the device.
   */
  static T *AllocateArrayDevice(const std::vector<T> &host_data) {
    T *device_array_data;
    size_t array_size = sizeof(T) * host_data.size();
    CUDA_ERROR_CHECK(cudaMalloc(&device_array_data, array_size))
    CUDA_ERROR_CHECK(cudaMemcpy(device_array_data, host_data.data(), array_size,
                                cudaMemcpyHostToDevice))

    return device_array_data;
  }

  /**
   * @brief Allocates memory on the device for a scalar and copies data from
   * the host.
   *
   * @param host_data Scalar data from the host to be copied to the device.
   *
   * @return Pointer to the allocated scalar on the device.
   */
  static T *AllocateScalarDevice(const T &host_data) {
    T *device_scalar_data;
    CUDA_ERROR_CHECK(cudaMalloc(&device_scalar_data, sizeof(T)))
    CUDA_ERROR_CHECK(cudaMemcpy(device_scalar_data, &host_data, sizeof(T),
                                cudaMemcpyHostToDevice))

    return device_scalar_data;
  }
};

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_MEMORY_MANAGER_HPP_
