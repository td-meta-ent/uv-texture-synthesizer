cmake_minimum_required(VERSION 3.26)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE PATH "Path to CUDA compiler" FORCE)
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "Set the CUDA architectures to native" FORCE)
set(CMAKE_CUDA_STANDARD 20 CACHE STRING "Set the CUDA standard version" FORCE)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "Set the C++ standard version" FORCE)

project(SurfaceRefinement LANGUAGES CXX CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON CACHE STRING "Enable CUDA Separate Compilation")

set(CUDA_INCLUDE_DIRS "/usr/local/cuda/targets/x86_64-linux/include" CACHE STRING "CUDA include directory" FORCE)
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")

# Locate the CUDA device runtime library. This is required for dynamic parallelism.
find_library(CUDA_DEV_RUNTIME_LIBRARY cudadevrt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
message(STATUS "CUDA_DEV_RUNTIME_LIBRARY: ${CUDA_DEV_RUNTIME_LIBRARY}")

find_package(cgal REQUIRED)
message(STATUS "cgal_INCLUDE_DIRS: ${cgal_INCLUDE_DIRS}")
message(STATUS "cgal_LIBRARIES: ${cgal_LIBRARIES}")

find_package(vcglib REQUIRED)
message(STATUS "vcglib_INCLUDE_DIRS: ${vcglib_INCLUDE_DIRS}")
message(STATUS "vcglib_LIBRARIES: ${vcglib_LIBRARIES}")

find_package(vcglib REQUIRED)
message(STATUS "vcglib_INCLUDE_DIRS: ${vcglib_INCLUDE_DIRS}")
message(STATUS "vcglib_LIBRARIES: ${vcglib_LIBRARIES}")

find_package(PCL 1.13.1 REQUIRED COMPONENTS common io filters surface)
message(STATUS "PCL_INCLUDE_DIRS: ${PCL_INCLUDE_DIRS}")
message(STATUS "PCL_LIBRARIES: ${PCL_LIBRARIES}")
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(Ceres REQUIRED)
message(STATUS "Ceres_INCLUDE_DIRS: ${Ceres_INCLUDE_DIRS}")
message(STATUS "Ceres_LIBRARIES: ${Ceres_LIBRARIES}")

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")

find_package(Open3D REQUIRED)
message(STATUS "Open3D_INCLUDE_DIRS: ${Open3D_INCLUDE_DIRS}")
message(STATUS "Open3D_LIBRARIES: ${Open3D_LIBRARIES}")

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3_INCLUDE_DIRS: ${Eigen3_INCLUDE_DIRS}")
message(STATUS "Eigen3_LIBRARIES: ${Eigen3_LIBRARIES}")

find_package(Boost REQUIRED COMPONENTS filesystem timer program_options)
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")

find_package(indicators REQUIRED)
message(STATUS "Indicators_INCLUDE_DIRS: ${indicators_INCLUDE_DIRS}")
message(STATUS "Indicators_LIBRARIES: ${indicators_LIBRARIES}")

find_package (glog REQUIRED)
message(STATUS "glog_INCLUDE_DIRS: ${glog_INCLUDE_DIRS}")
message(STATUS "glog_LIBRARIES: ${glog_LIBRARIES}")

find_package(cnpy REQUIRED)
message(STATUS "cnpy_INCLUDE_DIRS: ${cnpy_INCLUDE_DIRS}")
message(STATUS "cnpy_LIBRARIES: ${cnpy_LIBRARIES}")

find_package(OpenMP REQUIRED)
message(STATUS "OpenMP_INCLUDE_DIRS: ${OpenMP_INCLUDE_DIRS}")
message(STATUS "OpenMP_LIBRARIES: ${OpenMP_LIBRARIES}")

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu)
foreach(FILE ${SRC_FILES})
    message(STATUS "Found file: ${FILE}")
endforeach()

add_executable(SurfaceRefinement ${SRC_FILES})

target_include_directories(SurfaceRefinement PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${cgal_INCLUDE_DIRS}
        ${vcglib_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${Ceres_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Open3D_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${indicators_INCLUDE_DIRS}
        ${glog_INCLUDE_DIRS}
        ${cnpy_INCLUDE_DIRS}
        ${OpenMP_INCLUDE_DIRS}
        include/cpp
        include/cuda
)

target_link_libraries(SurfaceRefinement PRIVATE
        ${CUDA_DEV_RUNTIME_LIBRARY}
        ${cgal_LIBRARIES}
        ${vcglib_LIBRARIES}
        ${PCL_LIBRARIES}
        ${Ceres_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${Open3D_LIBRARIES}
        ${Eigen3_LIBRARIES}
        Boost::filesystem Boost::timer Boost::program_options
        ${indicators_LIBRARIES}
        ${glog_LIBRARIES}
        ${cnpy_LIBRARIES}
        ${OpenMP_LIBRARIES}
)
