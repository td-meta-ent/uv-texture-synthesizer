# Surface Refinement CUDA

Welcome to the Surface Refinement project. This is a sophisticated CUDA-based platform designed for the meticulous refinement of 3D surfaces leveraging both photometric and geometric consistency.

## System Prerequisites

### Nvidia GPU Drivers

Ensure you have the appropriate Nvidia drivers installed for optimal performance:

- **For WSL2**:
    - Download and install the recommended Nvidia driver for Windows Subsystem for Linux.

- **For Ubuntu**:
    - Use the package manager or Nvidia's official website to procure and install the appropriate driver for Ubuntu.

**Note**: Reference the `.env` file for the specific Nvidia driver version that has undergone rigorous testing with this project.

### Docker Environment

Docker containerization ensures a consistent and reproducible environment across platforms.

**Note**: The Docker version that has been thoroughly tested for compatibility with this project is stipulated in the `.env` file.

## Installation Procedure

This initiative harnesses the power of Docker for a seamless and automated installation paradigm. If you wish to bypass Docker and establish a local development environment, refer to the Docker commands contained within the Dockerfile as a guideline.

## Usage Guidelines

### Compiling the Project:

To compile the Surface Refinement project, utilize the following command:

```bash
/usr/bin/cmake -DCMAKE_BUILD_TYPE=Debug -G "CodeBlocks - Unix Makefiles" -S /root/surface-refinement -B /root/surface-refinement/cmake-build-debug
```

### Execution:

The Surface Refinement algorithm supports two modes: `curvature` and `linear_combination`. The `linear_combination` mode is a hybrid approach that integrates both curvature and photometric consistency aspects.

Invoke the Surface Refinement algorithm using the syntax below:

```bash
/root/surface-refinement/cmake-build-debug/SurfaceRefinement
    --inputMesh=/mnt/facer-nas01/231023/models/test/cut03/meshes/231023_test_06_cut03_00000.obj
    --mode=linear_combination
    --cameraDirPath=/mnt/facer-nas01/231023/boards/06/calibration
    --imageDirPath=/mnt/facer-nas01/231023/models/test/cut03/rectification
    --surfaceWeight=1
    --alpha=0.4
    --beta=0.01
    --delta=0.0001
    --damping=0.999
    --iterations=100
    --output=data/output/v5.0.0/face_part_meter_smoothed_lc.ply
```
