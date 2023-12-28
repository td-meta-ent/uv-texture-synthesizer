# UV Texture Synthesizer

Welcome to the UV Texture Synthesizer project.

## TODO

- Fix the Barycentric Filter in C++ Version:
  - Address and resolve the issues with the barycentric filter implementation in the C++ version of the project.

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

To compile the UV Texture Synthesizer project, utilize the following command:

```bash
/usr/bin/cmake -DCMAKE_BUILD_TYPE=Debug -G "CodeBlocks - Unix Makefiles" -S /root/uv-texture-synthesizer -B /root/uv-texture-synthesizer/cmake-build-debug
```

### Execution:

Invoke the UV Texture Synthesizer algorithm using the following syntax:

```bash
/root/uv-texture-synthesizer/cmake-build-debug/UVTextureSynthesizer
    --root_path="/root/uv-texture-synthesizer/data/input/v6.0.0/231102"
    --project_name="APC_Temp"
    --date="231102"
    --actor_name="test"
    --cut_number="00"
    --frame_number="0000"
    --time_stamp="000000_000010"
    --num_cameras=10
```
