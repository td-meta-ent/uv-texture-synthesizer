# Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
# This code is the property of Metaverse Entertainment Inc., a subsidiary of Netmarble Corporation.
# Unauthorized copying or reproduction of this code, in any form, is strictly prohibited.

# Use the ubuntu:22.04 image as the base image
FROM ubuntu:22.04

# Set the working directory to /root
WORKDIR /root

# No interactive frontend during docker build
ARG DEBIAN_FRONTEND=noninteractive

# Update the package repository
RUN apt-get update

########################################################################################################################
# Administration Utilities
# Utilities to administer system resources, manage user accounts, etc.
########################################################################################################################

# Install passwd - change and administer password and group data
RUN apt-get install --assume-yes passwd

# Set password for superuser
ARG ROOT_PASSWORD
RUN printf "%s %s " "${ROOT_PASSWORD}" "${ROOT_PASSWORD}" | \
    passwd

########################################################################################################################
# Web Software
# Web servers, browsers, proxies, download tools etc.
########################################################################################################################

# Install cURL - command line tool for transferring data with URL syntax
RUN apt-get install --assume-yes curl

# Install wget - retrieves files from the web
RUN apt-get install --assume-yes wget

# Install Google Chrome - Web Browser
RUN chrome_file="$(mktemp --suffix=.deb)" && \
    chrome_url="https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb" && \
    wget --progress=dot:giga --output-document="${chrome_file}" "${chrome_url}" && \
    apt install --assume-yes "${chrome_file}" && \
    rm "${chrome_file}"

# Create an alias 'chrome' to run Google Chrome without sandboxing and add it to .zshrc
RUN echo "alias chrome=\"google-chrome --no-sandbox\"" >>"${HOME}/.zshrc"

########################################################################################################################
# Utilities
# Utilities for file/disk manipulation, backup and archive tools, system monitoring, input systems, etc.
########################################################################################################################

# Install Glances - Curses-based monitoring tool
RUN apt-get install --assume-yes glances

# Install Neofetch - Shows Linux System Information with Distribution Logo
RUN apt-get install --assume-yes neofetch

# Install jq - Lightweight and flexible command-line JSON processor
RUN apt-get install --assume-yes jq

# Install XZ Utils - XZ-format compression utilities
RUN apt-get install --assume-yes xz-utils

# Install zip - Archiver for .zip files
RUN apt-get install --assume-yes zip

# Install unzip - De-archiver for .zip files
RUN apt-get install --assume-yes unzip

########################################################################################################################
# Development
# Development utilities, compilers, development environments, libraries, etc.
########################################################################################################################

# Custom script to install packages with specific versions
ADD pkg_install.sh /pkg_install.sh

# Install CMake - cross-platform, open-source make system
ARG CMAKE_VERSION
RUN CMAKE_PARENT_PATH="/usr" && \
    CMAKE_FILE="$(mktemp --suffix=.tar.gz)" && \
    CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" && \
    wget --progress=dot:giga --output-document="${CMAKE_FILE}" "${CMAKE_URL}" && \
    tar --extract --directory="${CMAKE_PARENT_PATH}" --strip-components=1 --file="${CMAKE_FILE}" && \
    rm "${CMAKE_FILE}"

# Install GNU Compiler Collection - GNU C compiler
ARG GCC_VERSION
RUN bash /pkg_install.sh gcc "${GCC_VERSION}"

# Install Make - utility for directing compilation
ARG MAKE_VERSION
RUN bash /pkg_install.sh make "${MAKE_VERSION}"

# Install GNU Debugger - GNU Debugger
ARG GDB_VERSION
RUN bash /pkg_install.sh gdb "${GDB_VERSION}"

# Install CUDA Toolkit - NVIDIA CUDA development toolkit
ENV PATH /usr/local/cuda/bin:$PATH
ARG CUDA_TOOLKIT_VERSION
RUN CUDA_FILE="$(mktemp --suffix=.run)" && \
    CUDA_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_TOOLKIT_VERSION}/local_installers/cuda_${CUDA_TOOLKIT_VERSION}_535.54.03_linux.run" && \
    wget --progress=dot:giga --output-document="${CUDA_FILE}" "${CUDA_URL}" && \
    chmod +x "${CUDA_FILE}" && \
    "${CUDA_FILE}" --silent --toolkit && \
    rm "${CUDA_FILE}"

# Remove custom script to install packages with specific versions
RUN rm /pkg_install.sh

# Install CLion - A Cross-Platform IDE for C and C++ by JetBrains
ARG CLION_VERSION
RUN clion_file="$(mktemp --suffix=.tar.gz)" && \
    clion_url="https://download-cdn.jetbrains.com/cpp/CLion-${CLION_VERSION}.tar.gz" && \
    wget --progress=dot:giga --output-document="${clion_file}" "${clion_url}" && \
    tar --extract --ungzip --file "${clion_file}" --directory=/opt && \
    rm "${clion_file}"

# Create an alias 'clion' to run CLion and add it to .zshrc
RUN echo "alias clion=\"/opt/clion-${CLION_VERSION}/bin/clion.sh\"" >>"${HOME}/.zshrc"

# Add environment variables to shell configuration file
RUN echo "cd ${PROJECT_PATH}" >>"${HOME}/.zshrc"

# Install Conan - C and C++ Open Source Package Manager
ARG CONAN_VERSION
RUN conan_file="$(mktemp --suffix=.deb)" && \
    conan_url="https://github.com/conan-io/conan/releases/download/${CONAN_VERSION}/conan-ubuntu-64.deb" && \
    wget --progress=dot:giga --output-document="${conan_file}" "${conan_url}" && \
    apt install --assume-yes "${conan_file}" && \
    rm "${conan_file}"

########################################################################################################################
# Editors
# Software to edit files. Programming environments.
########################################################################################################################

# Install NeoVim - heavily refactored Vim fork
RUN apt-get install --assume-yes neovim

########################################################################################################################
# Graphics
# Editors, viewers, converters... Everything to become an artist.
########################################################################################################################

# Install MeshLab - System for processing and editing triangular meshes
RUN apt-get install --assume-yes meshlab

########################################################################################################################
# Mathematics
# Math software.
########################################################################################################################

# Install gmsh - A 3D mesh generator with a high quality mesh generation algorithm
RUN apt-get install --assume-yes gmsh

########################################################################################################################
# Version Control Systems
# Version control systems and related utilities.
########################################################################################################################

# Install Git - fast, scalable, distributed revision control system
ARG GIT_USER_NAME
ARG GIT_USER_EMAIL
RUN apt-get install --assume-yes git && \
    git config --global user.name "${GIT_USER_NAME}" && \
    git config --global user.email "${GIT_USER_EMAIL}"

########################################################################################################################
# Shells
# Command shells. Friendly user interfaces for beginners.
########################################################################################################################

# Install Z shell - Shell with lots of features
RUN apt-get install --assume-yes zsh

# Change login shell for the superuser
RUN chsh --shell "$(which zsh)" root

# Install Oh My Zsh - framework for managing your Zsh configuration
ARG OH_MY_ZSH_URL="https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh"
RUN sh -c "$(curl --location ${OH_MY_ZSH_URL}) --unattended"
SHELL ["zsh", "-c"]

########################################################################################################################
# Network
# Daemons and clients to connect your system to the world.
########################################################################################################################

# Install OpenSSH server - secure shell (SSH) server, for secure access from remote machines
RUN apt-get install --assume-yes openssh-server

# Add custom sshd_config
ADD sshd_config /etc/ssh/sshd_config

# Configure authorized_keys for OpenSSH
ARG PUBLIC_KEY
RUN mkdir --parents "${HOME}/.ssh" && \
    echo "${PUBLIC_KEY}" >"${HOME}/.ssh/authorized_keys"

# Append the command 'cat ~/.ssh/id_rsa.pub' to the .zshrc file
RUN ssh-keygen -N "" -f "${HOME}/.ssh/id_rsa"

# Install rsync - fast, versatile, remote (and local) file-copying tool
RUN apt-get install --assume-yes rsync

# Install Xrdp - Remote Desktop Protocol (RDP) server
RUN apt-get install --assume-yes xrdp
RUN sed -i.bak '/fi/a #xrdp multiple users configuration startxfce4 ' /etc/xrdp/startwm.sh

########################################################################################################################
# Miscellaneous
# Miscellaneous utilities that didn't fit well anywhere else.
########################################################################################################################

# Install caffeine - Prevent the desktop becoming idle in full-screen mode
RUN apt-get install --assume-yes caffeine

########################################################################################################################
# Python
# Everything about Python, an interpreted, interactive object oriented language.
########################################################################################################################

# Download the Miniconda Installer
ARG CONDA_VERSION
ARG MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh"
RUN wget --progress=dot:giga --output-document=miniconda.sh "${MINICONDA_URL}"

# Install Conda - package, dependency and environment management for any language
RUN rm --force --recursive /opt/conda && \
    mkdir --parents /opt && \
    sh miniconda.sh -b -p /opt/conda && \
    ln --symbolic /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    rm miniconda.sh

# Enable Conda
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >>"${HOME}/.zshrc"

# Set the build arguments
ARG PROJECT_NAME
ARG PROJECT_PATH="/root/${PROJECT_NAME}"
ARG PYTHON_VERSION
ARG CONDA_BASE="/opt/conda"
ARG CONDA_BINARY="${CONDA_BASE}/bin/conda"
ARG CONDA_ENV="${PROJECT_NAME}-${PYTHON_VERSION}"
ARG PIP_BINARY="${CONDA_BASE}/envs/${CONDA_ENV}/bin/pip"

# Create a Conda environment
RUN "${CONDA_BINARY}" create --name "${CONDA_ENV}" python="${PYTHON_VERSION}"

# Install pre-commit - A framework for managing and maintaining multi-language pre-commit hooks
ARG PRE_COMMIT_VERSION
RUN "${PIP_BINARY}" install pre-commit=="${PRE_COMMIT_VERSION}"

# Install cpplint - C++ style checker
ARG CPP_LINT_VERSION
RUN "${PIP_BINARY}" install cpplint=="${CPP_LINT_VERSION}"

# Install shfmt - A shell parser, formatter, and interpreter (POSIX/Bash/mksh)
ARG SHFMT_VERSION
RUN "${CONDA_BINARY}" install --name "${CONDA_ENV}" --channel conda-forge go-shfmt=="${SHFMT_VERSION}"

# Set Conda environment as default
RUN echo "conda activate ${CONDA_ENV}" >>"${HOME}/.zshrc"

########################################################################################################################
# Xfce
# Xfce, a fast and lightweight Desktop Environment.
########################################################################################################################

# Install Xfce - Meta-package for the Xfce Lightweight Desktop Environment
RUN apt-get install --assume-yes xfce4

########################################################################################################################
# X Window System software
# X servers, libraries, fonts, window managers, terminal emulators and many related applications.
########################################################################################################################

# Install Xfce Terminal - Xfce terminal emulator
RUN apt-get install --assume-yes xfce4-terminal && \
    update-alternatives --set x-terminal-emulator /usr/bin/xfce4-terminal.wrapper

########################################################################################################################

# Install OpenCV dependencies
RUN apt-get install --assume-yes \
    libva-dev \
    libvdpau-dev \
    libgtk2.0-dev \
    libx11-xcb-dev \
    libfontenc-dev \
    libxaw7-dev \
    libxkbfile-dev \
    libxmu-dev \
    libxmuu-dev \
    libxpm-dev \
    libxres-dev \
    libxss-dev \
    libxt-dev \
    libxtst-dev \
    libxv-dev \
    libxvmc-dev \
    libxxf86vm-dev \
    libxcb-render-util0-dev \
    libxcb-xkb-dev \
    libxcb-icccm4-dev \
    libxcb-image0-dev \
    libxcb-keysyms1-dev \
    libxcb-randr0-dev \
    libxcb-shape0-dev \
    libxcb-sync-dev \
    libxcb-xfixes0-dev \
    libxcb-xinerama0-dev \
    libxcb-dri3-dev \
    libxcb-cursor-dev \
    libxcb-util-dev \
    libxcb-util0-dev

# Install Open3D dependencies
RUN apt-get install --assume-yes libglu1-mesa-dev libgl1-mesa-glx libc++-dev libc++abi-dev
RUN echo 'export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri/' >>"${HOME}/.zshrc" \

## Navigate to your conda environment's library directory
#cd /opt/conda/envs/surface-refinement-3.10/lib/
#
## Create a backup of the existing libstdc++.so.6 file
#mv libstdc++.so.6 libstdc++.so.6.bak
#
## Create a symbolic link to the system's libstdc++.so.6
#ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6

# Install Open3D - A Modern Library for 3D Data Processing
ARG OPEN3D_VERSION
RUN OPEN3D_PATH="/usr/local" && \
    OPEN3D_FILE="$(mktemp --suffix=.tar.xz)" && \
    OPEN3D_URL="https://github.com/isl-org/Open3D/releases/download/v${OPEN3D_VERSION}/open3d-devel-linux-x86_64-cxx11-abi-cuda-${OPEN3D_VERSION}.tar.xz" && \
    wget --progress=dot:giga --output-document="${OPEN3D_FILE}" "${OPEN3D_URL}" && \
    mkdir --parents "${OPEN3D_PATH}" && \
    tar --extract --directory="${OPEN3D_PATH}" --strip-components=1 --file="${OPEN3D_FILE}" && \
    rm "${OPEN3D_FILE}"

# Build project using Conan
ADD "${PROJECT_NAME}.tar.gz" "${PROJECT_PATH}"
WORKDIR "${PROJECT_PATH}"
RUN conan profile detect --force && \
    conan install . --output-folder=cmake-build-debug --build=missing --settings build_type=Debug --settings compiler.cppstd=20

RUN cd cmake-build-debug && \
    cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug && \
    cmake --build .

# Expose ports and add entrypoint script
EXPOSE 22
EXPOSE 3389
ADD entrypoint.sh /entrypoint.sh
ENTRYPOINT ["zsh", "/entrypoint.sh"]
