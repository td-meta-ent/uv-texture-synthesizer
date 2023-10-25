#!/bin/bash

# Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
# This code is the property of Metaverse Entertainment Inc., a subsidiary of Netmarble Corporation.
# Unauthorized copying or reproduction of this code, in any form, is strictly prohibited.

set -o errexit -o nounset -o pipefail

log() {
  local message=$1
  echo "$(date +'%Y-%m-%d %H:%M:%S') [INFO] $message"
}

get_distribution_info() {
  . /etc/os-release
  echo $ID$VERSION_ID
}

add_nvidia_key() {
  local keyring_path="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
  if [[ ! -e "$keyring_path" ]]; then
    log "Adding NVIDIA GPG key..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o $keyring_path
  else
    log "NVIDIA keyring already exists."
  fi
}

add_nvidia_repo() {
  local distribution="$1"
  local repo_path="/etc/apt/sources.list.d/nvidia-container-toolkit.list"

  if [[ -f "$repo_path" ]]; then
    log "NVIDIA repository for $distribution already exists."
  else
    log "Adding NVIDIA repository for $distribution..."
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list |
      sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" |
      tee "$repo_path" >/dev/null
  fi
}

install_nvidia_container_toolkit() {
  log "Updating apt sources..."
  apt-get update

  log "Installing NVIDIA container toolkit..."
  apt-get install -y nvidia-container-toolkit

  log "Configuring NVIDIA runtime for Docker..."
  nvidia-ctk runtime configure --runtime=docker
}

configure_docker_runtime() {
  # Check and create /etc/docker if it doesn't exist
  if [[ ! -d "/etc/docker" ]]; then
    log "Directory /etc/docker doesn't exist. Creating it..."
    mkdir -p /etc/docker
  fi

  log "Configuring NVIDIA runtime as the default for Docker..."
  echo '{
      "default-runtime": "nvidia",
      "runtimes": {
        "nvidia": {
          "path": "/usr/bin/nvidia-container-runtime",
          "runtimeArgs": []
        }
      }
    }' | tee /etc/docker/daemon.json >/dev/null
}

restart_docker_instructions() {
  local kernel_release=$(uname -r)
  log "It is recommended to restart the Docker daemon now."

  if [[ $kernel_release =~ "microsoft" ]] || [[ $kernel_release =~ "WSL" ]]; then
    log "You appear to be running WSL. To restart Docker Desktop, use the following PowerShell commands:"
    echo 'Stop-Process -Name "com.docker.backend" -Force'
    echo 'Start-Process -FilePath "C:\Program Files\Docker\Docker\Docker Desktop.exe"'
  else
    log "If you're on a Linux system, you can restart the Docker daemon with:"
    echo "systemctl restart docker"
  fi
}

main() {
  local distribution=$(get_distribution_info)
  add_nvidia_key
  add_nvidia_repo $distribution
  install_nvidia_container_toolkit
  configure_docker_runtime
  restart_docker_instructions
}

main
