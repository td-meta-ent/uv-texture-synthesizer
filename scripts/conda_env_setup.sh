#!/bin/bash

# Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
# This code is the property of Metaverse Entertainment Inc., a subsidiary of Netmarble Corporation.
# Unauthorized copying or reproduction of this code, in any form, is strictly prohibited.

set -o errexit -o nounset -o pipefail

script_directory="$(dirname "$(readlink -f "$0")")"
project_directory="$(dirname "${script_directory}")"

source "${project_directory}/.env"

log_message() {
  local log_text="$1"
  echo "$(date +'%Y-%m-%d %H:%M:%S') INFO: ${log_text}"
}

verify_or_define_conda_path() {
  if [[ "${CONDA_PATH}" && -d "${CONDA_PATH}" ]]; then
    log_message "Confirmed CONDA_PATH at $CONDA_PATH"
  else
    case "$(uname)" in
      Darwin)
        define_conda_path_macos
        ;;
      Linux)
        define_conda_path_linux
        ;;
      *NT*)
        define_conda_path_windows
        ;;
      *)
        log_message "Unsupported operating system"
        exit 1
        ;;
    esac
  fi
}

define_conda_path_macos() {
  local miniconda_path_macos="/opt/homebrew/Caskroom/miniconda/base"
  if [[ -d "${miniconda_path_macos}" ]]; then
    export CONDA_PATH="${miniconda_path_macos}"
    log_message "CONDA_PATH set to ${CONDA_PATH}"
  else
    log_message "Miniconda not found at ${miniconda_path_macos}"
    exit 1
  fi
}

define_conda_path_linux() {
  local potential_conda_paths_linux=("/opt/conda" "/home/${USER}/miniconda3")
  for each_path in "${potential_conda_paths_linux[@]}"; do
    if [[ -d "${each_path}" ]]; then
      export CONDA_PATH="${each_path}"
      log_message "CONDA_PATH set to ${CONDA_PATH}"
      return
    fi
  done
  log_message "Miniconda not found at ${potential_conda_paths_linux[*]}"
  exit 1
}

define_conda_path_windows() {
  local potential_conda_paths_windows=("/c/Users/${USERNAME}/Miniconda3" "/c/ProgramData/miniconda3")
  for each_path in "${potential_conda_paths_windows[@]}"; do
    if [[ -d "${each_path}" ]]; then
      export CONDA_PATH="${each_path}"
      log_message "CONDA_PATH set to ${CONDA_PATH}"
      return
    fi
  done
  log_message "Miniconda not found at ${potential_conda_paths_windows[*]}"
  exit 1
}

assign_conda_and_pip_binaries_and_env_name() {
  readonly ENV_NAME="${PROJECT_NAME}-${PYTHON_VERSION}"
  if [[ "$(uname)" == "Linux" ]] || [[ "$(uname)" == "Darwin" ]]; then
    readonly CONDA_BINARY="${CONDA_PATH}/bin/conda"
    readonly PIP_BINARY="${CONDA_PATH}/envs/${ENV_NAME}/bin/pip3"
  elif [[ "$(uname)" =~ "NT" ]]; then
    readonly CONDA_BINARY="${CONDA_PATH}/Scripts/conda.exe"
    readonly PIP_BINARY="${CONDA_PATH}/envs/${ENV_NAME}/Scripts/pip3.exe"
  else
    log_message "Unsupported operating system"
    exit 1
  fi
}

establish_conda_environment() {
  log_message "Starting creation of conda environment ${ENV_NAME}..."
  "${CONDA_BINARY}" create --name "${ENV_NAME}" --yes python="${PYTHON_VERSION}"
  log_message "Conda environment ${ENV_NAME} successfully created."
}

install_project_requirements() {
  log_message "Starting installation of project requirements..."
  "${PIP_BINARY}" install --requirement "${project_directory}/requirements.txt"
  log_message "Project requirements successfully installed."
}

install_go_shfmt() {
  log_message "Starting installation of go-shfmt..."
  "${CONDA_BINARY}" install --name "${ENV_NAME}" --channel conda-forge go-shfmt="${SHFMT_VERSION}" --yes
  log_message "go-shfmt successfully installed."
}

run_pre_commit_install() {
  log_message "Running pre-commit install..."
  pre_commit_binary="${CONDA_PATH}/envs/${ENV_NAME}/bin/pre-commit"
  if [[ -x "${pre_commit_binary}" ]]; then
    "${pre_commit_binary}" install
    log_message "Pre-commit install successful."
  else
    log_message "Error: pre-commit binary not found at ${pre_commit_binary}"
    exit 1
  fi
}

configure_project_environment() {
  log_message "Starting project environment configuration..."
  verify_or_define_conda_path
  assign_conda_and_pip_binaries_and_env_name
  establish_conda_environment
  install_project_requirements
  install_go_shfmt
  run_pre_commit_install
  log_message "Project environment configuration completed."
}

configure_project_environment
