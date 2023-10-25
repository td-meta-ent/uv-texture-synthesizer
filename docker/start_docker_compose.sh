#!/usr/bin/env bash

# Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
# This code is the property of Metaverse Entertainment Inc., a subsidiary of Netmarble Corporation.
# Unauthorized copying or reproduction of this code, in any form, is strictly prohibited.

# Enforce strict execution
set -o errexit -o nounset -o pipefail

SCRIPT_PATH_ABSOLUTE="$(readlink --canonicalize "${0}")"
SCRIPT_DIRECTORY="$(dirname "${SCRIPT_PATH_ABSOLUTE}")"

# Load environment variables from the .env file
set -a
source "${SCRIPT_DIRECTORY}/.env"
set +a

PROJECT_DIRECTORY="$(dirname "${SCRIPT_DIRECTORY}")"
ARCHIVE_FILENAME="${PROJECT_NAME}.tar.gz"

# Ensure Docker Compose is installed
ensure_docker_compose_installed() {
  if ! command -v docker-compose &>/dev/null; then
    echo "[INFO] Docker Compose not found. Installing..."

    curl -SL "https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-linux-x86_64" -o /usr/bin/docker-compose
    chmod +x /usr/bin/docker-compose

    docker-compose version
    echo "[INFO] Docker Compose was installed successfully."
  else
    echo "[INFO] Docker Compose is already installed."
  fi
}

# Ensure the VOLUME_PATH is existent
ensure_volume_path_exists() {
  if [[ ! -d "${VOLUME_PATH}" ]]; then
    echo "[INFO] VOLUME_PATH does not exist. Creating it..."
    mkdir -p "${VOLUME_PATH}"
    echo "[INFO] VOLUME_PATH was created successfully at ${VOLUME_PATH}."
  else
    echo "[INFO] VOLUME_PATH is already existent."
  fi
}

# Archive the project contents into a tarball
archive_project_content() {
  echo "[INFO] Starting the archiving process for the project content..."

  # Remove any existing archive if present
  [[ -f "${ARCHIVE_FILENAME}" ]] && rm -f "${ARCHIVE_FILENAME}"

  # Use a temporary file for archiving
  TEMP_ARCHIVE_FILE=$(mktemp)

  tar --create --gzip --file="${TEMP_ARCHIVE_FILE}" --directory="${PROJECT_DIRECTORY}" .

  mv "${TEMP_ARCHIVE_FILE}" "${SCRIPT_DIRECTORY}/${ARCHIVE_FILENAME}"

  echo "[INFO] Archive created successfully: ${ARCHIVE_FILENAME}"
}

# Deploy the project using Docker Compose
deploy_with_docker_compose() {
  echo "[INFO] Starting deployment with Docker Compose..."

  docker-compose --project-name="${PROJECT_NAME}" --file="docker-compose.yml" up --detach

  echo "[INFO] Deployment has concluded."
}

# Remove generated files
clear_generated_files() {
  echo "[INFO] Removing generated files..."

  [[ -f "${ARCHIVE_FILENAME}" ]] && rm "${ARCHIVE_FILENAME}"

  echo "[INFO] Generated files have been removed."
}

# Main execution function
main() {
  echo "[INFO] Beginning main execution..."

  ensure_docker_compose_installed
  ensure_volume_path_exists
  archive_project_content
  deploy_with_docker_compose
  clear_generated_files

  echo "[INFO] Main execution has concluded."
}

main
