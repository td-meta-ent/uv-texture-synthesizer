#!/bin/bash

# Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
# This code is the property of Metaverse Entertainment Inc., a subsidiary of Netmarble Corporation.
# Unauthorized copying or reproduction of this code, in any form, is strictly prohibited.

set -o errexit -o nounset -o pipefail

readonly PACKAGE_NAME=$1
readonly TARGET_VERSION=$2

available_version=$(apt-cache policy ${PACKAGE_NAME} | grep "${TARGET_VERSION}" | awk '{print $2; exit}')

echo "Attempting to install ${PACKAGE_NAME} version ${TARGET_VERSION}"

if [ -z "${available_version}" ]; then
  echo "No package found with target version ${TARGET_VERSION}"
  exit 1
else
  echo "Initiating installation for ${PACKAGE_NAME} version ${available_version} excluding recommended packages"
  apt-get install --no-install-recommends --assume-yes ${PACKAGE_NAME}=${available_version}
  if [ $? -eq 0 ]; then
    echo "Installation successful for ${PACKAGE_NAME} version ${available_version}"
  else
    echo "Installation failed for ${PACKAGE_NAME} version ${available_version}"
    exit 1
  fi
fi
