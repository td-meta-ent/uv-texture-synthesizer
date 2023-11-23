#!/bin/bash

# Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
# This code is the property of Metaverse Entertainment Inc., a subsidiary of Netmarble Corporation.
# Unauthorized copying or reproduction of this code, in any form, is strictly prohibited.

# Set options to exit immediately if any command in the script fails,
# if any variable is unset, and if any command in a pipeline fails
set -o errexit -o nounset -o pipefail

# This command restarts the ssh service
service ssh restart

# This command restarts the xrdp service using the init script
/etc/init.d/xrdp restart

# This command mounts the NAS share
set +e
mount.cifs //${NAS_IP_ADDRESS}/${NAS_SHARE_NAME} ${NAS_MOUNT_POINT} -o user=${NAS_USERNAME},password=${NAS_PASSWORD}
set -e

# This command starts the zsh shell
zsh
