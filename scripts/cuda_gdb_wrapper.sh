#!/bin/bash

# Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
# This code is the property of Metaverse Entertainment Inc., a subsidiary of Netmarble Corporation.
# Unauthorized copying or reproduction of this code, in any form, is strictly prohibited.

export PYTHON=/root/python3.6.15/bin/python3.6
export LD_LIBRARY_PATH=/root/python3.6.15/lib:$LD_LIBRARY_PATH
/usr/local/cuda/bin/cuda-gdb "$@"
