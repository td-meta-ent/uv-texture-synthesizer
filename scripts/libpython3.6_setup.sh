#!/bin/zsh

# Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
# This code is the property of Metaverse Entertainment Inc., a subsidiary of Netmarble Corporation.
# Unauthorized copying or reproduction of this code, in any form, is strictly prohibited.

# Define installation directory
INSTALL_DIR=$HOME/python3.6.15

# Download, extract, and navigate to the Python 3.6.15 source directory
wget https://www.python.org/ftp/python/3.6.15/Python-3.6.15.tar.xz
tar xvf Python-3.6.15.tar.xz
cd Python-3.6.15

# Configure Python without ensurepip and with optimizations if desired
./configure --prefix=$INSTALL_DIR --without-ensurepip --enable-optimizations --enable-shared

# Compile Python using parallel jobs
make -j$(nproc)
make altinstall

# Add the new Python installation to PATH in .zshrc
if ! grep -q "$INSTALL_DIR/bin" ~/.zshrc; then
  echo "export PATH=$INSTALL_DIR/bin:\$PATH" >>~/.zshrc
fi

echo "Python 3.6.15 installation complete. Please restart your terminal or source your ~/.zshrc."
