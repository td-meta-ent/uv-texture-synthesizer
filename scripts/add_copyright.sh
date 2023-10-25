#!/bin/bash

# Copyright (c) 2023 Metaverse Entertainment Inc. - All Rights Reserved.
# This code is the property of Metaverse Entertainment Inc., a subsidiary of Netmarble Corporation.
# Unauthorized copying or reproduction of this code, in any form, is strictly prohibited.

# Get the current year.
readonly CURRENT_YEAR=$(date +%Y)

# Define the copyright notice with the current year.
readonly COPYRIGHT_NOTICE="# Copyright (c) $CURRENT_YEAR Netmarble Corporation. All Rights Reserved.
# This code is the property of Metaverse Entertainment Inc., a subsidiary of Netmarble Corporation.
# Unauthorized copying or reproduction of this code, in any form, is strictly prohibited."
readonly CPP_COPYRIGHT_NOTICE="// Copyright (c) $CURRENT_YEAR Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of Netmarble Corporation.
// Unauthorized copying or reproduction of this code, in any form, is strictly prohibited."

# Function to check if the copyright notice is already present in a file, and append it if not.
# Args:
#     file_path: The path to the file where the notice should be checked and possibly appended.
check_and_append_notice() {
  local file_path=$1
  local file_extension="${file_path##*.}"
  local notice_to_use="$COPYRIGHT_NOTICE"

  # Decide on the appropriate comment style for the file extension.
  if [[ "$file_extension" == "cpp" || "$file_extension" == "hpp" || "$file_extension" == "cu" || "$file_extension" == "cuh" ]]; then
    notice_to_use="$CPP_COPYRIGHT_NOTICE"
  fi

  # Check if the copyright notice is already present in the file.
  if ! grep -Fxq "$notice_to_use" "$file_path"; then
    # If the file has a shebang, add the copyright notice below it.
    if head -n 1 "$file_path" | grep -q "^#!/"; then
      local shebang_line=$(head -n 1 "$file_path")
      local remaining_content=$(tail -n +2 "$file_path")
      echo "$shebang_line" >"$file_path"
      echo "" >>"$file_path"
      echo "$notice_to_use" >>"$file_path"
      echo "$remaining_content" >>"$file_path"
    else
      # If the file doesn't have a shebang, add the copyright notice at the beginning.
      local temp_file_content=$(cat "$file_path")
      echo -e "$notice_to_use\n\n$temp_file_content" >"$file_path"
    fi
  fi
}

# Run the function with all arguments passed to the script.
for file_path in "$@"; do
  check_and_append_notice "$file_path"
done
