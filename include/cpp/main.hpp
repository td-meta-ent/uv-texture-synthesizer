// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef UV_TEXTURE_SYNTHESIZER_MAIN_HPP_
#define UV_TEXTURE_SYNTHESIZER_MAIN_HPP_

#include <boost/filesystem/path.hpp>
#include <string>
#include <vector>

namespace uv_texture_synthesizer {

/**
 * @struct CommandArguments
 * @brief Holds command line arguments for the 3D model texture synthesis
 * process.
 */
struct CommandArguments {
  std::string root_path;
  std::string project_name;
  std::string date;
  std::string actor_name;
  std::string cut_number = "00";
  std::string frame_number = "0000";
  std::string time_stamp = "*";
  int num_cameras;
};

/**
 * @brief Processes the 3D model texture synthesis based on provided command
 * line arguments.
 *
 * @param arguments Command line arguments for the process.
 */
void ProcessTextureSynthesis(const CommandArguments &arguments);

}  // namespace uv_texture_synthesizer

#endif  // UV_TEXTURE_SYNTHESIZER_MAIN_HPP_
