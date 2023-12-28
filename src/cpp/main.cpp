// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "main.hpp"

#include <glog/logging.h>

#include <boost/program_options.hpp>
#include <iostream>

#include "camera_manager.hpp"
#include "image_manager.hpp"
#include "mesh.hpp"
#include "synthesizer.hpp"
#include "texture.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace uts = uv_texture_synthesizer;

namespace uv_texture_synthesizer {

void ProcessTextureSynthesis(const CommandArguments &arguments) {
  const auto start_time = std::chrono::high_resolution_clock::now();

  LOG(INFO) << "Starting texture synthesis process...";

  // Assuming these paths are relevant to the texture synthesis process
  fs::path root_path(arguments.root_path);
  std::string project_name = arguments.project_name;
  std::string date = arguments.date;
  std::string actor_name = arguments.actor_name;
  std::string cut_number = arguments.cut_number;
  std::string frame_number = arguments.frame_number;
  std::string time_stamp = arguments.time_stamp;
  int num_cameras = arguments.num_cameras;

  CameraManager camera_manager(root_path / "calibration", date, cut_number);

  Mesh mesh(root_path / "mesh/wrap_tri.obj", camera_manager, num_cameras);

  ImageManager image_manager(root_path / "rectification", project_name, date,
                             actor_name, cut_number, frame_number, time_stamp);

  Texture texture(root_path / "texture_pixel_info.h5", num_cameras);

  Synthesizer synthesizer(mesh, camera_manager, image_manager, texture,
                          num_cameras);
  synthesizer.LaunchSynthesis();

  auto duration = std::chrono::high_resolution_clock::now() - start_time;
  LOG(INFO)
      << "Texture synthesis completed in "
      << std::chrono::duration_cast<std::chrono::seconds>(duration).count()
      << " seconds.";
}

}  // namespace uv_texture_synthesizer

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);

  try {
    uts::CommandArguments arguments;

    // Define and parse command-line arguments
    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message")(
        "root_path", po::value<std::string>(&arguments.root_path)->required(),
        "Root path to the data directory")(
        "project_name",
        po::value<std::string>(&arguments.project_name)->required(),
        "Name of the project")(
        "date", po::value<std::string>(&arguments.date)->required(),
        "Date of the project data")(
        "actor_name", po::value<std::string>(&arguments.actor_name)->required(),
        "Name of the actor")(
        "cut_number",
        po::value<std::string>(&arguments.cut_number)->default_value("00"),
        "Cut number")(
        "frame_number",
        po::value<std::string>(&arguments.frame_number)->default_value("0000"),
        "Frame number")(
        "time_stamp",
        po::value<std::string>(&arguments.time_stamp)->default_value("*"),
        "Timestamp for file selection")(
        "num_cameras", po::value<int>(&arguments.num_cameras)->required(),
        "Number of cameras");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }

    po::notify(vm);

    // Process texture synthesis
    uts::ProcessTextureSynthesis(arguments);

  } catch (std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
