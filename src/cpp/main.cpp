// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include "main.hpp"

#include <glog/logging.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace sr = surface_refinement;

namespace surface_refinement {

void RefineMeshSurface(const CommandArguments &arguments) {
  const auto start_time = std::chrono::high_resolution_clock::now();

  LOG(INFO) << "Loading mesh, camera parameters, and images...";

  fs::path camera_dir_path(arguments.camera_dir_path);
  CameraManager camera_manager(camera_dir_path);

  fs::path input_path(arguments.input_mesh_path);
  Mesh mesh(input_path, camera_manager);

  fs::path image_dir_path(arguments.image_dir_path);
  ImageManager image_manager(image_dir_path, arguments.refinement_mode);

  LOG(INFO) << "Resources loaded in "
            << std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::high_resolution_clock::now() - start_time)
                   .count()
            << " seconds.";

  const auto refine_start_time = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Starting mesh surface refinement...";

  // Mesh Refinement Process
  Refiner refiner(mesh, camera_manager, image_manager, arguments.damping,
                  arguments.refinement_mode, arguments.number_of_iterations,
                  arguments.delta, arguments.surface_weight,
                  arguments.alpha_coefficient, arguments.beta_coefficient);
  std::vector<Eigen::Vector3d> curvature_adjusted_vertices =
      refiner.LaunchRefinement();

  auto duration = std::chrono::high_resolution_clock::now() - refine_start_time;
  auto seconds =
      std::chrono::duration_cast<std::chrono::seconds>(duration).count();
  auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

  LOG(INFO) << "Mesh surface refinement completed in " << seconds << "s ("
            << milliseconds << "ms).";

  const auto save_start_time = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Saving refined mesh...";
  fs::path output_mesh_path(arguments.output_mesh_path);
  mesh.SaveMesh(&curvature_adjusted_vertices, output_mesh_path);
  LOG(INFO) << "Mesh saved to " << arguments.output_mesh_path << " in "
            << std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::high_resolution_clock::now() - save_start_time)
                   .count()
            << " seconds.";

  LOG(INFO) << "Total execution time: "
            << std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::high_resolution_clock::now() - start_time)
                   .count()
            << " seconds.";
}

void UpdateRelativePaths(CommandArguments *arguments,
                         const fs::path &project_path) {
  if (!fs::path(arguments->input_mesh_path).is_absolute()) {
    arguments->input_mesh_path =
        (project_path / arguments->input_mesh_path).string();
  }

  if (!fs::path(arguments->camera_dir_path).is_absolute()) {
    arguments->camera_dir_path =
        (project_path / arguments->camera_dir_path).string();
  }

  if (!fs::path(arguments->image_dir_path).is_absolute()) {
    arguments->image_dir_path =
        (project_path / arguments->image_dir_path).string();
  }

  if (!fs::path(arguments->output_mesh_path).is_absolute()) {
    arguments->output_mesh_path =
        (project_path / arguments->output_mesh_path).string();
  }
}

}  // namespace surface_refinement

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);

  sr::CommandArguments arguments;
  po::options_description description("Options");
  description.add_options()("help,h", "Display help message")(
      "inputMesh,i",
      po::value<std::string>(&arguments.input_mesh_path)->required(),
      "Absolute path to input mesh")(
      "mode,m", po::value<std::string>(&arguments.refinement_mode)->required(),
      "Refinement mode: ['curvature', 'photometric', 'combined']")(
      "cameraDirPath,cd",
      po::value<std::string>(&arguments.camera_dir_path)->required(),
      "Absolute path to the camera parameters directory")(
      "imageDirPath,id",
      po::value<std::string>(&arguments.image_dir_path)->required(),
      "Absolute path to the image directory")(
      "surfaceWeight,w",
      po::value<double>(&arguments.surface_weight)->default_value(1.0),
      "Weight for surface refinement in combined mode")(
      "alpha,a",
      po::value<double>(&arguments.alpha_coefficient)->default_value(1.0),
      "Alpha coefficient for surface refinement")(
      "beta,b",
      po::value<double>(&arguments.beta_coefficient)->default_value(1.0),
      "Beta coefficient for photometric refinement")(
      "delta,d", po::value<double>(&arguments.delta)->default_value(1.0),
      "Delta resolution for refinement")(
      "damping,dm", po::value<double>(&arguments.damping)->default_value(0.99),
      "Damping factor for the photometric consistency")(
      "iterations,t",
      po::value<int>(&arguments.number_of_iterations)->default_value(1),
      "Number of refinement iterations")(
      "output,o",
      po::value<std::string>(&arguments.output_mesh_path)->required(),
      "Absolute path to output mesh");

  po::variables_map variable_map;
  try {
    po::store(po::parse_command_line(argc, argv, description), variable_map);
    if (variable_map.count("help")) {
      LOG(INFO) << description;
      return 1;
    }
    po::notify(variable_map);

    fs::path project_path = fs::absolute(argv[0]).parent_path().parent_path();
    LOG(INFO) << "Project path: " << project_path.string();
    sr::UpdateRelativePaths(&arguments, project_path);
  } catch (const po::error &e) {
    LOG(ERROR) << "Error: " << e.what();
    LOG(ERROR) << description;
    return 1;
  }

  sr::RefineMeshSurface(arguments);

  return 0;
}
