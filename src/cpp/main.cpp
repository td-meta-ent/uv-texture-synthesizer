// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include <glog/logging.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>

#include "image.hpp"
#include "mesh.hpp"
#include "refiner.hpp"
#include "shift_vector.hpp"

namespace po = boost::program_options;

struct CommandArguments {
  std::string input_mesh_path;
  std::string refinement_mode = "combined";
  std::string camera_parameters_path;
  std::string left_image_path;
  std::string right_image_path;
  std::string shift_vector_path;
  double surface_weight = 1.0;
  double alpha_coefficient = 1.0;
  double beta_coefficient = 1.0;
  double delta = 1.0;
  int number_of_iterations = 1;
  std::string output_mesh_path;
};

namespace surface_refinement {

void RefineMeshSurface(const CommandArguments &arguments) {
  const auto start_time = std::chrono::high_resolution_clock::now();

  LOG(INFO) << "Loading mesh, camera parameters, and images...";

  // Input Mesh
  boost::filesystem::path input_path(arguments.input_mesh_path);
  surface_refinement::Mesh mesh(input_path);

  // Shift Vector
  boost::filesystem::path npyFilePath(arguments.shift_vector_path);
  double scaleFactor = 22.5 / 1000 / 100;
  surface_refinement::ShiftVector shift_vector(
      npyFilePath, surface_refinement::Image::kDefaultImageWidth, scaleFactor);

  // Left Image
  boost::filesystem::path image_left_path(arguments.left_image_path);
  surface_refinement::Image image_left(image_left_path);

  // Right Image
  boost::filesystem::path image_right_path(arguments.right_image_path);
  surface_refinement::Image image_right(image_right_path);

  LOG(INFO) << "Loaded all resources in "
            << std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::high_resolution_clock::now() - start_time)
                   .count()
            << " seconds.";

  const auto refine_start_time = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Refining mesh surface...";

  surface_refinement::Refiner refiner(
      mesh.GetDeviceVertices(), mesh.GetNumVertices(),
      mesh.GetDeviceNumVertices(), mesh.GetDeviceTriangles(),
      mesh.GetNumTriangles(), mesh.GetDeviceNumTriangles(),
      mesh.GetDeviceTriangleProperties(), mesh.GetDeviceOneRingProperties(),
      mesh.GetDeviceOneRingIndices(), mesh.GetDeviceOneRingIndicesRowLengths(),
      image_left.GetDeviceImageMatrix(), image_right.GetDeviceImageMatrix(),
      shift_vector.GetDeviceDistance(), 1, arguments.refinement_mode,
      arguments.number_of_iterations, arguments.delta,
      arguments.alpha_coefficient, arguments.beta_coefficient);
  std::vector<Eigen::Vector3d> curvature_adjusted_vertices =
      refiner.LaunchRefinement();

  auto duration = std::chrono::high_resolution_clock::now() - refine_start_time;
  auto seconds =
      std::chrono::duration_cast<std::chrono::seconds>(duration).count();
  auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

  LOG(INFO) << "Mesh surface refined in " << seconds << "s (" << milliseconds
            << "ms).";

  const auto save_start_time = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "Saving mesh...";
  boost::filesystem::path output_mesh_path(arguments.output_mesh_path);
  mesh.SaveMesh(curvature_adjusted_vertices, output_mesh_path);
  LOG(INFO) << "Saved mesh to " << arguments.output_mesh_path << " in "
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

}  // namespace surface_refinement

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);

  CommandArguments arguments;
  po::options_description description("Options");
  description.add_options()("help,h", "Display help message")(
      "inputMesh,i",
      po::value<std::string>(&arguments.input_mesh_path)->required(),
      "Absolute path to input mesh")(
      "mode,m", po::value<std::string>(&arguments.refinement_mode)->required(),
      "Refinement mode: ['curvature', 'photometric', 'combined']")(
      "camera,c", po::value<std::string>(&arguments.camera_parameters_path),
      "Absolute path to camera parameters file")(
      "leftImage,l", po::value<std::string>(&arguments.left_image_path),
      "Absolute path to left image")(
      "rightImage,r", po::value<std::string>(&arguments.right_image_path),
      "Absolute path to right image")(
      "shiftVector,s", po::value<std::string>(&arguments.shift_vector_path),
      "Shift vector for the right camera")(
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

    boost::filesystem::path project_path =
        boost::filesystem::absolute(argv[0]).parent_path().parent_path();
    LOG(INFO) << "Project path: " << project_path.string();
    arguments.input_mesh_path =
        (project_path / arguments.input_mesh_path).string();
    arguments.left_image_path =
        (project_path / arguments.left_image_path).string();
    arguments.right_image_path =
        (project_path / arguments.right_image_path).string();
    arguments.shift_vector_path =
        (project_path / arguments.shift_vector_path).string();
    arguments.output_mesh_path =
        (project_path / arguments.output_mesh_path).string();
  } catch (const po::error &e) {
    LOG(ERROR) << "Error: " << e.what();
    LOG(ERROR) << description;
    return 1;
  }

  surface_refinement::RefineMeshSurface(arguments);

  return 0;
}
