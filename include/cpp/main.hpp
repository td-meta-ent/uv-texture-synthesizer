// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#ifndef SURFACE_REFINEMENT_MAIN_HPP_
#define SURFACE_REFINEMENT_MAIN_HPP_

#include <Eigen/Dense>
#include <boost/filesystem/path.hpp>
#include <chrono>
#include <string>

#include "camera_manager.hpp"
#include "image_manager.hpp"
#include "mesh.hpp"
#include "refiner.hpp"

namespace surface_refinement {

/**
 * @struct CommandArguments
 * @brief Holds command line arguments for the mesh refinement process.
 */
struct CommandArguments {
  std::string input_mesh_path;
  std::string refinement_mode = "linear_combination";
  std::string camera_dir_path;
  std::string image_dir_path;
  double surface_weight = 1.0;
  double alpha_coefficient = 1.0;
  double beta_coefficient = 1.0;
  double delta = 1.0;
  double damping = 0.99;
  int number_of_iterations = 1;
  std::string output_mesh_path;
};

/**
 * @brief Refines the surface of a mesh based on provided command line
 * arguments.
 *
 * @param arguments Command line arguments for mesh refinement.
 */
void RefineMeshSurface(const CommandArguments &arguments);

/**
 * @brief Updates relative paths in the command arguments to absolute paths
 * based on the project path.
 *
 * @param arguments Command arguments containing relative paths.
 * @param project_path Absolute path to the project directory.
 */
void UpdateRelativePaths(CommandArguments *arguments,
                         const boost::filesystem::path &project_path);

}  // namespace surface_refinement

#endif  // SURFACE_REFINEMENT_MAIN_HPP_
