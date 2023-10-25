// Copyright (c) 2023 Netmarble Corporation. All Rights Reserved.
// This code is the property of Metaverse Entertainment Inc., a subsidiary of
// Netmarble Corporation. Unauthorized copying or reproduction of this code, in
// any form, is strictly prohibited.

#include <CGAL/Bbox_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/PLY.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/repair_degeneracies.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/bounding_box.h>
#include <CGAL/squared_distance_3.h>
#include <glog/logging.h>

#include <boost/filesystem.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Surface_mesh<Point> Mesh;

//  MeshRepairUtils
//  meshRepair("/root/surface-refinement/data/input/v3.0.0/face_part.ply",
//  "/root/surface-refinement/data/input/v3.0.0/face_part_repaired.ply");
//
//  if(!meshRepair.loadMesh()) {
//    std::cerr << "Failed to load mesh. Exiting." << std::endl;
//    return -1;
//  }
//
//  meshRepair.uniformMeshResampling(11000);

//  if(!meshRepair.saveRepairedMesh()) {
//    std::cerr << "Failed to save repaired mesh. Exiting." << std::endl;
//    return -1;
//  }

class MeshRepairUtils {
 public:
  MeshRepairUtils(boost::filesystem::path inputFilePath,
                  boost::filesystem::path outputFilePath)
      : inputFilePath_(std::move(inputFilePath)),
        outputFilePath_(std::move(outputFilePath)) {
    LOG(INFO) << "Initializing MeshRepairUtils";
  }

  bool loadMesh() {
    LOG(INFO) << "Loading mesh from: " << inputFilePath_.string();
    bool success = CGAL::IO::read_PLY(inputFilePath_.string(), mesh_);
    if (success) {
      LOG(INFO) << "Number of vertices: " << mesh_.number_of_vertices();
      LOG(INFO) << "Number of faces: " << mesh_.number_of_faces();
    } else {
      LOG(ERROR) << "Error reading file";
    }
    return success;
  }

  void uniformMeshResampling(double target_edge_length) {
    auto start = std::chrono::high_resolution_clock::now();

    LOG(INFO) << "Starting uniform mesh resampling with target edge length: "
              << target_edge_length;

    CGAL::Polygon_mesh_processing::isotropic_remeshing(
        faces(mesh_), target_edge_length, mesh_,
        CGAL::Polygon_mesh_processing::parameters::number_of_iterations(3));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    LOG(INFO) << "Uniform mesh resampling completed in " << elapsed.count()
              << " seconds.";
    LOG(INFO) << "Number of vertices: " << mesh_.number_of_vertices();
    LOG(INFO) << "Number of faces: " << mesh_.number_of_faces();
  }

  bool saveRepairedMesh() {
    return CGAL::IO::write_PLY(outputFilePath_.string(), mesh_);
  }

 private:
  boost::filesystem::path inputFilePath_;
  boost::filesystem::path outputFilePath_;
  Mesh mesh_;
};
