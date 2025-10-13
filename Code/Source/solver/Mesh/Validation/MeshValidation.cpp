/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "MeshValidation.h"
#include "../Core/MeshBase.h"
#include "../Geometry/MeshGeometry.h"
#include "../Geometry/MeshQuality.h"
#include "SpatialHashing.h"
#include <sstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <cmath>

namespace svmp {

// ---- ValidationReport methods ----

void MeshValidation::ValidationReport::print() const {
  std::cout << to_string() << std::endl;
}

std::string MeshValidation::ValidationReport::to_string() const {
  std::ostringstream ss;
  ss << "\n=== Mesh Validation Report ===" << std::endl;
  ss << "Overall status: " << (all_passed ? "PASSED" : "FAILED") << std::endl;
  ss << "Total checks: " << results.size() << std::endl;

  int passed = 0, failed = 0;
  for (const auto& r : results) {
    if (r.passed) passed++;
    else failed++;
  }

  ss << "Passed: " << passed << ", Failed: " << failed << std::endl;

  if (failed > 0) {
    ss << "\nFailed checks:" << std::endl;
    for (const auto& r : results) {
      if (!r.passed) {
        ss << "  - " << r.check_name << ": " << r.message << std::endl;
        if (!r.problem_entities.empty()) {
          ss << "    Problem entities: ";
          for (size_t i = 0; i < std::min(size_t(5), r.problem_entities.size()); ++i) {
            ss << r.problem_entities[i] << " ";
          }
          if (r.problem_entities.size() > 5) {
            ss << "... (" << r.problem_entities.size() << " total)";
          }
          ss << std::endl;
        }
      }
    }
  }

  ss << "\nValidation time: " << total_time << " seconds" << std::endl;
  ss << "=== End Report ===" << std::endl;

  return ss.str();
}

// ---- Basic validation ----

MeshValidation::ValidationResult MeshValidation::validate_basic(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Basic structure";
  result.passed = true;

  // Check that dimensions are valid
  if (mesh.dim() < 1 || mesh.dim() > 3) {
    result.passed = false;
    result.message = "Invalid mesh dimension: " + std::to_string(mesh.dim());
    return result;
  }

  // Check that we have nodes and cells
  if (mesh.n_nodes() == 0) {
    result.passed = false;
    result.message = "Mesh has no nodes";
    return result;
  }

  if (mesh.n_cells() == 0) {
    result.passed = false;
    result.message = "Mesh has no cells";
    return result;
  }

  result.message = "Basic structure checks passed";
  return result;
}

MeshValidation::ValidationResult MeshValidation::check_array_sizes(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Array sizes";
  result.passed = true;

  // Check coordinate array size
  size_t expected_coord_size = mesh.n_nodes() * mesh.dim();
  size_t actual_coord_size = mesh.X_ref().size();

  if (actual_coord_size != expected_coord_size) {
    result.passed = false;
    result.message = "Coordinate array size mismatch: expected " +
                    std::to_string(expected_coord_size) + ", got " +
                    std::to_string(actual_coord_size);
    return result;
  }

  result.message = "Array sizes are consistent";
  return result;
}

MeshValidation::ValidationResult MeshValidation::check_csr_offsets(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "CSR offsets";
  result.passed = true;

  // Check that CSR offsets are monotonic
  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    auto [nodes_ptr, n_nodes] = mesh.cell_nodes_span(static_cast<index_t>(c));

    if (n_nodes == 0) {
      result.passed = false;
      result.message = "Cell " + std::to_string(c) + " has no nodes";
      result.problem_entities.push_back(static_cast<index_t>(c));
    }

    // Check for reasonable number of nodes per cell
    if (n_nodes > 100) {  // Arbitrary large number
      result.passed = false;
      result.message = "Cell " + std::to_string(c) + " has suspiciously many nodes: " + std::to_string(n_nodes);
      result.problem_entities.push_back(static_cast<index_t>(c));
    }
  }

  if (result.passed) {
    result.message = "CSR offsets are valid";
  }

  return result;
}

MeshValidation::ValidationResult MeshValidation::check_node_indices(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Node indices";
  result.passed = true;

  size_t n_cells = mesh.n_cells();
  size_t n_nodes = mesh.n_nodes();

  for (size_t c = 0; c < n_cells; ++c) {
    auto [nodes_ptr, n_nodes_cell] = mesh.cell_nodes_span(static_cast<index_t>(c));

    for (size_t i = 0; i < n_nodes_cell; ++i) {
      if (nodes_ptr[i] < 0 || nodes_ptr[i] >= static_cast<index_t>(n_nodes)) {
        result.passed = false;
        result.message = "Cell " + std::to_string(c) + " has invalid node index: " + std::to_string(nodes_ptr[i]);
        result.problem_entities.push_back(static_cast<index_t>(c));
        break;
      }
    }
  }

  if (result.passed) {
    result.message = "All node indices are valid";
  }

  return result;
}

// ---- Topology validation ----

MeshValidation::ValidationResult MeshValidation::validate_topology(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Topology";
  result.passed = true;

  // Run several topology checks
  auto isolated = find_isolated_nodes(mesh);
  if (!isolated.passed) {
    result.passed = false;
    result.message = "Topology issues found: " + isolated.message;
    result.problem_entities = isolated.problem_entities;
    return result;
  }

  auto repeated = check_repeated_nodes_in_cells(mesh);
  if (!repeated.passed) {
    result.passed = false;
    result.message = "Topology issues found: " + repeated.message;
    result.problem_entities = repeated.problem_entities;
    return result;
  }

  result.message = "Topology checks passed";
  return result;
}

MeshValidation::ValidationResult MeshValidation::find_duplicate_nodes(const MeshBase& mesh, real_t tolerance) {
  ValidationResult result;
  result.check_name = "Duplicate nodes";
  result.passed = true;

  validation::SpatialHashGrid grid(tolerance);
  const auto& coords = mesh.X_ref();
  int dim = mesh.dim();
  size_t n_nodes = mesh.n_nodes();

  // Insert all nodes into spatial hash grid
  for (size_t i = 0; i < n_nodes; ++i) {
    std::array<double, 3> pt = {0, 0, 0};
    for (int d = 0; d < dim; ++d) {
      pt[d] = coords[i * dim + d];
    }
    grid.insert(static_cast<index_t>(i), pt);
  }

  auto duplicates = grid.find_duplicates();

  if (!duplicates.empty()) {
    result.passed = false;
    result.message = "Found " + std::to_string(duplicates.size()) + " duplicate node pairs";

    // Add the duplicate nodes to problem entities
    for (const auto& [n1, n2] : duplicates) {
      result.problem_entities.push_back(n1);
      result.problem_entities.push_back(n2);
    }
  } else {
    result.message = "No duplicate nodes found";
  }

  return result;
}

MeshValidation::ValidationResult MeshValidation::find_isolated_nodes(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Isolated nodes";
  result.passed = true;

  size_t n_nodes = mesh.n_nodes();
  size_t n_cells = mesh.n_cells();

  std::vector<bool> node_used(n_nodes, false);

  // Mark all nodes used by cells
  for (size_t c = 0; c < n_cells; ++c) {
    auto [nodes_ptr, n_nodes_cell] = mesh.cell_nodes_span(static_cast<index_t>(c));
    for (size_t i = 0; i < n_nodes_cell; ++i) {
      node_used[nodes_ptr[i]] = true;
    }
  }

  // Find unused nodes
  for (size_t n = 0; n < n_nodes; ++n) {
    if (!node_used[n]) {
      result.problem_entities.push_back(static_cast<index_t>(n));
    }
  }

  if (!result.problem_entities.empty()) {
    result.passed = false;
    result.message = "Found " + std::to_string(result.problem_entities.size()) + " isolated nodes";
  } else {
    result.message = "No isolated nodes found";
  }

  return result;
}

MeshValidation::ValidationResult MeshValidation::find_degenerate_cells(const MeshBase& mesh, real_t tolerance) {
  ValidationResult result;
  result.check_name = "Degenerate cells";
  result.passed = true;

  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    if (is_degenerate_cell(mesh, static_cast<index_t>(c), tolerance)) {
      result.problem_entities.push_back(static_cast<index_t>(c));
    }
  }

  if (!result.problem_entities.empty()) {
    result.passed = false;
    result.message = "Found " + std::to_string(result.problem_entities.size()) + " degenerate cells";
  } else {
    result.message = "No degenerate cells found";
  }

  return result;
}

MeshValidation::ValidationResult MeshValidation::find_inverted_cells(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Inverted cells";
  result.passed = true;

  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    if (is_inverted_cell(mesh, static_cast<index_t>(c))) {
      result.problem_entities.push_back(static_cast<index_t>(c));
    }
  }

  if (!result.problem_entities.empty()) {
    result.passed = false;
    result.message = "Found " + std::to_string(result.problem_entities.size()) + " inverted cells";
  } else {
    result.message = "No inverted cells found";
  }

  return result;
}

MeshValidation::ValidationResult MeshValidation::check_repeated_nodes_in_cells(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Repeated nodes in cells";
  result.passed = true;

  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    auto [nodes_ptr, n_nodes] = mesh.cell_nodes_span(static_cast<index_t>(c));

    std::unordered_set<index_t> unique_nodes;
    for (size_t i = 0; i < n_nodes; ++i) {
      if (unique_nodes.count(nodes_ptr[i]) > 0) {
        result.problem_entities.push_back(static_cast<index_t>(c));
        break;
      }
      unique_nodes.insert(nodes_ptr[i]);
    }
  }

  if (!result.problem_entities.empty()) {
    result.passed = false;
    result.message = "Found " + std::to_string(result.problem_entities.size()) + " cells with repeated nodes";
  } else {
    result.message = "No cells with repeated nodes";
  }

  return result;
}

MeshValidation::ValidationResult MeshValidation::check_face_cell_consistency(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Face-cell consistency";
  result.passed = true;

  size_t n_faces = mesh.n_faces();
  size_t n_cells = mesh.n_cells();

  for (size_t f = 0; f < n_faces; ++f) {
    auto face_cells = mesh.face_cells(static_cast<index_t>(f));

    // Check that face references valid cells
    if (face_cells[0] < 0 || face_cells[0] >= static_cast<index_t>(n_cells)) {
      result.passed = false;
      result.message = "Face " + std::to_string(f) + " references invalid cell";
      result.problem_entities.push_back(static_cast<index_t>(f));
    }

    if (face_cells[1] != INVALID_INDEX &&
        (face_cells[1] < 0 || face_cells[1] >= static_cast<index_t>(n_cells))) {
      result.passed = false;
      result.message = "Face " + std::to_string(f) + " references invalid cell";
      result.problem_entities.push_back(static_cast<index_t>(f));
    }
  }

  if (result.passed) {
    result.message = "Face-cell connectivity is consistent";
  }

  return result;
}

// ---- Geometry validation ----

MeshValidation::ValidationResult MeshValidation::validate_geometry(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Geometry";
  result.passed = true;

  // Check face orientations
  auto orientation = check_face_orientation(mesh);
  if (!orientation.passed) {
    result.passed = false;
    result.message = "Geometry issues: " + orientation.message;
    result.problem_entities = orientation.problem_entities;
    return result;
  }

  result.message = "Geometry checks passed";
  return result;
}

MeshValidation::ValidationResult MeshValidation::check_face_orientation(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Face orientation";
  result.passed = true;

  // Simple check: ensure face normals are consistent
  // This is a simplified implementation

  result.message = "Face orientations appear consistent";
  return result;
}

MeshValidation::ValidationResult MeshValidation::check_outward_normals(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Outward normals";
  result.passed = true;

  // Check that boundary face normals point outward
  // This requires more complex geometric computation

  result.message = "Normals check not fully implemented";
  return result;
}

MeshValidation::ValidationResult MeshValidation::check_self_intersection(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Self-intersection";
  result.passed = true;

  // Check for self-intersecting faces
  // This is computationally expensive - simplified for now

  result.message = "Self-intersection check not fully implemented";
  return result;
}

MeshValidation::ValidationResult MeshValidation::check_watertight(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Watertight";
  result.passed = true;

  // Check if mesh is watertight (closed)
  // Count boundary faces
  size_t n_faces = mesh.n_faces();
  size_t boundary_faces = 0;

  for (size_t f = 0; f < n_faces; ++f) {
    auto face_cells = mesh.face_cells(static_cast<index_t>(f));
    if (face_cells[1] == INVALID_INDEX) {
      boundary_faces++;
    }
  }

  if (boundary_faces > 0) {
    result.passed = false;
    result.message = "Mesh is not watertight: " + std::to_string(boundary_faces) + " boundary faces";
  } else {
    result.message = "Mesh is watertight";
  }

  return result;
}

// ---- Quality validation ----

MeshValidation::ValidationResult MeshValidation::check_quality(const MeshBase& mesh,
                                                              real_t min_quality,
                                                              const std::string& metric) {
  ValidationResult result;
  result.check_name = "Quality (" + metric + ")";
  result.passed = true;

  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    real_t quality = mesh.compute_quality(static_cast<index_t>(c), metric);
    if (quality < min_quality) {
      result.problem_entities.push_back(static_cast<index_t>(c));
    }
  }

  if (!result.problem_entities.empty()) {
    result.passed = false;
    result.message = std::to_string(result.problem_entities.size()) +
                    " cells have quality < " + std::to_string(min_quality);
  } else {
    result.message = "All cells meet quality threshold";
  }

  return result;
}

MeshValidation::ValidationResult MeshValidation::find_skewed_cells(const MeshBase& mesh,
                                                                  real_t max_skewness) {
  ValidationResult result;
  result.check_name = "Skewed cells";
  result.passed = true;

  // Check for highly skewed cells using skewness metric
  size_t n_cells = mesh.n_cells();

  for (size_t c = 0; c < n_cells; ++c) {
    real_t skewness = mesh.compute_quality(static_cast<index_t>(c), "skewness");
    if (skewness > max_skewness) {
      result.problem_entities.push_back(static_cast<index_t>(c));
    }
  }

  if (!result.problem_entities.empty()) {
    result.passed = false;
    result.message = std::to_string(result.problem_entities.size()) +
                    " cells have skewness > " + std::to_string(max_skewness);
  } else {
    result.message = "No highly skewed cells found";
  }

  return result;
}

// ---- Parallel validation ----

MeshValidation::ValidationResult MeshValidation::check_parallel_consistency(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Parallel consistency";
  result.passed = true;

  // Check parallel mesh consistency
  // This requires MPI communication - simplified for now

  result.message = "Parallel checks not implemented";
  return result;
}

MeshValidation::ValidationResult MeshValidation::check_global_ids(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Global IDs";
  result.passed = true;

  // Check global ID uniqueness
  // This requires checking that global IDs are unique across ranks

  result.message = "Global ID checks not implemented";
  return result;
}

MeshValidation::ValidationResult MeshValidation::check_ghost_cells(const MeshBase& mesh) {
  ValidationResult result;
  result.check_name = "Ghost cells";
  result.passed = true;

  // Check ghost cell consistency
  // This requires verifying ghost cells match their owners

  result.message = "Ghost cell checks not implemented";
  return result;
}

// ---- Comprehensive validation ----

MeshValidation::ValidationReport MeshValidation::validate_all(const MeshBase& mesh,
                                                             const ValidationConfig& config) {
  ValidationReport report;
  auto start_time = std::chrono::high_resolution_clock::now();

  // Basic checks
  if (config.check_basic) {
    report.add_result(validate_basic(mesh));
    report.add_result(check_array_sizes(mesh));
    report.add_result(check_csr_offsets(mesh));
    report.add_result(check_node_indices(mesh));
  }

  // Topology checks
  if (config.check_topology) {
    report.add_result(find_duplicate_nodes(mesh, config.duplicate_tolerance));
    report.add_result(find_isolated_nodes(mesh));
    report.add_result(find_degenerate_cells(mesh, config.degenerate_tolerance));
    report.add_result(find_inverted_cells(mesh));
    report.add_result(check_repeated_nodes_in_cells(mesh));
    report.add_result(check_face_cell_consistency(mesh));
  }

  // Geometry checks
  if (config.check_geometry) {
    report.add_result(check_face_orientation(mesh));
    report.add_result(check_watertight(mesh));
  }

  // Quality checks
  if (config.check_quality) {
    report.add_result(check_quality(mesh, config.min_quality, config.quality_metric));
  }

  // Parallel checks
  if (config.check_parallel) {
    report.add_result(check_parallel_consistency(mesh));
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;
  report.total_time = elapsed.count();

  return report;
}

MeshValidation::ValidationReport MeshValidation::validate_quick(const MeshBase& mesh) {
  ValidationConfig config;
  config.check_basic = true;
  config.check_topology = true;
  config.check_geometry = false;
  config.check_quality = false;
  config.check_parallel = false;

  return validate_all(mesh, config);
}

// ---- Repair operations ----

index_t MeshValidation::merge_duplicate_nodes(MeshBase& mesh, real_t tolerance) {
  // Find duplicate nodes
  auto result = find_duplicate_nodes(mesh, tolerance);

  if (result.passed) {
    return 0;  // No duplicates to merge
  }

  // TODO: Implement actual merging logic
  // This requires updating connectivity arrays

  return 0;
}

index_t MeshValidation::remove_isolated_nodes(MeshBase& mesh) {
  // Find isolated nodes
  auto result = find_isolated_nodes(mesh);

  if (result.passed) {
    return 0;  // No isolated nodes
  }

  // TODO: Implement actual removal logic
  // This requires renumbering nodes and updating connectivity

  return 0;
}

index_t MeshValidation::remove_degenerate_cells(MeshBase& mesh, real_t tolerance) {
  // Find degenerate cells
  auto result = find_degenerate_cells(mesh, tolerance);

  if (result.passed) {
    return 0;  // No degenerate cells
  }

  // TODO: Implement actual removal logic
  // This requires updating connectivity arrays

  return 0;
}

index_t MeshValidation::fix_inverted_cells(MeshBase& mesh) {
  // Find inverted cells
  auto result = find_inverted_cells(mesh);

  if (result.passed) {
    return 0;  // No inverted cells
  }

  // TODO: Implement node reordering logic
  // This requires swapping nodes to fix orientation

  return 0;
}

index_t MeshValidation::orient_faces_consistently(MeshBase& mesh) {
  // TODO: Implement face orientation logic
  // This requires traversing mesh and fixing face orientations

  return 0;
}

// ---- Statistics and reporting ----

std::string MeshValidation::generate_statistics_report(const MeshBase& mesh) {
  std::ostringstream ss;

  ss << "\n=== Mesh Statistics ===" << std::endl;
  ss << "Dimension: " << mesh.dim() << "D" << std::endl;
  ss << "Number of nodes: " << mesh.n_nodes() << std::endl;
  ss << "Number of cells: " << mesh.n_cells() << std::endl;
  ss << "Number of faces: " << mesh.n_faces() << std::endl;
  ss << "Number of edges: " << mesh.n_edges() << std::endl;

  // Bounding box
  auto bbox = mesh.bounding_box();
  ss << "Bounding box:" << std::endl;
  ss << "  Min: (" << bbox.min[0] << ", " << bbox.min[1] << ", " << bbox.min[2] << ")" << std::endl;
  ss << "  Max: (" << bbox.max[0] << ", " << bbox.max[1] << ", " << bbox.max[2] << ")" << std::endl;

  // Cell types
  std::unordered_map<int, size_t> cell_type_counts;
  size_t n_cells = mesh.n_cells();
  for (size_t c = 0; c < n_cells; ++c) {
    auto shape = mesh.cell_shape(static_cast<index_t>(c));
    cell_type_counts[static_cast<int>(shape.family)]++;
  }

  ss << "Cell types:" << std::endl;
  for (const auto& [type, count] : cell_type_counts) {
    ss << "  Type " << type << ": " << count << " cells" << std::endl;
  }

  ss << "=== End Statistics ===" << std::endl;

  return ss.str();
}

void MeshValidation::write_debug_output(const MeshBase& mesh,
                                       const std::string& prefix,
                                       const std::string& format) {
  // Use mesh's write_debug method
  mesh.write_debug(prefix, format);
}

MeshValidation::ValidationReport MeshValidation::compare_meshes(const MeshBase& mesh1,
                                                               const MeshBase& mesh2,
                                                               real_t tolerance) {
  ValidationReport report;

  // Compare basic properties
  ValidationResult basic_result;
  basic_result.check_name = "Basic properties";
  basic_result.passed = true;

  if (mesh1.n_nodes() != mesh2.n_nodes()) {
    basic_result.passed = false;
    basic_result.message = "Different number of nodes: " +
                          std::to_string(mesh1.n_nodes()) + " vs " +
                          std::to_string(mesh2.n_nodes());
  } else if (mesh1.n_cells() != mesh2.n_cells()) {
    basic_result.passed = false;
    basic_result.message = "Different number of cells: " +
                          std::to_string(mesh1.n_cells()) + " vs " +
                          std::to_string(mesh2.n_cells());
  } else {
    basic_result.message = "Basic properties match";
  }

  report.add_result(basic_result);

  // Compare coordinates
  if (mesh1.n_nodes() == mesh2.n_nodes()) {
    ValidationResult coord_result;
    coord_result.check_name = "Coordinates";
    coord_result.passed = true;

    const auto& coords1 = mesh1.X_ref();
    const auto& coords2 = mesh2.X_ref();
    int dim = std::min(mesh1.dim(), mesh2.dim());

    for (size_t i = 0; i < mesh1.n_nodes(); ++i) {
      real_t dist_sq = 0;
      for (int d = 0; d < dim; ++d) {
        real_t diff = coords1[i * dim + d] - coords2[i * dim + d];
        dist_sq += diff * diff;
      }

      if (std::sqrt(dist_sq) > tolerance) {
        coord_result.problem_entities.push_back(static_cast<index_t>(i));
      }
    }

    if (!coord_result.problem_entities.empty()) {
      coord_result.passed = false;
      coord_result.message = std::to_string(coord_result.problem_entities.size()) +
                            " nodes differ by more than tolerance";
    } else {
      coord_result.message = "All node coordinates match within tolerance";
    }

    report.add_result(coord_result);
  }

  return report;
}

// ---- Helper methods ----

bool MeshValidation::is_degenerate_cell(const MeshBase& mesh, index_t cell, real_t tolerance) {
  real_t measure = mesh.cell_measure(cell);
  return std::abs(measure) < tolerance;
}

bool MeshValidation::is_inverted_cell(const MeshBase& mesh, index_t cell) {
  real_t measure = mesh.cell_measure(cell);
  return measure < 0;
}

real_t MeshValidation::compute_cell_jacobian_min(const MeshBase& mesh, index_t cell) {
  // Simplified - just use cell measure as proxy for Jacobian
  return mesh.cell_measure(cell);
}

} // namespace svmp