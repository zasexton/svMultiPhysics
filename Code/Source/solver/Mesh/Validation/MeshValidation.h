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

#ifndef SVMP_MESH_VALIDATION_H
#define SVMP_MESH_VALIDATION_H

#include "../Core/MeshTypes.h"
#include <vector>
#include <string>
#include <unordered_set>

namespace svmp {

// Forward declaration
class MeshBase;

/**
 * @brief Mesh validation and diagnostics
 *
 * This class provides comprehensive mesh validation including:
 * - Basic structural checks
 * - Topology validation
 * - Geometry validation
 * - Quality checks
 * - Parallel consistency checks
 */
class MeshValidation {
public:
  /**
   * @brief Validation result for a single check
   */
  struct ValidationResult {
    bool passed = true;
    std::string check_name;
    std::string message;
    std::vector<index_t> problem_entities; // Entities that failed check
  };

  /**
   * @brief Full validation report
   */
  struct ValidationReport {
    bool all_passed = true;
    std::vector<ValidationResult> results;
    double total_time = 0;  // Total validation time in seconds

    void add_result(const ValidationResult& result) {
      results.push_back(result);
      if (!result.passed) all_passed = false;
    }

    void print() const;
    std::string to_string() const;
  };

  // ---- Basic validation ----

  /**
   * @brief Validate basic mesh structure
   * @param mesh The mesh
   * @return Validation result
   */
  static ValidationResult validate_basic(const MeshBase& mesh);

  /**
   * @brief Check array sizes and offsets consistency
   * @param mesh The mesh
   * @return Validation result
   */
  static ValidationResult check_array_sizes(const MeshBase& mesh);

  /**
   * @brief Check CSR offset arrays
   * @param mesh The mesh
   * @return Validation result
   */
  static ValidationResult check_csr_offsets(const MeshBase& mesh);

  /**
   * @brief Check node indices are in valid range
   * @param mesh The mesh
   * @return Validation result
   */
  static ValidationResult check_node_indices(const MeshBase& mesh);

  // ---- Topology validation ----

  /**
   * @brief Validate mesh topology
   * @param mesh The mesh
   * @return Validation result
   */
  static ValidationResult validate_topology(const MeshBase& mesh);

  /**
   * @brief Find duplicate nodes
   * @param mesh The mesh
   * @param tolerance Geometric tolerance
   * @return Validation result with duplicate node pairs
   */
  static ValidationResult find_duplicate_nodes(const MeshBase& mesh,
                                              real_t tolerance = 1e-10);

  /**
   * @brief Find isolated nodes (not used by any cell)
   * @param mesh The mesh
   * @return Validation result with isolated node indices
   */
  static ValidationResult find_isolated_nodes(const MeshBase& mesh);

  /**
   * @brief Find degenerate cells (zero or negative volume)
   * @param mesh The mesh
   * @param tolerance Volume tolerance
   * @return Validation result with degenerate cell indices
   */
  static ValidationResult find_degenerate_cells(const MeshBase& mesh,
                                               real_t tolerance = 1e-12);

  /**
   * @brief Find inverted cells (negative Jacobian)
   * @param mesh The mesh
   * @return Validation result with inverted cell indices
   */
  static ValidationResult find_inverted_cells(const MeshBase& mesh);

  /**
   * @brief Check for repeated nodes within cells
   * @param mesh The mesh
   * @return Validation result
   */
  static ValidationResult check_repeated_nodes_in_cells(const MeshBase& mesh);

  /**
   * @brief Check face-cell consistency
   * @param mesh The mesh
   * @return Validation result
   */
  static ValidationResult check_face_cell_consistency(const MeshBase& mesh);

  // ---- Geometry validation ----

  /**
   * @brief Validate mesh geometry
   * @param mesh The mesh
   * @return Validation result
   */
  static ValidationResult validate_geometry(const MeshBase& mesh);

  /**
   * @brief Check face orientation consistency
   * @param mesh The mesh
   * @return Validation result
   */
  static ValidationResult check_face_orientation(const MeshBase& mesh);

  /**
   * @brief Check if normals point outward
   * @param mesh The mesh
   * @return Validation result
   */
  static ValidationResult check_outward_normals(const MeshBase& mesh);

  /**
   * @brief Check for self-intersecting faces
   * @param mesh The mesh
   * @return Validation result with intersecting face pairs
   */
  static ValidationResult check_self_intersection(const MeshBase& mesh);

  /**
   * @brief Check if mesh is watertight
   * @param mesh The mesh
   * @return Validation result
   */
  static ValidationResult check_watertight(const MeshBase& mesh);

  // ---- Quality validation ----

  /**
   * @brief Check mesh quality metrics
   * @param mesh The mesh
   * @param min_quality Minimum acceptable quality
   * @param metric Quality metric name
   * @return Validation result with poor quality cells
   */
  static ValidationResult check_quality(const MeshBase& mesh,
                                       real_t min_quality = 0.01,
                                       const std::string& metric = "aspect_ratio");

  /**
   * @brief Find highly skewed cells
   * @param mesh The mesh
   * @param max_skewness Maximum acceptable skewness
   * @return Validation result
   */
  static ValidationResult find_skewed_cells(const MeshBase& mesh,
                                          real_t max_skewness = 0.9);

  // ---- Parallel validation ----

  /**
   * @brief Check parallel mesh consistency
   * @param mesh The mesh
   * @return Validation result
   */
  static ValidationResult check_parallel_consistency(const MeshBase& mesh);

  /**
   * @brief Check global ID uniqueness
   * @param mesh The mesh
   * @return Validation result
   */
  static ValidationResult check_global_ids(const MeshBase& mesh);

  /**
   * @brief Check ghost cell consistency
   * @param mesh The mesh
   * @return Validation result
   */
  static ValidationResult check_ghost_cells(const MeshBase& mesh);

  // ---- Comprehensive validation ----

  /**
   * @brief Run all validation checks
   * @param mesh The mesh
   * @param config Validation configuration
   * @return Complete validation report
   */
  struct ValidationConfig {
    bool check_basic = true;
    bool check_topology = true;
    bool check_geometry = true;
    bool check_quality = true;
    bool check_parallel = false;
    real_t duplicate_tolerance = 1e-10;
    real_t degenerate_tolerance = 1e-12;
    real_t min_quality = 0.01;
    std::string quality_metric = "aspect_ratio";
  };

  static ValidationReport validate_all(const MeshBase& mesh,
                                      const ValidationConfig& config = ValidationConfig{});

  /**
   * @brief Quick validation (essential checks only)
   * @param mesh The mesh
   * @return Validation report
   */
  static ValidationReport validate_quick(const MeshBase& mesh);

  // ---- Repair operations ----

  /**
   * @brief Merge duplicate nodes
   * @param mesh The mesh
   * @param tolerance Merge tolerance
   * @return Number of nodes merged
   */
  static index_t merge_duplicate_nodes(MeshBase& mesh,
                                      real_t tolerance = 1e-10);

  /**
   * @brief Remove isolated nodes
   * @param mesh The mesh
   * @return Number of nodes removed
   */
  static index_t remove_isolated_nodes(MeshBase& mesh);

  /**
   * @brief Remove degenerate cells
   * @param mesh The mesh
   * @param tolerance Volume tolerance
   * @return Number of cells removed
   */
  static index_t remove_degenerate_cells(MeshBase& mesh,
                                        real_t tolerance = 1e-12);

  /**
   * @brief Fix inverted cells by reordering nodes
   * @param mesh The mesh
   * @return Number of cells fixed
   */
  static index_t fix_inverted_cells(MeshBase& mesh);

  /**
   * @brief Orient faces consistently
   * @param mesh The mesh
   * @return Number of faces reoriented
   */
  static index_t orient_faces_consistently(MeshBase& mesh);

  // ---- Statistics and reporting ----

  /**
   * @brief Generate mesh statistics report
   * @param mesh The mesh
   * @return Statistics string
   */
  static std::string generate_statistics_report(const MeshBase& mesh);

  /**
   * @brief Write detailed debug information
   * @param mesh The mesh
   * @param prefix File prefix
   * @param format Output format (vtk, vtu)
   */
  static void write_debug_output(const MeshBase& mesh,
                                const std::string& prefix,
                                const std::string& format = "vtu");

  /**
   * @brief Compare two meshes for differences
   * @param mesh1 First mesh
   * @param mesh2 Second mesh
   * @param tolerance Geometric tolerance
   * @return Comparison report
   */
  static ValidationReport compare_meshes(const MeshBase& mesh1,
                                        const MeshBase& mesh2,
                                        real_t tolerance = 1e-10);

private:
  // Helper methods
  static bool is_degenerate_cell(const MeshBase& mesh, index_t cell, real_t tolerance);
  static bool is_inverted_cell(const MeshBase& mesh, index_t cell);
  static real_t compute_cell_jacobian_min(const MeshBase& mesh, index_t cell);
};

} // namespace svmp

#endif // SVMP_MESH_VALIDATION_H