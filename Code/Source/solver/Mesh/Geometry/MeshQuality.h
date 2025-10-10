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

#ifndef SVMP_MESH_QUALITY_H
#define SVMP_MESH_QUALITY_H

#include "../Core/MeshTypes.h"
#include "../Topology/CellShape.h"
#include <array>
#include <vector>
#include <string>
#include <utility>

namespace svmp {

// Forward declaration
class MeshBase;

/**
 * @brief Mesh quality metrics computation
 *
 * This class provides various quality metrics for mesh cells including:
 * - Aspect ratio
 * - Skewness
 * - Jacobian quality
 * - Edge length ratios
 * - Angle metrics
 * - Volume/area ratios
 */
class MeshQuality {
public:
  // Available quality metrics
  enum class Metric {
    AspectRatio,        // Ratio of max to min edge/dimension
    Skewness,          // Deviation from ideal element shape
    Jacobian,          // Min/max Jacobian ratio
    EdgeRatio,         // Max/min edge length ratio
    MinAngle,          // Minimum angle in element
    MaxAngle,          // Maximum angle in element
    Warpage,           // Deviation from planar (for quads/hexes)
    Taper,             // Taper ratio (quads/hexes)
    Stretch,           // Stretch factor
    DiagonalRatio,     // Diagonal length ratio (quads/hexes)
    ConditionNumber,   // Condition number of Jacobian matrix
    ScaledJacobian,    // Scaled Jacobian
    ShapeIndex,        // Shape quality index
    RelativeSizeSquared, // Relative size squared
    ShapeAndSize       // Combined shape and size metric
  };

  // Convert metric enum to string
  static std::string metric_name(Metric m);

  // Convert string to metric enum
  static Metric metric_from_name(const std::string& name);

  // Main interface: compute quality for a single cell
  static real_t compute(const MeshBase& mesh, index_t cell, Metric metric,
                       Configuration cfg = Configuration::Reference);

  // Compute quality for a single cell by metric name
  static real_t compute(const MeshBase& mesh, index_t cell, const std::string& metric,
                       Configuration cfg = Configuration::Reference);

  // Compute quality for all cells
  static std::vector<real_t> compute_all(const MeshBase& mesh, Metric metric,
                                        Configuration cfg = Configuration::Reference);

  // Get global min/max quality
  static std::pair<real_t,real_t> global_range(const MeshBase& mesh, Metric metric,
                                              Configuration cfg = Configuration::Reference);

  // Get statistics for quality metric
  struct QualityStats {
    real_t min = 0;
    real_t max = 0;
    real_t mean = 0;
    real_t std_dev = 0;
    size_t count_poor = 0;     // Count below threshold
    size_t count_good = 0;     // Count above threshold
    size_t count_excellent = 0; // Count in excellent range
  };

  static QualityStats compute_statistics(const MeshBase& mesh, Metric metric,
                                        Configuration cfg = Configuration::Reference);

  // Quality thresholds for different metrics
  struct QualityThresholds {
    real_t poor = 0;       // Below this is poor quality
    real_t acceptable = 0;  // Above this is acceptable
    real_t good = 0;       // Above this is good
    real_t excellent = 0;  // Above this is excellent
  };

  static QualityThresholds get_thresholds(Metric metric, CellFamily family);

  // Check if cell quality is acceptable
  static bool is_acceptable(const MeshBase& mesh, index_t cell, Metric metric,
                           Configuration cfg = Configuration::Reference);

  // Find cells with poor quality
  static std::vector<index_t> find_poor_quality_cells(const MeshBase& mesh, Metric metric,
                                                     real_t threshold,
                                                     Configuration cfg = Configuration::Reference);

private:
  // Specific quality metric implementations
  static real_t compute_aspect_ratio(const MeshBase& mesh, index_t cell, Configuration cfg);
  static real_t compute_skewness(const MeshBase& mesh, index_t cell, Configuration cfg);
  static real_t compute_jacobian_quality(const MeshBase& mesh, index_t cell, Configuration cfg);
  static real_t compute_edge_ratio(const MeshBase& mesh, index_t cell, Configuration cfg);
  static real_t compute_min_angle(const MeshBase& mesh, index_t cell, Configuration cfg);
  static real_t compute_max_angle(const MeshBase& mesh, index_t cell, Configuration cfg);
  static real_t compute_warpage(const MeshBase& mesh, index_t cell, Configuration cfg);
  static real_t compute_taper(const MeshBase& mesh, index_t cell, Configuration cfg);
  static real_t compute_stretch(const MeshBase& mesh, index_t cell, Configuration cfg);
  static real_t compute_diagonal_ratio(const MeshBase& mesh, index_t cell, Configuration cfg);
  static real_t compute_condition_number(const MeshBase& mesh, index_t cell, Configuration cfg);
  static real_t compute_scaled_jacobian(const MeshBase& mesh, index_t cell, Configuration cfg);
  static real_t compute_shape_index(const MeshBase& mesh, index_t cell, Configuration cfg);
  static real_t compute_relative_size_squared(const MeshBase& mesh, index_t cell, Configuration cfg);
  static real_t compute_shape_and_size(const MeshBase& mesh, index_t cell, Configuration cfg);

  // Helper: get cell vertices
  static std::vector<std::array<real_t,3>> get_cell_vertices(const MeshBase& mesh, index_t cell,
                                                             Configuration cfg);

  // Helper: compute edge lengths
  static std::vector<real_t> compute_edge_lengths(const std::vector<std::array<real_t,3>>& vertices,
                                                  const CellShape& shape);

  // Helper: compute angles (for 2D elements)
  static std::vector<real_t> compute_angles_2d(const std::vector<std::array<real_t,3>>& vertices);

  // Helper: compute angles (for 3D elements at vertices)
  static std::vector<real_t> compute_angles_3d(const std::vector<std::array<real_t,3>>& vertices,
                                              const CellShape& shape);

  // Helper: compute face normals
  static std::vector<std::array<real_t,3>> compute_face_normals(const std::vector<std::array<real_t,3>>& vertices,
                                                               const CellShape& shape);
};

} // namespace svmp

#endif // SVMP_MESH_QUALITY_H