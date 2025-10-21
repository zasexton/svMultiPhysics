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

#ifndef SVMP_ADAPTIVITY_OPTIONS_H
#define SVMP_ADAPTIVITY_OPTIONS_H

#include <cstddef>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace svmp {

/**
 * @brief Unified options for mesh adaptivity operations
 *
 * Contains all configuration parameters for error estimation, marking,
 * refinement, coarsening, field transfer, and quality control.
 */
struct AdaptivityOptions {
  // ====================
  // General Settings
  // ====================

  /** Enable refinement operations */
  bool enable_refinement = true;

  /** Enable coarsening operations */
  bool enable_coarsening = false;

  /** Maximum refinement level (0 = original mesh) */
  size_t max_refinement_level = 5;

  /** Minimum refinement level (for coarsening) */
  size_t min_refinement_level = 0;

  /** Maximum number of elements allowed after refinement */
  size_t max_element_count = std::numeric_limits<size_t>::max();

  /** Minimum number of elements to maintain */
  size_t min_element_count = 1;

  /** Create new mesh or modify in place */
  bool create_new_mesh = true;

  // ====================
  // Error Estimation
  // ====================

  /** Type of error estimator */
  enum class EstimatorType {
    GRADIENT_RECOVERY,    // Zienkiewicz-Zhu gradient recovery
    JUMP_INDICATOR,      // Inter-element jump indicator
    RESIDUAL_BASED,      // Residual-based estimator
    USER_FIELD,          // User-provided field
    MULTI_CRITERIA       // Weighted combination
  };

  EstimatorType estimator_type = EstimatorType::GRADIENT_RECOVERY;

  /** Field name for user-provided error indicator */
  std::string user_field_name;

  /** Weights for multi-criteria aggregation */
  std::vector<double> estimator_weights;

  /** Power for L^p error norm (2.0 for L2) */
  double error_norm_power = 2.0;

  // ====================
  // Marking Strategy
  // ====================

  /** Type of marking strategy */
  enum class MarkingStrategy {
    FIXED_FRACTION,      // Mark fixed fraction (bulk/Dörfler marking)
    THRESHOLD_ABSOLUTE,  // Mark based on absolute threshold
    THRESHOLD_RELATIVE,  // Mark based on relative threshold
    FIXED_COUNT,        // Mark fixed number of elements
    REGION_AWARE        // Mark with region/label constraints
  };

  MarkingStrategy marking_strategy = MarkingStrategy::FIXED_FRACTION;

  /** Fraction of elements to refine (for FIXED_FRACTION) */
  double refine_fraction = 0.3;

  /** Fraction of elements to coarsen (for FIXED_FRACTION) */
  double coarsen_fraction = 0.1;

  /** Absolute error threshold for refinement */
  double refine_threshold = 1.0;

  /** Absolute error threshold for coarsening */
  double coarsen_threshold = 0.01;

  /** Number of elements to refine (for FIXED_COUNT) */
  size_t refine_count = 0;

  /** Number of elements to coarsen (for FIXED_COUNT) */
  size_t coarsen_count = 0;

  /** Region labels to include in marking (empty = all) */
  std::set<int> include_regions;

  /** Region labels to exclude from marking */
  std::set<int> exclude_regions;

  /** Boundary labels to exclude from marking */
  std::set<int> exclude_boundaries;

  // ====================
  // Refinement Patterns
  // ====================

  /** Type of refinement pattern */
  enum class RefinementPattern {
    RED,           // Regular refinement (all edges split)
    GREEN,         // Compatibility refinement
    BLUE,          // Alternative compatibility
    RED_GREEN,     // Red with green closure
    ADAPTIVE,      // Choose pattern based on marking
    HIERARCHICAL   // Hierarchical refinement (future)
  };

  RefinementPattern refinement_pattern = RefinementPattern::RED_GREEN;

  /** Use bisection for simplices (triangles/tetrahedra) */
  bool use_bisection = false;

  /** Allow anisotropic refinement (directional) */
  bool allow_anisotropic = false;

  /** Preferred anisotropic directions (if enabled) */
  std::vector<size_t> anisotropic_directions;

  // ====================
  // Coarsening Options
  // ====================

  /** Only coarsen elements that were previously refined */
  bool safe_coarsening_only = true;

  /** Check quality before accepting coarsening */
  bool check_coarsening_quality = true;

  /** Minimum quality ratio to accept coarsening */
  double min_coarsening_quality = 0.1;

  // ====================
  // Conformity Control
  // ====================

  /** How to handle non-conformity */
  enum class ConformityMode {
    ENFORCE_CONFORMING,   // Add closure elements
    ALLOW_HANGING_NODES,  // Create constraints for hanging nodes
    MINIMAL_CLOSURE       // Minimal refinement for conformity
  };

  ConformityMode conformity_mode = ConformityMode::ENFORCE_CONFORMING;

  /** Maximum closure iterations */
  size_t max_closure_iterations = 10;

  /** Maximum hanging level difference */
  size_t max_hanging_level = 1;

  // ====================
  // Field Transfer
  // ====================

  /** Prolongation (coarse to fine) method */
  enum class ProlongationMethod {
    COPY,              // Direct copy for vertex fields
    LINEAR_INTERP,     // Linear interpolation
    HIGH_ORDER_INTERP, // High-order interpolation
    CONSERVATIVE       // Conservative transfer
  };

  ProlongationMethod prolongation_method = ProlongationMethod::LINEAR_INTERP;

  /** Restriction (fine to coarse) method */
  enum class RestrictionMethod {
    AVERAGE,          // Simple averaging
    VOLUME_WEIGHTED,  // Volume-weighted average
    INJECTION,        // Direct injection (take one value)
    CONSERVATIVE      // Conservative restriction
  };

  RestrictionMethod restriction_method = RestrictionMethod::VOLUME_WEIGHTED;

  /** Fields to transfer (empty = all) */
  std::vector<std::string> transfer_fields;

  /** Fields to skip during transfer */
  std::vector<std::string> skip_fields;

  /** Preserve integral quantities during transfer */
  bool preserve_integrals = false;

  // ====================
  // Quality Control
  // ====================

  /** Check mesh quality after adaptivity */
  bool check_quality = true;

  /** Minimum acceptable element quality */
  double min_quality = 0.01;

  /** Quality measure to use */
  enum class QualityMeasure {
    ASPECT_RATIO,
    SKEWNESS,
    JACOBIAN,
    SCALED_JACOBIAN,
    SHAPE,
    VOLUME
  };

  QualityMeasure quality_measure = QualityMeasure::SCALED_JACOBIAN;

  /** Attempt smoothing if quality is poor */
  bool enable_smoothing = false;

  /** Maximum smoothing iterations */
  size_t max_smoothing_iterations = 10;

  /** Rollback adaptivity if quality check fails */
  bool rollback_on_poor_quality = false;

  // ====================
  // Parallel/Distributed
  // ====================

  /** Synchronize marking across MPI ranks */
  bool sync_marking = true;

  /** How to merge marks from different ranks */
  enum class MarkMergeStrategy {
    UNION,      // Mark if any rank marks
    INTERSECTION, // Mark only if all ranks mark
    MAJORITY    // Mark if majority of ranks mark
  };

  MarkMergeStrategy mark_merge_strategy = MarkMergeStrategy::UNION;

  /** Rebalance after adaptivity */
  bool rebalance_after_adapt = true;

  /** Target load imbalance factor */
  double load_imbalance_factor = 1.1;

  // ====================
  // Debug/Diagnostic
  // ====================

  /** Verbosity level (0=quiet, 1=normal, 2=verbose) */
  int verbosity = 1;

  /** Write intermediate meshes */
  bool write_intermediate_meshes = false;

  /** Output directory for intermediate meshes */
  std::string output_directory = "./adaptivity_output";

  /** Collect timing statistics */
  bool collect_timings = false;

  /** Validate connectivity after each step */
  bool validate_connectivity = false;

  /** Check parent-child relationships */
  bool track_provenance = false;

  // ====================
  // Constructor and Methods
  // ====================

  /** Default constructor with sensible defaults */
  AdaptivityOptions() = default;

  /**
   * @brief Validate options for consistency
   * @return True if options are valid, false otherwise
   */
  bool validate() const {
    // Basic validation
    if (max_refinement_level < min_refinement_level) return false;
    if (max_element_count < min_element_count) return false;
    if (refine_fraction < 0.0 || refine_fraction > 1.0) return false;
    if (coarsen_fraction < 0.0 || coarsen_fraction > 1.0) return false;
    if (refine_fraction + coarsen_fraction > 1.0) return false;
    if (min_quality < 0.0 || min_quality > 1.0) return false;
    if (load_imbalance_factor < 1.0) return false;

    return true;
  }

  /**
   * @brief Create options for pure refinement
   */
  static AdaptivityOptions refinement_only() {
    AdaptivityOptions opts;
    opts.enable_refinement = true;
    opts.enable_coarsening = false;
    return opts;
  }

  /**
   * @brief Create options for pure coarsening
   */
  static AdaptivityOptions coarsening_only() {
    AdaptivityOptions opts;
    opts.enable_refinement = false;
    opts.enable_coarsening = true;
    return opts;
  }

  /**
   * @brief Create options for balanced h-adaptivity
   */
  static AdaptivityOptions h_adaptivity() {
    AdaptivityOptions opts;
    opts.enable_refinement = true;
    opts.enable_coarsening = true;
    opts.refine_fraction = 0.3;
    opts.coarsen_fraction = 0.1;
    return opts;
  }
};

} // namespace svmp

#endif // SVMP_ADAPTIVITY_OPTIONS_H