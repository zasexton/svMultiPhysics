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

#ifndef SVMP_ADAPTIVITY_MANAGER_H
#define SVMP_ADAPTIVITY_MANAGER_H

#include "Options.h"
#include "ErrorEstimator.h"
#include "Marker.h"
#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace svmp {

// Forward declarations
class MeshBase;
class MeshFields;
class RefinementRule;
class FieldTransfer;
class QualityChecker;
class ConformityEnforcer;

/**
 * @brief Result of adaptivity operation
 */
struct AdaptivityResult {
  /** Success status */
  bool success = false;

  /** Adapted mesh (if create_new_mesh was true) */
  std::unique_ptr<MeshBase> adapted_mesh;

  /** Number of refinement steps performed */
  size_t refinement_steps = 0;

  /** Number of coarsening steps performed */
  size_t coarsening_steps = 0;

  /** Number of elements before adaptation */
  size_t initial_element_count = 0;

  /** Number of elements after adaptation */
  size_t final_element_count = 0;

  /** Number of vertices before adaptation */
  size_t initial_vertex_count = 0;

  /** Number of vertices after adaptation */
  size_t final_vertex_count = 0;

  /** Number of elements marked for refinement */
  size_t num_refined = 0;

  /** Number of elements marked for coarsening */
  size_t num_coarsened = 0;

  /** Global error before adaptation */
  double initial_error = 0.0;

  /** Global error after adaptation (estimated) */
  double final_error = 0.0;

  /** Minimum element quality after adaptation */
  double min_quality = 0.0;

  /** Average element quality after adaptation */
  double avg_quality = 0.0;

  /** Total time for adaptation */
  std::chrono::duration<double> total_time;

  /** Time breakdown by phase */
  struct TimingBreakdown {
    std::chrono::duration<double> estimation;
    std::chrono::duration<double> marking;
    std::chrono::duration<double> refinement;
    std::chrono::duration<double> coarsening;
    std::chrono::duration<double> conformity;
    std::chrono::duration<double> field_transfer;
    std::chrono::duration<double> quality_check;
    std::chrono::duration<double> finalization;
  } timing;

  /** Error messages if failed */
  std::vector<std::string> error_messages;

  /** Warning messages */
  std::vector<std::string> warning_messages;

  /**
   * @brief Generate summary report
   */
  std::string summary() const;
};

/**
 * @brief Main orchestrator for mesh adaptivity operations
 *
 * Coordinates the complete adaptivity pipeline: error estimation,
 * marking, refinement/coarsening, conformity enforcement, field transfer,
 * and quality checking.
 */
class AdaptivityManager {
public:
  /**
   * @brief Constructor with options
   */
  explicit AdaptivityManager(const AdaptivityOptions& options = {});

  /**
   * @brief Destructor
   */
  ~AdaptivityManager();

  /**
   * @brief Perform adaptive mesh refinement/coarsening
   *
   * @param mesh The mesh to adapt
   * @param fields Optional fields for error estimation and transfer
   * @return Result of adaptation
   */
  AdaptivityResult adapt(MeshBase& mesh, MeshFields* fields = nullptr);

  /**
   * @brief Perform single refinement step
   *
   * @param mesh The mesh to refine
   * @param marks Elements marked for refinement
   * @param fields Optional fields for transfer
   * @return Result of refinement
   */
  AdaptivityResult refine(
      MeshBase& mesh,
      const std::vector<bool>& marks,
      MeshFields* fields = nullptr);

  /**
   * @brief Perform single coarsening step
   *
   * @param mesh The mesh to coarsen
   * @param marks Elements marked for coarsening
   * @param fields Optional fields for transfer
   * @return Result of coarsening
   */
  AdaptivityResult coarsen(
      MeshBase& mesh,
      const std::vector<bool>& marks,
      MeshFields* fields = nullptr);

  // Configuration methods

  /**
   * @brief Set adaptivity options
   */
  void set_options(const AdaptivityOptions& options);

  /**
   * @brief Get current options
   */
  const AdaptivityOptions& get_options() const { return options_; }

  /**
   * @brief Set custom error estimator
   */
  void set_error_estimator(std::unique_ptr<ErrorEstimator> estimator);

  /**
   * @brief Set custom marker
   */
  void set_marker(std::unique_ptr<Marker> marker);

  /**
   * @brief Set custom field transfer
   */
  void set_field_transfer(std::unique_ptr<FieldTransfer> transfer);

  /**
   * @brief Set custom quality checker
   */
  void set_quality_checker(std::unique_ptr<QualityChecker> checker);

  /**
   * @brief Set custom conformity enforcer
   */
  void set_conformity_enforcer(std::unique_ptr<ConformityEnforcer> enforcer);

  // Query methods

  /**
   * @brief Get last error indicators
   */
  const std::vector<double>& get_last_indicators() const {
    return last_indicators_;
  }

  /**
   * @brief Get last marking decisions
   */
  const std::vector<MarkType>& get_last_marks() const {
    return last_marks_;
  }

  /**
   * @brief Check if adaptation is needed based on current error
   */
  bool needs_adaptation(
      const MeshBase& mesh,
      const MeshFields* fields = nullptr) const;

  /**
   * @brief Estimate the result of adaptation without performing it
   */
  AdaptivityResult estimate_adaptation(
      const MeshBase& mesh,
      const MeshFields* fields = nullptr) const;

private:
  // Pipeline stages

  /**
   * @brief Estimate error indicators
   */
  std::vector<double> estimate_error(
      const MeshBase& mesh,
      const MeshFields* fields);

  /**
   * @brief Mark elements for refinement/coarsening
   */
  std::vector<MarkType> mark_elements(
      const std::vector<double>& indicators,
      const MeshBase& mesh);

  /**
   * @brief Perform refinement on marked elements
   */
  std::unique_ptr<MeshBase> perform_refinement(
      const MeshBase& mesh,
      const std::vector<MarkType>& marks,
      AdaptivityResult& result);

  /**
   * @brief Perform coarsening on marked elements
   */
  std::unique_ptr<MeshBase> perform_coarsening(
      const MeshBase& mesh,
      const std::vector<MarkType>& marks,
      AdaptivityResult& result);

  /**
   * @brief Enforce mesh conformity
   */
  void enforce_conformity(
      MeshBase& mesh,
      AdaptivityResult& result);

  /**
   * @brief Transfer fields to adapted mesh
   */
  void transfer_fields(
      const MeshBase& old_mesh,
      MeshBase& new_mesh,
      MeshFields* old_fields,
      MeshFields* new_fields,
      AdaptivityResult& result);

  /**
   * @brief Check mesh quality
   */
  bool check_quality(
      const MeshBase& mesh,
      AdaptivityResult& result);

  /**
   * @brief Finalize adapted mesh
   */
  void finalize_mesh(
      MeshBase& mesh,
      AdaptivityResult& result);

  /**
   * @brief Emit mesh events
   */
  void emit_events(MeshBase& mesh) const;

  /**
   * @brief Write intermediate mesh if requested
   */
  void write_intermediate_mesh(
      const MeshBase& mesh,
      const std::string& stage_name) const;

  // Data members
  AdaptivityOptions options_;
  std::unique_ptr<ErrorEstimator> error_estimator_;
  std::unique_ptr<Marker> marker_;
  std::unique_ptr<FieldTransfer> field_transfer_;
  std::unique_ptr<QualityChecker> quality_checker_;
  std::unique_ptr<ConformityEnforcer> conformity_enforcer_;

  // Cached data from last adaptation
  std::vector<double> last_indicators_;
  std::vector<MarkType> last_marks_;

  // Statistics
  size_t total_adaptations_ = 0;
  size_t total_refinements_ = 0;
  size_t total_coarsenings_ = 0;
};

/**
 * @brief Builder pattern for AdaptivityManager configuration
 */
class AdaptivityManagerBuilder {
public:
  AdaptivityManagerBuilder& with_options(const AdaptivityOptions& options);

  AdaptivityManagerBuilder& with_error_estimator(
      std::unique_ptr<ErrorEstimator> estimator);

  AdaptivityManagerBuilder& with_marker(std::unique_ptr<Marker> marker);

  AdaptivityManagerBuilder& with_field_transfer(
      std::unique_ptr<FieldTransfer> transfer);

  AdaptivityManagerBuilder& with_quality_checker(
      std::unique_ptr<QualityChecker> checker);

  AdaptivityManagerBuilder& with_conformity_enforcer(
      std::unique_ptr<ConformityEnforcer> enforcer);

  std::unique_ptr<AdaptivityManager> build();

private:
  AdaptivityOptions options_;
  std::unique_ptr<ErrorEstimator> error_estimator_;
  std::unique_ptr<Marker> marker_;
  std::unique_ptr<FieldTransfer> field_transfer_;
  std::unique_ptr<QualityChecker> quality_checker_;
  std::unique_ptr<ConformityEnforcer> conformity_enforcer_;
};

/**
 * @brief Utility functions for adaptivity
 */
class AdaptivityUtils {
public:
  /**
   * @brief Perform uniform refinement on entire mesh
   */
  static AdaptivityResult uniform_refinement(
      MeshBase& mesh,
      size_t num_levels = 1,
      MeshFields* fields = nullptr);

  /**
   * @brief Perform uniform coarsening on entire mesh
   */
  static AdaptivityResult uniform_coarsening(
      MeshBase& mesh,
      size_t num_levels = 1,
      MeshFields* fields = nullptr);

  /**
   * @brief Perform local refinement in a region
   */
  static AdaptivityResult local_refinement(
      MeshBase& mesh,
      const std::function<bool(const std::array<double, 3>&)>& region_predicate,
      size_t num_levels = 1,
      MeshFields* fields = nullptr);

  /**
   * @brief Check if mesh has been adapted
   */
  static bool is_adapted(const MeshBase& mesh);

  /**
   * @brief Get refinement level statistics
   */
  struct LevelStats {
    size_t min_level = 0;
    size_t max_level = 0;
    std::vector<size_t> element_count_per_level;
    double avg_level = 0.0;
  };

  static LevelStats get_level_stats(const MeshBase& mesh);

  /**
   * @brief Write adaptivity metrics to field
   */
  static void write_metrics_to_fields(
      MeshBase& mesh,
      const std::vector<double>& error_indicators,
      const std::vector<MarkType>& marks);
};

} // namespace svmp

#endif // SVMP_ADAPTIVITY_MANAGER_H
