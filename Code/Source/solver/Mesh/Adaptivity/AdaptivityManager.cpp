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

#include "AdaptivityManager.h"
#include "Conformity.h"
#include "FieldTransfer.h"
#include "QualityGuards.h"
#include "../Core/MeshBase.h"
#include <algorithm>
#include <sstream>

namespace svmp {

std::string AdaptivityResult::summary() const {
  std::ostringstream ss;
  ss << "AdaptivityResult{success=" << (success ? "true" : "false")
     << ", elements=" << initial_element_count << "->" << final_element_count
     << ", vertices=" << initial_vertex_count << "->" << final_vertex_count
     << ", refined=" << num_refined << ", coarsened=" << num_coarsened
     << ", min_quality=" << min_quality << ", avg_quality=" << avg_quality
     << "}";
  return ss.str();
}

AdaptivityManager::AdaptivityManager(const AdaptivityOptions& options)
    : options_(options) {
  if (!error_estimator_) {
    error_estimator_ = ErrorEstimatorFactory::create(options_);
  }
  if (!marker_) {
    marker_ = MarkerFactory::create(options_);
  }
  if (!field_transfer_) {
    field_transfer_ = FieldTransferFactory::create(options_);
  }
  if (!quality_checker_) {
    quality_checker_ = QualityCheckerFactory::create(options_);
  }
  if (!conformity_enforcer_) {
    conformity_enforcer_ = ConformityEnforcerFactory::create(options_);
  }
}

AdaptivityManager::~AdaptivityManager() = default;

AdaptivityResult AdaptivityManager::adapt(MeshBase& mesh, MeshFields* fields) {
  AdaptivityResult result;
  result.initial_element_count = mesh.n_cells();
  result.initial_vertex_count = mesh.n_vertices();

  // Compute error indicators + marks (but do not modify the mesh yet).
  last_indicators_ = estimate_error(mesh, fields);
  last_marks_ = mark_elements(last_indicators_, mesh);

  // Update counts/statistics.
  auto stats = MarkerUtils::count_marks(last_marks_);
  result.num_refined = stats.num_marked_refine;
  result.num_coarsened = stats.num_marked_coarsen;

  result.initial_error = ErrorEstimatorUtils::compute_global_error(last_indicators_, options_.error_norm_power);
  result.final_error = result.initial_error;

  if (options_.check_quality && quality_checker_) {
    const auto q = quality_checker_->compute_mesh_quality(mesh, options_);
    result.min_quality = q.min_quality;
    result.avg_quality = q.avg_quality;
  }

  result.final_element_count = mesh.n_cells();
  result.final_vertex_count = mesh.n_vertices();
  result.success = true;

  // The full refinement/coarsening pipeline is not yet implemented.
  result.warning_messages.push_back("Adaptivity pipeline is not yet implemented (analysis-only run).");
  return result;
}

AdaptivityResult AdaptivityManager::refine(MeshBase& mesh,
                                          const std::vector<bool>& marks,
                                          MeshFields* fields) {
  (void)mesh;
  (void)marks;
  (void)fields;
  AdaptivityResult result;
  result.success = false;
  result.warning_messages.push_back("Refine() is not yet implemented.");
  return result;
}

AdaptivityResult AdaptivityManager::coarsen(MeshBase& mesh,
                                           const std::vector<bool>& marks,
                                           MeshFields* fields) {
  (void)mesh;
  (void)marks;
  (void)fields;
  AdaptivityResult result;
  result.success = false;
  result.warning_messages.push_back("Coarsen() is not yet implemented.");
  return result;
}

void AdaptivityManager::set_options(const AdaptivityOptions& options) {
  options_ = options;
}

void AdaptivityManager::set_error_estimator(std::unique_ptr<ErrorEstimator> estimator) {
  error_estimator_ = std::move(estimator);
}

void AdaptivityManager::set_marker(std::unique_ptr<Marker> marker) {
  marker_ = std::move(marker);
}

void AdaptivityManager::set_field_transfer(std::unique_ptr<FieldTransfer> transfer) {
  field_transfer_ = std::move(transfer);
}

void AdaptivityManager::set_quality_checker(std::unique_ptr<QualityChecker> checker) {
  quality_checker_ = std::move(checker);
}

void AdaptivityManager::set_conformity_enforcer(std::unique_ptr<ConformityEnforcer> enforcer) {
  conformity_enforcer_ = std::move(enforcer);
}

bool AdaptivityManager::needs_adaptation(const MeshBase& mesh, const MeshFields* fields) const {
  (void)fields;
  if (!error_estimator_) {
    return false;
  }
  const auto indicators = error_estimator_->estimate(mesh, fields, options_);
  if (indicators.empty()) {
    return false;
  }
  const double max_val = *std::max_element(indicators.begin(), indicators.end());
  if (options_.enable_refinement && max_val >= options_.refine_threshold) {
    return true;
  }
  return false;
}

AdaptivityResult AdaptivityManager::estimate_adaptation(const MeshBase& mesh, const MeshFields* fields) const {
  AdaptivityResult result;
  result.initial_element_count = mesh.n_cells();
  result.initial_vertex_count = mesh.n_vertices();
  if (error_estimator_) {
    const auto indicators = error_estimator_->estimate(mesh, fields, options_);
    result.initial_error = ErrorEstimatorUtils::compute_global_error(indicators, options_.error_norm_power);
    result.final_error = result.initial_error;
  }
  result.final_element_count = result.initial_element_count;
  result.final_vertex_count = result.initial_vertex_count;
  result.success = true;
  result.warning_messages.push_back("Estimate-only: refinement/coarsening not yet implemented.");
  return result;
}

std::vector<double> AdaptivityManager::estimate_error(const MeshBase& mesh, const MeshFields* fields) {
  if (!error_estimator_) {
    return std::vector<double>(mesh.n_cells(), 0.0);
  }
  return error_estimator_->estimate(mesh, fields, options_);
}

std::vector<MarkType> AdaptivityManager::mark_elements(const std::vector<double>& indicators,
                                                       const MeshBase& mesh) {
  if (!marker_) {
    return std::vector<MarkType>(indicators.size(), MarkType::NONE);
  }
  auto marks = marker_->mark(indicators, mesh, options_);
  MarkerUtils::apply_constraints(marks, mesh, options_);
  return marks;
}

std::unique_ptr<MeshBase> AdaptivityManager::perform_refinement(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks,
    AdaptivityResult& result) {
  (void)mesh;
  (void)marks;
  (void)result;
  return nullptr;
}

std::unique_ptr<MeshBase> AdaptivityManager::perform_coarsening(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks,
    AdaptivityResult& result) {
  (void)mesh;
  (void)marks;
  (void)result;
  return nullptr;
}

void AdaptivityManager::enforce_conformity(MeshBase& mesh,
                                          AdaptivityResult& result) {
  (void)mesh;
  (void)result;
}

void AdaptivityManager::transfer_fields(const MeshBase& old_mesh,
                                       MeshBase& new_mesh,
                                       MeshFields* old_fields,
                                       MeshFields* new_fields,
                                       AdaptivityResult& result) {
  (void)old_mesh;
  (void)new_mesh;
  (void)old_fields;
  (void)new_fields;
  (void)result;
}

bool AdaptivityManager::check_quality(const MeshBase& mesh, AdaptivityResult& result) {
  (void)mesh;
  (void)result;
  return true;
}

void AdaptivityManager::finalize_mesh(MeshBase& mesh, AdaptivityResult& result) {
  (void)mesh;
  (void)result;
}

void AdaptivityManager::emit_events(MeshBase& mesh) const {
  (void)mesh;
}

void AdaptivityManager::write_intermediate_mesh(const MeshBase& mesh, const std::string& stage_name) const {
  (void)mesh;
  (void)stage_name;
}

AdaptivityManagerBuilder& AdaptivityManagerBuilder::with_options(const AdaptivityOptions& options) {
  options_ = options;
  return *this;
}

AdaptivityManagerBuilder& AdaptivityManagerBuilder::with_error_estimator(std::unique_ptr<ErrorEstimator> estimator) {
  error_estimator_ = std::move(estimator);
  return *this;
}

AdaptivityManagerBuilder& AdaptivityManagerBuilder::with_marker(std::unique_ptr<Marker> marker) {
  marker_ = std::move(marker);
  return *this;
}

AdaptivityManagerBuilder& AdaptivityManagerBuilder::with_field_transfer(std::unique_ptr<FieldTransfer> transfer) {
  field_transfer_ = std::move(transfer);
  return *this;
}

AdaptivityManagerBuilder& AdaptivityManagerBuilder::with_quality_checker(std::unique_ptr<QualityChecker> checker) {
  quality_checker_ = std::move(checker);
  return *this;
}

AdaptivityManagerBuilder& AdaptivityManagerBuilder::with_conformity_enforcer(std::unique_ptr<ConformityEnforcer> enforcer) {
  conformity_enforcer_ = std::move(enforcer);
  return *this;
}

std::unique_ptr<AdaptivityManager> AdaptivityManagerBuilder::build() {
  auto manager = std::make_unique<AdaptivityManager>(options_);
  if (error_estimator_) manager->set_error_estimator(std::move(error_estimator_));
  if (marker_) manager->set_marker(std::move(marker_));
  if (field_transfer_) manager->set_field_transfer(std::move(field_transfer_));
  if (quality_checker_) manager->set_quality_checker(std::move(quality_checker_));
  if (conformity_enforcer_) manager->set_conformity_enforcer(std::move(conformity_enforcer_));
  return manager;
}

AdaptivityResult AdaptivityUtils::uniform_refinement(MeshBase& mesh, size_t num_levels, MeshFields* fields) {
  (void)num_levels;
  AdaptivityOptions opts;
  opts.enable_refinement = true;
  opts.enable_coarsening = false;
  AdaptivityManager manager(opts);
  return manager.adapt(mesh, fields);
}

AdaptivityResult AdaptivityUtils::uniform_coarsening(MeshBase& mesh, size_t num_levels, MeshFields* fields) {
  (void)num_levels;
  (void)fields;
  AdaptivityResult result;
  result.initial_element_count = mesh.n_cells();
  result.initial_vertex_count = mesh.n_vertices();
  result.final_element_count = result.initial_element_count;
  result.final_vertex_count = result.initial_vertex_count;
  result.success = false;
  result.warning_messages.push_back("Uniform coarsening is not yet implemented.");
  return result;
}

AdaptivityResult AdaptivityUtils::local_refinement(
    MeshBase& mesh,
    const std::function<bool(const std::array<double, 3>&)>& region_predicate,
    size_t num_levels,
    MeshFields* fields) {
  (void)mesh;
  (void)region_predicate;
  (void)num_levels;
  (void)fields;
  AdaptivityResult result;
  result.success = false;
  result.warning_messages.push_back("Local refinement is not yet implemented.");
  return result;
}

bool AdaptivityUtils::is_adapted(const MeshBase& mesh) {
  (void)mesh;
  return false;
}

AdaptivityUtils::LevelStats AdaptivityUtils::get_level_stats(const MeshBase& mesh) {
  (void)mesh;
  LevelStats stats;
  stats.min_level = 0;
  stats.max_level = 0;
  stats.avg_level = 0.0;
  stats.element_count_per_level = {mesh.n_cells()};
  return stats;
}

void AdaptivityUtils::write_metrics_to_fields(
    MeshBase& mesh,
    const std::vector<double>& error_indicators,
    const std::vector<MarkType>& marks) {
  ErrorEstimatorUtils::write_to_field(mesh, "error_indicator", error_indicators);
  MarkerUtils::write_marks_to_field(mesh, "refinement_marks", marks);
}

} // namespace svmp
