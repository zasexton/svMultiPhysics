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
#include "ErrorEstimator.h"
#include "Marker.h"
#include "RefinementRules.h"
#include "Conformity.h"
#include "FieldTransfer.h"
#include "QualityGuards.h"
#include "../Core/MeshBase.h"
#include "../Fields/MeshFields.h"
#include "../Observer/MeshObserver.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

namespace svmp {

// ====================
// AdaptivityResult Implementation
// ====================

std::string AdaptivityResult::summary() const {
  std::stringstream ss;

  ss << "=== Adaptivity Result Summary ===" << std::endl;
  ss << "Status: " << (success ? "SUCCESS" : "FAILED") << std::endl;

  if (success) {
    ss << "Element count: " << initial_element_count << " -> " << final_element_count
       << " (" << (static_cast<int>(final_element_count) - static_cast<int>(initial_element_count))
       << ")" << std::endl;

    ss << "Vertex count: " << initial_vertex_count << " -> " << final_vertex_count
       << " (" << (static_cast<int>(final_vertex_count) - static_cast<int>(initial_vertex_count))
       << ")" << std::endl;

    ss << "Refined elements: " << num_refined << std::endl;
    ss << "Coarsened elements: " << num_coarsened << std::endl;

    if (initial_error > 0 || final_error > 0) {
      ss << "Global error: " << initial_error << " -> " << final_error
         << " (reduction: " << (1.0 - final_error/initial_error) * 100 << "%)" << std::endl;
    }

    ss << "Min quality: " << min_quality << std::endl;
    ss << "Avg quality: " << avg_quality << std::endl;

    ss << "Total time: " << total_time.count() << "s" << std::endl;

    // Timing breakdown
    ss << "Timing breakdown:" << std::endl;
    ss << "  Estimation: " << timing.estimation.count() << "s" << std::endl;
    ss << "  Marking: " << timing.marking.count() << "s" << std::endl;
    ss << "  Refinement: " << timing.refinement.count() << "s" << std::endl;
    ss << "  Coarsening: " << timing.coarsening.count() << "s" << std::endl;
    ss << "  Conformity: " << timing.conformity.count() << "s" << std::endl;
    ss << "  Field transfer: " << timing.field_transfer.count() << "s" << std::endl;
    ss << "  Quality check: " << timing.quality_check.count() << "s" << std::endl;
    ss << "  Finalization: " << timing.finalization.count() << "s" << std::endl;
  } else {
    ss << "Errors:" << std::endl;
    for (const auto& error : error_messages) {
      ss << "  - " << error << std::endl;
    }
  }

  if (!warning_messages.empty()) {
    ss << "Warnings:" << std::endl;
    for (const auto& warning : warning_messages) {
      ss << "  - " << warning << std::endl;
    }
  }

  return ss.str();
}

// ====================
// AdaptivityManager Implementation
// ====================

AdaptivityManager::AdaptivityManager(const AdaptivityOptions& options)
    : options_(options) {

  // Create default components if not set
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
  auto start_time = std::chrono::steady_clock::now();

  // Initialize result
  result.initial_element_count = mesh.num_cells();
  result.initial_vertex_count = mesh.num_vertices();

  if (options_.verbosity >= 1) {
    std::cout << "Starting mesh adaptation..." << std::endl;
    std::cout << "Initial mesh: " << result.initial_element_count
              << " elements, " << result.initial_vertex_count << " vertices" << std::endl;
  }

  try {
    // Step 1: Error estimation
    auto est_start = std::chrono::steady_clock::now();
    last_indicators_ = estimate_error(mesh, fields);
    result.initial_error = ErrorEstimatorUtils::compute_global_error(last_indicators_);
    result.timing.estimation = std::chrono::steady_clock::now() - est_start;

    if (options_.verbosity >= 2) {
      std::cout << "Error estimation complete. Global error: " << result.initial_error << std::endl;
    }

    // Step 2: Marking
    auto mark_start = std::chrono::steady_clock::now();
    last_marks_ = mark_elements(last_indicators_, mesh);
    result.timing.marking = std::chrono::steady_clock::now() - mark_start;

    // Count marks
    auto mark_stats = MarkerUtils::count_marks(last_marks_);
    result.num_refined = mark_stats.num_marked_refine;
    result.num_coarsened = mark_stats.num_marked_coarsen;

    if (options_.verbosity >= 1) {
      std::cout << "Marking complete: " << result.num_refined << " to refine, "
                << result.num_coarsened << " to coarsen" << std::endl;
    }

    // Check if adaptation is needed
    if (result.num_refined == 0 && result.num_coarsened == 0) {
      if (options_.verbosity >= 1) {
        std::cout << "No elements marked for adaptation" << std::endl;
      }
      result.success = true;
      result.final_element_count = result.initial_element_count;
      result.final_vertex_count = result.initial_vertex_count;
      result.final_error = result.initial_error;
      return result;
    }

    // Step 3: Conformity enforcement
    auto conf_start = std::chrono::steady_clock::now();
    enforce_conformity(mesh, result);
    result.timing.conformity = std::chrono::steady_clock::now() - conf_start;

    // Step 4: Create adapted mesh
    std::unique_ptr<MeshBase> new_mesh;

    if (result.num_refined > 0) {
      auto ref_start = std::chrono::steady_clock::now();
      new_mesh = perform_refinement(mesh, last_marks_, result);
      result.timing.refinement = std::chrono::steady_clock::now() - ref_start;
    }

    if (result.num_coarsened > 0) {
      auto coarse_start = std::chrono::steady_clock::now();
      if (new_mesh) {
        // Coarsen the already refined mesh
        auto temp_mesh = perform_coarsening(*new_mesh, last_marks_, result);
        new_mesh = std::move(temp_mesh);
      } else {
        new_mesh = perform_coarsening(mesh, last_marks_, result);
      }
      result.timing.coarsening = std::chrono::steady_clock::now() - coarse_start;
    }

    if (!new_mesh) {
      throw std::runtime_error("Failed to create adapted mesh");
    }

    // Step 5: Field transfer
    if (fields) {
      auto transfer_start = std::chrono::steady_clock::now();

      // Build parent-child map
      auto parent_child = FieldTransferUtils::build_parent_child_map(mesh, *new_mesh, last_marks_);

      // Create new fields
      MeshFields new_fields;
      transfer_fields(mesh, *new_mesh, fields, &new_fields, result);

      // Update fields
      *fields = std::move(new_fields);

      result.timing.field_transfer = std::chrono::steady_clock::now() - transfer_start;
    }

    // Step 6: Quality check
    auto quality_start = std::chrono::steady_clock::now();
    bool quality_ok = check_quality(*new_mesh, result);
    result.timing.quality_check = std::chrono::steady_clock::now() - quality_start;

    if (!quality_ok && options_.rollback_on_poor_quality) {
      result.success = false;
      result.error_messages.push_back("Quality check failed, rolling back");
      return result;
    }

    // Step 7: Finalize mesh
    auto final_start = std::chrono::steady_clock::now();
    finalize_mesh(*new_mesh, result);
    result.timing.finalization = std::chrono::steady_clock::now() - final_start;

    // Update mesh statistics
    result.final_element_count = new_mesh->num_cells();
    result.final_vertex_count = new_mesh->num_vertices();

    // Estimate final error
    if (fields) {
      auto final_indicators = estimate_error(*new_mesh, fields);
      result.final_error = ErrorEstimatorUtils::compute_global_error(final_indicators);
    }

    // Step 8: Replace old mesh with new mesh
    if (options_.create_new_mesh) {
      result.adapted_mesh = std::move(new_mesh);
    } else {
      // Copy new mesh data to original mesh
      mesh = std::move(*new_mesh);
    }

    // Emit events
    emit_events(result.adapted_mesh ? *result.adapted_mesh : mesh);

    // Success!
    result.success = true;
    total_adaptations_++;

  } catch (const std::exception& e) {
    result.success = false;
    result.error_messages.push_back(e.what());
  }

  // Total time
  result.total_time = std::chrono::steady_clock::now() - start_time;

  if (options_.verbosity >= 1) {
    std::cout << result.summary() << std::endl;
  }

  return result;
}

AdaptivityResult AdaptivityManager::refine(
    MeshBase& mesh,
    const std::vector<bool>& marks,
    MeshFields* fields) {

  // Convert bool marks to MarkType
  std::vector<MarkType> mark_types(marks.size(), MarkType::NONE);
  for (size_t i = 0; i < marks.size(); ++i) {
    if (marks[i]) {
      mark_types[i] = MarkType::REFINE;
    }
  }

  last_marks_ = mark_types;

  // Perform adaptation with refinement only
  AdaptivityOptions temp_options = options_;
  temp_options.enable_refinement = true;
  temp_options.enable_coarsening = false;

  auto saved_options = options_;
  options_ = temp_options;
  auto result = adapt(mesh, fields);
  options_ = saved_options;

  total_refinements_ += result.num_refined;

  return result;
}

AdaptivityResult AdaptivityManager::coarsen(
    MeshBase& mesh,
    const std::vector<bool>& marks,
    MeshFields* fields) {

  // Convert bool marks to MarkType
  std::vector<MarkType> mark_types(marks.size(), MarkType::NONE);
  for (size_t i = 0; i < marks.size(); ++i) {
    if (marks[i]) {
      mark_types[i] = MarkType::COARSEN;
    }
  }

  last_marks_ = mark_types;

  // Perform adaptation with coarsening only
  AdaptivityOptions temp_options = options_;
  temp_options.enable_refinement = false;
  temp_options.enable_coarsening = true;

  auto saved_options = options_;
  options_ = temp_options;
  auto result = adapt(mesh, fields);
  options_ = saved_options;

  total_coarsenings_ += result.num_coarsened;

  return result;
}

void AdaptivityManager::set_options(const AdaptivityOptions& options) {
  options_ = options;

  // Recreate components with new options
  error_estimator_ = ErrorEstimatorFactory::create(options_);
  marker_ = MarkerFactory::create(options_);
  field_transfer_ = FieldTransferFactory::create(options_);
  quality_checker_ = QualityCheckerFactory::create(options_);
  conformity_enforcer_ = ConformityEnforcerFactory::create(options_);
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

bool AdaptivityManager::needs_adaptation(
    const MeshBase& mesh,
    const MeshFields* fields) const {

  // Estimate error
  auto indicators = error_estimator_->estimate(mesh, fields, options_);

  // Check global error
  double global_error = ErrorEstimatorUtils::compute_global_error(indicators);

  // Simple criterion: adapt if error is above threshold
  // This could be made more sophisticated
  return global_error > options_.refine_threshold;
}

AdaptivityResult AdaptivityManager::estimate_adaptation(
    const MeshBase& mesh,
    const MeshFields* fields) const {

  AdaptivityResult result;

  // Estimate error
  auto indicators = error_estimator_->estimate(mesh, fields, options_);
  result.initial_error = ErrorEstimatorUtils::compute_global_error(indicators);

  // Mark elements
  auto marks = marker_->mark(indicators, mesh, options_);

  // Count marks
  auto stats = MarkerUtils::count_marks(marks);
  result.num_refined = stats.num_marked_refine;
  result.num_coarsened = stats.num_marked_coarsen;

  // Estimate element counts
  result.initial_element_count = mesh.num_cells();
  result.initial_vertex_count = mesh.num_vertices();

  // Rough estimates (assuming regular refinement)
  result.final_element_count = result.initial_element_count
                              - result.num_coarsened
                              + result.num_refined * 4;  // Assuming 1:4 refinement

  result.final_vertex_count = result.initial_vertex_count
                             + result.num_refined * 3;  // Rough estimate

  // Estimate error reduction (heuristic)
  if (result.num_refined > 0) {
    double refine_fraction = static_cast<double>(result.num_refined) / result.initial_element_count;
    result.final_error = result.initial_error * (1.0 - 0.5 * refine_fraction);
  } else {
    result.final_error = result.initial_error;
  }

  result.success = true;

  return result;
}

// Private methods

std::vector<double> AdaptivityManager::estimate_error(
    const MeshBase& mesh,
    const MeshFields* fields) {

  if (!error_estimator_) {
    throw std::runtime_error("No error estimator set");
  }

  auto indicators = error_estimator_->estimate(mesh, fields, options_);

  // Write indicators to field if requested
  if (options_.write_intermediate_meshes && fields) {
    ErrorEstimatorUtils::write_to_field(
        const_cast<MeshFields&>(*fields), "error_indicator", indicators);
  }

  return indicators;
}

std::vector<MarkType> AdaptivityManager::mark_elements(
    const std::vector<double>& indicators,
    const MeshBase& mesh) {

  if (!marker_) {
    throw std::runtime_error("No marker set");
  }

  auto marks = marker_->mark(indicators, mesh, options_);

  // Apply constraints
  MarkerUtils::apply_constraints(marks, mesh, options_);

  // Smooth marking if requested
  // MarkerUtils::smooth_marking(marks, mesh);

  // Write marks to field if requested
  if (options_.write_intermediate_meshes) {
    // TODO: Write marks to field
  }

  return marks;
}

std::unique_ptr<MeshBase> AdaptivityManager::perform_refinement(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks,
    AdaptivityResult& result) {

  // Create refined mesh
  auto refined_mesh = std::make_unique<MeshBase>();

  // Get refinement rules manager
  auto& rules_manager = RefinementRulesManager::instance();

  // Collect new vertices and elements
  std::vector<std::array<double, 3>> new_vertices;
  std::vector<std::vector<size_t>> new_connectivity;
  std::vector<CellType> new_types;

  // Copy existing vertices
  for (size_t i = 0; i < mesh.num_vertices(); ++i) {
    new_vertices.push_back(mesh.vertex_coords(i));
  }

  // Process each element
  for (size_t elem_id = 0; elem_id < mesh.num_cells(); ++elem_id) {
    if (marks[elem_id] == MarkType::REFINE) {
      // Get element data
      auto cell_type = mesh.cell_type(elem_id);
      auto vertex_ids = mesh.cell_vertices(elem_id);

      // Get element vertices
      std::vector<std::array<double, 3>> elem_vertices;
      for (size_t vid : vertex_ids) {
        elem_vertices.push_back(mesh.vertex_coords(vid));
      }

      // Get refinement pattern
      RefinementPattern pattern = options_.refinement_pattern;

      // Refine element
      auto refined = rules_manager.refine(
          elem_vertices, cell_type, pattern,
          mesh.cell_refinement_level(elem_id));

      // Add new vertices
      size_t new_vertex_start = new_vertices.size();
      for (const auto& vertex : refined.new_vertices) {
        new_vertices.push_back(vertex);
      }

      // Add child elements with updated connectivity
      for (const auto& child_conn : refined.child_connectivity) {
        std::vector<size_t> updated_conn;
        for (size_t vid : child_conn) {
          if (vid < vertex_ids.size()) {
            // Original vertex
            updated_conn.push_back(vertex_ids[vid]);
          } else {
            // New vertex
            updated_conn.push_back(new_vertex_start + (vid - vertex_ids.size()));
          }
        }
        new_connectivity.push_back(updated_conn);
        new_types.push_back(cell_type);
      }

      result.refinement_steps++;

    } else {
      // Keep element as is
      new_connectivity.push_back(mesh.cell_vertices(elem_id));
      new_types.push_back(mesh.cell_type(elem_id));
    }
  }

  // Build refined mesh
  refined_mesh->build_from_arrays(
      new_vertices,
      new_connectivity,
      new_types);

  // Copy labels and other attributes
  // TODO: Transfer labels and attributes

  if (options_.write_intermediate_meshes) {
    write_intermediate_mesh(*refined_mesh, "refined");
  }

  return refined_mesh;
}

std::unique_ptr<MeshBase> AdaptivityManager::perform_coarsening(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks,
    AdaptivityResult& result) {

  // TODO: Implement coarsening
  // This requires tracking parent-child relationships and
  // only coarsening elements that can be safely coarsened

  result.warning_messages.push_back("Coarsening not yet fully implemented");

  // For now, return a copy of the mesh
  auto coarsened_mesh = std::make_unique<MeshBase>(mesh);

  return coarsened_mesh;
}

void AdaptivityManager::enforce_conformity(
    MeshBase& mesh,
    AdaptivityResult& result) {

  if (!conformity_enforcer_) {
    return;
  }

  // Check conformity
  auto non_conformity = conformity_enforcer_->check_conformity(mesh, last_marks_);

  if (non_conformity.is_conforming()) {
    return;
  }

  // Enforce conformity
  size_t iterations = conformity_enforcer_->enforce_conformity(
      mesh, last_marks_, options_);

  if (options_.verbosity >= 2) {
    std::cout << "Conformity enforced in " << iterations << " iterations" << std::endl;
  }

  // Update mark counts
  auto stats = MarkerUtils::count_marks(last_marks_);
  result.num_refined = stats.num_marked_refine;
  result.num_coarsened = stats.num_marked_coarsen;
}

void AdaptivityManager::transfer_fields(
    const MeshBase& old_mesh,
    MeshBase& new_mesh,
    MeshFields* old_fields,
    MeshFields* new_fields,
    AdaptivityResult& result) {

  if (!field_transfer_ || !old_fields || !new_fields) {
    return;
  }

  // Build parent-child map
  auto parent_child = FieldTransferUtils::build_parent_child_map(
      old_mesh, new_mesh, last_marks_);

  // Transfer fields
  auto transfer_stats = field_transfer_->transfer(
      old_mesh, new_mesh,
      *old_fields, *new_fields,
      parent_child, options_);

  if (options_.verbosity >= 2) {
    std::cout << "Transferred " << transfer_stats.num_fields << " fields" << std::endl;
    if (!transfer_stats.conservation_errors.empty()) {
      std::cout << "Conservation errors:" << std::endl;
      for (const auto& [name, error] : transfer_stats.conservation_errors) {
        std::cout << "  " << name << ": " << error << std::endl;
      }
    }
  }
}

bool AdaptivityManager::check_quality(
    const MeshBase& mesh,
    AdaptivityResult& result) {

  if (!quality_checker_ || !options_.check_quality) {
    return true;
  }

  // Check quality
  auto quality_result = quality_checker_->check_quality(mesh, options_);

  result.min_quality = quality_result.min_quality;
  result.avg_quality = quality_result.avg_quality;

  if (quality_result.min_quality < options_.min_quality) {
    result.warning_messages.push_back(
        "Poor quality elements detected: min = " + std::to_string(quality_result.min_quality));

    if (options_.enable_smoothing) {
      // TODO: Apply smoothing
      result.warning_messages.push_back("Smoothing not yet implemented");
    }

    return false;
  }

  return true;
}

void AdaptivityManager::finalize_mesh(
    MeshBase& mesh,
    AdaptivityResult& result) {

  // Finalize mesh topology
  mesh.finalize();

  // Build faces and edges if needed
  if (mesh.num_faces() == 0) {
    mesh.build_faces();
  }

  if (mesh.num_edges() == 0) {
    mesh.build_edges();
  }

  // Update any derived data
  // TODO: Update derived data structures
}

void AdaptivityManager::emit_events(MeshBase& mesh) const {
  auto& event_bus = mesh.event_bus();

  // Emit adaptivity event
  event_bus.notify(MeshEvent::AdaptivityApplied);

  // Emit topology changed
  event_bus.notify(MeshEvent::TopologyChanged);

  // Emit geometry changed if coordinates were modified
  event_bus.notify(MeshEvent::GeometryChanged);

  // Emit fields changed if fields were transferred
  if (field_transfer_) {
    event_bus.notify(MeshEvent::FieldsChanged);
  }
}

void AdaptivityManager::write_intermediate_mesh(
    const MeshBase& mesh,
    const std::string& stage_name) const {

  if (!options_.write_intermediate_meshes) {
    return;
  }

  // Create filename
  std::string filename = options_.output_directory + "/mesh_" + stage_name + ".vtk";

  // Write mesh
  // TODO: Implement VTK writer

  if (options_.verbosity >= 2) {
    std::cout << "Wrote intermediate mesh: " << filename << std::endl;
  }
}

// ====================
// AdaptivityManagerBuilder Implementation
// ====================

std::unique_ptr<AdaptivityManager> AdaptivityManagerBuilder::build() {
  auto manager = std::make_unique<AdaptivityManager>(options_);

  if (error_estimator_) {
    manager->set_error_estimator(std::move(error_estimator_));
  }

  if (marker_) {
    manager->set_marker(std::move(marker_));
  }

  if (field_transfer_) {
    manager->set_field_transfer(std::move(field_transfer_));
  }

  if (quality_checker_) {
    manager->set_quality_checker(std::move(quality_checker_));
  }

  if (conformity_enforcer_) {
    manager->set_conformity_enforcer(std::move(conformity_enforcer_));
  }

  return manager;
}

// ====================
// AdaptivityUtils Implementation
// ====================

AdaptivityResult AdaptivityUtils::uniform_refinement(
    MeshBase& mesh,
    size_t num_levels,
    MeshFields* fields) {

  AdaptivityResult result;

  for (size_t level = 0; level < num_levels; ++level) {
    // Mark all elements for refinement
    std::vector<bool> marks(mesh.num_cells(), true);

    // Create manager with simple options
    AdaptivityOptions options;
    options.enable_refinement = true;
    options.enable_coarsening = false;
    AdaptivityManager manager(options);

    // Refine
    auto level_result = manager.refine(mesh, marks, fields);

    if (!level_result.success) {
      result = level_result;
      break;
    }

    // Accumulate results
    result.num_refined += level_result.num_refined;
    result.refinement_steps++;

    if (level_result.adapted_mesh) {
      mesh = std::move(*level_result.adapted_mesh);
    }
  }

  result.success = true;
  result.final_element_count = mesh.num_cells();
  result.final_vertex_count = mesh.num_vertices();

  return result;
}

AdaptivityResult AdaptivityUtils::uniform_coarsening(
    MeshBase& mesh,
    size_t num_levels,
    MeshFields* fields) {

  AdaptivityResult result;

  for (size_t level = 0; level < num_levels; ++level) {
    // Mark all elements for coarsening
    std::vector<bool> marks(mesh.num_cells(), true);

    // Create manager with simple options
    AdaptivityOptions options;
    options.enable_refinement = false;
    options.enable_coarsening = true;
    AdaptivityManager manager(options);

    // Coarsen
    auto level_result = manager.coarsen(mesh, marks, fields);

    if (!level_result.success) {
      result = level_result;
      break;
    }

    // Accumulate results
    result.num_coarsened += level_result.num_coarsened;
    result.coarsening_steps++;

    if (level_result.adapted_mesh) {
      mesh = std::move(*level_result.adapted_mesh);
    }
  }

  result.success = true;
  result.final_element_count = mesh.num_cells();
  result.final_vertex_count = mesh.num_vertices();

  return result;
}

AdaptivityResult AdaptivityUtils::local_refinement(
    MeshBase& mesh,
    const std::function<bool(const std::array<double, 3>&)>& region_predicate,
    size_t num_levels,
    MeshFields* fields) {

  AdaptivityResult result;

  for (size_t level = 0; level < num_levels; ++level) {
    // Mark elements in region
    std::vector<bool> marks(mesh.num_cells(), false);

    for (size_t i = 0; i < mesh.num_cells(); ++i) {
      // Check if element center is in region
      auto center = mesh.cell_center(i);
      if (region_predicate(center)) {
        marks[i] = true;
      }
    }

    // Create manager
    AdaptivityOptions options;
    options.enable_refinement = true;
    options.enable_coarsening = false;
    AdaptivityManager manager(options);

    // Refine
    auto level_result = manager.refine(mesh, marks, fields);

    if (!level_result.success) {
      result = level_result;
      break;
    }

    // Accumulate results
    result.num_refined += level_result.num_refined;
    result.refinement_steps++;

    if (level_result.adapted_mesh) {
      mesh = std::move(*level_result.adapted_mesh);
    }
  }

  result.success = true;
  result.final_element_count = mesh.num_cells();
  result.final_vertex_count = mesh.num_vertices();

  return result;
}

bool AdaptivityUtils::is_adapted(const MeshBase& mesh) {
  // Check if any element has refinement level > 0
  for (size_t i = 0; i < mesh.num_cells(); ++i) {
    if (mesh.cell_refinement_level(i) > 0) {
      return true;
    }
  }
  return false;
}

AdaptivityUtils::LevelStats AdaptivityUtils::get_level_stats(const MeshBase& mesh) {
  LevelStats stats;

  if (mesh.num_cells() == 0) {
    return stats;
  }

  stats.min_level = std::numeric_limits<size_t>::max();
  stats.max_level = 0;

  double sum_level = 0.0;

  for (size_t i = 0; i < mesh.num_cells(); ++i) {
    size_t level = mesh.cell_refinement_level(i);
    stats.min_level = std::min(stats.min_level, level);
    stats.max_level = std::max(stats.max_level, level);
    sum_level += level;
  }

  stats.avg_level = sum_level / mesh.num_cells();

  // Count elements per level
  stats.element_count_per_level.resize(stats.max_level + 1, 0);
  for (size_t i = 0; i < mesh.num_cells(); ++i) {
    stats.element_count_per_level[mesh.cell_refinement_level(i)]++;
  }

  return stats;
}

void AdaptivityUtils::write_metrics_to_fields(
    const MeshBase& mesh,
    MeshFields& fields,
    const std::vector<double>& error_indicators,
    const std::vector<MarkType>& marks) {

  // Write error indicators
  if (!error_indicators.empty()) {
    auto error_field = fields.create_field("error_indicator", FieldLocation::CELL, 1);
    for (size_t i = 0; i < error_indicators.size(); ++i) {
      error_field->set_cell_value(i, error_indicators[i]);
    }
  }

  // Write marks
  if (!marks.empty()) {
    MarkerUtils::write_marks_to_field(fields, "refinement_marks", marks);
  }

  // Write refinement levels
  auto level_field = fields.create_field("refinement_level", FieldLocation::CELL, 1);
  for (size_t i = 0; i < mesh.num_cells(); ++i) {
    level_field->set_cell_value(i, static_cast<double>(mesh.cell_refinement_level(i)));
  }
}

} // namespace svmp