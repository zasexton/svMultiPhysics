/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/LevelSetPhaseGeometryReconciliation.h"

#include "LevelSet/LevelSetConservativePhaseState.h"
#include "LevelSet/LevelSetConservativePhaseWorkflow.h"
#include "LevelSet/LevelSetGeometry.h"
#include "LevelSet/LevelSetVolume.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace level_set {

// Moved from Application/Core/ApplicationDriver.cpp (lines 24052-24075 at 7d9ddc8a)
// with bodies, messages and collective order unchanged.

ConservativePhaseMomentMismatch conservativePhaseMomentMismatch(
    std::span<const svmp::FE::Real> current,
    std::span<const svmp::FE::Real> target)
{
  if (current.size() != target.size()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Conservative phase moment vectors have incompatible layouts.");
  }
  ConservativePhaseMomentMismatch mismatch;
  svmp::FE::Real squared_norm{0.0};
  for (std::size_t node = 0u; node < current.size(); ++node) {
    const auto residual = target[node] - current[node];
    if (!std::isfinite(residual)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase geometry reconciliation found a non-finite local moment residual.");
    }
    mismatch.maximum_nodal_residual = std::max(
        mismatch.maximum_nodal_residual, std::abs(residual));
    squared_norm += residual * residual;
    mismatch.total_residual += residual;
  }
  mismatch.residual_norm = std::sqrt(squared_norm);
  return mismatch;
}

// Moved from Application/Core/ApplicationDriver.cpp (lines 25605-25912 at 7d9ddc8a)
// with bodies, messages and collective order unchanged.

ConservativePhaseGeometryReconciliationResult
reconcileConservativePhaseGeometry(
    const ConservativePhaseWorkflowHost& host,
    ConservativePhaseRequest& request,
    std::span<const svmp::FE::Real> target_liquid_phase_mass,
    std::vector<svmp::FE::Real>& candidate,
    ConservativePhaseGeometryTransaction& transaction,
    std::span<const std::uint8_t> contact_protected_nodes)
{
  ConservativePhaseGeometryReconciliationResult result;
  if (!host.fe_system) {
    result.diagnostic =
        "Conservative phase geometry reconciliation requires an FE system";
    return result;
  }
  auto& system = *host.fe_system;
  auto& graph = requireCurrentConservativePhaseGraph(
      system, request, /*allow_graph_rebuild=*/false);
  if (target_liquid_phase_mass.size() != graph.nodes) {
    result.diagnostic =
        "Conservative phase geometry reconciliation target does not match the phase graph";
    return result;
  }
  if (!contact_protected_nodes.empty() &&
      contact_protected_nodes.size() != graph.nodes) {
    result.diagnostic =
        "Conservative phase geometry reconciliation contact mask does not match the phase graph";
    return result;
  }
  result.contact_protected_nodes = static_cast<std::size_t>(std::count_if(
      contact_protected_nodes.begin(),
      contact_protected_nodes.end(),
      [](std::uint8_t value) { return value != 0u; }));
  const auto phase_field = system.findFieldByName(
      request.configuration->transport.conservative_phase.liquid_indicator.field_name);
  const auto level_set_field = system.findFieldByName(
      request.configuration->transport.level_set.field_name);
  if (phase_field == svmp::FE::INVALID_FIELD_ID ||
      level_set_field == svmp::FE::INVALID_FIELD_ID) {
    result.diagnostic =
        "Conservative phase geometry reconciliation could not resolve its phase or level-set field";
    return result;
  }
  const auto raw_level_set_offset =
      system.fieldDofOffset(level_set_field);
  const auto raw_level_set_count =
      system.fieldDofHandler(level_set_field).getNumDofs();
  if (raw_level_set_offset < 0 || raw_level_set_count < 0) {
    result.diagnostic =
        "Conservative phase geometry reconciliation found an invalid level-set layout";
    return result;
  }
  const auto level_set_offset =
      static_cast<std::size_t>(raw_level_set_offset);
  const auto level_set_count =
      static_cast<std::size_t>(raw_level_set_count);
  if (level_set_count != graph.nodes ||
      level_set_offset > candidate.size() ||
      level_set_count > candidate.size() - level_set_offset) {
    result.diagnostic =
        "Conservative phase geometry reconciliation requires identical P1 phase and level-set layouts";
    return result;
  }

  const auto projection_options = conservativePhaseProjectionOptions(
      system, request);
  const auto volume_options = volumeOptionsForCutMaintenance(
      request.configuration->volume_cut_request,
      request.configuration->transport.level_set.field_name,
      request.configuration->isovalue);
  const auto moment_tolerance =
      request.configuration->transport.conservative_phase.geometry_measure_tolerance *
      std::max({svmp::FE::Real{1.0}, graph.physical_measure,
                std::accumulate(target_liquid_phase_mass.begin(),
                                target_liquid_phase_mass.end(),
                                svmp::FE::Real{0.0})});
  if (!std::isfinite(moment_tolerance) ||
      !(moment_tolerance > svmp::FE::Real{0.0})) {
    result.diagnostic =
        "Conservative phase geometry reconciliation has an invalid local moment tolerance";
    return result;
  }

  ++result.geometry_refresh_requests;
  result.geometry_rebuilds +=
      transaction.refresh(candidate).refreshed ? 1 : 0;
  for (int iteration = 0;
       iteration <
       request.configuration->transport.conservative_phase.geometry_correction_max_iterations;
       ++iteration) {
    const auto projection = projectCurrentConservativePhaseGeometry(
        system, request, /*allow_graph_rebuild=*/false);
    if (!projection.success) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase local geometry projection failed: " +
          projection.diagnostic);
    }
    const auto mismatch = conservativePhaseMomentMismatch(
        projection.liquid_phase_mass, target_liquid_phase_mass);
    if (iteration == 0) {
      result.initial_residual_norm = mismatch.residual_norm;
    }
    result.final_residual_norm = mismatch.residual_norm;
    result.maximum_final_nodal_residual =
        mismatch.maximum_nodal_residual;
    result.final_total_residual = mismatch.total_residual;
    if (mismatch.maximum_nodal_residual <= moment_tolerance &&
        std::abs(mismatch.total_residual) <= moment_tolerance) {
      result.success = true;
      result.target_reached = true;
      result.diagnostic = "ok";
      return result;
    }

    const auto sensitivity =
        svmp::FE::level_set::buildLevelSetP1PhaseGeometrySensitivity(
            system,
            level_set_field,
            phase_field,
            graph,
            projection_options,
            candidate);
    if (!sensitivity.success) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Conservative phase local geometry sensitivity failed: " +
          sensitivity.diagnostic);
    }
    if (iteration == 0) {
      result.allowed_interface_displacement =
          request.configuration->transport.conservative_phase
              .maximum_geometry_displacement_fraction *
          sensitivity.minimum_cell_node_distance;
    }
    const auto current_level_set = std::span<const svmp::FE::Real>(
        candidate.data() + level_set_offset, level_set_count);
    svmp::FE::level_set::LevelSetP1PhaseGeometryCorrectionOptions
        correction_options;
    correction_options.invariant_tolerance =
        request.configuration->transport.conservative_phase.invariant_tolerance;
    correction_options.relative_linear_tolerance = std::max(
        request.configuration->transport.conservative_phase.invariant_tolerance,
        std::min(svmp::FE::Real{1.0e-8},
                 request.configuration->transport.conservative_phase.geometry_measure_tolerance));
    auto correction =
        svmp::FE::level_set::solveLevelSetP1PhaseGeometryCorrection(
            sensitivity,
            projection_options.liquid_side,
            current_level_set,
            projection.liquid_phase_mass,
            target_liquid_phase_mass,
            correction_options);
    if (!correction.success) {
      result.diagnostic =
          "Conservative phase local geometry correction has no admissible shape update: " +
          correction.diagnostic;
      return result;
    }

    for (std::size_t node = 0u;
         node < correction.level_set_increment.size(); ++node) {
      if (!contact_protected_nodes.empty() &&
          contact_protected_nodes[node] != 0u) {
        result.maximum_removed_contact_increment = std::max(
            result.maximum_removed_contact_increment,
            std::abs(correction.level_set_increment[node]));
        correction.level_set_increment[node] = svmp::FE::Real{0.0};
      }
    }

    svmp::FE::Real maximum_increment{0.0};
    for (const auto increment : correction.level_set_increment) {
      maximum_increment = std::max(
          maximum_increment, std::abs(increment));
    }
    if (!(maximum_increment > svmp::FE::Real{0.0}) ||
        !std::isfinite(maximum_increment)) {
      result.diagnostic = result.contact_protected_nodes > 0u
          ? "Conservative phase local geometry correction cannot close the nodal moments without changing accepted wall-contact parent cells"
          : "Conservative phase local geometry correction produced a zero or non-finite update";
      return result;
    }
    const auto full_step_displacement =
        maximum_increment / sensitivity.minimum_level_set_gradient;
    const auto remaining_displacement =
        result.allowed_interface_displacement -
        result.accumulated_interface_displacement_bound;
    if (!(remaining_displacement > svmp::FE::Real{0.0})) {
      result.limited_by_displacement = true;
      result.diagnostic =
          "Conservative phase local geometry correction exhausted its cumulative displacement contract";
      return result;
    }
    svmp::FE::Real step_scale = std::min(
        svmp::FE::Real{1.0},
        remaining_displacement / full_step_displacement);
    if (step_scale < svmp::FE::Real{1.0}) {
      result.limited_by_displacement = true;
    }

    const auto sign_margin =
        request.configuration->transport.conservative_phase.invariant_tolerance *
        std::max(svmp::FE::Real{1.0},
                 std::abs(*std::max_element(
                     current_level_set.begin(), current_level_set.end(),
                     [](svmp::FE::Real first, svmp::FE::Real second) {
                       return std::abs(first) < std::abs(second);
                     })));
    for (std::size_t node = 0u; node < level_set_count; ++node) {
      const auto signed_value =
          current_level_set[node] - volume_options.isovalue;
      const auto increment = correction.level_set_increment[node];
      if (signed_value * increment < svmp::FE::Real{0.0}) {
        const auto safe_magnitude =
            std::abs(signed_value) - sign_margin;
        if (!(safe_magnitude > svmp::FE::Real{0.0})) {
          result.limited_by_topology = true;
          result.diagnostic =
              "Conservative phase local geometry correction encountered a level-set node inside its topology margin";
          return result;
        }
        step_scale = std::min(
            step_scale,
            svmp::FE::Real{0.9} * safe_magnitude /
                std::abs(increment));
      }
    }
    if (!(step_scale > svmp::FE::Real{0.0}) ||
        !std::isfinite(step_scale)) {
      result.limited_by_topology = true;
      result.diagnostic =
          "Conservative phase local geometry correction has no topology-stable step";
      return result;
    }

    bool accepted_line_search = false;
    constexpr int maximum_line_search_evaluations = 16;
    for (int line_search = 0;
         line_search < maximum_line_search_evaluations;
         ++line_search) {
      ++result.line_search_evaluations;
      auto trial = candidate;
      for (std::size_t node = 0u; node < level_set_count; ++node) {
        trial[level_set_offset + node] =
            current_level_set[node] +
            step_scale * correction.level_set_increment[node];
      }
      svmp::FE::level_set::LevelSetP1PhaseProjectionResult
          trial_projection;
      try {
        ++result.geometry_refresh_requests;
        result.geometry_rebuilds +=
            transaction.refresh(trial).refreshed ? 1 : 0;
        trial_projection = projectCurrentConservativePhaseGeometry(
            system, request, /*allow_graph_rebuild=*/false);
      } catch (const std::exception& exception) {
        ++result.rejected_geometry_trials;
        result.last_rejected_trial_diagnostic = exception.what();
        step_scale *= svmp::FE::Real{0.5};
        continue;
      }
      if (!trial_projection.success) {
        ++result.rejected_geometry_trials;
        result.last_rejected_trial_diagnostic =
            trial_projection.diagnostic;
        step_scale *= svmp::FE::Real{0.5};
        continue;
      }
      const auto trial_mismatch = conservativePhaseMomentMismatch(
          trial_projection.liquid_phase_mass,
          target_liquid_phase_mass);
      const bool trial_reached =
          trial_mismatch.maximum_nodal_residual <= moment_tolerance &&
          std::abs(trial_mismatch.total_residual) <= moment_tolerance;
      const bool sufficient_decrease =
          trial_mismatch.residual_norm <
          mismatch.residual_norm *
              (svmp::FE::Real{1.0} -
               svmp::FE::Real{1.0e-4} * step_scale);
      if (trial_reached || sufficient_decrease) {
        candidate = std::move(trial);
        result.accumulated_interface_displacement_bound +=
            step_scale * full_step_displacement;
        result.final_residual_norm = trial_mismatch.residual_norm;
        result.maximum_final_nodal_residual =
            trial_mismatch.maximum_nodal_residual;
        result.final_total_residual = trial_mismatch.total_residual;
        result.iterations = iteration + 1;
        accepted_line_search = true;
        break;
      }
      step_scale *= svmp::FE::Real{0.5};
    }
    if (!accepted_line_search) {
      ++result.geometry_refresh_requests;
      result.geometry_rebuilds +=
          transaction.refresh(candidate).refreshed ? 1 : 0;
      result.diagnostic =
          "Conservative phase local geometry correction line search did not reduce the nodal moment residual";
      if (!result.last_rejected_trial_diagnostic.empty()) {
        result.diagnostic += ": " +
                             result.last_rejected_trial_diagnostic;
      }
      return result;
    }
  }

  result.diagnostic =
      "Conservative phase local geometry correction did not reach its nodal moment tolerance within the bounded iteration count";
  return result;
}

} // namespace level_set
} // namespace FE
} // namespace svmp
