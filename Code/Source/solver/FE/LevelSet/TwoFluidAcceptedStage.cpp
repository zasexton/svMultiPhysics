/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "LevelSet/TwoFluidAcceptedStage.h"

#include "Assembly/Assembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Geometry/CutQuadratureMapping.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Interfaces/LevelSetInterfaceDomain.h"
#include "Interfaces/SnapshotFieldEvaluator.h"
#include "LevelSet/LevelSetOptions.h"
#include "Systems/FESystemHandle.h"
#include "Systems/SnapshotFieldAccess.h"
#include "Systems/SystemCollectives.h"
#include "Systems/TwoFluidStageRecords.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace svmp::FE::level_set {

using svmp::FE::interfaces::makeSnapshotScalarFieldEvaluator;
using svmp::FE::interfaces::makeSnapshotVectorFieldEvaluator;
using svmp::FE::systems::snapshotFieldAccess;
using svmp::FE::systems::FESystemHandle;
using svmp::FE::systems::collectiveAnyTrue;
using svmp::FE::systems::collectiveMaxDouble;
using svmp::FE::systems::collectiveMinDouble;
using svmp::FE::systems::collectiveMinMaxUint64;
using svmp::FE::systems::collectiveSumDouble;
using svmp::FE::systems::collectiveSumSize;
using svmp::FE::systems::requireCollectivePhasePreparation;

std::vector<svmp::FE::systems::AcceptedTwoFluidStageDiagnosticState>
evaluateCurrentTwoFluidStageDiagnostics(
    const FESystemHandle& sim,
    std::span<const svmp::FE::systems::
                        TwoFluidAcceptedStageDiagnosticDeclaration>
        declarations,
    std::span<const svmp::FE::Real> stage_solution,
    svmp::FE::Real effective_dt,
    const svmp::MeshComm& comm)
{
  const auto local_count = static_cast<double>(declarations.size());
  if (collectiveMinDouble(local_count, comm) !=
      collectiveMaxDouble(local_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Two-fluid diagnostic declaration coverage differs across the FE communicator.");
  }
  if (declarations.empty()) {
    return {};
  }
  const auto* context = sim.fe_system->cutIntegrationContext();
  if (collectiveMinDouble(context != nullptr ? 1.0 : 0.0, comm) != 1.0 ||
      collectiveMinDouble(stage_solution.empty() ? 0.0 : 1.0, comm) != 1.0 ||
      collectiveMinDouble(static_cast<double>(stage_solution.size()), comm) !=
          collectiveMaxDouble(static_cast<double>(stage_solution.size()), comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Two-fluid stage diagnostics require an authoritative cut context and a communicator-consistent stage solution.");
  }
  context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
      sim.fe_system->meshAccess());

  std::vector<svmp::FE::systems::AcceptedTwoFluidStageDiagnosticState>
      states;
  states.reserve(declarations.size());
  for (const auto& declaration : declarations) {
    const auto expected_source_id =
        svmp::FE::interfaces::LevelSetInterfaceSource::fromField(
            declaration.level_set_field)
            .identifier();
    const auto& snapshots = context->freeSurfaceGeometrySnapshots();
    const auto found = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().interface_marker ==
                     declaration.interface_marker &&
                 candidate->revision().domain_id ==
                     declaration.geometry_domain_id &&
                 candidate->revision().source_id == expected_source_id &&
                 candidate->revision().isovalue ==
                     declaration.level_set_isovalue;
        });
    if (collectiveMinDouble(found != snapshots.end() ? 1.0 : 0.0, comm) !=
        1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Two-fluid stage diagnostics cannot resolve the declared authoritative snapshot for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    const auto& snapshot = **found;
    const auto [minimum_snapshot_revision, maximum_snapshot_revision] =
        collectiveMinMaxUint64(
            snapshot.revision().snapshot_revision_key, comm);
    if (minimum_snapshot_revision == 0u ||
        minimum_snapshot_revision != maximum_snapshot_revision) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Two-fluid snapshot revision differs across the FE communicator.");
    }

    std::optional<
        svmp::FE::interfaces::IncompressibleTwoFluidDiagnosticAccumulator>
        local_accumulator;
    std::exception_ptr local_evaluation_failure;
    try {
      svmp::FE::interfaces::IncompressibleTwoFluidPhaseEvaluators negative{
          .velocity = makeSnapshotVectorFieldEvaluator(
              snapshotFieldAccess(*sim.fe_system, declaration.negative_velocity_field),
              stage_solution,
              "Two-fluid negative-phase stage"),
          .pressure = makeSnapshotScalarFieldEvaluator(
              snapshotFieldAccess(*sim.fe_system, declaration.negative_pressure_field),
              stage_solution,
              "Two-fluid negative-phase stage"),
      };
      svmp::FE::interfaces::IncompressibleTwoFluidPhaseEvaluators positive{
          .velocity = makeSnapshotVectorFieldEvaluator(
              snapshotFieldAccess(*sim.fe_system, declaration.positive_velocity_field),
              stage_solution,
              "Two-fluid positive-phase stage"),
          .pressure = makeSnapshotScalarFieldEvaluator(
              snapshotFieldAccess(*sim.fe_system, declaration.positive_pressure_field),
              stage_solution,
              "Two-fluid positive-phase stage"),
      };
      auto cell_measures =
          std::make_shared<std::map<svmp::FE::GlobalIndex, svmp::FE::Real>>();
      auto* const mesh = &sim.fe_system->meshAccess();
      svmp::FE::interfaces::IncompressibleTwoFluidCellMeasureEvaluator
          cell_measure;
      cell_measure.physical_cell_measure =
          [mesh, cell_measures](svmp::FE::GlobalIndex cell) {
            const auto existing = cell_measures->find(cell);
            if (existing != cell_measures->end()) {
              return existing->second;
            }
            const auto value =
                svmp::FE::geometry::physicalCellMeasureFromMapping(
                    *mesh, cell);
            cell_measures->emplace(cell, value);
            return value;
          };
      local_accumulator.emplace(
          svmp::FE::interfaces::
              evaluateLocalIncompressibleTwoFluidDiagnostics(
                  snapshot,
                  declaration.parameters,
                  negative,
                  positive,
                  cell_measure,
                  effective_dt));
    } catch (...) {
      local_evaluation_failure = std::current_exception();
    }
    if (collectiveAnyTrue(local_evaluation_failure != nullptr, comm)) {
      if (local_evaluation_failure != nullptr) {
        std::rethrow_exception(local_evaluation_failure);
      }
      throw std::runtime_error(
          "[svMultiPhysics::Application] Another FE communicator rank could not evaluate the two-fluid stage diagnostics.");
    }
    auto global = std::move(*local_accumulator);

    const auto global_sum = [&comm](svmp::FE::Real value) {
      return static_cast<svmp::FE::Real>(
          collectiveSumDouble(static_cast<double>(value), comm));
    };
    global.owned_interface_quadrature_point_count = collectiveSumSize(
        global.owned_interface_quadrature_point_count, comm);
    global.interface_measure = global_sum(global.interface_measure);
    global.velocity_jump_squared =
        global_sum(global.velocity_jump_squared);
    global.normal_velocity_jump_squared =
        global_sum(global.normal_velocity_jump_squared);
    global.tangential_velocity_jump_squared =
        global_sum(global.tangential_velocity_jump_squared);
    global.negative_normal_flux = global_sum(global.negative_normal_flux);
    global.positive_normal_flux = global_sum(global.positive_normal_flux);
    global.traction_jump_squared =
        global_sum(global.traction_jump_squared);
    global.viscous_traction_jump_squared =
        global_sum(global.viscous_traction_jump_squared);
    global.traction_jump_normal_integral =
        global_sum(global.traction_jump_normal_integral);
    global.pressure_jump_integral =
        global_sum(global.pressure_jump_integral);
    global.pressure_jump_squared =
        global_sum(global.pressure_jump_squared);
    global.surface_energy_work = global_sum(global.surface_energy_work);
    global.nitsche_consistency_work =
        global_sum(global.nitsche_consistency_work);
    global.nitsche_adjoint_work =
        global_sum(global.nitsche_adjoint_work);
    global.nitsche_penalty_work =
        global_sum(global.nitsche_penalty_work);
    for (auto* vector : {&global.interface_normal_integral,
                         &global.negative_traction_integral,
                         &global.positive_traction_integral,
                         &global.traction_jump_integral,
                         &global.negative_viscous_traction_integral,
                         &global.positive_viscous_traction_integral,
                         &global.viscous_traction_jump_integral}) {
      for (auto& value : *vector) {
        value = global_sum(value);
      }
    }
    const auto reduce_optional = [&](std::optional<svmp::FE::Real>& value) {
      const double present = value.has_value() ? 1.0 : 0.0;
      if (collectiveMinDouble(present, comm) !=
          collectiveMaxDouble(present, comm)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Two-fluid prescribed diagnostic applicability differs across the FE communicator.");
      }
      if (value.has_value()) {
        value = global_sum(*value);
      }
    };
    reduce_optional(global.prescribed_stress_jump_residual_squared);
    reduce_optional(global.prescribed_pressure_jump_error_squared);
    reduce_optional(
        global.prescribed_viscous_traction_jump_error_squared);
    const auto reduce_phase = [&](auto& phase) {
      phase.owned_quadrature_point_count =
          collectiveSumSize(phase.owned_quadrature_point_count, comm);
      phase.volume = global_sum(phase.volume);
      phase.velocity_squared_integral =
          global_sum(phase.velocity_squared_integral);
      phase.maximum_velocity_squared =
          collectiveMaxDouble(phase.maximum_velocity_squared, comm);
      phase.pressure_integral = global_sum(phase.pressure_integral);
      phase.pressure_squared_integral =
          global_sum(phase.pressure_squared_integral);
      phase.hydrostatic_residual_squared =
          global_sum(phase.hydrostatic_residual_squared);
      for (auto& value : phase.velocity_integral) {
        value = global_sum(value);
      }
      for (auto* vector : {&phase.pressure_gradient_integral,
                           &phase.hydrostatic_residual_integral}) {
        for (auto& value : *vector) {
          value = global_sum(value);
        }
      }
    };
    reduce_phase(global.negative_phase);
    reduce_phase(global.positive_phase);
    states.push_back(
        svmp::FE::systems::AcceptedTwoFluidStageDiagnosticState{
            .interface_marker = declaration.interface_marker,
            .geometry_revision = snapshot.revision(),
            .local_mesh_revision = snapshot.localMeshRevision(),
            .diagnostics =
                svmp::FE::interfaces::
                    finalizeIncompressibleTwoFluidDiagnostics(
                        global, declaration.parameters),
        });
  }
  return states;
}

TwoFluidMomentumReconciliationBatch
buildCurrentTwoFluidMomentumReconciliations(
    const FESystemHandle& sim,
    std::span<const svmp::FE::level_set::ConservativePhaseRequest>
        maintenance_requests,
    std::span<const svmp::FE::Real> corrected_solution,
    std::uint64_t corrected_algebraic_revision,
    svmp::FE::Real effective_dt,
    const svmp::MeshComm& comm)
{
  const auto declarations =
      svmp::FE::systems::twoFluidStageRecords(*sim.fe_system).twoFluidAcceptedStageDiagnosticDeclarations();
  const auto pending =
      svmp::FE::systems::twoFluidStageRecords(*sim.fe_system).pendingTwoFluidAcceptedStageDiagnostics();
  TwoFluidMomentumReconciliationBatch batch;
  std::vector<svmp::FE::systems::
                  TwoFluidAcceptedStageDiagnosticDeclaration>
      required_declarations;
  std::vector<std::size_t> required_indices;
  std::vector<svmp::FE::Real> tolerances;
  std::exception_ptr local_preparation_failure;
  try {
    if (pending.size() != declarations.size() ||
        corrected_algebraic_revision == 0u ||
        !std::isfinite(effective_dt) ||
        !(effective_dt > svmp::FE::Real{0.0})) {
      throw std::runtime_error(
          "Two-fluid momentum reconciliation requires one complete pending diagnostic group and a valid corrected endpoint revision.");
    }
    for (std::size_t index = 0u; index < declarations.size(); ++index) {
      const auto& declaration = declarations[index];
      if (!declaration
               .require_conservative_phase_momentum_reconciliation) {
        continue;
      }
      const auto& level_set_name =
          sim.fe_system->fieldRecord(declaration.level_set_field).name;
      const auto matches = [&](const auto& request) {
        return request.configuration->transport.conservative_phase.enabled &&
               request.configuration->transport.conservative_phase.reconcile_geometry &&
               request.configuration->transport.level_set.field_name == level_set_name &&
               request.configuration->isovalue == declaration.level_set_isovalue &&
               request.configuration->transport.velocity.source ==
                   svmp::FE::level_set::LevelSetVelocitySource::
                       MaterialInterfacePhasePair &&
               request.configuration->transport.velocity.material_interface_marker ==
                   declaration.interface_marker;
      };
      const auto count = static_cast<std::size_t>(std::count_if(
          maintenance_requests.begin(),
          maintenance_requests.end(),
          matches));
      if (count != 1u ||
          pending[index].declaration != declaration) {
        throw std::runtime_error(
            "Two-fluid momentum reconciliation requires exactly one matching conservative material-interface maintenance request.");
      }
      const auto request = std::find_if(
          maintenance_requests.begin(), maintenance_requests.end(), matches);
      const auto tolerance =
          request->configuration->transport.conservative_phase.momentum_relative_tolerance;
      if (!std::isfinite(tolerance) ||
          !(tolerance > svmp::FE::Real{0.0})) {
        throw std::runtime_error(
            "Two-fluid momentum reconciliation tolerance must be positive and finite.");
      }
      required_declarations.push_back(declaration);
      required_indices.push_back(index);
      tolerances.push_back(tolerance);
    }
  } catch (...) {
    local_preparation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_preparation_failure,
      comm,
      "two_fluid_momentum_reconciliation_preparation");
  if (required_declarations.empty()) {
    return batch;
  }

  auto corrected_states = evaluateCurrentTwoFluidStageDiagnostics(
      sim,
      required_declarations,
      corrected_solution,
      effective_dt,
      comm);
  local_preparation_failure = nullptr;
  try {
    batch.records.reserve(required_declarations.size());
    for (std::size_t required = 0u;
         required < required_declarations.size();
         ++required) {
      const auto& raw = pending[required_indices[required]];
      const auto& corrected = corrected_states[required];
      auto reconciliation = svmp::FE::interfaces::
          buildIncompressibleTwoFluidMomentumReconciliation(
              raw.declaration.interface_marker,
              raw.state.geometry_revision,
              raw.state.diagnostics,
              raw.stage.state_revision,
              corrected.geometry_revision,
              corrected.diagnostics,
              corrected_algebraic_revision,
              tolerances[required]);
      batch.satisfied = batch.satisfied && reconciliation.satisfied;
      batch.records.push_back(std::move(reconciliation));
    }
  } catch (...) {
    local_preparation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_preparation_failure,
      comm,
      "two_fluid_momentum_reconciliation_evaluation");
  return batch;
}

svmp::FE::systems::TwoFluidAcceptedStageSolveTelemetry
makeTwoFluidAcceptedStageSolveTelemetry(
    const svmp::FE::timestepping::NewtonReport& report)
{
  const auto reason_or = [](const std::string& reason,
                            const char* fallback) {
    return reason.find_first_not_of(" \t\r\n") == std::string::npos
               ? std::string(fallback)
               : reason;
  };
  std::optional<svmp::FE::Real> outer_raw_contraction_ratio;
  if (std::isfinite(report.outer_raw_contraction_ratio)) {
    outer_raw_contraction_ratio = static_cast<svmp::FE::Real>(
        report.outer_raw_contraction_ratio);
  }
  const bool linear_attempted =
      report.iterations > 0 || report.inner_iterations_total > 0;
  const bool exact_entry_convergence =
      !linear_attempted && report.converged;
  return svmp::FE::systems::TwoFluidAcceptedStageSolveTelemetry{
      .nonlinear =
          svmp::FE::systems::TwoFluidAcceptedStageNonlinearTelemetry{
              .converged = report.converged,
              .iterations = report.iterations,
              .outer_iterations = report.outer_iterations,
              .inner_iterations_total = report.inner_iterations_total,
              .outer_state_change_norm = static_cast<svmp::FE::Real>(
                  report.outer_state_change_norm),
              .outer_dynamic_relaxation_enabled =
                  report.outer_dynamic_relaxation_enabled,
              .outer_dynamic_relaxation_updates =
                  report.outer_dynamic_relaxation_updates,
              .outer_dynamic_relaxation_safeguards =
                  report.outer_dynamic_relaxation_safeguards,
              .outer_dynamic_relaxation_resets =
                  report.outer_dynamic_relaxation_resets,
              .outer_dynamic_relaxation_factor =
                  static_cast<svmp::FE::Real>(
                      report.outer_dynamic_relaxation_factor),
              .outer_relaxed_state_change_norm =
                  static_cast<svmp::FE::Real>(
                      report.outer_relaxed_state_change_norm),
              .outer_raw_contraction_ratio =
                  outer_raw_contraction_ratio,
              .initial_residual_norm = static_cast<svmp::FE::Real>(
                  report.residual_norm0),
              .final_residual_norm = static_cast<svmp::FE::Real>(
                  report.residual_norm),
              .initial_field_residual_norm =
                  static_cast<svmp::FE::Real>(
                      report.field_residual_norm0),
              .final_field_residual_norm =
                  static_cast<svmp::FE::Real>(
                      report.field_residual_norm),
              .initial_auxiliary_residual_norm =
                  static_cast<svmp::FE::Real>(
                      report.auxiliary_residual_norm0),
              .final_auxiliary_residual_norm =
                  static_cast<svmp::FE::Real>(
                      report.auxiliary_residual_norm),
              .component_residual_convergence =
                  report.component_residual_convergence,
              .reason = report.converged ? "converged" : "not_converged",
          },
      .linear =
          svmp::FE::systems::TwoFluidAcceptedStageLinearTelemetry{
              .attempted = linear_attempted,
              .converged =
                  report.linear.converged || exact_entry_convergence,
              .numerical_breakdown = report.linear.numerical_breakdown,
              .iterations = report.linear.iterations,
              .initial_residual_norm =
                  report.linear.initial_residual_norm,
              .final_residual_norm = report.linear.final_residual_norm,
              .relative_residual = report.linear.relative_residual,
              .collective_calls = report.linear.collective_calls,
              .collective_words = report.linear.collective_words,
              .blockschur_outer_iterations =
                  report.linear.blockschur_outer_iterations,
              .blockschur_collective_calls_max_per_outer =
                  report.linear.blockschur_collective_calls_max_per_outer,
              .blockschur_momentum_solve_calls =
                  report.linear.blockschur_momentum_solve_calls,
              .blockschur_momentum_iterations =
                  report.linear.blockschur_momentum_iterations,
              .blockschur_momentum_restart_cycles =
                  report.linear.blockschur_momentum_restart_cycles,
              .blockschur_momentum_collective_calls =
                  report.linear.blockschur_momentum_collective_calls,
              .blockschur_momentum_collective_words =
                  report.linear.blockschur_momentum_collective_words,
              .blockschur_schur_solve_calls =
                  report.linear.blockschur_schur_solve_calls,
              .blockschur_schur_iterations =
                  report.linear.blockschur_schur_iterations,
              .blockschur_true_residual_retries =
                  report.linear.blockschur_true_residual_retries,
              .blockschur_schur_collective_calls =
                  report.linear.blockschur_schur_collective_calls,
              .blockschur_schur_collective_words =
                  report.linear.blockschur_schur_collective_words,
              .reason = exact_entry_convergence
                            ? "not_required_exact_entry_convergence"
                            : reason_or(
                                  report.linear.message,
                                  report.linear.converged
                                      ? "converged"
                                      : "not_converged"),
          },
  };
}

} // namespace svmp::FE::level_set
