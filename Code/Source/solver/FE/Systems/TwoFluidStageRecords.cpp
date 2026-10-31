/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Systems/TwoFluidStageRecords.h"

#include "Core/LogRecord.h"

#include "Core/FEException.h"
#include "Core/Logger.h"
#include "Systems/MeshBoundaryRecordSupport.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp::FE::systems {

// The record code moved from FE/Systems/FESystem.cpp and keeps the FE names
// it used there.
using namespace svmp::FE;
using namespace svmp::FE::systems;
using namespace svmp::FE::systems::record_support;

namespace {

void appendTwoFluidDiagnosticParameters(
    MeshBoundaryConsensusTokens& tokens,
    const interfaces::IncompressibleTwoFluidDiagnosticParameters& parameters)
{
    appendMeshBoundaryIntegral(tokens, parameters.dimension);
    appendMeshBoundaryIntegral(tokens, parameters.interface_marker);
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(parameters.negative_density));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(parameters.positive_density));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(parameters.negative_viscosity));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(parameters.positive_viscosity));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(parameters.nitsche_gamma));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(parameters.surface_tension));
    appendMeshBoundaryToken(
        tokens, parameters.include_transient_penalty ? 1u : 0u);
    appendMeshBoundaryToken(
        tokens, parameters.prescribed_pressure_jump.has_value() ? 1u : 0u);
    if (parameters.prescribed_pressure_jump.has_value()) {
        appendMeshBoundaryToken(
            tokens,
            generatedBoundaryTraceRealBits(
                *parameters.prescribed_pressure_jump));
    }
    appendMeshBoundaryToken(
        tokens,
        parameters.prescribed_viscous_traction_jump.has_value() ? 1u : 0u);
    if (parameters.prescribed_viscous_traction_jump.has_value()) {
        for (const auto value :
             *parameters.prescribed_viscous_traction_jump) {
            appendMeshBoundaryToken(
                tokens, generatedBoundaryTraceRealBits(value));
        }
    }
    for (const auto value : parameters.body_force) {
        appendMeshBoundaryToken(
            tokens, generatedBoundaryTraceRealBits(value));
    }
}

void appendTwoFluidDiagnosticDeclaration(
    MeshBoundaryConsensusTokens& tokens,
    const TwoFluidAcceptedStageDiagnosticDeclaration& declaration)
{
    appendMeshBoundaryIntegral(tokens, declaration.interface_marker);
    appendMeshBoundaryIntegral(tokens, declaration.level_set_field);
    appendMeshBoundaryIntegral(tokens, declaration.negative_velocity_field);
    appendMeshBoundaryIntegral(tokens, declaration.negative_pressure_field);
    appendMeshBoundaryIntegral(tokens, declaration.positive_velocity_field);
    appendMeshBoundaryIntegral(tokens, declaration.positive_pressure_field);
    appendMeshBoundaryString(tokens, declaration.operator_tag);
    appendMeshBoundaryString(tokens, declaration.geometry_domain_id);
    appendMeshBoundaryToken(
        tokens,
        generatedBoundaryTraceRealBits(declaration.level_set_isovalue));
    appendMeshBoundaryToken(
        tokens,
        declaration.require_conservative_phase_momentum_reconciliation
            ? 1u
            : 0u);
    appendMeshBoundaryToken(
        tokens, declaration.require_accepted_stage_numerics ? 1u : 0u);
    appendMeshBoundaryToken(
        tokens,
        generatedBoundaryTraceRealBits(
            declaration.pressure_stabilization_coefficient));
    appendTwoFluidDiagnosticParameters(tokens, declaration.parameters);
    appendMeshBoundaryString(tokens, declaration.owner_component);
}

void appendTwoFluidGeometryRevision(
    MeshBoundaryConsensusTokens& tokens,
    const interfaces::FreeSurfaceGeometryRevision& revision)
{
    appendMeshBoundaryString(tokens, revision.source_id);
    appendMeshBoundaryString(tokens, revision.domain_id);
    appendMeshBoundaryIntegral(tokens, revision.interface_marker);
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(revision.isovalue));
    appendMeshBoundaryToken(tokens, revision.source_layout_revision);
    appendMeshBoundaryToken(tokens, revision.source_value_revision);
    appendMeshBoundaryToken(tokens, revision.mesh_geometry_revision);
    appendMeshBoundaryToken(tokens, revision.mesh_topology_revision);
    appendMeshBoundaryToken(tokens, revision.ownership_revision);
    appendMeshBoundaryToken(tokens, revision.numbering_revision);
    appendMeshBoundaryToken(tokens, revision.quadrature_policy_key);
    appendMeshBoundaryToken(tokens, revision.snapshot_revision_key);
}

void appendTwoFluidPhaseDiagnosticState(
    MeshBoundaryConsensusTokens& tokens,
    const interfaces::IncompressibleTwoFluidPhaseDiagnosticState& state)
{
    appendMeshBoundaryIntegral(tokens, state.side);
    appendMeshBoundaryToken(
        tokens, static_cast<std::uint64_t>(state.quadrature_point_count));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(state.density));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(state.volume));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(state.mass));
    for (const auto value : state.momentum) {
        appendMeshBoundaryToken(
            tokens, generatedBoundaryTraceRealBits(value));
    }
    for (const auto value : state.mean_velocity) {
        appendMeshBoundaryToken(
            tokens, generatedBoundaryTraceRealBits(value));
    }
    appendMeshBoundaryToken(
        tokens,
        generatedBoundaryTraceRealBits(state.maximum_quadrature_speed));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(state.kinetic_energy));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(state.pressure_integral));
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(state.mean_pressure));
    appendMeshBoundaryToken(
        tokens,
        generatedBoundaryTraceRealBits(state.pressure_squared_integral));
    for (const auto& vector :
         {state.pressure_gradient_integral,
          state.body_force_density_integral,
          state.hydrostatic_residual_integral}) {
        for (const auto value : vector) {
            appendMeshBoundaryToken(
                tokens, generatedBoundaryTraceRealBits(value));
        }
    }
    appendMeshBoundaryToken(
        tokens,
        generatedBoundaryTraceRealBits(
            state.hydrostatic_residual_squared));
}

void appendTwoFluidDiagnosticState(
    MeshBoundaryConsensusTokens& tokens,
    const interfaces::IncompressibleTwoFluidDiagnosticState& state)
{
    appendMeshBoundaryToken(tokens, state.snapshot_revision_key);
    appendMeshBoundaryToken(
        tokens,
        state.transient_penalty_effective_dt.has_value() ? 1u : 0u);
    if (state.transient_penalty_effective_dt.has_value()) {
        appendMeshBoundaryToken(
            tokens,
            generatedBoundaryTraceRealBits(
                *state.transient_penalty_effective_dt));
    }
    appendMeshBoundaryToken(
        tokens,
        static_cast<std::uint64_t>(state.interface_quadrature_point_count));
    const std::array<Real, 21> scalars{
        state.interface_measure,
        state.velocity_jump_squared,
        state.normal_velocity_jump_squared,
        state.tangential_velocity_jump_squared,
        state.negative_normal_flux,
        state.positive_normal_flux,
        state.normal_flux_jump,
        state.negative_mass_flux,
        state.positive_mass_flux,
        state.traction_jump_normal_integral,
        state.traction_jump_squared,
        state.viscous_traction_jump_squared,
        state.pressure_jump_integral,
        state.mean_pressure_jump,
        state.pressure_jump_squared,
        state.surface_energy_work,
        state.nitsche_consistency_work,
        state.nitsche_adjoint_work,
        state.nitsche_penalty_work,
        state.negative_phase.volume,
        state.positive_phase.volume};
    for (const auto value : scalars) {
        appendMeshBoundaryToken(
            tokens, generatedBoundaryTraceRealBits(value));
    }
    for (const auto& vector : {state.interface_normal_integral,
                               state.negative_traction_integral,
                               state.positive_traction_integral,
                               state.traction_jump_integral,
                               state.negative_viscous_traction_integral,
                               state.positive_viscous_traction_integral,
                               state.viscous_traction_jump_integral}) {
        for (const auto value : vector) {
            appendMeshBoundaryToken(
                tokens, generatedBoundaryTraceRealBits(value));
        }
    }
    appendMeshBoundaryToken(
        tokens,
        state.prescribed_stress_jump_residual_squared.has_value() ? 1u : 0u);
    if (state.prescribed_stress_jump_residual_squared.has_value()) {
        appendMeshBoundaryToken(
            tokens,
            generatedBoundaryTraceRealBits(
                *state.prescribed_stress_jump_residual_squared));
    }
    appendMeshBoundaryToken(
        tokens,
        state.prescribed_pressure_jump_error_squared.has_value() ? 1u : 0u);
    if (state.prescribed_pressure_jump_error_squared.has_value()) {
        appendMeshBoundaryToken(
            tokens,
            generatedBoundaryTraceRealBits(
                *state.prescribed_pressure_jump_error_squared));
    }
    appendMeshBoundaryToken(
        tokens,
        state.prescribed_viscous_traction_jump_error_squared.has_value()
            ? 1u
            : 0u);
    if (state.prescribed_viscous_traction_jump_error_squared.has_value()) {
        appendMeshBoundaryToken(
            tokens,
            generatedBoundaryTraceRealBits(
                *state.prescribed_viscous_traction_jump_error_squared));
    }
    appendTwoFluidPhaseDiagnosticState(tokens, state.negative_phase);
    appendTwoFluidPhaseDiagnosticState(tokens, state.positive_phase);
}

void appendTwoFluidPhaseMomentumReconciliation(
    MeshBoundaryConsensusTokens& tokens,
    const interfaces::IncompressibleTwoFluidPhaseMomentumReconciliation& phase)
{
    appendMeshBoundaryIntegral(tokens, phase.side);
    const std::array<Real, 7> scalars{
        phase.density,
        phase.raw_volume,
        phase.corrected_volume,
        phase.raw_mass,
        phase.corrected_mass,
        phase.momentum_delta_norm,
        phase.momentum_reference_norm};
    for (const auto value : scalars) {
        appendMeshBoundaryToken(
            tokens, generatedBoundaryTraceRealBits(value));
    }
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(phase.allowed_momentum_delta));
    for (const auto& vector : {phase.raw_momentum,
                               phase.corrected_momentum,
                               phase.momentum_delta}) {
        for (const auto value : vector) {
            appendMeshBoundaryToken(
                tokens, generatedBoundaryTraceRealBits(value));
        }
    }
    appendMeshBoundaryToken(tokens, phase.satisfied ? 1u : 0u);
}

void appendTwoFluidMomentumReconciliation(
    MeshBoundaryConsensusTokens& tokens,
    const interfaces::IncompressibleTwoFluidMomentumReconciliation& record)
{
    appendMeshBoundaryIntegral(tokens, record.interface_marker);
    appendTwoFluidGeometryRevision(tokens, record.raw_geometry_revision);
    appendTwoFluidGeometryRevision(tokens, record.corrected_geometry_revision);
    appendMeshBoundaryToken(tokens, record.raw_algebraic_revision);
    appendMeshBoundaryToken(tokens, record.corrected_algebraic_revision);
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(record.relative_tolerance));
    appendMeshBoundaryToken(tokens, record.velocity_update_applied ? 1u : 0u);
    appendMeshBoundaryToken(tokens, record.satisfied ? 1u : 0u);
    appendTwoFluidPhaseMomentumReconciliation(
        tokens, record.negative_phase);
    appendTwoFluidPhaseMomentumReconciliation(
        tokens, record.positive_phase);
}

void appendTwoFluidAcceptedStageAggregationGeometry(
    MeshBoundaryConsensusTokens& tokens,
    const TwoFluidAcceptedStageAggregationGeometryTelemetry& geometry)
{
    appendMeshBoundaryIntegral(tokens, geometry.kind);
    appendMeshBoundaryToken(tokens, geometry.available ? 1u : 0u);
    appendMeshBoundaryToken(
        tokens,
        geometry.communicator_fingerprint_consensus_validated ? 1u : 0u);
    appendMeshBoundaryString(tokens, geometry.source_id);
    appendMeshBoundaryString(tokens, geometry.domain_id);
    appendMeshBoundaryIntegral(tokens, geometry.interface_marker);
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(geometry.isovalue));
    appendMeshBoundaryToken(tokens, geometry.source_layout_revision);
    appendMeshBoundaryToken(tokens, geometry.source_value_revision);
    appendMeshBoundaryToken(tokens, geometry.quadrature_policy_key);
    appendMeshBoundaryToken(tokens, geometry.snapshot_revision_key);
    appendMeshBoundaryToken(
        tokens, geometry.distributed_mesh_geometry_revision);
    appendMeshBoundaryToken(
        tokens, geometry.distributed_mesh_topology_revision);
    appendMeshBoundaryToken(tokens, geometry.distributed_ownership_revision);
    appendMeshBoundaryToken(tokens, geometry.distributed_numbering_revision);
    appendMeshBoundaryToken(tokens, geometry.canonical_fingerprint);
}

void appendTwoFluidAcceptedStageAggregationTransition(
    MeshBoundaryConsensusTokens& tokens,
    const TwoFluidAcceptedStageAggregationTransitionTelemetry& transition)
{
    appendTwoFluidAcceptedStageAggregationGeometry(
        tokens, transition.geometry_identity_before);
    appendTwoFluidAcceptedStageAggregationGeometry(
        tokens, transition.geometry_identity_after);
    const std::array<std::uint64_t, 4> fingerprints{
        transition.canonical_feature_class_fingerprint_before,
        transition.canonical_feature_class_fingerprint_after,
        transition.canonical_slave_set_fingerprint_before,
        transition.canonical_slave_set_fingerprint_after};
    for (const auto value : fingerprints) {
        appendMeshBoundaryToken(tokens, value);
    }
    const std::array<std::size_t, 12> counts{
        transition.canonical_active_feature_count_before,
        transition.canonical_active_feature_count_after,
        transition.canonical_features_entered,
        transition.canonical_features_exited,
        transition.canonical_features_persisted,
        transition.canonical_feature_classification_changes,
        transition.canonical_features_became_rooted,
        transition.canonical_features_became_rootless,
        transition.canonical_aggregate_slaves_before,
        transition.canonical_aggregate_slaves_after,
        transition.canonical_aggregate_slaves_entered,
        transition.canonical_aggregate_slaves_left};
    for (const auto value : counts) {
        appendMeshBoundaryToken(
            tokens, static_cast<std::uint64_t>(value));
    }
    appendMeshBoundaryToken(
        tokens,
        generatedBoundaryTraceRealBits(
            transition.canonical_rootless_active_physical_volume_before));
    appendMeshBoundaryToken(
        tokens,
        generatedBoundaryTraceRealBits(
            transition.canonical_rootless_active_physical_volume_after));
    appendMeshBoundaryToken(
        tokens,
        generatedBoundaryTraceRealBits(
            transition.canonical_rootless_active_physical_volume_delta));
    appendMeshBoundaryToken(
        tokens, transition.canonical_topology_changed ? 1u : 0u);
}

void appendTwoFluidAcceptedStageAggregation(
    MeshBoundaryConsensusTokens& tokens,
    const TwoFluidAcceptedStageAggregationTelemetry& aggregation)
{
    appendTwoFluidAcceptedStageAggregationGeometry(
        tokens, aggregation.geometry_identity);
    appendMeshBoundaryToken(
        tokens, aggregation.canonical_feature_class_fingerprint);
    appendMeshBoundaryToken(
        tokens, aggregation.canonical_slave_set_fingerprint);
    const std::array<std::size_t, 14> counts{
        aggregation.maximum_root_path_length,
        aggregation.maximum_observed_root_path,
        aggregation.root_path_guard_rejections,
        aggregation.extrapolation_guard_rejections,
        aggregation.line_guard_rejections,
        aggregation.canonical_candidate_vertices,
        aggregation.canonical_rooted_candidate_vertices,
        aggregation.canonical_rootless_candidate_vertices,
        aggregation.canonical_owned_aggregate_dofs,
        aggregation.canonical_owned_pinned_dofs,
        aggregation.canonical_strong_suppressed_dofs,
        aggregation.canonical_active_feature_count,
        aggregation.canonical_rooted_active_feature_count,
        aggregation.canonical_rootless_active_feature_count};
    for (const auto value : counts) {
        appendMeshBoundaryToken(tokens, static_cast<std::uint64_t>(value));
    }
    const std::array<Real, 7> scalars{
        aggregation.maximum_reference_extrapolation_distance,
        aggregation.maximum_observed_reference_extrapolation,
        aggregation.maximum_absolute_coefficient,
        aggregation.maximum_observed_absolute_coefficient,
        aggregation.maximum_row_l1_norm,
        aggregation.maximum_observed_row_l1_norm,
        aggregation.canonical_rootless_active_physical_volume};
    for (const auto value : scalars) {
        appendMeshBoundaryToken(
            tokens, generatedBoundaryTraceRealBits(value));
    }
    appendMeshBoundaryToken(
        tokens,
        aggregation.canonical_topology_transition.has_value() ? 1u : 0u);
    if (aggregation.canonical_topology_transition.has_value()) {
        appendTwoFluidAcceptedStageAggregationTransition(
            tokens, *aggregation.canonical_topology_transition);
    }
}

void appendTwoFluidAcceptedStageSolveTelemetry(
    MeshBoundaryConsensusTokens& tokens,
    const TwoFluidAcceptedStageSolveTelemetry& solve)
{
    const auto& nonlinear = solve.nonlinear;
    appendMeshBoundaryToken(tokens, nonlinear.converged ? 1u : 0u);
    appendMeshBoundaryIntegral(tokens, nonlinear.iterations);
    appendMeshBoundaryIntegral(tokens, nonlinear.outer_iterations);
    appendMeshBoundaryIntegral(tokens, nonlinear.inner_iterations_total);
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(nonlinear.outer_state_change_norm));
    appendMeshBoundaryToken(
        tokens, nonlinear.outer_dynamic_relaxation_enabled ? 1u : 0u);
    appendMeshBoundaryIntegral(
        tokens, nonlinear.outer_dynamic_relaxation_updates);
    appendMeshBoundaryIntegral(
        tokens, nonlinear.outer_dynamic_relaxation_safeguards);
    appendMeshBoundaryIntegral(
        tokens, nonlinear.outer_dynamic_relaxation_resets);
    appendMeshBoundaryToken(
        tokens,
        generatedBoundaryTraceRealBits(
            nonlinear.outer_dynamic_relaxation_factor));
    appendMeshBoundaryToken(
        tokens,
        generatedBoundaryTraceRealBits(
            nonlinear.outer_relaxed_state_change_norm));
    appendMeshBoundaryToken(
        tokens, nonlinear.outer_raw_contraction_ratio.has_value() ? 1u : 0u);
    if (nonlinear.outer_raw_contraction_ratio.has_value()) {
        appendMeshBoundaryToken(
            tokens,
            generatedBoundaryTraceRealBits(
                *nonlinear.outer_raw_contraction_ratio));
    }
    const std::array<Real, 6> nonlinear_residuals{
        nonlinear.initial_residual_norm,
        nonlinear.final_residual_norm,
        nonlinear.initial_field_residual_norm,
        nonlinear.final_field_residual_norm,
        nonlinear.initial_auxiliary_residual_norm,
        nonlinear.final_auxiliary_residual_norm};
    for (const auto value : nonlinear_residuals) {
        appendMeshBoundaryToken(
            tokens, generatedBoundaryTraceRealBits(value));
    }
    appendMeshBoundaryToken(
        tokens, nonlinear.component_residual_convergence ? 1u : 0u);
    appendMeshBoundaryString(tokens, nonlinear.reason);

    const auto& linear = solve.linear;
    appendMeshBoundaryToken(tokens, linear.attempted ? 1u : 0u);
    appendMeshBoundaryToken(tokens, linear.converged ? 1u : 0u);
    appendMeshBoundaryToken(tokens, linear.numerical_breakdown ? 1u : 0u);
    appendMeshBoundaryIntegral(tokens, linear.iterations);
    for (const auto value : {linear.initial_residual_norm,
                             linear.final_residual_norm,
                             linear.relative_residual}) {
        appendMeshBoundaryToken(
            tokens, generatedBoundaryTraceRealBits(value));
    }
    appendMeshBoundaryToken(tokens, linear.collective_calls);
    appendMeshBoundaryToken(tokens, linear.collective_words);
    appendMeshBoundaryIntegral(tokens, linear.blockschur_outer_iterations);
    appendMeshBoundaryToken(
        tokens, linear.blockschur_collective_calls_max_per_outer);
    appendMeshBoundaryIntegral(
        tokens, linear.blockschur_momentum_solve_calls);
    appendMeshBoundaryIntegral(
        tokens, linear.blockschur_momentum_iterations);
    appendMeshBoundaryIntegral(
        tokens, linear.blockschur_momentum_restart_cycles);
    appendMeshBoundaryToken(
        tokens, linear.blockschur_momentum_collective_calls);
    appendMeshBoundaryToken(
        tokens, linear.blockschur_momentum_collective_words);
    appendMeshBoundaryIntegral(
        tokens, linear.blockschur_schur_solve_calls);
    appendMeshBoundaryIntegral(
        tokens, linear.blockschur_schur_iterations);
    appendMeshBoundaryIntegral(
        tokens, linear.blockschur_true_residual_retries);
    appendMeshBoundaryToken(
        tokens, linear.blockschur_schur_collective_calls);
    appendMeshBoundaryToken(
        tokens, linear.blockschur_schur_collective_words);
    appendMeshBoundaryString(tokens, linear.reason);
}

void appendTwoFluidAcceptedStagePhaseNumerics(
    MeshBoundaryConsensusTokens& tokens,
    const TwoFluidAcceptedStagePhaseNumerics& phase)
{
    appendMeshBoundaryIntegral(tokens, phase.side);
    appendTwoFluidAcceptedStageAggregation(
        tokens, phase.velocity_aggregation);
    appendTwoFluidAcceptedStageAggregation(
        tokens, phase.pressure_aggregation);
    const auto& stabilization = phase.pressure_stabilization;
    appendMeshBoundaryToken(
        tokens, generatedBoundaryTraceRealBits(stabilization.coefficient));
    appendMeshBoundaryToken(
        tokens, stabilization.accepted_stage_work.has_value() ? 1u : 0u);
    if (stabilization.accepted_stage_work.has_value()) {
        appendMeshBoundaryToken(
            tokens,
            generatedBoundaryTraceRealBits(
                *stabilization.accepted_stage_work));
    }
    appendMeshBoundaryString(tokens, stabilization.work_reason);
    appendMeshBoundaryIntegral(tokens, phase.linear_iteration_scope);
    appendMeshBoundaryToken(
        tokens, phase.linear_iterations.has_value() ? 1u : 0u);
    if (phase.linear_iterations.has_value()) {
        appendMeshBoundaryIntegral(tokens, *phase.linear_iterations);
    }
    appendMeshBoundaryString(tokens, phase.linear_reason);
    appendMeshBoundaryIntegral(tokens, phase.nonlinear_iteration_scope);
    appendMeshBoundaryToken(
        tokens, phase.nonlinear_iterations.has_value() ? 1u : 0u);
    if (phase.nonlinear_iterations.has_value()) {
        appendMeshBoundaryIntegral(tokens, *phase.nonlinear_iterations);
    }
    appendMeshBoundaryString(tokens, phase.nonlinear_reason);
}

void appendTwoFluidAcceptedStageNumerics(
    MeshBoundaryConsensusTokens& tokens,
    const TwoFluidAcceptedStageNumerics& numerics)
{
    appendTwoFluidAcceptedStageSolveTelemetry(tokens, numerics.solve);
    appendTwoFluidAcceptedStagePhaseNumerics(
        tokens, numerics.negative_phase);
    appendTwoFluidAcceptedStagePhaseNumerics(
        tokens, numerics.positive_phase);
}

void appendTwoFluidAcceptedStageRecord(
    MeshBoundaryConsensusTokens& tokens,
    const TwoFluidAcceptedStageDiagnosticHistoryRecord& record)
{
    appendTwoFluidDiagnosticDeclaration(tokens, record.declaration);
    appendOperatorStageMeasurementMetadata(tokens, record.stage);
    appendMeshBoundaryIntegral(tokens, record.state.interface_marker);
    appendTwoFluidGeometryRevision(tokens, record.state.geometry_revision);
    // local_mesh_revision is rank-local staleness provenance. It is checked
    // against the live operator-stage stamp but excluded from MPI consensus.
    appendTwoFluidDiagnosticState(tokens, record.state.diagnostics);
    appendMeshBoundaryToken(
        tokens, record.momentum_reconciliation.has_value() ? 1u : 0u);
    if (record.momentum_reconciliation.has_value()) {
        appendTwoFluidMomentumReconciliation(
            tokens, *record.momentum_reconciliation);
    }
    appendMeshBoundaryToken(tokens, record.numerics.has_value() ? 1u : 0u);
    if (record.numerics.has_value()) {
        appendTwoFluidAcceptedStageNumerics(tokens, *record.numerics);
    }
}

[[nodiscard]] bool twoFluidTelemetryStringPresent(
    const std::string& value) noexcept
{
    return value.find_first_not_of(" \t\r\n") != std::string::npos;
}

[[nodiscard]] bool twoFluidTelemetryFiniteNonnegative(Real value) noexcept
{
    return std::isfinite(value) && value >= Real{0.0};
}

[[nodiscard]] bool twoFluidTelemetryWithinGuard(
    Real observed,
    Real maximum) noexcept
{
    if (!twoFluidTelemetryFiniteNonnegative(observed) ||
        !twoFluidTelemetryFiniteNonnegative(maximum)) {
        return false;
    }
    const Real scale = std::max({Real{1.0}, observed, maximum});
    return observed <=
           maximum + Real{1024.0} * std::numeric_limits<Real>::epsilon() *
                         scale;
}

[[nodiscard]] TwoFluidAcceptedStageAggregationGeometryTelemetry
compactTwoFluidAcceptedStageAggregationGeometry(
    const constraints::SmallCutAggregationCanonicalGeometryIdentity& source)
{
    return TwoFluidAcceptedStageAggregationGeometryTelemetry{
        .kind = source.kind,
        .available = source.available,
        .communicator_fingerprint_consensus_validated =
            source.communicator_fingerprint_consensus_validated,
        .source_id = source.source_id,
        .domain_id = source.domain_id,
        .interface_marker = source.interface_marker,
        .isovalue = source.isovalue,
        .source_layout_revision = source.source_layout_revision,
        .source_value_revision = source.source_value_revision,
        .quadrature_policy_key = source.quadrature_policy_key,
        .snapshot_revision_key = source.snapshot_revision_key,
        .distributed_mesh_geometry_revision =
            source.distributed_mesh_geometry_revision,
        .distributed_mesh_topology_revision =
            source.distributed_mesh_topology_revision,
        .distributed_ownership_revision =
            source.distributed_ownership_revision,
        .distributed_numbering_revision =
            source.distributed_numbering_revision,
        .canonical_fingerprint = source.canonical_fingerprint,
    };
}

void validateTwoFluidAcceptedStageAggregationGeometry(
    const TwoFluidAcceptedStageAggregationGeometryTelemetry& identity,
    const interfaces::FreeSurfaceGeometryRevision& expected)
{
    const bool common_identity_matches =
        identity.available &&
        identity.communicator_fingerprint_consensus_validated &&
        identity.source_id == expected.source_id &&
        identity.domain_id == expected.domain_id &&
        identity.interface_marker == expected.interface_marker &&
        identity.isovalue == expected.isovalue &&
        identity.source_layout_revision == expected.source_layout_revision &&
        identity.source_value_revision == expected.source_value_revision &&
        identity.quadrature_policy_key == expected.quadrature_policy_key;
    const bool authoritative =
        identity.kind == constraints::
                             SmallCutAggregationGeometryIdentityKind::
                                 AuthoritativeFreeSurfaceSnapshot &&
        identity.snapshot_revision_key == expected.snapshot_revision_key &&
        identity.distributed_mesh_geometry_revision ==
            expected.mesh_geometry_revision &&
        identity.distributed_mesh_topology_revision ==
            expected.mesh_topology_revision &&
        identity.distributed_ownership_revision ==
            expected.ownership_revision &&
        identity.distributed_numbering_revision ==
            expected.numbering_revision;
    const bool generated =
        identity.kind == constraints::
                             SmallCutAggregationGeometryIdentityKind::
                                 GeneratedPublicationSource &&
        identity.snapshot_revision_key == 0u &&
        identity.distributed_mesh_geometry_revision == 0u &&
        identity.distributed_mesh_topology_revision == 0u &&
        identity.distributed_ownership_revision == 0u &&
        identity.distributed_numbering_revision == 0u;
    FE_THROW_IF(
        !common_identity_matches || (!authoritative && !generated),
        InvalidArgumentException,
        "FESystem::bindPendingTwoFluidAcceptedStageNumerics: aggregation geometry identity does not certify the pending interface source state");
}

[[nodiscard]] TwoFluidAcceptedStageAggregationTransitionTelemetry
compactTwoFluidAcceptedStageAggregationTransition(
    const constraints::SmallCutAggregationTopologyTransitionReport& source)
{
    FE_THROW_IF(
        !twoFluidTelemetryFiniteNonnegative(
            source.canonical_rootless_active_physical_volume_before) ||
            !twoFluidTelemetryFiniteNonnegative(
                source.canonical_rootless_active_physical_volume_after) ||
            !std::isfinite(
                source.canonical_rootless_active_physical_volume_delta),
        InvalidArgumentException,
        "FESystem::bindPendingTwoFluidAcceptedStageNumerics: aggregation topology-transition volumes are invalid");
    return TwoFluidAcceptedStageAggregationTransitionTelemetry{
        .geometry_identity_before =
            compactTwoFluidAcceptedStageAggregationGeometry(
                source.geometry_identity_before),
        .geometry_identity_after =
            compactTwoFluidAcceptedStageAggregationGeometry(
                source.geometry_identity_after),
        .canonical_feature_class_fingerprint_before =
            source.canonical_feature_class_fingerprint_before,
        .canonical_feature_class_fingerprint_after =
            source.canonical_feature_class_fingerprint_after,
        .canonical_slave_set_fingerprint_before =
            source.canonical_slave_set_fingerprint_before,
        .canonical_slave_set_fingerprint_after =
            source.canonical_slave_set_fingerprint_after,
        .canonical_active_feature_count_before =
            source.canonical_active_feature_count_before,
        .canonical_active_feature_count_after =
            source.canonical_active_feature_count_after,
        .canonical_features_entered = source.canonical_features_entered,
        .canonical_features_exited = source.canonical_features_exited,
        .canonical_features_persisted = source.canonical_features_persisted,
        .canonical_feature_classification_changes =
            source.canonical_feature_classification_changes,
        .canonical_features_became_rooted =
            source.canonical_features_became_rooted,
        .canonical_features_became_rootless =
            source.canonical_features_became_rootless,
        .canonical_aggregate_slaves_before =
            source.canonical_aggregate_slaves_before,
        .canonical_aggregate_slaves_after =
            source.canonical_aggregate_slaves_after,
        .canonical_aggregate_slaves_entered =
            source.canonical_aggregate_slaves_entered,
        .canonical_aggregate_slaves_left =
            source.canonical_aggregate_slaves_left,
        .canonical_rootless_active_physical_volume_before =
            source.canonical_rootless_active_physical_volume_before,
        .canonical_rootless_active_physical_volume_after =
            source.canonical_rootless_active_physical_volume_after,
        .canonical_rootless_active_physical_volume_delta =
            source.canonical_rootless_active_physical_volume_delta,
        .canonical_topology_changed = source.canonical_topology_changed,
    };
}

[[nodiscard]] TwoFluidAcceptedStageAggregationTelemetry
compactTwoFluidAcceptedStageAggregation(
    const constraints::SmallCutAggregationRefreshReport& source,
    const interfaces::FreeSurfaceGeometryRevision& expected_geometry)
{
    const auto geometry =
        compactTwoFluidAcceptedStageAggregationGeometry(
            source.geometry_identity);
    validateTwoFluidAcceptedStageAggregationGeometry(
        geometry, expected_geometry);
    FE_THROW_IF(
        source.maximum_observed_root_path >
                source.maximum_root_path_length ||
            !twoFluidTelemetryWithinGuard(
                source.maximum_observed_reference_extrapolation,
                source.maximum_reference_extrapolation_distance) ||
            !twoFluidTelemetryWithinGuard(
                source.maximum_observed_absolute_coefficient,
                source.maximum_absolute_coefficient) ||
            !twoFluidTelemetryWithinGuard(
                source.maximum_observed_row_l1_norm,
                source.maximum_row_l1_norm) ||
            source.canonical_rooted_candidate_vertices +
                    source.canonical_rootless_candidate_vertices !=
                source.canonical_candidate_vertices ||
            source.canonical_rooted_active_feature_count +
                    source.canonical_rootless_active_feature_count !=
                source.canonical_active_feature_count ||
            !twoFluidTelemetryFiniteNonnegative(
                source.canonical_rootless_active_physical_volume),
        InvalidArgumentException,
        "FESystem::bindPendingTwoFluidAcceptedStageNumerics: aggregation guards or canonical counts are invalid");

    TwoFluidAcceptedStageAggregationTelemetry compact{
        .geometry_identity = geometry,
        .canonical_feature_class_fingerprint =
            source.canonical_feature_class_fingerprint,
        .canonical_slave_set_fingerprint =
            source.canonical_slave_set_fingerprint,
        .maximum_root_path_length = source.maximum_root_path_length,
        .maximum_observed_root_path = source.maximum_observed_root_path,
        .root_path_guard_rejections = source.root_path_guard_rejections,
        .maximum_reference_extrapolation_distance =
            source.maximum_reference_extrapolation_distance,
        .maximum_observed_reference_extrapolation =
            source.maximum_observed_reference_extrapolation,
        .extrapolation_guard_rejections =
            source.extrapolation_guard_rejections,
        .maximum_absolute_coefficient =
            source.maximum_absolute_coefficient,
        .maximum_observed_absolute_coefficient =
            source.maximum_observed_absolute_coefficient,
        .maximum_row_l1_norm = source.maximum_row_l1_norm,
        .maximum_observed_row_l1_norm =
            source.maximum_observed_row_l1_norm,
        .line_guard_rejections = source.line_guard_rejections,
        .canonical_candidate_vertices =
            source.canonical_candidate_vertices,
        .canonical_rooted_candidate_vertices =
            source.canonical_rooted_candidate_vertices,
        .canonical_rootless_candidate_vertices =
            source.canonical_rootless_candidate_vertices,
        .canonical_owned_aggregate_dofs =
            source.canonical_owned_aggregate_dofs,
        .canonical_owned_pinned_dofs =
            source.canonical_owned_pinned_dofs,
        .canonical_strong_suppressed_dofs =
            source.canonical_strong_suppressed_dofs,
        .canonical_active_feature_count =
            source.canonical_active_feature_count,
        .canonical_rooted_active_feature_count =
            source.canonical_rooted_active_feature_count,
        .canonical_rootless_active_feature_count =
            source.canonical_rootless_active_feature_count,
        .canonical_rootless_active_physical_volume =
            source.canonical_rootless_active_physical_volume,
    };
    if (source.canonical_topology_transition.has_value()) {
        compact.canonical_topology_transition =
            compactTwoFluidAcceptedStageAggregationTransition(
                *source.canonical_topology_transition);
        FE_THROW_IF(
            compact.canonical_topology_transition->geometry_identity_after !=
                compact.geometry_identity,
            InvalidArgumentException,
            "FESystem::bindPendingTwoFluidAcceptedStageNumerics: aggregation topology transition does not end at the report geometry identity");
    }
    return compact;
}

void validateTwoFluidAcceptedStageSolveTelemetry(
    const TwoFluidAcceptedStageSolveTelemetry& solve)
{
    const auto& nonlinear = solve.nonlinear;
    const std::array<int, 6> nonlinear_counts{
        nonlinear.iterations,
        nonlinear.outer_iterations,
        nonlinear.inner_iterations_total,
        nonlinear.outer_dynamic_relaxation_updates,
        nonlinear.outer_dynamic_relaxation_safeguards,
        nonlinear.outer_dynamic_relaxation_resets};
    const std::array<Real, 10> nonlinear_scalars{
        nonlinear.outer_state_change_norm,
        nonlinear.outer_dynamic_relaxation_factor,
        nonlinear.outer_relaxed_state_change_norm,
        nonlinear.initial_residual_norm,
        nonlinear.final_residual_norm,
        nonlinear.initial_field_residual_norm,
        nonlinear.final_field_residual_norm,
        nonlinear.initial_auxiliary_residual_norm,
        nonlinear.final_auxiliary_residual_norm,
        nonlinear.outer_raw_contraction_ratio.value_or(Real{0.0})};
    FE_THROW_IF(
        !nonlinear.converged ||
            std::any_of(
                nonlinear_counts.begin(), nonlinear_counts.end(),
                [](int value) { return value < 0; }) ||
            std::any_of(
                nonlinear_scalars.begin(),
                nonlinear_scalars.end(),
                [](Real value) {
                    return !twoFluidTelemetryFiniteNonnegative(value);
                }) ||
            !twoFluidTelemetryStringPresent(nonlinear.reason),
        InvalidArgumentException,
        "FESystem::bindPendingTwoFluidAcceptedStageNumerics: nonlinear solve evidence is invalid or unconverged");

    const auto& linear = solve.linear;
    const std::array<int, 8> linear_counts{
        linear.iterations,
        linear.blockschur_outer_iterations,
        linear.blockschur_momentum_solve_calls,
        linear.blockschur_momentum_iterations,
        linear.blockschur_momentum_restart_cycles,
        linear.blockschur_schur_solve_calls,
        linear.blockschur_schur_iterations,
        linear.blockschur_true_residual_retries};
    const std::array<std::uint64_t, 7> linear_collective_counts{
        linear.collective_calls,
        linear.collective_words,
        linear.blockschur_collective_calls_max_per_outer,
        linear.blockschur_momentum_collective_calls,
        linear.blockschur_momentum_collective_words,
        linear.blockschur_schur_collective_calls,
        linear.blockschur_schur_collective_words};
    const std::array<Real, 3> linear_scalars{
        linear.initial_residual_norm,
        linear.final_residual_norm,
        linear.relative_residual};
    const bool valid_unattempted_exact_entry =
        nonlinear.iterations == 0 &&
        nonlinear.inner_iterations_total == 0 &&
        std::all_of(
            linear_counts.begin(), linear_counts.end(),
            [](int value) { return value == 0; }) &&
        std::all_of(
            linear_collective_counts.begin(),
            linear_collective_counts.end(),
            [](std::uint64_t value) { return value == 0u; }) &&
        std::all_of(
            linear_scalars.begin(), linear_scalars.end(),
            [](Real value) { return value == Real{0.0}; });
    FE_THROW_IF(
        !linear.converged ||
            linear.numerical_breakdown ||
            std::any_of(
                linear_counts.begin(), linear_counts.end(),
                [](int value) { return value < 0; }) ||
            std::any_of(
                linear_scalars.begin(), linear_scalars.end(),
                [](Real value) {
                    return !twoFluidTelemetryFiniteNonnegative(value);
                }) ||
            !twoFluidTelemetryStringPresent(linear.reason) ||
            (!linear.attempted && !valid_unattempted_exact_entry),
        InvalidArgumentException,
        "FESystem::bindPendingTwoFluidAcceptedStageNumerics: linear solve evidence is invalid, unjustifiably unattempted, unconverged, or numerically broken down");
}

} // namespace

TwoFluidStageRecords::TwoFluidStageRecords(FESystem& system)
    : system_(&system),
      two_fluid_accepted_stage_declarations_(
          system.attachment<TwoFluidStageRecordStore>().declarations),
      pending_two_fluid_accepted_stage_diagnostics_(
          system.attachment<TwoFluidStageRecordStore>().pending),
      two_fluid_accepted_stage_diagnostic_history_(
          system.attachment<TwoFluidStageRecordStore>().history),
      two_fluid_accepted_stage_declarations_frozen_(
          system.attachment<TwoFluidStageRecordStore>().declarations_frozen),
      two_fluid_accepted_stage_transaction_active_(
          system.attachment<TwoFluidStageRecordStore>().transaction_active),
      field_registry_{&system}
{
}

ConstTwoFluidStageRecords::ConstTwoFluidStageRecords(
    const FESystem& system) noexcept
    : store_(system.findAttachment<TwoFluidStageRecordStore>())
{
}

std::span<const TwoFluidAcceptedStageDiagnosticDeclaration>
ConstTwoFluidStageRecords::twoFluidAcceptedStageDiagnosticDeclarations()
    const noexcept
{
  if (store_ == nullptr) {
    return {};
  }
  return store_->declarations;
}

std::span<const TwoFluidAcceptedStageDiagnosticHistoryRecord>
ConstTwoFluidStageRecords::pendingTwoFluidAcceptedStageDiagnostics()
    const noexcept
{
  if (store_ == nullptr) {
    return {};
  }
  return store_->pending;
}

std::span<const TwoFluidAcceptedStageDiagnosticHistoryRecord>
ConstTwoFluidStageRecords::twoFluidAcceptedStageDiagnosticHistory()
    const noexcept
{
  if (store_ == nullptr) {
    return {};
  }
  return store_->history;
}

TwoFluidStageRecords twoFluidStageRecords(FESystem& system)
{
  return TwoFluidStageRecords(system);
}

ConstTwoFluidStageRecords twoFluidStageRecords(const FESystem& system) noexcept
{
  return ConstTwoFluidStageRecords(system);
}

void TwoFluidStageRecords::declareTwoFluidAcceptedStageDiagnostics(
    TwoFluidAcceptedStageDiagnosticDeclaration declaration)
{
    FE_THROW_IF(
        isSetup() || two_fluid_accepted_stage_declarations_frozen_ ||
            two_fluid_accepted_stage_transaction_active_ ||
            !pending_two_fluid_accepted_stage_diagnostics_.empty() ||
            !two_fluid_accepted_stage_diagnostic_history_.empty(),
        InvalidArgumentException,
        "FESystem::declareTwoFluidAcceptedStageDiagnostics: declarations "
        "must be complete before setup or any stage/history transaction");
    const auto& parameters = declaration.parameters;
    const auto finite_positive = [](Real value) {
        return std::isfinite(value) && value > Real{0.0};
    };
    const auto finite_vector = [](const std::array<Real, 3>& values) {
        return std::all_of(values.begin(), values.end(), [](Real value) {
            return std::isfinite(value);
        });
    };
    FE_THROW_IF(
        declaration.interface_marker < 0 ||
            declaration.interface_marker != parameters.interface_marker ||
            (parameters.dimension != 2 && parameters.dimension != 3) ||
            declaration.operator_tag.empty() ||
            declaration.geometry_domain_id.find_first_not_of(" \t\r\n") ==
                std::string::npos ||
            declaration.owner_component.find_first_not_of(" \t\r\n") ==
                std::string::npos ||
            !std::isfinite(declaration.level_set_isovalue) ||
            !finite_positive(parameters.negative_density) ||
            !finite_positive(parameters.positive_density) ||
            !finite_positive(parameters.negative_viscosity) ||
            !finite_positive(parameters.positive_viscosity) ||
            !finite_positive(parameters.nitsche_gamma) ||
            !std::isfinite(parameters.surface_tension) ||
            parameters.surface_tension < Real{0.0} ||
            (declaration.require_accepted_stage_numerics &&
             !finite_positive(
                 declaration.pressure_stabilization_coefficient)) ||
            (!declaration.require_accepted_stage_numerics &&
             (!std::isfinite(
                  declaration.pressure_stabilization_coefficient) ||
              declaration.pressure_stabilization_coefficient <
                  Real{0.0})) ||
            (parameters.prescribed_pressure_jump.has_value() &&
             !std::isfinite(*parameters.prescribed_pressure_jump)) ||
            !finite_vector(parameters.body_force) ||
            (parameters.dimension == 2 &&
             parameters.body_force[2] != Real{0.0}) ||
            (parameters.prescribed_viscous_traction_jump.has_value() &&
             (!finite_vector(
                  *parameters.prescribed_viscous_traction_jump) ||
              (parameters.dimension == 2 &&
               (*parameters.prescribed_viscous_traction_jump)[2] !=
                   Real{0.0}))),
        InvalidArgumentException,
        "FESystem::declareTwoFluidAcceptedStageDiagnostics: declaration "
        "metadata and material/interface coefficients are invalid");

    const std::array<FieldId, 5> fields{
        declaration.level_set_field,
        declaration.negative_velocity_field,
        declaration.negative_pressure_field,
        declaration.positive_velocity_field,
        declaration.positive_pressure_field};
    for (std::size_t index = 0u; index < fields.size(); ++index) {
        FE_THROW_IF(
            !field_registry_.has(fields[index]),
            InvalidArgumentException,
            "FESystem::declareTwoFluidAcceptedStageDiagnostics: every "
            "declared field must already exist");
        for (std::size_t other = index + 1u; other < fields.size(); ++other) {
            FE_THROW_IF(
                fields[index] == fields[other],
                InvalidArgumentException,
                "FESystem::declareTwoFluidAcceptedStageDiagnostics: phase "
                "and level-set fields must be distinct");
        }
    }
    const auto& level_set = field_registry_.get(declaration.level_set_field);
    const auto& negative_velocity =
        field_registry_.get(declaration.negative_velocity_field);
    const auto& negative_pressure =
        field_registry_.get(declaration.negative_pressure_field);
    const auto& positive_velocity =
        field_registry_.get(declaration.positive_velocity_field);
    const auto& positive_pressure =
        field_registry_.get(declaration.positive_pressure_field);
    FE_THROW_IF(
        level_set.components != 1 || level_set.space == nullptr ||
            level_set.space->value_dimension() != 1 ||
            level_set.space->topological_dimension() !=
                parameters.dimension ||
            negative_velocity.source_kind != FieldSourceKind::Unknown ||
            positive_velocity.source_kind != FieldSourceKind::Unknown ||
            negative_pressure.source_kind != FieldSourceKind::Unknown ||
            positive_pressure.source_kind != FieldSourceKind::Unknown ||
            negative_velocity.components != parameters.dimension ||
            positive_velocity.components != parameters.dimension ||
            negative_pressure.components != 1 ||
            positive_pressure.components != 1 ||
            negative_velocity.space == nullptr ||
            positive_velocity.space == nullptr ||
            negative_pressure.space == nullptr ||
            positive_pressure.space == nullptr,
        InvalidArgumentException,
        "FESystem::declareTwoFluidAcceptedStageDiagnostics: field roles and "
        "dimensions do not describe two unknown velocity/pressure pairs and "
        "one scalar level set");

    const auto existing = std::find_if(
        two_fluid_accepted_stage_declarations_.begin(),
        two_fluid_accepted_stage_declarations_.end(),
        [&](const auto& candidate) {
            return candidate.interface_marker == declaration.interface_marker;
        });
    if (existing != two_fluid_accepted_stage_declarations_.end()) {
        FE_THROW_IF(
            *existing != declaration,
            InvalidArgumentException,
            "FESystem::declareTwoFluidAcceptedStageDiagnostics: interface "
            "marker already has a conflicting owner");
        return;
    }
    two_fluid_accepted_stage_declarations_.push_back(std::move(declaration));
    std::sort(
        two_fluid_accepted_stage_declarations_.begin(),
        two_fluid_accepted_stage_declarations_.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.interface_marker < rhs.interface_marker;
        });
}

void TwoFluidStageRecords::stageTwoFluidAcceptedStageDiagnostics(
    OperatorStageMeasurementMetadata metadata,
    std::span<const AcceptedTwoFluidStageDiagnosticState> states)
{
    const auto collective = meshBoundaryCollectiveContext(*system_);
    MeshBoundaryConsensusTokens local_tokens;
    MeshBoundaryConsensusTokens minimum_tokens;
    MeshBoundaryConsensusTokens maximum_tokens;
    std::exception_ptr local_exception;
    std::vector<TwoFluidAcceptedStageDiagnosticHistoryRecord> staged;

    try {
        std::sort(
            metadata.derivative_fields.begin(),
            metadata.derivative_fields.end());
        metadata.derivative_fields.erase(
            std::unique(metadata.derivative_fields.begin(),
                        metadata.derivative_fields.end()),
            metadata.derivative_fields.end());
        const bool endpoint_scheme =
            metadata.scheme_name == "BackwardEuler" ||
            metadata.scheme_name == "DG0";
        const bool generalized_alpha =
            metadata.scheme_name == "GeneralizedAlphaFirstOrder";
        FE_THROW_IF(
            !isSetup() || !dofHandler().isFinalized() ||
                two_fluid_accepted_stage_transaction_active_ ||
                !pending_two_fluid_accepted_stage_diagnostics_.empty() ||
                (!endpoint_scheme && !generalized_alpha) ||
                metadata.temporal_order != 1 ||
                metadata.prospective_accepted_step == 0u ||
                !std::isfinite(metadata.step_start_time) ||
                !std::isfinite(metadata.step_end_time) ||
                !std::isfinite(metadata.state_time) ||
                !std::isfinite(metadata.rate_time) ||
                !std::isfinite(metadata.dt) || !(metadata.dt > Real{0.0}) ||
                metadata.state_revision == 0u || metadata.rate_revision == 0u ||
                states.size() !=
                    two_fluid_accepted_stage_declarations_.size(),
            InvalidArgumentException,
            "FESystem::stageTwoFluidAcceptedStageDiagnostics: setup, "
            "operator-stage metadata, pending state, or declaration coverage "
            "is invalid");
        FE_THROW_IF(
            !operatorStageTimesMatch(
                metadata.step_end_time,
                metadata.step_start_time + metadata.dt),
            InvalidArgumentException,
            "FESystem::stageTwoFluidAcceptedStageDiagnostics: step end does "
            "not match start plus dt");
        if (endpoint_scheme) {
            FE_THROW_IF(
                metadata.generalized_alpha.has_value() ||
                    !operatorStageTimesMatch(
                        metadata.state_time, metadata.step_end_time) ||
                    !operatorStageTimesMatch(
                        metadata.rate_time, metadata.step_end_time),
                InvalidArgumentException,
                "FESystem::stageTwoFluidAcceptedStageDiagnostics: endpoint "
                "schemes require endpoint state/rate times");
        } else {
            FE_THROW_IF(
                !metadata.generalized_alpha.has_value() ||
                    !std::isfinite(metadata.generalized_alpha->alpha_f) ||
                    !std::isfinite(metadata.generalized_alpha->alpha_m) ||
                    !operatorStageTimesMatch(
                        metadata.state_time,
                        metadata.step_start_time +
                            metadata.generalized_alpha->alpha_f * metadata.dt) ||
                    !operatorStageTimesMatch(
                        metadata.rate_time,
                        metadata.step_start_time +
                            metadata.generalized_alpha->alpha_m * metadata.dt),
                InvalidArgumentException,
                "FESystem::stageTwoFluidAcceptedStageDiagnostics: "
                "generalized-alpha times do not match their coordinates");
        }
        for (const auto field : metadata.derivative_fields) {
            FE_THROW_IF(
                !field_registry_.has(field),
                InvalidArgumentException,
                "FESystem::stageTwoFluidAcceptedStageDiagnostics: derivative "
                "metadata names an unknown field");
        }
        if (!two_fluid_accepted_stage_declarations_.empty()) {
            FE_THROW_IF(
                !metadata.expected_stage_geometry.has_value() ||
                    !meshAccess().revisionTrackingAvailable() ||
                    captureOperatorStageGeometry(meshAccess()) !=
                        *metadata.expected_stage_geometry,
                InvalidArgumentException,
                "FESystem::stageTwoFluidAcceptedStageDiagnostics: live "
                "geometry does not match the operator-stage observation");
        }

        const auto finite_vector = [](const std::array<Real, 3>& values) {
            return std::all_of(values.begin(), values.end(), [](Real value) {
                return std::isfinite(value);
            });
        };
        const auto near = [](Real lhs, Real rhs) {
            if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
                return false;
            }
            const Real scale = std::max(
                {Real{1.0}, std::abs(lhs), std::abs(rhs)});
            return std::abs(lhs - rhs) <=
                   Real{1024.0} * std::numeric_limits<Real>::epsilon() * scale;
        };
        const auto near_difference = [](Real observed,
                                        Real negative,
                                        Real positive) {
            if (!std::isfinite(observed) || !std::isfinite(negative) ||
                !std::isfinite(positive)) {
                return false;
            }
            const Real scale = std::max(
                {Real{1.0},
                 std::abs(observed),
                 std::abs(negative),
                 std::abs(positive)});
            return std::abs(observed - (negative - positive)) <=
                   Real{2048.0} * std::numeric_limits<Real>::epsilon() * scale;
        };
        const auto bounded_moment = [](Real moment_squared, Real bound) {
            const Real scale = std::max(
                {Real{1.0}, std::abs(moment_squared), std::abs(bound)});
            return std::isfinite(moment_squared) && std::isfinite(bound) &&
                   moment_squared <=
                       bound + Real{2048.0} *
                                   std::numeric_limits<Real>::epsilon() * scale;
        };
        const auto squared_norm = [](const std::array<Real, 3>& values) {
            Real result{0.0};
            for (const auto value : values) {
                result += value * value;
            }
            return result;
        };
        const auto dot = [](const std::array<Real, 3>& lhs,
                            const std::array<Real, 3>& rhs) {
            Real result{0.0};
            for (std::size_t component = 0u; component < 3u; ++component) {
                result += lhs[component] * rhs[component];
            }
            return result;
        };
        for (std::size_t index = 0u; index < states.size(); ++index) {
            const auto& declaration =
                two_fluid_accepted_stage_declarations_[index];
            const auto& supplied = states[index];
            const auto& revision = supplied.geometry_revision;
            const auto& local_revision = supplied.local_mesh_revision;
            const auto& state = supplied.diagnostics;
            const auto& parameters = declaration.parameters;
            const auto& expected = *metadata.expected_stage_geometry;
            const auto expected_source_id =
                interfaces::LevelSetInterfaceSource::fromField(
                    declaration.level_set_field)
                    .identifier();
            const std::array<Real, 19> scalars{
                state.interface_measure,
                state.velocity_jump_squared,
                state.normal_velocity_jump_squared,
                state.tangential_velocity_jump_squared,
                state.negative_normal_flux,
                state.positive_normal_flux,
                state.normal_flux_jump,
                state.negative_mass_flux,
                state.positive_mass_flux,
                state.traction_jump_normal_integral,
                state.traction_jump_squared,
                state.viscous_traction_jump_squared,
                state.pressure_jump_integral,
                state.mean_pressure_jump,
                state.pressure_jump_squared,
                state.surface_energy_work,
                state.nitsche_consistency_work,
                state.nitsche_adjoint_work,
                state.nitsche_penalty_work};
            FE_THROW_IF(
                (index != 0u &&
                 declaration.interface_marker <=
                     two_fluid_accepted_stage_declarations_[index - 1u]
                         .interface_marker) ||
                    supplied.interface_marker != declaration.interface_marker ||
                    revision.source_id != expected_source_id ||
                    revision.interface_marker != declaration.interface_marker ||
                    revision.domain_id != declaration.geometry_domain_id ||
                    revision.isovalue != declaration.level_set_isovalue ||
                    !revision.complete() ||
                    revision.snapshot_revision_key !=
                        state.snapshot_revision_key ||
                    local_revision.mesh_geometry_revision !=
                        expected.geometry_revision ||
                    local_revision.mesh_topology_revision !=
                        expected.topology_revision ||
                    local_revision.ownership_revision !=
                        expected.ownership_revision ||
                    local_revision.numbering_revision !=
                        expected.numbering_revision ||
                    state.interface_quadrature_point_count == 0u ||
                    !(state.interface_measure > Real{0.0}) ||
                    !std::all_of(
                        scalars.begin(), scalars.end(), [](Real value) {
                            return std::isfinite(value);
                        }) ||
                    !finite_vector(state.interface_normal_integral) ||
                    !finite_vector(state.negative_traction_integral) ||
                    !finite_vector(state.positive_traction_integral) ||
                    !finite_vector(state.traction_jump_integral) ||
                    !finite_vector(
                        state.negative_viscous_traction_integral) ||
                    !finite_vector(
                        state.positive_viscous_traction_integral) ||
                    !finite_vector(state.viscous_traction_jump_integral) ||
                    state.velocity_jump_squared < Real{0.0} ||
                    state.normal_velocity_jump_squared < Real{0.0} ||
                    state.tangential_velocity_jump_squared < Real{0.0} ||
                    state.traction_jump_squared < Real{0.0} ||
                    state.viscous_traction_jump_squared < Real{0.0} ||
                    state.pressure_jump_squared < Real{0.0} ||
                    state.nitsche_penalty_work < Real{0.0} ||
                    !near(
                        state.velocity_jump_squared,
                        state.normal_velocity_jump_squared +
                            state.tangential_velocity_jump_squared) ||
                    !near(
                        state.normal_flux_jump,
                        state.negative_normal_flux -
                            state.positive_normal_flux) ||
                    !near(
                        state.negative_mass_flux,
                        parameters.negative_density *
                            state.negative_normal_flux) ||
                    !near(
                        state.positive_mass_flux,
                        parameters.positive_density *
                            state.positive_normal_flux) ||
                    !near(
                        state.mean_pressure_jump,
                        state.pressure_jump_integral /
                            state.interface_measure) ||
                    !near(
                        state.nitsche_consistency_work,
                        state.nitsche_adjoint_work) ||
                    state.transient_penalty_effective_dt.has_value() !=
                        parameters.include_transient_penalty ||
                    (state.transient_penalty_effective_dt.has_value() &&
                     (!std::isfinite(
                          *state.transient_penalty_effective_dt) ||
                      !(*state.transient_penalty_effective_dt > Real{0.0}) ||
                      !near(
                          *state.transient_penalty_effective_dt,
                          metadata.state_time -
                              metadata.step_start_time))) ||
                    !bounded_moment(
                        state.normal_flux_jump * state.normal_flux_jump,
                        state.interface_measure *
                            state.normal_velocity_jump_squared) ||
                    !bounded_moment(
                        state.pressure_jump_integral *
                            state.pressure_jump_integral,
                        state.interface_measure *
                            state.pressure_jump_squared) ||
                    !bounded_moment(
                        squared_norm(state.traction_jump_integral),
                        state.interface_measure *
                            state.traction_jump_squared) ||
                    !bounded_moment(
                        squared_norm(state.interface_normal_integral),
                        state.interface_measure *
                            state.interface_measure) ||
                    !bounded_moment(
                        squared_norm(state.viscous_traction_jump_integral),
                        state.interface_measure *
                            state.viscous_traction_jump_squared) ||
                    state.prescribed_pressure_jump_error_squared.has_value() !=
                        parameters.prescribed_pressure_jump.has_value() ||
                    state.prescribed_viscous_traction_jump_error_squared
                            .has_value() !=
                        parameters.prescribed_viscous_traction_jump
                            .has_value() ||
                    state.prescribed_stress_jump_residual_squared.has_value() !=
                        (parameters.prescribed_pressure_jump.has_value() ||
                         parameters.prescribed_viscous_traction_jump
                             .has_value()),
                InvalidArgumentException,
                "FESystem::stageTwoFluidAcceptedStageDiagnostics: supplied "
                "interface state violates its declaration, snapshot, raw "
                "identity, or applicability contract");
            const auto validate_phase = [&](const auto& phase,
                                            geometry::CutIntegrationSide side,
                                            Real density) {
                if (phase.side != side ||
                    phase.quadrature_point_count == 0u ||
                    phase.density != density ||
                    !std::isfinite(phase.volume) ||
                    !(phase.volume > Real{0.0}) ||
                    !std::isfinite(phase.mass) ||
                    !near(phase.mass, density * phase.volume) ||
                    !finite_vector(phase.momentum) ||
                    !finite_vector(phase.mean_velocity) ||
                    !std::isfinite(phase.maximum_quadrature_speed) ||
                    phase.maximum_quadrature_speed < Real{0.0} ||
                    !std::isfinite(phase.kinetic_energy) ||
                    phase.kinetic_energy < Real{0.0} ||
                    !std::isfinite(phase.pressure_integral) ||
                    !std::isfinite(phase.mean_pressure) ||
                    !std::isfinite(phase.pressure_squared_integral) ||
                    phase.pressure_squared_integral < Real{0.0} ||
                    !finite_vector(phase.pressure_gradient_integral) ||
                    !finite_vector(phase.body_force_density_integral) ||
                    !finite_vector(phase.hydrostatic_residual_integral) ||
                    !std::isfinite(phase.hydrostatic_residual_squared) ||
                    phase.hydrostatic_residual_squared < Real{0.0} ||
                    !near(
                        phase.mean_pressure,
                        phase.pressure_integral / phase.volume) ||
                    !bounded_moment(
                        squared_norm(phase.mean_velocity),
                        phase.maximum_quadrature_speed *
                            phase.maximum_quadrature_speed) ||
                    !bounded_moment(
                        Real{2.0} * phase.kinetic_energy / density,
                        phase.volume * phase.maximum_quadrature_speed *
                            phase.maximum_quadrature_speed) ||
                    !bounded_moment(
                        squared_norm(phase.momentum),
                        Real{2.0} * phase.mass * phase.kinetic_energy) ||
                    !bounded_moment(
                        phase.pressure_integral * phase.pressure_integral,
                        phase.volume * phase.pressure_squared_integral) ||
                    !bounded_moment(
                        squared_norm(
                            phase.hydrostatic_residual_integral),
                        phase.volume *
                            phase.hydrostatic_residual_squared)) {
                    return false;
                }
                for (std::size_t component = 0u; component < 3u;
                     ++component) {
                    if (!near(
                            phase.momentum[component],
                            density * phase.volume *
                                phase.mean_velocity[component])) {
                        return false;
                    }
                    const Real expected_body_force =
                        density * parameters.body_force[component] *
                        phase.volume;
                    if (!near(
                            phase.body_force_density_integral[component],
                            expected_body_force) ||
                        !near_difference(
                            phase.hydrostatic_residual_integral[component],
                            phase.pressure_gradient_integral[component],
                            phase.body_force_density_integral[component])) {
                        return false;
                    }
                }
                return true;
            };
            FE_THROW_IF(
                !validate_phase(
                    state.negative_phase,
                    geometry::CutIntegrationSide::Negative,
                    parameters.negative_density) ||
                    !validate_phase(
                        state.positive_phase,
                        geometry::CutIntegrationSide::Positive,
                        parameters.positive_density) ||
                    (state.prescribed_pressure_jump_error_squared.has_value() &&
                     (!std::isfinite(
                          *state.prescribed_pressure_jump_error_squared) ||
                      *state.prescribed_pressure_jump_error_squared <
                          Real{0.0})) ||
                    (state.prescribed_stress_jump_residual_squared.has_value() &&
                     (!std::isfinite(
                          *state.prescribed_stress_jump_residual_squared) ||
                      *state.prescribed_stress_jump_residual_squared <
                          Real{0.0})) ||
                    (state.prescribed_viscous_traction_jump_error_squared
                         .has_value() &&
                     (!std::isfinite(
                          *state
                               .prescribed_viscous_traction_jump_error_squared) ||
                      *state.prescribed_viscous_traction_jump_error_squared <
                          Real{0.0})),
                InvalidArgumentException,
                "FESystem::stageTwoFluidAcceptedStageDiagnostics: supplied "
                "phase state or prescribed-target error is invalid");
            for (std::size_t component = 0u; component < 3u; ++component) {
                FE_THROW_IF(
                    !near_difference(
                        state.traction_jump_integral[component],
                        state.negative_traction_integral[component],
                        state.positive_traction_integral[component]),
                    InvalidArgumentException,
                    "FESystem::stageTwoFluidAcceptedStageDiagnostics: "
                    "traction-jump integral does not equal the phase "
                    "traction difference");
                FE_THROW_IF(
                    !near_difference(
                        state.viscous_traction_jump_integral[component],
                        state.negative_viscous_traction_integral[component],
                        state.positive_viscous_traction_integral[component]),
                    InvalidArgumentException,
                    "FESystem::stageTwoFluidAcceptedStageDiagnostics: "
                    "viscous-traction-jump integral does not equal the phase "
                    "viscous-traction difference");
            }
            if (parameters.prescribed_pressure_jump.has_value()) {
                const Real target = *parameters.prescribed_pressure_jump;
                const Real expected_pressure_error =
                    state.pressure_jump_squared -
                    Real{2.0} * target * state.pressure_jump_integral +
                    target * target * state.interface_measure;
                FE_THROW_IF(
                    !near(
                        *state.prescribed_pressure_jump_error_squared,
                        expected_pressure_error),
                    InvalidArgumentException,
                    "FESystem::stageTwoFluidAcceptedStageDiagnostics: "
                    "prescribed-pressure-jump error does not match its raw "
                    "moments");
            }
            if (parameters.prescribed_viscous_traction_jump.has_value()) {
                const auto& target =
                    *parameters.prescribed_viscous_traction_jump;
                const Real expected_viscous_error =
                    state.viscous_traction_jump_squared -
                    Real{2.0} *
                        dot(target, state.viscous_traction_jump_integral) +
                    squared_norm(target) * state.interface_measure;
                FE_THROW_IF(
                    !near(
                        *state
                             .prescribed_viscous_traction_jump_error_squared,
                        expected_viscous_error),
                    InvalidArgumentException,
                    "FESystem::stageTwoFluidAcceptedStageDiagnostics: "
                    "prescribed-viscous-traction-jump error does not match "
                    "its raw moments");
            }
            if (state.prescribed_stress_jump_residual_squared.has_value()) {
                const Real pressure_target =
                    parameters.prescribed_pressure_jump.value_or(Real{0.0});
                const std::array<Real, 3> viscous_target =
                    parameters.prescribed_viscous_traction_jump.value_or(
                        std::array<Real, 3>{});
                const Real expected_stress_error =
                    state.traction_jump_squared +
                    squared_norm(viscous_target) * state.interface_measure +
                    pressure_target * pressure_target *
                        state.interface_measure -
                    Real{2.0} *
                        dot(viscous_target, state.traction_jump_integral) +
                    Real{2.0} * pressure_target *
                        state.traction_jump_normal_integral -
                    Real{2.0} * pressure_target *
                        dot(viscous_target, state.interface_normal_integral);
                FE_THROW_IF(
                    !near(
                        *state.prescribed_stress_jump_residual_squared,
                        expected_stress_error),
                    InvalidArgumentException,
                    "FESystem::stageTwoFluidAcceptedStageDiagnostics: "
                    "composed stress-jump residual does not match its raw "
                    "moments");
            }
        }

        appendMeshBoundaryToken(
            local_tokens,
            static_cast<std::uint64_t>(
                two_fluid_accepted_stage_declarations_.size()));
        appendMeshBoundaryToken(
            local_tokens,
            two_fluid_accepted_stage_declarations_frozen_ ? 1u : 0u);
        for (const auto& declaration :
             two_fluid_accepted_stage_declarations_) {
            appendTwoFluidDiagnosticDeclaration(local_tokens, declaration);
        }
        appendOperatorStageMeasurementMetadata(local_tokens, metadata);
        appendMeshBoundaryToken(
            local_tokens,
            static_cast<std::uint64_t>(
                two_fluid_accepted_stage_diagnostic_history_.size()));
        appendMeshBoundaryToken(
            local_tokens, static_cast<std::uint64_t>(states.size()));
        for (const auto& state : states) {
            appendMeshBoundaryIntegral(local_tokens, state.interface_marker);
            appendTwoFluidGeometryRevision(
                local_tokens, state.geometry_revision);
            appendTwoFluidDiagnosticState(local_tokens, state.diagnostics);
        }
        FE_THROW_IF(
            local_tokens.size() >
                static_cast<std::size_t>(std::numeric_limits<int>::max()),
            InvalidArgumentException,
            "FESystem::stageTwoFluidAcceptedStageDiagnostics: consensus "
            "token sequence exceeds the MPI count range");
        minimum_tokens = local_tokens;
        maximum_tokens = local_tokens;
    } catch (...) {
        local_exception = std::current_exception();
    }

    if (coordinateMeshBoundaryLocalFailure(
            collective,
            local_exception != nullptr,
            "FESystem::stageTwoFluidAcceptedStageDiagnostics/preflight")) {
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw InvalidArgumentException(
            "FESystem::stageTwoFluidAcceptedStageDiagnostics: another active "
            "communicator rank rejected the stage");
    }
    requireMeshBoundaryTokenConsensus(
        collective,
        local_tokens,
        minimum_tokens,
        maximum_tokens,
        "FESystem::stageTwoFluidAcceptedStageDiagnostics/consensus");
    if (two_fluid_accepted_stage_declarations_.empty()) {
        return;
    }

    local_exception = nullptr;
    two_fluid_accepted_stage_transaction_active_ = true;
    try {
        staged.reserve(states.size());
        for (std::size_t index = 0u; index < states.size(); ++index) {
            staged.push_back(TwoFluidAcceptedStageDiagnosticHistoryRecord{
                .declaration =
                    two_fluid_accepted_stage_declarations_[index],
                .stage = metadata,
                .state = states[index],
            });
        }
        two_fluid_accepted_stage_diagnostic_history_.reserve(
            two_fluid_accepted_stage_diagnostic_history_.size() +
            staged.size());
    } catch (...) {
        local_exception = std::current_exception();
    }
    if (coordinateMeshBoundaryLocalFailure(
            collective,
            local_exception != nullptr,
            "FESystem::stageTwoFluidAcceptedStageDiagnostics/reserve")) {
        two_fluid_accepted_stage_transaction_active_ = false;
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw InvalidArgumentException(
            "FESystem::stageTwoFluidAcceptedStageDiagnostics: another active "
            "communicator rank could not stage the complete group");
    }
    pending_two_fluid_accepted_stage_diagnostics_ = std::move(staged);
    two_fluid_accepted_stage_declarations_frozen_ = true;
    two_fluid_accepted_stage_transaction_active_ = false;
}

void TwoFluidStageRecords::bindPendingTwoFluidMomentumReconciliations(
    std::span<const interfaces::IncompressibleTwoFluidMomentumReconciliation>
        reconciliations)
{
    const auto collective = meshBoundaryCollectiveContext(*system_);
    MeshBoundaryConsensusTokens local_tokens;
    MeshBoundaryConsensusTokens minimum_tokens;
    MeshBoundaryConsensusTokens maximum_tokens;
    std::exception_ptr local_exception;
    std::vector<interfaces::IncompressibleTwoFluidMomentumReconciliation>
        staged;

    try {
        const auto required_count = static_cast<std::size_t>(std::count_if(
            two_fluid_accepted_stage_declarations_.begin(),
            two_fluid_accepted_stage_declarations_.end(),
            [](const auto& declaration) {
                return declaration
                    .require_conservative_phase_momentum_reconciliation;
            }));
        FE_THROW_IF(
            !isSetup() || two_fluid_accepted_stage_transaction_active_ ||
                pending_two_fluid_accepted_stage_diagnostics_.size() !=
                    two_fluid_accepted_stage_declarations_.size() ||
                reconciliations.size() != required_count,
            InvalidArgumentException,
            "FESystem::bindPendingTwoFluidMomentumReconciliations: pending group or required reconciliation coverage is invalid");

        staged.assign(reconciliations.begin(), reconciliations.end());
        std::size_t reconciliation_index = 0u;
        for (std::size_t index = 0u;
             index < pending_two_fluid_accepted_stage_diagnostics_.size();
             ++index) {
            const auto& pending =
                pending_two_fluid_accepted_stage_diagnostics_[index];
            const bool required = pending.declaration
                                      .require_conservative_phase_momentum_reconciliation;
            FE_THROW_IF(
                pending.momentum_reconciliation.has_value(),
                InvalidArgumentException,
                "FESystem::bindPendingTwoFluidMomentumReconciliations: pending group is already bound");
            if (!required) {
                continue;
            }

            auto& reconciliation = staged[reconciliation_index++];
            try {
                interfaces::validateIncompressibleTwoFluidMomentumReconciliation(
                    reconciliation);
            } catch (const std::invalid_argument& error) {
                throw InvalidArgumentException(
                    std::string(
                        "FESystem::bindPendingTwoFluidMomentumReconciliations: ") +
                    error.what());
            }
            const auto& raw = pending.state.diagnostics;
            const auto& raw_revision = pending.state.geometry_revision;
            const auto& negative = reconciliation.negative_phase;
            const auto& positive = reconciliation.positive_phase;
            const bool raw_revision_matches =
                reconciliation.raw_geometry_revision.sameSourceState(
                    raw_revision) &&
                reconciliation.raw_geometry_revision.snapshot_revision_key ==
                    raw_revision.snapshot_revision_key;
            const bool negative_matches =
                negative.side == raw.negative_phase.side &&
                negative.density == raw.negative_phase.density &&
                negative.raw_volume == raw.negative_phase.volume &&
                negative.raw_mass == raw.negative_phase.mass &&
                negative.raw_momentum == raw.negative_phase.momentum;
            const bool positive_matches =
                positive.side == raw.positive_phase.side &&
                positive.density == raw.positive_phase.density &&
                positive.raw_volume == raw.positive_phase.volume &&
                positive.raw_mass == raw.positive_phase.mass &&
                positive.raw_momentum == raw.positive_phase.momentum;
            FE_THROW_IF(
                !reconciliation.satisfied ||
                    reconciliation.interface_marker !=
                        pending.declaration.interface_marker ||
                    !raw_revision_matches ||
                    reconciliation.raw_algebraic_revision !=
                        pending.stage.state_revision ||
                    !negative_matches || !positive_matches,
                InvalidArgumentException,
                "FESystem::bindPendingTwoFluidMomentumReconciliations: reconciliation does not match the pending raw stage or failed its phasewise gate");
        }

        appendMeshBoundaryToken(
            local_tokens,
            static_cast<std::uint64_t>(reconciliations.size()));
        for (const auto& reconciliation : staged) {
            appendTwoFluidMomentumReconciliation(
                local_tokens, reconciliation);
        }
        FE_THROW_IF(
            local_tokens.size() >
                static_cast<std::size_t>(std::numeric_limits<int>::max()),
            InvalidArgumentException,
            "FESystem::bindPendingTwoFluidMomentumReconciliations: consensus token sequence exceeds the MPI count range");
        minimum_tokens = local_tokens;
        maximum_tokens = local_tokens;
    } catch (...) {
        local_exception = std::current_exception();
    }

    if (coordinateMeshBoundaryLocalFailure(
            collective,
            local_exception != nullptr,
            "FESystem::bindPendingTwoFluidMomentumReconciliations/preflight")) {
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw InvalidArgumentException(
            "FESystem::bindPendingTwoFluidMomentumReconciliations: another active communicator rank rejected the binding");
    }
    requireMeshBoundaryTokenConsensus(
        collective,
        local_tokens,
        minimum_tokens,
        maximum_tokens,
        "FESystem::bindPendingTwoFluidMomentumReconciliations/consensus");

    static_assert(std::is_nothrow_move_constructible_v<
                  interfaces::IncompressibleTwoFluidMomentumReconciliation>);
    two_fluid_accepted_stage_transaction_active_ = true;
    std::size_t reconciliation_index = 0u;
    for (auto& pending : pending_two_fluid_accepted_stage_diagnostics_) {
        if (pending.declaration
                .require_conservative_phase_momentum_reconciliation) {
            pending.momentum_reconciliation.emplace(
                std::move(staged[reconciliation_index++]));
        }
    }
    two_fluid_accepted_stage_transaction_active_ = false;
}

void TwoFluidStageRecords::bindPendingTwoFluidAcceptedStageNumerics(
    TwoFluidAcceptedStageSolveTelemetry solve,
    std::span<const constraints::SmallCutAggregationRefreshReport>
        aggregation_reports)
{
    const auto collective = meshBoundaryCollectiveContext(*system_);
    MeshBoundaryConsensusTokens local_tokens;
    MeshBoundaryConsensusTokens minimum_tokens;
    MeshBoundaryConsensusTokens maximum_tokens;
    std::exception_ptr local_exception;
    std::vector<std::pair<std::size_t, TwoFluidAcceptedStageNumerics>>
        staged;

    try {
        const auto required_count = static_cast<std::size_t>(std::count_if(
            two_fluid_accepted_stage_declarations_.begin(),
            two_fluid_accepted_stage_declarations_.end(),
            [](const auto& declaration) {
                return declaration.require_accepted_stage_numerics;
            }));
        FE_THROW_IF(
            !isSetup() || two_fluid_accepted_stage_transaction_active_ ||
                pending_two_fluid_accepted_stage_diagnostics_.size() !=
                    two_fluid_accepted_stage_declarations_.size() ||
                required_count == 0u,
            InvalidArgumentException,
            "FESystem::bindPendingTwoFluidAcceptedStageNumerics: pending group or required numerical-evidence coverage is invalid");
        validateTwoFluidAcceptedStageSolveTelemetry(solve);
        staged.reserve(required_count);

        for (std::size_t record_index = 0u;
             record_index <
                 pending_two_fluid_accepted_stage_diagnostics_.size();
             ++record_index) {
            const auto& pending =
                pending_two_fluid_accepted_stage_diagnostics_[record_index];
            FE_THROW_IF(
                pending.numerics.has_value(),
                InvalidArgumentException,
                "FESystem::bindPendingTwoFluidAcceptedStageNumerics: pending group is already bound");
            if (!pending.declaration.require_accepted_stage_numerics) {
                continue;
            }

            const auto& declaration = pending.declaration;
            const std::array<FieldId, 4> expected_fields{
                declaration.negative_velocity_field,
                declaration.negative_pressure_field,
                declaration.positive_velocity_field,
                declaration.positive_pressure_field};
            const std::array<geometry::CutIntegrationSide, 4>
                expected_sides{
                    geometry::CutIntegrationSide::Negative,
                    geometry::CutIntegrationSide::Negative,
                    geometry::CutIntegrationSide::Positive,
                    geometry::CutIntegrationSide::Positive};
            std::array<
                const constraints::SmallCutAggregationRefreshReport*, 4>
                matched{};
            for (const auto& report : aggregation_reports) {
                if (report.interface_marker !=
                    declaration.interface_marker) {
                    continue;
                }
                const auto field = std::find(
                    expected_fields.begin(),
                    expected_fields.end(),
                    report.field);
                FE_THROW_IF(
                    field == expected_fields.end(),
                    InvalidArgumentException,
                    "FESystem::bindPendingTwoFluidAcceptedStageNumerics: unexpected aggregation field for a required interface marker");
                const auto index = static_cast<std::size_t>(
                    std::distance(expected_fields.begin(), field));
                FE_THROW_IF(
                    matched[index] != nullptr ||
                        report.active_side != expected_sides[index],
                    InvalidArgumentException,
                    "FESystem::bindPendingTwoFluidAcceptedStageNumerics: duplicate or wrong-side aggregation evidence for a required phase field");
                matched[index] = &report;
            }
            FE_THROW_IF(
                std::any_of(
                    matched.begin(), matched.end(),
                    [](const auto* report) { return report == nullptr; }),
                InvalidArgumentException,
                "FESystem::bindPendingTwoFluidAcceptedStageNumerics: required velocity/pressure aggregation evidence is incomplete");

            std::array<TwoFluidAcceptedStageAggregationTelemetry, 4>
                compact;
            for (std::size_t index = 0u; index < compact.size(); ++index) {
                compact[index] = compactTwoFluidAcceptedStageAggregation(
                    *matched[index], pending.state.geometry_revision);
                FE_THROW_IF(
                    index != 0u &&
                        compact[index].geometry_identity !=
                            compact.front().geometry_identity,
                    InvalidArgumentException,
                    "FESystem::bindPendingTwoFluidAcceptedStageNumerics: phase aggregation reports do not share one canonical geometry identity");
            }

            const auto pressure_stabilization = [&declaration] {
                return TwoFluidAcceptedStagePressureStabilizationTelemetry{
                    .coefficient =
                        declaration.pressure_stabilization_coefficient,
                    .accepted_stage_work = std::nullopt,
                    .work_reason =
                        "not_separately_resolved_by_coupled_operator",
                };
            };
            const auto phase = [&](geometry::CutIntegrationSide side,
                                   std::size_t velocity_index,
                                   std::size_t pressure_index) {
                return TwoFluidAcceptedStagePhaseNumerics{
                    .side = side,
                    .velocity_aggregation =
                        std::move(compact[velocity_index]),
                    .pressure_aggregation =
                        std::move(compact[pressure_index]),
                    .pressure_stabilization = pressure_stabilization(),
                    .linear_iteration_scope =
                        TwoFluidAcceptedStageLinearIterationScope::
                            SharedCoupledSolve,
                    .linear_iterations = std::nullopt,
                    .linear_reason =
                        "not_individually_resolved_by_shared_coupled_backend",
                    .nonlinear_iteration_scope =
                        TwoFluidAcceptedStageNonlinearIterationScope::
                            SharedCoupledSolve,
                    .nonlinear_iterations = std::nullopt,
                    .nonlinear_reason =
                        "not_individually_resolved_by_shared_coupled_backend",
                };
            };
            staged.emplace_back(
                record_index,
                TwoFluidAcceptedStageNumerics{
                    .solve = solve,
                    .negative_phase = phase(
                        geometry::CutIntegrationSide::Negative, 0u, 1u),
                    .positive_phase = phase(
                        geometry::CutIntegrationSide::Positive, 2u, 3u),
                });
        }

        appendMeshBoundaryToken(
            local_tokens, static_cast<std::uint64_t>(staged.size()));
        for (const auto& [record_index, numerics] : staged) {
            appendMeshBoundaryToken(
                local_tokens, static_cast<std::uint64_t>(record_index));
            appendTwoFluidAcceptedStageNumerics(local_tokens, numerics);
        }
        FE_THROW_IF(
            local_tokens.size() >
                static_cast<std::size_t>(std::numeric_limits<int>::max()),
            InvalidArgumentException,
            "FESystem::bindPendingTwoFluidAcceptedStageNumerics: consensus token sequence exceeds the MPI count range");
        minimum_tokens = local_tokens;
        maximum_tokens = local_tokens;
    } catch (...) {
        local_exception = std::current_exception();
    }

    if (coordinateMeshBoundaryLocalFailure(
            collective,
            local_exception != nullptr,
            "FESystem::bindPendingTwoFluidAcceptedStageNumerics/preflight")) {
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw InvalidArgumentException(
            "FESystem::bindPendingTwoFluidAcceptedStageNumerics: another active communicator rank rejected the numerical-evidence binding");
    }
    requireMeshBoundaryTokenConsensus(
        collective,
        local_tokens,
        minimum_tokens,
        maximum_tokens,
        "FESystem::bindPendingTwoFluidAcceptedStageNumerics/consensus");

    static_assert(std::is_nothrow_move_constructible_v<
                  TwoFluidAcceptedStageNumerics>);
    two_fluid_accepted_stage_transaction_active_ = true;
    for (auto& [record_index, numerics] : staged) {
        pending_two_fluid_accepted_stage_diagnostics_[record_index]
            .numerics.emplace(std::move(numerics));
    }
    two_fluid_accepted_stage_transaction_active_ = false;
}

void TwoFluidStageRecords::discardPendingTwoFluidAcceptedStageDiagnostics() noexcept
{
    pending_two_fluid_accepted_stage_diagnostics_.clear();
    two_fluid_accepted_stage_transaction_active_ = false;
}

void TwoFluidStageRecords::commitPendingTwoFluidAcceptedStageDiagnostics(
    std::uint64_t accepted_step,
    Real accepted_time,
    Real dt)
{
    const auto collective = meshBoundaryCollectiveContext(*system_);
    MeshBoundaryConsensusTokens local_tokens;
    MeshBoundaryConsensusTokens minimum_tokens;
    MeshBoundaryConsensusTokens maximum_tokens;
    std::exception_ptr local_exception;
    bool identical_replay = false;
    const auto declaration_count =
        two_fluid_accepted_stage_declarations_.size();

    try {
        FE_THROW_IF(
            two_fluid_accepted_stage_transaction_active_ ||
                (declaration_count == 0u
                     ? !pending_two_fluid_accepted_stage_diagnostics_.empty()
                     : pending_two_fluid_accepted_stage_diagnostics_.size() !=
                           declaration_count),
            InvalidArgumentException,
            "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics: "
            "pending group or transaction state is incomplete");
        appendMeshBoundaryToken(
            local_tokens, static_cast<std::uint64_t>(declaration_count));
        for (const auto& declaration :
             two_fluid_accepted_stage_declarations_) {
            appendTwoFluidDiagnosticDeclaration(local_tokens, declaration);
        }
        if (declaration_count != 0u) {
            FE_THROW_IF(
                !isSetup() || !dofHandler().isFinalized() ||
                    !std::isfinite(accepted_time) || !std::isfinite(dt) ||
                    !(dt > Real{0.0}),
                InvalidArgumentException,
                "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics: "
                "accepted metadata or setup is invalid");
            const auto& pending_stage =
                pending_two_fluid_accepted_stage_diagnostics_.front().stage;
            FE_THROW_IF(
                accepted_step != pending_stage.prospective_accepted_step ||
                    accepted_time != pending_stage.step_end_time ||
                    dt != pending_stage.dt,
                InvalidArgumentException,
                "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics: "
                "accepted step/time/dt do not match the pending stage");
            for (std::size_t index = 0u; index < declaration_count; ++index) {
                const auto& pending =
                    pending_two_fluid_accepted_stage_diagnostics_[index];
                FE_THROW_IF(
                    pending.declaration !=
                            two_fluid_accepted_stage_declarations_[index] ||
                        pending.stage != pending_stage ||
                        pending.state.interface_marker !=
                            pending.declaration.interface_marker ||
                        pending.declaration
                                .require_conservative_phase_momentum_reconciliation !=
                            pending.momentum_reconciliation.has_value() ||
                        pending.declaration.require_accepted_stage_numerics !=
                            pending.numerics.has_value(),
                    InvalidArgumentException,
                    "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics: "
                    "pending group no longer matches its declaration or "
                    "shared stage, or required momentum/numerical evidence is missing");
                if (pending.momentum_reconciliation.has_value()) {
                    try {
                        interfaces::validateIncompressibleTwoFluidMomentumReconciliation(
                            *pending.momentum_reconciliation);
                    } catch (const std::invalid_argument& error) {
                        throw InvalidArgumentException(
                            std::string(
                                "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics: ") +
                            error.what());
                    }
                    FE_THROW_IF(
                        !pending.momentum_reconciliation->satisfied ||
                            pending.momentum_reconciliation
                                    ->interface_marker !=
                                pending.declaration.interface_marker ||
                            pending.momentum_reconciliation
                                    ->raw_algebraic_revision !=
                                pending.stage.state_revision,
                        InvalidArgumentException,
                        "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics: momentum reconciliation does not certify the pending raw stage");
                }
                if (pending.numerics.has_value()) {
                    validateTwoFluidAcceptedStageSolveTelemetry(
                        pending.numerics->solve);
                    const auto phase_matches = [&](const auto& phase,
                                                   auto side) {
                        return phase.side == side &&
                               phase.pressure_stabilization.coefficient ==
                                   pending.declaration
                                       .pressure_stabilization_coefficient &&
                               !phase.pressure_stabilization
                                    .accepted_stage_work.has_value() &&
                               phase.linear_iteration_scope ==
                                   TwoFluidAcceptedStageLinearIterationScope::
                                       SharedCoupledSolve &&
                               !phase.linear_iterations.has_value() &&
                               phase.nonlinear_iteration_scope ==
                                   TwoFluidAcceptedStageNonlinearIterationScope::
                                       SharedCoupledSolve &&
                               !phase.nonlinear_iterations.has_value();
                    };
                    FE_THROW_IF(
                        !phase_matches(
                            pending.numerics->negative_phase,
                            geometry::CutIntegrationSide::Negative) ||
                            !phase_matches(
                                pending.numerics->positive_phase,
                                geometry::CutIntegrationSide::Positive),
                        InvalidArgumentException,
                        "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics: phase numerical evidence no longer matches its declaration or shared-solve scope");
                }
                appendTwoFluidAcceptedStageRecord(local_tokens, pending);
            }
            appendMeshBoundaryToken(local_tokens, accepted_step);
            appendMeshBoundaryToken(
                local_tokens,
                generatedBoundaryTraceRealBits(accepted_time));
            appendMeshBoundaryToken(
                local_tokens, generatedBoundaryTraceRealBits(dt));
            FE_THROW_IF(
                two_fluid_accepted_stage_diagnostic_history_.size() %
                        declaration_count !=
                    0u,
                InvalidArgumentException,
                "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics: "
                "accepted history contains an incomplete group");
            if (!two_fluid_accepted_stage_diagnostic_history_.empty()) {
                const auto latest_begin =
                    two_fluid_accepted_stage_diagnostic_history_.size() -
                    declaration_count;
                const auto& latest_stage =
                    two_fluid_accepted_stage_diagnostic_history_[latest_begin]
                        .stage;
                FE_THROW_IF(
                    accepted_step < latest_stage.prospective_accepted_step,
                    InvalidArgumentException,
                    "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics: "
                    "accepted history step must be monotone");
                if (accepted_step == latest_stage.prospective_accepted_step) {
                    identical_replay = true;
                    for (std::size_t index = 0u; index < declaration_count;
                         ++index) {
                        MeshBoundaryConsensusTokens pending_tokens;
                        MeshBoundaryConsensusTokens latest_tokens;
                        appendTwoFluidAcceptedStageRecord(
                            pending_tokens,
                            pending_two_fluid_accepted_stage_diagnostics_[index]);
                        appendTwoFluidAcceptedStageRecord(
                            latest_tokens,
                            two_fluid_accepted_stage_diagnostic_history_[
                                latest_begin + index]);
                        identical_replay =
                            identical_replay &&
                            pending_tokens == latest_tokens &&
                            pending_two_fluid_accepted_stage_diagnostics_[index]
                                    .state.local_mesh_revision ==
                                two_fluid_accepted_stage_diagnostic_history_[
                                    latest_begin + index]
                                    .state.local_mesh_revision &&
                            pending_two_fluid_accepted_stage_diagnostics_[index]
                                    .stage.expected_stage_geometry ==
                                two_fluid_accepted_stage_diagnostic_history_[
                                    latest_begin + index]
                                    .stage.expected_stage_geometry;
                    }
                    FE_THROW_IF(
                        !identical_replay,
                        InvalidArgumentException,
                        "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics: "
                        "an accepted step cannot be replayed with different "
                        "diagnostics or provenance");
                } else {
                    FE_THROW_IF(
                        accepted_time < latest_stage.step_end_time,
                        InvalidArgumentException,
                        "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics: "
                        "accepted history time must be monotone");
                }
            }
            FE_THROW_IF(
                !identical_replay &&
                    two_fluid_accepted_stage_diagnostic_history_.capacity() <
                        two_fluid_accepted_stage_diagnostic_history_.size() +
                            declaration_count,
                InvalidArgumentException,
                "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics: "
                "staging did not reserve allocation-free publication");
        }
        appendMeshBoundaryToken(
            local_tokens, identical_replay ? 1u : 0u);
        FE_THROW_IF(
            local_tokens.size() >
                static_cast<std::size_t>(std::numeric_limits<int>::max()),
            InvalidArgumentException,
            "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics: "
            "consensus token sequence exceeds the MPI count range");
        minimum_tokens = local_tokens;
        maximum_tokens = local_tokens;
    } catch (...) {
        local_exception = std::current_exception();
    }
    if (coordinateMeshBoundaryLocalFailure(
            collective,
            local_exception != nullptr,
            "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics/preflight")) {
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw InvalidArgumentException(
            "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics: another "
            "active communicator rank rejected publication");
    }
    requireMeshBoundaryTokenConsensus(
        collective,
        local_tokens,
        minimum_tokens,
        maximum_tokens,
        "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics/consensus");
    if (declaration_count == 0u) {
        return;
    }
    if (identical_replay) {
        pending_two_fluid_accepted_stage_diagnostics_.clear();
        return;
    }

    static_assert(std::is_nothrow_move_constructible_v<
                  TwoFluidAcceptedStageDiagnosticHistoryRecord>);
    static_assert(std::is_nothrow_move_assignable_v<
                  TwoFluidAcceptedStageDiagnosticHistoryRecord>);
    const auto history_size =
        two_fluid_accepted_stage_diagnostic_history_.size();
    std::size_t appended_count = 0u;
    local_exception = nullptr;
    two_fluid_accepted_stage_transaction_active_ = true;
    try {
        for (auto& record : pending_two_fluid_accepted_stage_diagnostics_) {
            two_fluid_accepted_stage_diagnostic_history_.push_back(
                std::move(record));
            ++appended_count;
        }
    } catch (...) {
        local_exception = std::current_exception();
    }
    if (coordinateMeshBoundaryLocalFailure(
            collective,
            local_exception != nullptr,
            "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics/publish")) {
        for (std::size_t index = 0u; index < appended_count; ++index) {
            pending_two_fluid_accepted_stage_diagnostics_[index] = std::move(
                two_fluid_accepted_stage_diagnostic_history_[history_size +
                                                              index]);
        }
        two_fluid_accepted_stage_diagnostic_history_.resize(history_size);
        two_fluid_accepted_stage_transaction_active_ = false;
        if (local_exception != nullptr) {
            std::rethrow_exception(local_exception);
        }
        throw InvalidArgumentException(
            "FESystem::commitPendingTwoFluidAcceptedStageDiagnostics: another "
            "active communicator rank failed publication");
    }
    pending_two_fluid_accepted_stage_diagnostics_.clear();
    two_fluid_accepted_stage_transaction_active_ = false;

    bool emit_diagnostics = true;
#if FE_HAS_MPI
    if (collective.size > 1) {
        int rank = 0;
        MPI_Comm_rank(collective.communicator, &rank);
        emit_diagnostics = rank == 0;
    }
#endif
    if (!emit_diagnostics) {
        return;
    }

    try {
        for (std::size_t index = history_size;
             index < two_fluid_accepted_stage_diagnostic_history_.size();
             ++index) {
            const auto& record =
                two_fluid_accepted_stage_diagnostic_history_[index];
            const auto& state = record.state.diagnostics;
            // The accepted two-fluid interface record (map entry R1.9); rank 0 prints it.
            LogRecord log_line("TwoFluidStageRecords", "accepted two-fluid interface diagnostics");
            log_line.diagnostic("two_fluid_interface_history");
            log_line
                    .text("semantics", "operator_stage")
                    .quoted("scheme", record.stage.scheme_name)
                    .value("accepted_step", accepted_step)
                    .value("attempt", record.stage.prospective_attempt)
                    .value("state_time", record.stage.state_time)
                    .value("interface_marker", record.state.interface_marker)
                    .value("snapshot_revision", state.snapshot_revision_key);
            if (state.transient_penalty_effective_dt.has_value()) {
                log_line.real("transient_penalty_effective_dt", *state.transient_penalty_effective_dt);
            } else {
                log_line.text("transient_penalty_effective_dt", "not_applicable");
            }
            log_line
                    .value("interface_quadrature_points", state.interface_quadrature_point_count)
                    .value("interface_measure", state.interface_measure)
                    .value("interface_normal_integral_0", state.interface_normal_integral[0])
                    .value("interface_normal_integral_1", state.interface_normal_integral[1])
                    .value("interface_normal_integral_2", state.interface_normal_integral[2])
                    .value("velocity_jump_sq", state.velocity_jump_squared)
                    .value("velocity_jump_normal_sq", state.normal_velocity_jump_squared)
                    .value("velocity_jump_tangential_sq", state.tangential_velocity_jump_squared)
                    .value("negative_normal_flux", state.negative_normal_flux)
                    .value("positive_normal_flux", state.positive_normal_flux)
                    .value("normal_flux_jump", state.normal_flux_jump)
                    .value("negative_mass_flux", state.negative_mass_flux)
                    .value("positive_mass_flux", state.positive_mass_flux)
                    .value("mean_pressure_jump", state.mean_pressure_jump)
                    .value("pressure_jump_sq", state.pressure_jump_squared)
                    .value("pressure_jump_integral", state.pressure_jump_integral)
                    .value("traction_jump_sq", state.traction_jump_squared)
                    .value("traction_jump_normal_integral", state.traction_jump_normal_integral)
                    .value("negative_traction_integral_0", state.negative_traction_integral[0])
                    .value("negative_traction_integral_1", state.negative_traction_integral[1])
                    .value("negative_traction_integral_2", state.negative_traction_integral[2])
                    .value("positive_traction_integral_0", state.positive_traction_integral[0])
                    .value("positive_traction_integral_1", state.positive_traction_integral[1])
                    .value("positive_traction_integral_2", state.positive_traction_integral[2])
                    .value("traction_jump_integral_0", state.traction_jump_integral[0])
                    .value("traction_jump_integral_1", state.traction_jump_integral[1])
                    .value("traction_jump_integral_2", state.traction_jump_integral[2])
                    .value("viscous_traction_jump_sq", state.viscous_traction_jump_squared)
                    .value("negative_viscous_traction_integral_0", state.negative_viscous_traction_integral[0])
                    .value("negative_viscous_traction_integral_1", state.negative_viscous_traction_integral[1])
                    .value("negative_viscous_traction_integral_2", state.negative_viscous_traction_integral[2])
                    .value("positive_viscous_traction_integral_0", state.positive_viscous_traction_integral[0])
                    .value("positive_viscous_traction_integral_1", state.positive_viscous_traction_integral[1])
                    .value("positive_viscous_traction_integral_2", state.positive_viscous_traction_integral[2])
                    .value("viscous_traction_jump_integral_0", state.viscous_traction_jump_integral[0])
                    .value("viscous_traction_jump_integral_1", state.viscous_traction_jump_integral[1])
                    .value("viscous_traction_jump_integral_2", state.viscous_traction_jump_integral[2])
                    .value("surface_energy_work", state.surface_energy_work)
                    .value("nitsche_consistency_work", state.nitsche_consistency_work)
                    .value("nitsche_adjoint_work", state.nitsche_adjoint_work)
                    .value("nitsche_penalty_work", state.nitsche_penalty_work)
                    .value("negative_phase_quadrature_points", state.negative_phase.quadrature_point_count)
                    .value("negative_density", state.negative_phase.density)
                    .value("negative_volume", state.negative_phase.volume)
                    .value("negative_mass", state.negative_phase.mass)
                    .value("negative_momentum_0", state.negative_phase.momentum[0])
                    .value("negative_momentum_1", state.negative_phase.momentum[1])
                    .value("negative_momentum_2", state.negative_phase.momentum[2])
                    .value("negative_mean_velocity_0", state.negative_phase.mean_velocity[0])
                    .value("negative_mean_velocity_1", state.negative_phase.mean_velocity[1])
                    .value("negative_mean_velocity_2", state.negative_phase.mean_velocity[2])
                    .value("negative_maximum_quadrature_speed", state.negative_phase.maximum_quadrature_speed)
                    .value("negative_kinetic_energy", state.negative_phase.kinetic_energy)
                    .value("negative_pressure_integral", state.negative_phase.pressure_integral)
                    .value("negative_mean_pressure", state.negative_phase.mean_pressure)
                    .value("negative_pressure_squared_integral", state.negative_phase.pressure_squared_integral)
                    .value("negative_pressure_gradient_integral_0", state.negative_phase.pressure_gradient_integral[0])
                    .value("negative_pressure_gradient_integral_1", state.negative_phase.pressure_gradient_integral[1])
                    .value("negative_pressure_gradient_integral_2", state.negative_phase.pressure_gradient_integral[2])
                    .value("negative_body_force_density_integral_0", state.negative_phase.body_force_density_integral[0])
                    .value("negative_body_force_density_integral_1", state.negative_phase.body_force_density_integral[1])
                    .value("negative_body_force_density_integral_2", state.negative_phase.body_force_density_integral[2])
                    .value("negative_hydrostatic_residual_integral_0", state.negative_phase.hydrostatic_residual_integral[0])
                    .value("negative_hydrostatic_residual_integral_1", state.negative_phase.hydrostatic_residual_integral[1])
                    .value("negative_hydrostatic_residual_integral_2", state.negative_phase.hydrostatic_residual_integral[2])
                    .value("negative_hydrostatic_residual_sq", state.negative_phase.hydrostatic_residual_squared)
                    .value("positive_phase_quadrature_points", state.positive_phase.quadrature_point_count)
                    .value("positive_density", state.positive_phase.density)
                    .value("positive_volume", state.positive_phase.volume)
                    .value("positive_mass", state.positive_phase.mass)
                    .value("positive_momentum_0", state.positive_phase.momentum[0])
                    .value("positive_momentum_1", state.positive_phase.momentum[1])
                    .value("positive_momentum_2", state.positive_phase.momentum[2])
                    .value("positive_mean_velocity_0", state.positive_phase.mean_velocity[0])
                    .value("positive_mean_velocity_1", state.positive_phase.mean_velocity[1])
                    .value("positive_mean_velocity_2", state.positive_phase.mean_velocity[2])
                    .value("positive_maximum_quadrature_speed", state.positive_phase.maximum_quadrature_speed)
                    .value("positive_kinetic_energy", state.positive_phase.kinetic_energy)
                    .value("positive_pressure_integral", state.positive_phase.pressure_integral)
                    .value("positive_mean_pressure", state.positive_phase.mean_pressure)
                    .value("positive_pressure_squared_integral", state.positive_phase.pressure_squared_integral)
                    .value("positive_pressure_gradient_integral_0", state.positive_phase.pressure_gradient_integral[0])
                    .value("positive_pressure_gradient_integral_1", state.positive_phase.pressure_gradient_integral[1])
                    .value("positive_pressure_gradient_integral_2", state.positive_phase.pressure_gradient_integral[2])
                    .value("positive_body_force_density_integral_0", state.positive_phase.body_force_density_integral[0])
                    .value("positive_body_force_density_integral_1", state.positive_phase.body_force_density_integral[1])
                    .value("positive_body_force_density_integral_2", state.positive_phase.body_force_density_integral[2])
                    .value("positive_hydrostatic_residual_integral_0", state.positive_phase.hydrostatic_residual_integral[0])
                    .value("positive_hydrostatic_residual_integral_1", state.positive_phase.hydrostatic_residual_integral[1])
                    .value("positive_hydrostatic_residual_integral_2", state.positive_phase.hydrostatic_residual_integral[2])
                    .value("positive_hydrostatic_residual_sq", state.positive_phase.hydrostatic_residual_squared)
                    .boolean("prescribed_pressure_jump_applicable", state.prescribed_pressure_jump_error_squared.has_value());
            if (state.prescribed_pressure_jump_error_squared.has_value()) {
                log_line
                    .value("prescribed_pressure_jump_target", *record.declaration.parameters
                            .prescribed_pressure_jump)
                    .value("prescribed_pressure_jump_error_sq", *state.prescribed_pressure_jump_error_squared);
            }
            log_line
                    .boolean("prescribed_viscous_traction_jump_applicable", state
                                .prescribed_viscous_traction_jump_error_squared
                                .has_value());
            if (state.prescribed_viscous_traction_jump_error_squared
                    .has_value()) {
                const auto& target =
                    *record.declaration.parameters
                         .prescribed_viscous_traction_jump;
                log_line
                    .value("prescribed_viscous_traction_jump_target_0", target[0])
                    .value("prescribed_viscous_traction_jump_target_1", target[1])
                    .value("prescribed_viscous_traction_jump_target_2", target[2])
                    .value("prescribed_viscous_traction_jump_error_sq", *state
                            .prescribed_viscous_traction_jump_error_squared);
            }
            log_line
                    .boolean("prescribed_stress_jump_residual_applicable", state.prescribed_stress_jump_residual_squared
                                .has_value());
            if (state.prescribed_stress_jump_residual_squared.has_value()) {
                log_line
                    .value("prescribed_stress_jump_residual_sq", *state.prescribed_stress_jump_residual_squared);
            }
            log_line
                    .boolean("momentum_reconciliation_applicable", record.momentum_reconciliation.has_value());
            if (record.momentum_reconciliation.has_value()) {
                const auto& reconciliation =
                    *record.momentum_reconciliation;
                log_line
                    .value("corrected_snapshot_revision", reconciliation.corrected_geometry_revision
                           .snapshot_revision_key)
                    .value("raw_algebraic_revision", reconciliation.raw_algebraic_revision)
                    .value("corrected_algebraic_revision", reconciliation.corrected_algebraic_revision)
                    .value("momentum_relative_tolerance", reconciliation.relative_tolerance)
                    .boolean("velocity_update_applied", false)
                    .value("negative_momentum_delta_norm", reconciliation.negative_phase.momentum_delta_norm)
                    .value("negative_allowed_momentum_delta", reconciliation.negative_phase
                           .allowed_momentum_delta)
                    .value("positive_momentum_delta_norm", reconciliation.positive_phase.momentum_delta_norm)
                    .value("positive_allowed_momentum_delta", reconciliation.positive_phase
                           .allowed_momentum_delta)
                    .boolean("momentum_reconciliation_satisfied", true);
            }
            log_line
                    .boolean("accepted_stage_numerics_applicable", record.numerics.has_value());
            if (record.numerics.has_value()) {
                const auto& numerics = *record.numerics;
                log_line
                    .boolean("nonlinear_converged", numerics.solve.nonlinear.converged)
                    .value("nonlinear_iterations", numerics.solve.nonlinear.iterations)
                    .value("nonlinear_initial_residual_norm", numerics.solve.nonlinear.initial_residual_norm)
                    .value("nonlinear_final_residual_norm", numerics.solve.nonlinear.final_residual_norm)
                    .quoted("nonlinear_reason", numerics.solve.nonlinear.reason)
                    .boolean("linear_converged", numerics.solve.linear.converged)
                    .value("linear_iterations", numerics.solve.linear.iterations)
                    .value("linear_relative_residual", numerics.solve.linear.relative_residual)
                    .quoted("linear_reason", numerics.solve.linear.reason)
                    .value("blockschur_outer_iterations", numerics.solve.linear.blockschur_outer_iterations)
                    .value("negative_velocity_aggregate_dofs", numerics.negative_phase.velocity_aggregation
                           .canonical_owned_aggregate_dofs)
                    .value("negative_pressure_aggregate_dofs", numerics.negative_phase.pressure_aggregation
                           .canonical_owned_aggregate_dofs)
                    .value("positive_velocity_aggregate_dofs", numerics.positive_phase.velocity_aggregation
                           .canonical_owned_aggregate_dofs)
                    .value("positive_pressure_aggregate_dofs", numerics.positive_phase.pressure_aggregation
                           .canonical_owned_aggregate_dofs);
                // Fixed fields of the shared coupled solve, as key=value texts
                // (the WP-10 capability-boundary contract reads these texts).
                for (const std::string_view field : {std::string_view("pressure_stabilization_work=unavailable"),
                                                     std::string_view("phase_iteration_scope=shared_coupled_solve")}) {
                    const auto equals = field.find('=');
                    log_line.text(std::string(field.substr(0, equals)), field.substr(equals + 1));
                }
            }
            logRecord(log_line);
        }
    } catch (...) {
        // Accepted semantic history is authoritative; diagnostics are best effort.
    }
}


} // namespace svmp::FE::systems
