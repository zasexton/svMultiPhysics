/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_TWO_FLUID_RECORD_TYPES_H
#define SVMP_FE_SYSTEMS_TWO_FLUID_RECORD_TYPES_H

/**
 * @file TwoFluidRecordTypes.h
 * @brief Data types of the two-fluid accepted-stage diagnostic records:
 *        declarations, stage states, solve telemetry, aggregation evidence,
 *        numerics and history records.
 *
 * This header is a staging point.  The records and their logic belong to
 * the Navier-Stokes two-fluid physics (Physics/Formulations/NavierStokes/
 * TwoFluid); these plain data types left FESystem.h first and keep their
 * namespace for now.
 */

#include "Constraints/SmallCutAggregationConstraint.h"
#include "Core/Types.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Interfaces/IncompressibleTwoFluidDiagnostics.h"
#include "Systems/OperatorRegistry.h"
#include "Systems/OperatorStageMetadata.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/** Setup-time owner for one sharp incompressible two-fluid history stream. */
struct TwoFluidAcceptedStageDiagnosticDeclaration {
    int interface_marker{-1};
    FieldId level_set_field{INVALID_FIELD_ID};
    FieldId negative_velocity_field{INVALID_FIELD_ID};
    FieldId negative_pressure_field{INVALID_FIELD_ID};
    FieldId positive_velocity_field{INVALID_FIELD_ID};
    FieldId positive_pressure_field{INVALID_FIELD_ID};
    OperatorTag operator_tag{};
    std::string geometry_domain_id{};
    Real level_set_isovalue{0.0};
    bool require_conservative_phase_momentum_reconciliation{false};
    bool require_accepted_stage_numerics{false};
    Real pressure_stabilization_coefficient{0.0};
    interfaces::IncompressibleTwoFluidDiagnosticParameters parameters{};
    std::string owner_component{};

    [[nodiscard]] friend bool operator==(
        const TwoFluidAcceptedStageDiagnosticDeclaration&,
        const TwoFluidAcceptedStageDiagnosticDeclaration&) = default;
};

/**
 * One already reduced state bound to its immutable communicator-canonical
 * snapshot revision and the rank-local mesh stamp used to validate it.
 */
struct AcceptedTwoFluidStageDiagnosticState {
    int interface_marker{-1};
    interfaces::FreeSurfaceGeometryRevision geometry_revision{};
    interfaces::FreeSurfaceGeometryLocalMeshRevision local_mesh_revision{};
    interfaces::IncompressibleTwoFluidDiagnosticState diagnostics{};
};

/** Communicator-canonical nonlinear evidence for one accepted coupled solve. */
struct TwoFluidAcceptedStageNonlinearTelemetry {
    bool converged{false};
    int iterations{0};
    int outer_iterations{0};
    int inner_iterations_total{0};
    Real outer_state_change_norm{0.0};
    bool outer_dynamic_relaxation_enabled{false};
    int outer_dynamic_relaxation_updates{0};
    int outer_dynamic_relaxation_safeguards{0};
    int outer_dynamic_relaxation_resets{0};
    Real outer_dynamic_relaxation_factor{1.0};
    Real outer_relaxed_state_change_norm{0.0};
    std::optional<Real> outer_raw_contraction_ratio{};
    Real initial_residual_norm{0.0};
    Real final_residual_norm{0.0};
    Real initial_field_residual_norm{0.0};
    Real final_field_residual_norm{0.0};
    Real initial_auxiliary_residual_norm{0.0};
    Real final_auxiliary_residual_norm{0.0};
    bool component_residual_convergence{false};
    std::string reason{};

    [[nodiscard]] friend bool operator==(
        const TwoFluidAcceptedStageNonlinearTelemetry&,
        const TwoFluidAcceptedStageNonlinearTelemetry&) = default;
};

/** Timing-free linear evidence for one accepted coupled solve. */
struct TwoFluidAcceptedStageLinearTelemetry {
    bool attempted{true};
    bool converged{false};
    bool numerical_breakdown{false};
    int iterations{0};
    Real initial_residual_norm{0.0};
    Real final_residual_norm{0.0};
    Real relative_residual{0.0};
    std::uint64_t collective_calls{0u};
    std::uint64_t collective_words{0u};
    int blockschur_outer_iterations{0};
    std::uint64_t blockschur_collective_calls_max_per_outer{0u};
    int blockschur_momentum_solve_calls{0};
    int blockschur_momentum_iterations{0};
    int blockschur_momentum_restart_cycles{0};
    std::uint64_t blockschur_momentum_collective_calls{0u};
    std::uint64_t blockschur_momentum_collective_words{0u};
    int blockschur_schur_solve_calls{0};
    int blockschur_schur_iterations{0};
    int blockschur_true_residual_retries{0};
    std::uint64_t blockschur_schur_collective_calls{0u};
    std::uint64_t blockschur_schur_collective_words{0u};
    std::string reason{};

    [[nodiscard]] friend bool operator==(
        const TwoFluidAcceptedStageLinearTelemetry&,
        const TwoFluidAcceptedStageLinearTelemetry&) = default;
};

struct TwoFluidAcceptedStageSolveTelemetry {
    TwoFluidAcceptedStageNonlinearTelemetry nonlinear{};
    TwoFluidAcceptedStageLinearTelemetry linear{};

    [[nodiscard]] friend bool operator==(
        const TwoFluidAcceptedStageSolveTelemetry&,
        const TwoFluidAcceptedStageSolveTelemetry&) = default;
};

/** Exact rank-invariant subset of an aggregation publication identity. */
struct TwoFluidAcceptedStageAggregationGeometryTelemetry {
    constraints::SmallCutAggregationGeometryIdentityKind kind{
        constraints::SmallCutAggregationGeometryIdentityKind::Unavailable};
    bool available{false};
    bool communicator_fingerprint_consensus_validated{false};
    std::string source_id{};
    std::string domain_id{};
    int interface_marker{-1};
    Real isovalue{0.0};
    std::uint64_t source_layout_revision{0u};
    std::uint64_t source_value_revision{0u};
    std::uint64_t quadrature_policy_key{0u};
    std::uint64_t snapshot_revision_key{0u};
    std::uint64_t distributed_mesh_geometry_revision{0u};
    std::uint64_t distributed_mesh_topology_revision{0u};
    std::uint64_t distributed_ownership_revision{0u};
    std::uint64_t distributed_numbering_revision{0u};
    std::uint64_t canonical_fingerprint{0u};

    [[nodiscard]] friend bool operator==(
        const TwoFluidAcceptedStageAggregationGeometryTelemetry&,
        const TwoFluidAcceptedStageAggregationGeometryTelemetry&) = default;
};

/** Vector-free canonical subset of a successive aggregation transition. */
struct TwoFluidAcceptedStageAggregationTransitionTelemetry {
    TwoFluidAcceptedStageAggregationGeometryTelemetry
        geometry_identity_before{};
    TwoFluidAcceptedStageAggregationGeometryTelemetry
        geometry_identity_after{};
    std::uint64_t canonical_feature_class_fingerprint_before{0u};
    std::uint64_t canonical_feature_class_fingerprint_after{0u};
    std::uint64_t canonical_slave_set_fingerprint_before{0u};
    std::uint64_t canonical_slave_set_fingerprint_after{0u};
    std::size_t canonical_active_feature_count_before{0u};
    std::size_t canonical_active_feature_count_after{0u};
    std::size_t canonical_features_entered{0u};
    std::size_t canonical_features_exited{0u};
    std::size_t canonical_features_persisted{0u};
    std::size_t canonical_feature_classification_changes{0u};
    std::size_t canonical_features_became_rooted{0u};
    std::size_t canonical_features_became_rootless{0u};
    std::size_t canonical_aggregate_slaves_before{0u};
    std::size_t canonical_aggregate_slaves_after{0u};
    std::size_t canonical_aggregate_slaves_entered{0u};
    std::size_t canonical_aggregate_slaves_left{0u};
    Real canonical_rootless_active_physical_volume_before{0.0};
    Real canonical_rootless_active_physical_volume_after{0.0};
    Real canonical_rootless_active_physical_volume_delta{0.0};
    bool canonical_topology_changed{false};

    [[nodiscard]] friend bool operator==(
        const TwoFluidAcceptedStageAggregationTransitionTelemetry&,
        const TwoFluidAcceptedStageAggregationTransitionTelemetry&) = default;
};

/** Canonical guard, topology, and aggregate-count evidence for one field. */
struct TwoFluidAcceptedStageAggregationTelemetry {
    TwoFluidAcceptedStageAggregationGeometryTelemetry geometry_identity{};
    std::uint64_t canonical_feature_class_fingerprint{0u};
    std::uint64_t canonical_slave_set_fingerprint{0u};
    std::size_t maximum_root_path_length{0u};
    std::size_t maximum_observed_root_path{0u};
    std::size_t root_path_guard_rejections{0u};
    Real maximum_reference_extrapolation_distance{0.0};
    Real maximum_observed_reference_extrapolation{0.0};
    std::size_t extrapolation_guard_rejections{0u};
    Real maximum_absolute_coefficient{0.0};
    Real maximum_observed_absolute_coefficient{0.0};
    Real maximum_row_l1_norm{0.0};
    Real maximum_observed_row_l1_norm{0.0};
    std::size_t line_guard_rejections{0u};
    std::size_t canonical_candidate_vertices{0u};
    std::size_t canonical_rooted_candidate_vertices{0u};
    std::size_t canonical_rootless_candidate_vertices{0u};
    std::size_t canonical_owned_aggregate_dofs{0u};
    std::size_t canonical_owned_pinned_dofs{0u};
    std::size_t canonical_strong_suppressed_dofs{0u};
    std::size_t canonical_active_feature_count{0u};
    std::size_t canonical_rooted_active_feature_count{0u};
    std::size_t canonical_rootless_active_feature_count{0u};
    Real canonical_rootless_active_physical_volume{0.0};
    std::optional<TwoFluidAcceptedStageAggregationTransitionTelemetry>
        canonical_topology_transition{};

    [[nodiscard]] friend bool operator==(
        const TwoFluidAcceptedStageAggregationTelemetry&,
        const TwoFluidAcceptedStageAggregationTelemetry&) = default;
};

struct TwoFluidAcceptedStagePressureStabilizationTelemetry {
    Real coefficient{0.0};
    std::optional<Real> accepted_stage_work{};
    std::string work_reason{};

    [[nodiscard]] friend bool operator==(
        const TwoFluidAcceptedStagePressureStabilizationTelemetry&,
        const TwoFluidAcceptedStagePressureStabilizationTelemetry&) = default;
};

enum class TwoFluidAcceptedStageLinearIterationScope : std::uint8_t {
    SharedCoupledSolve,
    PhaseResolved,
};

enum class TwoFluidAcceptedStageNonlinearIterationScope : std::uint8_t {
    SharedCoupledSolve,
    PhaseResolved,
};

struct TwoFluidAcceptedStagePhaseNumerics {
    geometry::CutIntegrationSide side{
        geometry::CutIntegrationSide::Negative};
    TwoFluidAcceptedStageAggregationTelemetry velocity_aggregation{};
    TwoFluidAcceptedStageAggregationTelemetry pressure_aggregation{};
    TwoFluidAcceptedStagePressureStabilizationTelemetry
        pressure_stabilization{};
    TwoFluidAcceptedStageLinearIterationScope linear_iteration_scope{
        TwoFluidAcceptedStageLinearIterationScope::SharedCoupledSolve};
    std::optional<int> linear_iterations{};
    std::string linear_reason{};
    TwoFluidAcceptedStageNonlinearIterationScope nonlinear_iteration_scope{
        TwoFluidAcceptedStageNonlinearIterationScope::SharedCoupledSolve};
    std::optional<int> nonlinear_iterations{};
    std::string nonlinear_reason{};

    [[nodiscard]] friend bool operator==(
        const TwoFluidAcceptedStagePhaseNumerics&,
        const TwoFluidAcceptedStagePhaseNumerics&) = default;
};

struct TwoFluidAcceptedStageNumerics {
    TwoFluidAcceptedStageSolveTelemetry solve{};
    TwoFluidAcceptedStagePhaseNumerics negative_phase{};
    TwoFluidAcceptedStagePhaseNumerics positive_phase{};

    [[nodiscard]] friend bool operator==(
        const TwoFluidAcceptedStageNumerics&,
        const TwoFluidAcceptedStageNumerics&) = default;
};

/** Pending or accepted two-fluid diagnostics at the converged operator stage. */
struct TwoFluidAcceptedStageDiagnosticHistoryRecord {
    TwoFluidAcceptedStageDiagnosticDeclaration declaration{};
    OperatorStageMeasurementMetadata stage{};
    AcceptedTwoFluidStageDiagnosticState state{};
    std::optional<interfaces::IncompressibleTwoFluidMomentumReconciliation>
        momentum_reconciliation{};
    std::optional<TwoFluidAcceptedStageNumerics> numerics{};
};

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_TWO_FLUID_RECORD_TYPES_H
