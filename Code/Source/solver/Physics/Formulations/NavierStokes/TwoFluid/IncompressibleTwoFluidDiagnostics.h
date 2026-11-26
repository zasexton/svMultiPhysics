/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_TWO_FLUID_INCOMPRESSIBLETWOFLUIDDIAGNOSTICS_H
#define SVMP_PHYSICS_TWO_FLUID_INCOMPRESSIBLETWOFLUIDDIAGNOSTICS_H

/**
 * @file IncompressibleTwoFluidDiagnostics.h
 * @brief Raw accepted-stage measures for a sharp two-fluid interface.
 */

#include "FE/Interfaces/LevelSetGeometrySnapshot.h"

#include <array>
#include <cstddef>
#include <functional>
#include <optional>

namespace svmp::Physics::formulations::navier_stokes::two_fluid {

/** Immutable coefficients shared by the production interface form and its diagnostics. */
struct IncompressibleTwoFluidDiagnosticParameters {
    int dimension{0};
    int interface_marker{-1};
    FE::Real negative_density{0.0};
    FE::Real positive_density{0.0};
    FE::Real negative_viscosity{0.0};
    FE::Real positive_viscosity{0.0};
    FE::Real nitsche_gamma{0.0};
    FE::Real surface_tension{0.0};
    bool include_transient_penalty{true};
    /** Expected p_minus-p_plus. Absence is distinct from a zero target. */
    std::optional<FE::Real> prescribed_pressure_jump{};
    /** Expected (tau_minus-tau_plus)n in global physical components. */
    std::optional<std::array<FE::Real, 3>> prescribed_viscous_traction_jump{};
    /** Constant acceleration used by both phase momentum operators. */
    std::array<FE::Real, 3> body_force{{0.0, 0.0, 0.0}};

    [[nodiscard]] friend bool operator==(
        const IncompressibleTwoFluidDiagnosticParameters&,
        const IncompressibleTwoFluidDiagnosticParameters&) = default;
};

/** Point evaluators for one phase field pair on the shared background cell. */
struct IncompressibleTwoFluidPhaseEvaluators {
    FE::interfaces::FreeSurfaceDiscreteFunctionalVectorEvaluator velocity{};
    FE::interfaces::LevelSetGeometryScalarEvaluator pressure{};
};

/**
 * Geometry input needed to reproduce h_n = 2 |K| / |Gamma cap K| from the
 * production Nitsche form. The callback returns the full physical parent-cell
 * measure, not a cut-phase measure.
 */
struct IncompressibleTwoFluidCellMeasureEvaluator {
    std::function<FE::Real(FE::GlobalIndex)> physical_cell_measure{};

    [[nodiscard]] bool canEvaluate() const noexcept {
        return static_cast<bool>(physical_cell_measure);
    }
};

/** Reducible phase-volume quantities prior to communicator reduction. */
struct IncompressibleTwoFluidPhaseDiagnosticAccumulator {
    std::size_t owned_quadrature_point_count{0u};
    FE::Real volume{0.0};
    std::array<FE::Real, 3> velocity_integral{{0.0, 0.0, 0.0}};
    FE::Real velocity_squared_integral{0.0};
    FE::Real maximum_velocity_squared{0.0};
    FE::Real pressure_integral{0.0};
    FE::Real pressure_squared_integral{0.0};
    std::array<FE::Real, 3> pressure_gradient_integral{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> hydrostatic_residual_integral{{0.0, 0.0, 0.0}};
    FE::Real hydrostatic_residual_squared{0.0};
};

/** Reducible interface and phase quantities prior to communicator reduction. */
struct IncompressibleTwoFluidDiagnosticAccumulator {
    std::uint64_t snapshot_revision_key{0u};
    /** Assembly-stage effective step used only by the transient penalty. */
    std::optional<FE::Real> transient_penalty_effective_dt{};
    std::size_t owned_interface_quadrature_point_count{0u};
    FE::Real interface_measure{0.0};
    std::array<FE::Real, 3> interface_normal_integral{{0.0, 0.0, 0.0}};
    FE::Real velocity_jump_squared{0.0};
    FE::Real normal_velocity_jump_squared{0.0};
    FE::Real tangential_velocity_jump_squared{0.0};
    FE::Real negative_normal_flux{0.0};
    FE::Real positive_normal_flux{0.0};
    std::array<FE::Real, 3> negative_traction_integral{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> positive_traction_integral{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> traction_jump_integral{{0.0, 0.0, 0.0}};
    FE::Real traction_jump_normal_integral{0.0};
    FE::Real traction_jump_squared{0.0};
    std::array<FE::Real, 3> negative_viscous_traction_integral{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> positive_viscous_traction_integral{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> viscous_traction_jump_integral{{0.0, 0.0, 0.0}};
    FE::Real viscous_traction_jump_squared{0.0};
    std::optional<FE::Real> prescribed_stress_jump_residual_squared{};
    FE::Real pressure_jump_integral{0.0};
    FE::Real pressure_jump_squared{0.0};
    std::optional<FE::Real> prescribed_pressure_jump_error_squared{};
    std::optional<FE::Real> prescribed_viscous_traction_jump_error_squared{};
    FE::Real surface_energy_work{0.0};
    FE::Real nitsche_consistency_work{0.0};
    FE::Real nitsche_adjoint_work{0.0};
    FE::Real nitsche_penalty_work{0.0};
    IncompressibleTwoFluidPhaseDiagnosticAccumulator negative_phase{};
    IncompressibleTwoFluidPhaseDiagnosticAccumulator positive_phase{};
};

/** Final phase quantities after communicator reduction and normalization. */
struct IncompressibleTwoFluidPhaseDiagnosticState {
    FE::geometry::CutIntegrationSide side{
        FE::geometry::CutIntegrationSide::Negative};
    std::size_t quadrature_point_count{0u};
    FE::Real density{0.0};
    FE::Real volume{0.0};
    FE::Real mass{0.0};
    std::array<FE::Real, 3> momentum{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> mean_velocity{{0.0, 0.0, 0.0}};
    FE::Real maximum_quadrature_speed{0.0};
    FE::Real kinetic_energy{0.0};
    FE::Real pressure_integral{0.0};
    FE::Real mean_pressure{0.0};
    FE::Real pressure_squared_integral{0.0};
    std::array<FE::Real, 3> pressure_gradient_integral{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> body_force_density_integral{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> hydrostatic_residual_integral{{0.0, 0.0, 0.0}};
    FE::Real hydrostatic_residual_squared{0.0};
};

/**
 * Communicator-global raw accepted-stage record. Integrals remain available
 * alongside normalized quantities so downstream gates do not have to infer
 * whether a displayed zero was measured or inapplicable.
 */
struct IncompressibleTwoFluidDiagnosticState {
    std::uint64_t snapshot_revision_key{0u};
    /** Absence denotes a genuinely steady interface penalty. */
    std::optional<FE::Real> transient_penalty_effective_dt{};
    std::size_t interface_quadrature_point_count{0u};
    FE::Real interface_measure{0.0};
    std::array<FE::Real, 3> interface_normal_integral{{0.0, 0.0, 0.0}};
    FE::Real velocity_jump_squared{0.0};
    FE::Real normal_velocity_jump_squared{0.0};
    FE::Real tangential_velocity_jump_squared{0.0};
    FE::Real negative_normal_flux{0.0};
    FE::Real positive_normal_flux{0.0};
    FE::Real normal_flux_jump{0.0};
    FE::Real negative_mass_flux{0.0};
    FE::Real positive_mass_flux{0.0};
    std::array<FE::Real, 3> negative_traction_integral{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> positive_traction_integral{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> traction_jump_integral{{0.0, 0.0, 0.0}};
    FE::Real traction_jump_normal_integral{0.0};
    FE::Real traction_jump_squared{0.0};
    std::array<FE::Real, 3> negative_viscous_traction_integral{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> positive_viscous_traction_integral{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> viscous_traction_jump_integral{{0.0, 0.0, 0.0}};
    FE::Real viscous_traction_jump_squared{0.0};
    std::optional<FE::Real> prescribed_stress_jump_residual_squared{};
    FE::Real pressure_jump_integral{0.0};
    FE::Real mean_pressure_jump{0.0};
    FE::Real pressure_jump_squared{0.0};
    std::optional<FE::Real> prescribed_pressure_jump_error_squared{};
    std::optional<FE::Real> prescribed_viscous_traction_jump_error_squared{};
    FE::Real surface_energy_work{0.0};
    FE::Real nitsche_consistency_work{0.0};
    FE::Real nitsche_adjoint_work{0.0};
    FE::Real nitsche_penalty_work{0.0};
    IncompressibleTwoFluidPhaseDiagnosticState negative_phase{};
    IncompressibleTwoFluidPhaseDiagnosticState positive_phase{};
};

/** One phase's before/after momentum contract across phase maintenance. */
struct IncompressibleTwoFluidPhaseMomentumReconciliation {
    FE::geometry::CutIntegrationSide side{
        FE::geometry::CutIntegrationSide::Negative};
    FE::Real density{0.0};
    FE::Real raw_volume{0.0};
    FE::Real corrected_volume{0.0};
    FE::Real raw_mass{0.0};
    FE::Real corrected_mass{0.0};
    std::array<FE::Real, 3> raw_momentum{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> corrected_momentum{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> momentum_delta{{0.0, 0.0, 0.0}};
    FE::Real momentum_delta_norm{0.0};
    FE::Real momentum_reference_norm{0.0};
    FE::Real allowed_momentum_delta{0.0};
    bool satisfied{false};

    [[nodiscard]] friend bool operator==(
        const IncompressibleTwoFluidPhaseMomentumReconciliation&,
        const IncompressibleTwoFluidPhaseMomentumReconciliation&) = default;
};

/**
 * Accepted-step phase/momentum contract for conservative phase maintenance.
 *
 * The initial supported policy never applies a hidden velocity correction.
 * A geometry/phase correction is accepted only when both phasewise momentum
 * deltas satisfy the declared relative gate.
 */
struct IncompressibleTwoFluidMomentumReconciliation {
    int interface_marker{-1};
    FE::interfaces::LevelSetGeometryRevision raw_geometry_revision{};
    FE::interfaces::LevelSetGeometryRevision corrected_geometry_revision{};
    std::uint64_t raw_algebraic_revision{0u};
    std::uint64_t corrected_algebraic_revision{0u};
    FE::Real relative_tolerance{0.0};
    bool velocity_update_applied{false};
    bool satisfied{false};
    IncompressibleTwoFluidPhaseMomentumReconciliation negative_phase{};
    IncompressibleTwoFluidPhaseMomentumReconciliation positive_phase{};

};

/** Evaluate rank-owned additive quantities on one immutable snapshot. */
[[nodiscard]] IncompressibleTwoFluidDiagnosticAccumulator
evaluateLocalIncompressibleTwoFluidDiagnostics(
    const FE::interfaces::LevelSetGeometrySnapshot& snapshot,
    const IncompressibleTwoFluidDiagnosticParameters& parameters,
    const IncompressibleTwoFluidPhaseEvaluators& negative_phase,
    const IncompressibleTwoFluidPhaseEvaluators& positive_phase,
    const IncompressibleTwoFluidCellMeasureEvaluator& cell_measure,
    FE::Real effective_dt);

/** Normalize one already-reduced accumulator and enforce its exact identities. */
[[nodiscard]] IncompressibleTwoFluidDiagnosticState
finalizeIncompressibleTwoFluidDiagnostics(
    const IncompressibleTwoFluidDiagnosticAccumulator& accumulator,
    const IncompressibleTwoFluidDiagnosticParameters& parameters);

/** Build the explicit no-hidden-update phasewise momentum contract. */
[[nodiscard]] IncompressibleTwoFluidMomentumReconciliation
buildIncompressibleTwoFluidMomentumReconciliation(
    int interface_marker,
    const FE::interfaces::LevelSetGeometryRevision& raw_geometry_revision,
    const IncompressibleTwoFluidDiagnosticState& raw,
    std::uint64_t raw_algebraic_revision,
    const FE::interfaces::LevelSetGeometryRevision& corrected_geometry_revision,
    const IncompressibleTwoFluidDiagnosticState& corrected,
    std::uint64_t corrected_algebraic_revision,
    FE::Real relative_tolerance);

/** Reject any inconsistent or non-finite reconciliation record. */
void validateIncompressibleTwoFluidMomentumReconciliation(
    const IncompressibleTwoFluidMomentumReconciliation& reconciliation);

} // namespace svmp::Physics::formulations::navier_stokes::two_fluid

#endif // SVMP_PHYSICS_TWO_FLUID_INCOMPRESSIBLETWOFLUIDDIAGNOSTICS_H
