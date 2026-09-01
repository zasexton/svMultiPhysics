/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_INTERFACES_INCOMPRESSIBLETWOFLUIDDIAGNOSTICS_H
#define SVMP_FE_INTERFACES_INCOMPRESSIBLETWOFLUIDDIAGNOSTICS_H

/**
 * @file IncompressibleTwoFluidDiagnostics.h
 * @brief Raw accepted-stage measures for a sharp two-fluid interface.
 */

#include "Interfaces/FreeSurfaceGeometrySnapshot.h"

#include <array>
#include <cstddef>
#include <functional>
#include <optional>

namespace svmp::FE::interfaces {

/** Immutable coefficients shared by the production interface form and its diagnostics. */
struct IncompressibleTwoFluidDiagnosticParameters {
    int dimension{0};
    int interface_marker{-1};
    Real negative_density{0.0};
    Real positive_density{0.0};
    Real negative_viscosity{0.0};
    Real positive_viscosity{0.0};
    Real nitsche_gamma{0.0};
    Real surface_tension{0.0};
    bool include_transient_penalty{true};
    /** Expected p_minus-p_plus. Absence is distinct from a zero target. */
    std::optional<Real> prescribed_pressure_jump{};
    /** Expected (tau_minus-tau_plus)n in global physical components. */
    std::optional<std::array<Real, 3>> prescribed_viscous_traction_jump{};

    [[nodiscard]] friend bool operator==(
        const IncompressibleTwoFluidDiagnosticParameters&,
        const IncompressibleTwoFluidDiagnosticParameters&) = default;
};

/** Point evaluators for one phase field pair on the shared background cell. */
struct IncompressibleTwoFluidPhaseEvaluators {
    FreeSurfaceDiscreteFunctionalVectorEvaluator velocity{};
    FreeSurfaceGeometryScalarEvaluator pressure{};
};

/**
 * Geometry input needed to reproduce h_n = 2 |K| / |Gamma cap K| from the
 * production Nitsche form. The callback returns the full physical parent-cell
 * measure, not a cut-phase measure.
 */
struct IncompressibleTwoFluidCellMeasureEvaluator {
    std::function<Real(GlobalIndex)> physical_cell_measure{};

    [[nodiscard]] bool canEvaluate() const noexcept {
        return static_cast<bool>(physical_cell_measure);
    }
};

/** Additive phase-volume quantities prior to communicator reduction. */
struct IncompressibleTwoFluidPhaseDiagnosticAccumulator {
    std::size_t owned_quadrature_point_count{0u};
    Real volume{0.0};
    std::array<Real, 3> velocity_integral{{0.0, 0.0, 0.0}};
    Real velocity_squared_integral{0.0};
};

/** Additive interface and phase quantities prior to communicator reduction. */
struct IncompressibleTwoFluidDiagnosticAccumulator {
    std::uint64_t snapshot_revision_key{0u};
    /** Assembly-stage effective step used only by the transient penalty. */
    std::optional<Real> transient_penalty_effective_dt{};
    std::size_t owned_interface_quadrature_point_count{0u};
    Real interface_measure{0.0};
    std::array<Real, 3> interface_normal_integral{{0.0, 0.0, 0.0}};
    Real velocity_jump_squared{0.0};
    Real normal_velocity_jump_squared{0.0};
    Real tangential_velocity_jump_squared{0.0};
    Real negative_normal_flux{0.0};
    Real positive_normal_flux{0.0};
    std::array<Real, 3> negative_traction_integral{{0.0, 0.0, 0.0}};
    std::array<Real, 3> positive_traction_integral{{0.0, 0.0, 0.0}};
    std::array<Real, 3> traction_jump_integral{{0.0, 0.0, 0.0}};
    Real traction_jump_normal_integral{0.0};
    Real traction_jump_squared{0.0};
    std::array<Real, 3> negative_viscous_traction_integral{{0.0, 0.0, 0.0}};
    std::array<Real, 3> positive_viscous_traction_integral{{0.0, 0.0, 0.0}};
    std::array<Real, 3> viscous_traction_jump_integral{{0.0, 0.0, 0.0}};
    Real viscous_traction_jump_squared{0.0};
    std::optional<Real> prescribed_stress_jump_residual_squared{};
    Real pressure_jump_integral{0.0};
    Real pressure_jump_squared{0.0};
    std::optional<Real> prescribed_pressure_jump_error_squared{};
    std::optional<Real> prescribed_viscous_traction_jump_error_squared{};
    Real surface_energy_work{0.0};
    Real nitsche_consistency_work{0.0};
    Real nitsche_adjoint_work{0.0};
    Real nitsche_penalty_work{0.0};
    IncompressibleTwoFluidPhaseDiagnosticAccumulator negative_phase{};
    IncompressibleTwoFluidPhaseDiagnosticAccumulator positive_phase{};
};

/** Final phase quantities after communicator reduction and normalization. */
struct IncompressibleTwoFluidPhaseDiagnosticState {
    geometry::CutIntegrationSide side{
        geometry::CutIntegrationSide::Negative};
    std::size_t quadrature_point_count{0u};
    Real density{0.0};
    Real volume{0.0};
    Real mass{0.0};
    std::array<Real, 3> momentum{{0.0, 0.0, 0.0}};
    Real kinetic_energy{0.0};
};

/**
 * Communicator-global raw accepted-stage record. Integrals remain available
 * alongside normalized quantities so downstream gates do not have to infer
 * whether a displayed zero was measured or inapplicable.
 */
struct IncompressibleTwoFluidDiagnosticState {
    std::uint64_t snapshot_revision_key{0u};
    /** Absence denotes a genuinely steady interface penalty. */
    std::optional<Real> transient_penalty_effective_dt{};
    std::size_t interface_quadrature_point_count{0u};
    Real interface_measure{0.0};
    std::array<Real, 3> interface_normal_integral{{0.0, 0.0, 0.0}};
    Real velocity_jump_squared{0.0};
    Real normal_velocity_jump_squared{0.0};
    Real tangential_velocity_jump_squared{0.0};
    Real negative_normal_flux{0.0};
    Real positive_normal_flux{0.0};
    Real normal_flux_jump{0.0};
    Real negative_mass_flux{0.0};
    Real positive_mass_flux{0.0};
    std::array<Real, 3> negative_traction_integral{{0.0, 0.0, 0.0}};
    std::array<Real, 3> positive_traction_integral{{0.0, 0.0, 0.0}};
    std::array<Real, 3> traction_jump_integral{{0.0, 0.0, 0.0}};
    Real traction_jump_normal_integral{0.0};
    Real traction_jump_squared{0.0};
    std::array<Real, 3> negative_viscous_traction_integral{{0.0, 0.0, 0.0}};
    std::array<Real, 3> positive_viscous_traction_integral{{0.0, 0.0, 0.0}};
    std::array<Real, 3> viscous_traction_jump_integral{{0.0, 0.0, 0.0}};
    Real viscous_traction_jump_squared{0.0};
    std::optional<Real> prescribed_stress_jump_residual_squared{};
    Real pressure_jump_integral{0.0};
    Real mean_pressure_jump{0.0};
    Real pressure_jump_squared{0.0};
    std::optional<Real> prescribed_pressure_jump_error_squared{};
    std::optional<Real> prescribed_viscous_traction_jump_error_squared{};
    Real surface_energy_work{0.0};
    Real nitsche_consistency_work{0.0};
    Real nitsche_adjoint_work{0.0};
    Real nitsche_penalty_work{0.0};
    IncompressibleTwoFluidPhaseDiagnosticState negative_phase{};
    IncompressibleTwoFluidPhaseDiagnosticState positive_phase{};
};

/** One phase's before/after momentum contract across phase maintenance. */
struct IncompressibleTwoFluidPhaseMomentumReconciliation {
    geometry::CutIntegrationSide side{
        geometry::CutIntegrationSide::Negative};
    Real density{0.0};
    Real raw_volume{0.0};
    Real corrected_volume{0.0};
    Real raw_mass{0.0};
    Real corrected_mass{0.0};
    std::array<Real, 3> raw_momentum{{0.0, 0.0, 0.0}};
    std::array<Real, 3> corrected_momentum{{0.0, 0.0, 0.0}};
    std::array<Real, 3> momentum_delta{{0.0, 0.0, 0.0}};
    Real momentum_delta_norm{0.0};
    Real momentum_reference_norm{0.0};
    Real allowed_momentum_delta{0.0};
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
    FreeSurfaceGeometryRevision raw_geometry_revision{};
    FreeSurfaceGeometryRevision corrected_geometry_revision{};
    std::uint64_t raw_algebraic_revision{0u};
    std::uint64_t corrected_algebraic_revision{0u};
    Real relative_tolerance{0.0};
    bool velocity_update_applied{false};
    bool satisfied{false};
    IncompressibleTwoFluidPhaseMomentumReconciliation negative_phase{};
    IncompressibleTwoFluidPhaseMomentumReconciliation positive_phase{};

};

/** Evaluate rank-owned additive quantities on one immutable snapshot. */
[[nodiscard]] IncompressibleTwoFluidDiagnosticAccumulator
evaluateLocalIncompressibleTwoFluidDiagnostics(
    const FreeSurfaceGeometrySnapshot& snapshot,
    const IncompressibleTwoFluidDiagnosticParameters& parameters,
    const IncompressibleTwoFluidPhaseEvaluators& negative_phase,
    const IncompressibleTwoFluidPhaseEvaluators& positive_phase,
    const IncompressibleTwoFluidCellMeasureEvaluator& cell_measure,
    Real effective_dt);

/** Normalize one already-reduced accumulator and enforce its exact identities. */
[[nodiscard]] IncompressibleTwoFluidDiagnosticState
finalizeIncompressibleTwoFluidDiagnostics(
    const IncompressibleTwoFluidDiagnosticAccumulator& accumulator,
    const IncompressibleTwoFluidDiagnosticParameters& parameters);

/** Build the explicit no-hidden-update phasewise momentum contract. */
[[nodiscard]] IncompressibleTwoFluidMomentumReconciliation
buildIncompressibleTwoFluidMomentumReconciliation(
    int interface_marker,
    const FreeSurfaceGeometryRevision& raw_geometry_revision,
    const IncompressibleTwoFluidDiagnosticState& raw,
    std::uint64_t raw_algebraic_revision,
    const FreeSurfaceGeometryRevision& corrected_geometry_revision,
    const IncompressibleTwoFluidDiagnosticState& corrected,
    std::uint64_t corrected_algebraic_revision,
    Real relative_tolerance);

/** Reject any inconsistent or non-finite reconciliation record. */
void validateIncompressibleTwoFluidMomentumReconciliation(
    const IncompressibleTwoFluidMomentumReconciliation& reconciliation);

} // namespace svmp::FE::interfaces

#endif // SVMP_FE_INTERFACES_INCOMPRESSIBLETWOFLUIDDIAGNOSTICS_H
