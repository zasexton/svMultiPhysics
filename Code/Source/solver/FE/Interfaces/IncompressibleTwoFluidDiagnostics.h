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
    std::optional<Real> prescribed_stress_jump_residual_squared{};
    Real pressure_jump_integral{0.0};
    Real pressure_jump_squared{0.0};
    std::optional<Real> prescribed_pressure_jump_error_squared{};
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
    std::optional<Real> prescribed_stress_jump_residual_squared{};
    Real pressure_jump_integral{0.0};
    Real mean_pressure_jump{0.0};
    Real pressure_jump_squared{0.0};
    std::optional<Real> prescribed_pressure_jump_error_squared{};
    Real surface_energy_work{0.0};
    Real nitsche_consistency_work{0.0};
    Real nitsche_adjoint_work{0.0};
    Real nitsche_penalty_work{0.0};
    IncompressibleTwoFluidPhaseDiagnosticState negative_phase{};
    IncompressibleTwoFluidPhaseDiagnosticState positive_phase{};
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

} // namespace svmp::FE::interfaces

#endif // SVMP_FE_INTERFACES_INCOMPRESSIBLETWOFLUIDDIAGNOSTICS_H
