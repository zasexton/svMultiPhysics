#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Public level-set option structs, source enums, and cadence helpers.
 */

#include "Core/Types.h"
#include "Forms/BoundaryConditions.h"
#include "LevelSet/LevelSetConservativePhaseRegions.h"
#include "Spaces/FunctionSpace.h"

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace svmp::FE::level_set {

using ScalarValue = forms::bc::ScalarValue;

enum class LevelSetFieldSource {
    Unknown,
    PrescribedData
};

enum class LevelSetVelocitySource {
    CoupledField,
    PrescribedData,
    ConstantVector,
    MaterialInterfacePhasePair
};

enum class LevelSetTransportForm {
    Advective,
    ConservativeDivergence
};

enum class LevelSetPhaseSide {
    Negative,
    Positive
};

struct LevelSetFieldOptions {
    std::string field_name{"level_set"};
    LevelSetFieldSource source{LevelSetFieldSource::Unknown};
    bool auto_register_field{true};
};

struct LevelSetVelocityOptions {
    std::string field_name{"Velocity"};
    LevelSetVelocitySource source{LevelSetVelocitySource::CoupledField};
    bool auto_register_field{false};
    std::shared_ptr<const spaces::FunctionSpace> space{};
    std::array<Real, 3> constant_value{0.0, 0.0, 0.0};
    // Required only for MaterialInterfacePhasePair. The matching declaration
    // owns both phase fields, the level-set source, and the trace weights.
    int material_interface_marker{-1};
    // Non-empty only for the monolithic wet-side extension path.  field_name
    // is then an algebraic extension unknown E, while this names the physical
    // velocity u used by the frozen sparse extension constraints E=P(phi)u.
    // A state-dependent extension must use this algebraic path; treating E as
    // PrescribedData would silently omit dR_phi/du.
    std::string algebraic_extension_source_field_name{};
};

struct LevelSetSUPGOptions {
    bool enabled{false};
    Real tau_scale{0.5};
    Real velocity_epsilon{1.0e-12};
    // Tezduyar-style transient contribution to the inverse stabilization
    // time scale.  The metric contribution is u . (J^-T J^-1) u.
    Real transient_scale{2.0};
};

/**
 * @brief Accepted-candidate invariant-domain control for P1 H1 level sets.
 *
 * The limiter bounds every transported coefficient by the extrema of the
 * previous accepted solution on its one-ring cell patch.  This is a nodal
 * invariant-domain projection, not an FCT scheme: it is deliberately
 * classified as nonconservative and can reduce local accuracy where active.
 * A one-ring domain of dependence is only physically admissible for a cell
 * Courant number no larger than @ref maximum_courant, so the production path
 * rejects (and, with an adaptive time loop, retries) larger steps.
 *
 * Boundary faces not declared as inflow or outflow are treated as
 * impermeable.  Their advecting normal velocity is checked before the
 * projection; the scalar limiter never hides an incompatible velocity wall
 * condition.
 */
struct LevelSetBoundPreservingOptions {
    bool enabled{false};
    Real bound_tolerance{1.0e-12};
    Real sign_tolerance{1.0e-12};
    Real maximum_courant{1.0};
    Real courant_tolerance{1.0e-12};
    bool enforce_courant_limit{true};
    bool enforce_impermeable_boundaries{true};
    Real impermeable_normal_velocity_tolerance{1.0e-10};
};

struct LevelSetInterfaceKinematicOptions {
    bool enabled{false};
    int interface_marker{-1};
    Real weight_scale{1.0};
};

/**
 * @brief Accepted-step reconciliation of the transported P1 level set with
 * the kinematic interface flux of its own transport velocity.
 *
 * A Galerkin transport step moves the discrete zero set consistently with the
 * velocity only in an L2(Omega) sense; where the velocity varies on the mesh
 * scale (a moving contact line) the sharp liquid measure drifts.  When enabled,
 * every accepted step applies the local, parameter-free correction described
 * in LevelSetKinematicReconciliation.h: the trapezoidal volume change of the
 * step equals the trapezoidal interface flux of the transport velocity.  It is
 * not a volume target and not a global shift.
 */
struct LevelSetKinematicReconciliationOptions {
    bool enabled{false};
};

/**
 * @brief Accepted-step one-ring bounds on sign-definite patches.
 *
 * Disabled by default.  When enabled, every accepted step clamps each node
 * whose whole patch lies in one phase into the range of the previous values
 * over its patch (LevelSetSignDefinitePatchBounds.h), the local maximum
 * principle of exact transport.  Nodes of cut cells are never changed, so the
 * interface and the liquid measure are untouched; a node away from the
 * interface cannot cross the isovalue.  There is no numerical parameter.
 */
struct LevelSetSignDefinitePatchBoundsOptions {
    bool enabled{false};
};

/**
 * @brief Accepted-step interface-preserving redistancing (opt-in; Triangle3
 *        and Tetra4).
 *
 * Disabled by default.  When enabled, every accepted step, after the
 * kinematic reconciliation and the sign-definite patch bounds, replaces the
 * level set by the signed distance to its own zero polygon, with the polygon
 * ends on the boundary (contact points) continued by rays
 * (LevelSetInterfacePreservingRedistancing.h), or on Tetra4 meshes to its own
 * zero surface, with the open surface edges on the boundary (contact lines)
 * continued by half-strips in the surface plane
 * (LevelSetInterfacePreservingRedistancing3D.h), and then restores the lumped
 * interface position node by node with the kinematic reconciliation at zero
 * velocity.  It keeps |grad phi| = 1 next to moving contact lines, where the
 * transport otherwise steepens and flattens the field without bound.  There
 * is no numerical parameter.
 */
struct LevelSetInterfacePreservingRedistancingOptions {
    bool enabled{false};
};

struct LevelSetInflowBoundary {
    // The transported level set is defined on the background space, so this
    // condition intentionally covers the complete physical inlet. It is not
    // a liquid-only generated-active boundary condition.
    int boundary_marker{-1};
    ScalarValue value{0.0};
    Real penalty_scale{1.0};
};

struct LevelSetOutflowBoundary {
    int boundary_marker{-1};
};

struct LevelSetBoundaryOptions {
    std::vector<LevelSetInflowBoundary> inflow{};
    std::vector<LevelSetOutflowBoundary> outflow{};
};

enum class LevelSetConservativePhaseBoundaryFluxPolicy {
    ClosedDomainDiscreteQFluxOnly,
    GloballyBalancedDiscreteQFlux
};

[[nodiscard]] constexpr const char*
levelSetConservativePhaseBoundaryFluxPolicyName(
    LevelSetConservativePhaseBoundaryFluxPolicy policy) noexcept
{
    switch (policy) {
        case LevelSetConservativePhaseBoundaryFluxPolicy::
            ClosedDomainDiscreteQFluxOnly:
            return "closed_domain_discrete_q_flux_only";
        case LevelSetConservativePhaseBoundaryFluxPolicy::
            GloballyBalancedDiscreteQFlux:
            return "globally_balanced_discrete_q_flux";
    }
    return "unknown";
}

[[nodiscard]] constexpr const char*
levelSetConservativePhaseBoundaryFluxScope(
    LevelSetConservativePhaseBoundaryFluxPolicy policy) noexcept
{
    switch (policy) {
        case LevelSetConservativePhaseBoundaryFluxPolicy::
            ClosedDomainDiscreteQFluxOnly:
            return "discrete_q_flux_not_pointwise_velocity_normal";
        case LevelSetConservativePhaseBoundaryFluxPolicy::
            GloballyBalancedDiscreteQFlux:
            return "globally_balanced_discrete_q_flux_not_pointwise_velocity_normal";
    }
    return "unknown";
}

[[nodiscard]] inline bool
levelSetConservativePhaseBoundaryFluxSatisfied(
    LevelSetConservativePhaseBoundaryFluxPolicy policy,
    Real maximum_nodal_boundary_mass_transfer,
    Real total_boundary_mass_transfer,
    Real boundary_mass_tolerance) noexcept
{
    if (!std::isfinite(maximum_nodal_boundary_mass_transfer) ||
        !std::isfinite(total_boundary_mass_transfer) ||
        !std::isfinite(boundary_mass_tolerance) ||
        maximum_nodal_boundary_mass_transfer < Real{0.0} ||
        boundary_mass_tolerance < Real{0.0} ||
        std::abs(total_boundary_mass_transfer) >
            boundary_mass_tolerance) {
        return false;
    }
    switch (policy) {
        case LevelSetConservativePhaseBoundaryFluxPolicy::
            ClosedDomainDiscreteQFluxOnly:
            return maximum_nodal_boundary_mass_transfer <=
                   boundary_mass_tolerance;
        case LevelSetConservativePhaseBoundaryFluxPolicy::
            GloballyBalancedDiscreteQFlux:
            return true;
    }
    return false;
}

/**
 * @brief Explicit conservative P1 phase state coupled to level-set geometry.
 *
 * The level set remains a provisional geometry representation. The additional
 * scalar indicator is held at its previous accepted value during the
 * monolithic solve, advanced by a conservative graph stage before acceptance,
 * and then used as the phase-measure authority for geometry reconciliation.
 */
struct LevelSetConservativePhaseOptions {
    bool enabled{false};
    LevelSetFieldOptions liquid_indicator{
        .field_name = "liquid_indicator",
        .source = LevelSetFieldSource::Unknown,
        .auto_register_field = true,
    };
    LevelSetPhaseSide liquid_side{LevelSetPhaseSide::Negative};
    Real invariant_tolerance{1.0e-12};
    Real component_activity_tolerance{1.0e-8};
    Real maximum_courant{1.0};
    bool enforce_courant_limit{true};
    bool require_constant_preservation{true};
    // Relative phasewise momentum gate applied after geometry reconciliation.
    // The current policy is fail-closed and never changes velocity implicitly.
    Real momentum_relative_tolerance{1.0e-10};
    bool write_flux_artifacts{false};
    int flux_artifact_cadence_steps{1};
    bool classify_nonprimary_components_as_satellites{false};
    std::vector<LevelSetPhaseRegionBox> fixed_flux_regions{};
    LevelSetConservativePhaseBoundaryFluxPolicy boundary_flux_policy{
        LevelSetConservativePhaseBoundaryFluxPolicy::
            ClosedDomainDiscreteQFluxOnly};
    Real impermeable_normal_velocity_tolerance{1.0e-10};
    // The current conservative split does not implement a pointwise u.n wall
    // check. Translators retain this bit so an explicit legacy request can
    // fail closed instead of silently treating the value as an active policy.
    bool pointwise_impermeable_velocity_tolerance_explicitly_requested{false};
    bool reconcile_geometry{true};
    Real geometry_measure_tolerance{1.0e-10};
    int geometry_correction_max_iterations{50};
    Real maximum_geometry_displacement_fraction{0.1};
};

struct LevelSetTransportOptions {
    std::string operator_tag{"level_set"};
    LevelSetTransportForm transport_form{LevelSetTransportForm::Advective};
    LevelSetFieldOptions level_set{};
    LevelSetVelocityOptions velocity{};
    LevelSetSUPGOptions supg{};
    LevelSetBoundPreservingOptions bound_preserving{};
    LevelSetInterfaceKinematicOptions interface_kinematic{};
    LevelSetBoundaryOptions boundaries{};
    LevelSetConservativePhaseOptions conservative_phase{};
    LevelSetKinematicReconciliationOptions kinematic_reconciliation{};
    LevelSetSignDefinitePatchBoundsOptions sign_definite_patch_bounds{};
    LevelSetInterfacePreservingRedistancingOptions interface_preserving_redistancing{};
};

enum class LevelSetConservationDiagnostic {
    PlainAdvectionNotConservative,
    ConservativeDivergenceAdvectionNotLocallyConservative,
    ConservativePhaseIndicatorLocallyConservative
};

[[nodiscard]] LevelSetConservationDiagnostic levelSetConservationDiagnostic(
    LevelSetTransportForm transport_form) noexcept;

[[nodiscard]] LevelSetConservationDiagnostic levelSetConservationDiagnostic(
    const LevelSetTransportOptions& options) noexcept;

[[nodiscard]] const char* levelSetConservationDiagnosticName(
    LevelSetConservationDiagnostic diagnostic) noexcept;

[[nodiscard]] bool shouldApplyLevelSetKinematicReconciliation(
    const LevelSetKinematicReconciliationOptions& options,
    int completed_step_index) noexcept;

[[nodiscard]] bool shouldApplyLevelSetSignDefinitePatchBounds(
    const LevelSetSignDefinitePatchBoundsOptions& options,
    int completed_step_index) noexcept;

[[nodiscard]] bool shouldApplyLevelSetInterfacePreservingRedistancing(
    const LevelSetInterfacePreservingRedistancingOptions& options,
    int completed_step_index) noexcept;

} // namespace svmp::FE::level_set
