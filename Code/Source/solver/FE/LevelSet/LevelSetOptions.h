#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Public level-set option structs, source enums, and cadence helpers.
 */

#include "Core/Types.h"
#include "Forms/BoundaryConditions.h"
#include "Spaces/FunctionSpace.h"

#include <array>
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
    ConstantVector
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
    // Residual-based isotropic discontinuity capturing within the SUPG path.
    // It has a separate switch so smooth manufactured solutions can disable it
    // while steep level-set fronts retain cross-stream control.
    bool discontinuity_capturing_enabled{false};
    Real discontinuity_capturing_scale{0.1};
    Real gradient_epsilon{1.0e-12};
    // Smooth |R| only inside the DC diffusivity as
    // sqrt(R^2+eps_R^2)-eps_R.  This preserves zero diffusivity and selects
    // the symmetric/Clarke derivative zero at R=0, avoiding the arbitrary
    // one-sided abs tangent used by generic FormExpr::abs.
    Real discontinuity_capturing_residual_epsilon{1.0e-12};
    // Caps artificial diffusivity at this multiple of h*|u| + h^2/dt.  A
    // positive cap prevents a large transient residual from erasing the
    // interface.
    Real discontinuity_capturing_max_courant{0.5};
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

enum class LevelSetReinitializationMethod {
    HamiltonJacobiPDE,
    FastMarching,
    Projection
};

struct LevelSetReinitializationOptions {
    bool enabled{false};
    LevelSetReinitializationMethod method{LevelSetReinitializationMethod::Projection};
    int cadence_steps{1};
    int max_iterations{10};
    Real pseudo_time_step_scale{0.3};
    Real interface_band_width{3.0};
    Real signed_distance_tolerance{1.0e-6};
    // Deprecated compatibility knob.  The interface-preserving projection no
    // longer freezes a near-interface band.  Zero and negative values select
    // the topology-preserving algorithm; positive values are rejected so a
    // configuration cannot silently request the old freeze behavior.
    Real preserve_band_width{0.0};
    // Maximum allowed geometric motion of an original edge/interface
    // crossing.  The projection line-search enforces this fail-closed guard;
    // it is not a target displacement.
    Real max_zero_set_displacement{1.0e-10};
};

struct LevelSetVolumeCorrectionOptions {
    bool enabled{false};
    int cadence_steps{1};
    bool use_initial_negative_volume_as_target{true};
    Real target_negative_volume{0.0};
    Real volume_tolerance{1.0e-10};
    int max_iterations{50};
    // Treat the global shift as a fallback: do nothing until the volume error
    // exceeds this fraction of total volume (in addition to volume_tolerance).
    Real minimum_relative_volume_error{1.0e-6};
    // A correction may move no edge zero-crossing farther than this fraction
    // of the smallest mesh edge.  This also conservatively bounds contact-line
    // motion, since wall contact crossings are a subset of edge crossings.
    Real maximum_interface_displacement_fraction{0.1};
    // Sum the certified maximum displacement from every applied event and
    // fail closed before that path-length bound exceeds this fraction of the
    // smallest edge observed by the application.  This prevents many
    // individually small global shifts from producing unbounded cumulative
    // interface/contact-line drift.
    Real maximum_cumulative_interface_displacement_fraction{1.0};
};

struct LevelSetInflowBoundary {
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
    Real impermeable_normal_velocity_tolerance{1.0e-10};
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
    LevelSetReinitializationOptions reinitialization{};
    LevelSetVolumeCorrectionOptions volume_correction{};
    LevelSetBoundaryOptions boundaries{};
    LevelSetConservativePhaseOptions conservative_phase{};
};

enum class LevelSetConservationDiagnostic {
    PlainAdvectionNotConservative,
    ConservativeDivergenceAdvectionNotLocallyConservative,
    ReinitializedAdvectionNotConservative,
    VolumeCorrectedAdvectionNotLocallyConservative,
    ConservativePhaseIndicatorLocallyConservative
};

[[nodiscard]] LevelSetConservationDiagnostic levelSetConservationDiagnostic(
    LevelSetTransportForm transport_form,
    const LevelSetReinitializationOptions& reinitialization,
    const LevelSetVolumeCorrectionOptions& volume_correction) noexcept;

[[nodiscard]] LevelSetConservationDiagnostic levelSetConservationDiagnostic(
    const LevelSetTransportOptions& options) noexcept;

[[nodiscard]] const char* levelSetConservationDiagnosticName(
    LevelSetConservationDiagnostic diagnostic) noexcept;

[[nodiscard]] bool shouldReinitializeLevelSet(
    const LevelSetReinitializationOptions& options,
    int completed_step_index) noexcept;

[[nodiscard]] bool shouldApplyLevelSetVolumeCorrection(
    const LevelSetVolumeCorrectionOptions& options,
    int completed_step_index) noexcept;

} // namespace svmp::FE::level_set
