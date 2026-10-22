/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_FREE_SURFACE_RECORD_TYPES_H
#define SVMP_FE_SYSTEMS_FREE_SURFACE_RECORD_TYPES_H

/**
 * @file FreeSurfaceRecordTypes.h
 * @brief Data types of the free-surface accepted-step records: functional,
 *        residual-work and exterior-boundary energy declarations, accepted
 *        functional states, contact stages and history records.
 *
 * This header is a staging point.  The records and their logic belong to
 * the Navier-Stokes free-surface physics (Physics/Formulations/NavierStokes/
 * FreeSurface); these plain data types left FESystem.h first and keep their
 * namespace for now.  The accepted contact-stage state and its
 * generalized-alpha provenance are also read by FE/LevelSet.
 */

#include "Core/Types.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Systems/OperatorRegistry.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

/** Static one-phase capillary-balance construction selected by the owner. */
enum class FreeSurfaceCapillaryBalanceMethod : std::uint8_t {
    Unselected,
    /** Snapshot energy variation plus a fixed-volume stationary geometry. */
    DiscreteEnergyVolumeStationarity,
    /** Total surface-plus-Young-wall energy gradient represented as traction. */
    KinematicAreaGradientEnergyTraction
};

/** Evidence boundary for the selected capillary-balance construction. */
enum class FreeSurfaceCapillaryBalanceQualification : std::uint8_t {
    Unselected,
    /** Low-level identities only; static-cap qualification is still open. */
    PrerequisiteOnly,
    Qualified
};

/**
 * Residual components whose accepted operator-stage pairing contributes to
 * the free-surface work account.
 *
 * These channels contain only assembled residual work.  Stored-energy,
 * physical-dissipation, maintenance, aggregation-projection, extension, and
 * pruning contributions have separate owners and must not be folded into one
 * of these declarations.
 */
enum class FreeSurfaceResidualWorkChannel : std::uint8_t {
    Convection,
    PressureContinuity,
    NonconservativeBodyForce,
    WeakBoundary,
    VmsPspg,
    CutStabilization,
    GhostPenalty
};

[[nodiscard]] const char* freeSurfaceResidualWorkChannelName(
    FreeSurfaceResidualWorkChannel channel) noexcept;

enum class FreeSurfaceResidualWorkApplicability : std::uint8_t {
    Produced,
    NotApplicable
};

/**
 * One setup-time owner for an accepted-stage residual-work channel.
 *
 * A produced channel names an installed operator containing exactly that
 * residual component.  A channel that is not active in the effective
 * formulation remains explicit through NotApplicable and carries no operator
 * tag.  The signed step work is evaluated later as minus dt times the
 * constrained residual paired with the converged operator-stage state.
 */
struct FreeSurfaceResidualWorkDeclaration {
    FreeSurfaceResidualWorkChannel channel{
        FreeSurfaceResidualWorkChannel::Convection};
    FreeSurfaceResidualWorkApplicability applicability{
        FreeSurfaceResidualWorkApplicability::NotApplicable};
    OperatorTag operator_tag{};
    std::string owner_component{};

    friend bool operator==(
        const FreeSurfaceResidualWorkDeclaration&,
        const FreeSurfaceResidualWorkDeclaration&) = default;
};

enum class FreeSurfaceExternalBoundaryEnergyApplicability : std::uint8_t {
    Unresolved,
    NotApplicable
};

/**
 * Setup-time proof boundary for external work not represented by the
 * prescribed exterior-pressure or residual-work producers.
 *
 * NotApplicable is allowed only when the owning physics module found no
 * configured route for the channel. Unresolved keeps the complete energy
 * record fail closed until a distinct accepted-stage work producer exists.
 * Homogeneous velocity data are recorded separately because prescribed wall
 * motion can inject energy even on a geometrically closed boundary.
 */
struct FreeSurfaceExternalBoundaryEnergyDeclaration {
    FreeSurfaceExternalBoundaryEnergyApplicability imposed_traction{
        FreeSurfaceExternalBoundaryEnergyApplicability::Unresolved};
    FreeSurfaceExternalBoundaryEnergyApplicability open_boundary_flux{
        FreeSurfaceExternalBoundaryEnergyApplicability::Unresolved};
    bool homogeneous_velocity_data{false};
    bool pressure_dirichlet_absent{false};
    std::string owner_component{};

    friend bool operator==(
        const FreeSurfaceExternalBoundaryEnergyDeclaration&,
        const FreeSurfaceExternalBoundaryEnergyDeclaration&) = default;
};

struct FreeSurfaceDiscreteFunctionalDeclaration {
    int interface_marker{-1};
    FieldId level_set_field{INVALID_FIELD_ID};
    /** Prescribed curvature carrying the selected total energy gradient. */
    FieldId curvature_field{INVALID_FIELD_ID};
    FieldId velocity_field{INVALID_FIELD_ID};
    std::string geometry_domain_id{};
    interfaces::FreeSurfaceDiscreteFunctionalParameters parameters{};
    std::optional<interfaces::FreeSurfaceActiveVolumeEnergyParameters>
        active_volume_energy_parameters{};
    /**
     * The active-volume gravitational acceleration is the complete
     * velocity-independent body acceleration present in the production
     * static momentum residual.  Static equilibrium certification fails
     * closed unless the owning physics module makes this declaration.
     */
    bool static_conservative_body_force_complete{false};
    std::optional<
        interfaces::FreeSurfaceActiveVolumeDissipationParameters>
        active_volume_dissipation_parameters{};
    std::optional<
        interfaces::FreeSurfaceExternalPressurePowerParameters>
        external_pressure_power_parameters{};
    bool endpoint_functional_power_enabled{false};
    FreeSurfaceCapillaryBalanceMethod capillary_balance_method{
        FreeSurfaceCapillaryBalanceMethod::Unselected};
    FreeSurfaceCapillaryBalanceQualification
        capillary_balance_qualification{
            FreeSurfaceCapillaryBalanceQualification::Unselected};
    std::string owner_component{};
};

/**
 * Authentic first-order generalized-alpha parameters retained with an
 * accepted contact stage.  This is optional only for endpoint stages.  The
 * supported TimeLoop rho-infinity family has alpha_f == gamma in [0.5, 1]
 * and alpha_m == 2*alpha_f - 0.5 in [0.5, 1.5].
 */
struct FreeSurfaceFirstOrderGeneralizedAlphaProvenance {
    Real alpha_m{1.0};
    Real alpha_f{1.0};
    Real gamma{1.0};
    Real dt{0.0};

    [[nodiscard]] friend bool operator==(
        const FreeSurfaceFirstOrderGeneralizedAlphaProvenance&,
        const FreeSurfaceFirstOrderGeneralizedAlphaProvenance&) = default;
};

struct FreeSurfaceAcceptedContactStageState {
    Real stage_time{0.0};
    Real stage_alpha_f{1.0};
    std::optional<FreeSurfaceFirstOrderGeneralizedAlphaProvenance>
        first_order_generalized_alpha{};
    // Communicator-consistent fingerprints of FE-ordered algebraic content.
    std::uint64_t previous_state_revision{0};
    std::uint64_t endpoint_state_revision{0};
    // Composite fingerprint over the previous/endpoint content, accepted
    // snapshot, stage time/parameters, and stage solution.
    std::uint64_t stage_state_revision{0};
    interfaces::FreeSurfaceGeometryRevision geometry_revision{};
    interfaces::FreeSurfaceDynamicContactState state{};
};

/**
 * Accepted-stage contact-centroid kinematics derived from two consecutive
 * history records.
 *
 * This is a secant velocity of the measure-weighted contact-position
 * centroid projected onto a common oriented footprint direction.  It is not
 * a pointwise contact-line velocity.  The previous accepted record and stage
 * revisions make the comparison with the instantaneous fluid-trace speed
 * reproducible.
 */
struct FreeSurfaceAcceptedContactLineKinematics {
    int boundary_marker{-1};
    std::uint64_t previous_accepted_step{0};
    Real previous_accepted_time{0.0};
    Real previous_stage_time{0.0};
    std::uint64_t previous_stage_state_revision{0};
    std::uint64_t previous_snapshot_revision_key{0};
    Real stage_time_interval{0.0};
    std::array<Real, 3> previous_mean_contact_position{
        {0.0, 0.0, 0.0}};
    std::array<Real, 3> projection_direction{{0.0, 0.0, 0.0}};
    Real projected_contact_centroid_speed{0.0};
    Real mean_fluid_contact_speed{0.0};
    Real fluid_minus_geometric_contact_speed{0.0};
};

struct AcceptedFreeSurfaceDiscreteFunctionalState {
    int interface_marker{-1};
    interfaces::FreeSurfaceGeometryRevision geometry_revision{};
    // Communicator-consistent topology-only fingerprint for the exact
    // authoritative snapshot rules used by this state.
    std::uint64_t cut_topology_revision{0};
    // Communicator-wide truth value for any pruning in the authoritative
    // snapshot. False is the proof required to mark pruning work inapplicable.
    bool geometry_pruning_occurred{false};
    interfaces::FreeSurfaceDiscreteFunctionalState state{};
    std::optional<
        interfaces::FreeSurfaceDiscreteFunctionalVariationState>
        endpoint_functional_power{};
    std::optional<interfaces::FreeSurfaceActiveVolumeEnergyState>
        active_volume_energy{};
    std::optional<
        interfaces::FreeSurfaceActiveVolumeDissipationState>
        active_volume_dissipation{};
    std::optional<
        interfaces::FreeSurfaceExternalPressurePowerState>
        external_pressure_power{};
    std::optional<
        interfaces::FreeSurfaceBackwardEulerKineticWorkState>
        backward_euler_kinetic_work{};
    std::optional<FreeSurfaceAcceptedContactStageState> contact_stage{};
    std::vector<FreeSurfaceAcceptedContactLineKinematics>
        contact_line_kinematics{};
};

struct FreeSurfaceDiscreteFunctionalHistoryRecord {
    std::uint64_t accepted_step{0};
    Real accepted_time{0.0};
    Real dt{0.0};
    // Communicator-consistent fingerprints before and after maintenance.
    std::uint64_t pre_maintenance_endpoint_state_revision{0};
    std::uint64_t state_revision{0};
    std::optional<std::uint64_t> extension_map_revision{};
    FreeSurfaceDiscreteFunctionalDeclaration declaration{};
    interfaces::FreeSurfaceGeometryRevision geometry_revision{};
    std::uint64_t cut_topology_revision{0};
    bool geometry_pruning_occurred{false};
    interfaces::FreeSurfaceDiscreteFunctionalState state{};
    std::optional<
        interfaces::FreeSurfaceDiscreteFunctionalVariationState>
        endpoint_functional_power{};
    std::optional<interfaces::FreeSurfaceActiveVolumeEnergyState>
        active_volume_energy{};
    std::optional<
        interfaces::FreeSurfaceActiveVolumeDissipationState>
        active_volume_dissipation{};
    std::optional<
        interfaces::FreeSurfaceExternalPressurePowerState>
        external_pressure_power{};
    std::optional<
        interfaces::FreeSurfaceBackwardEulerKineticWorkState>
        backward_euler_kinetic_work{};
    std::optional<FreeSurfaceAcceptedContactStageState> contact_stage{};
    std::vector<FreeSurfaceAcceptedContactLineKinematics>
        contact_line_kinematics{};
};

[[nodiscard]] std::vector<FreeSurfaceAcceptedContactLineKinematics>
deriveFreeSurfaceAcceptedContactLineKinematics(
    const FreeSurfaceDiscreteFunctionalHistoryRecord& previous,
    std::uint64_t accepted_step,
    const FreeSurfaceAcceptedContactStageState& current);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_FREE_SURFACE_RECORD_TYPES_H
