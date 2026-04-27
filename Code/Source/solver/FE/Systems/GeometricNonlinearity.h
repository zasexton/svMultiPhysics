/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_GEOMETRIC_NONLINEARITY_H
#define SVMP_FE_SYSTEMS_GEOMETRIC_NONLINEARITY_H

/**
 * @file GeometricNonlinearity.h
 * @brief Physics-neutral nonlinear geometry transaction policy.
 */

#include "Geometry/FiniteDeformationKinematics.h"
#include "Systems/GeometryTransaction.h"

#include <cstdint>

namespace svmp {
namespace FE {
namespace systems {

enum class GeometricNonlinearityUpdatePoint : std::uint8_t {
    TrialIterate,
    AcceptedNonlinearState,
    AcceptedTimeStep,
    RolledBackTrial,
    AcceptedRemeshOrRezoneState
};

enum class GeometricNonlinearityStateField : std::uint8_t {
    Displacement,
    CurrentCoordinates,
    DeformationGradient,
    Strain,
    Stress,
    HistoryState
};

struct ArcLengthContinuationOptions {
    bool enabled{false};
    Real initial_radius{1.0};
    Real min_radius{1.0e-8};
    Real max_radius{1.0e8};
    bool adapt_radius{true};
};

struct GeometricNonlinearityPolicy {
    bool enabled{false};
    geometry::FiniteDeformationReferencePolicy reference_policy{
        geometry::FiniteDeformationReferencePolicy::TotalLagrangian};
    bool update_current_coordinates_on_trial{true};
    bool update_strain_and_stress_on_trial{true};
    bool history_updates_are_trial_until_commit{true};
    bool rollback_geometry_on_line_search_reject{true};
    bool rollback_coupled_state_on_line_search_reject{true};
    bool allow_lagged_kinematics{false};
    bool reset_displacement_after_reference_rebase{true};
    ArcLengthContinuationOptions arc_length{};
};

struct GeometricNonlinearityTransactionEvent {
    GeometricNonlinearityUpdatePoint update_point{
        GeometricNonlinearityUpdatePoint::TrialIterate};
    GeometryTransactionState geometry_state{GeometryTransactionState::Trial};
    bool line_search_rejected{false};
    bool nonlinear_step_accepted{false};
    bool time_step_accepted{false};
};

[[nodiscard]] constexpr bool isAcceptedGeometricState(
    GeometricNonlinearityUpdatePoint update_point) noexcept
{
    return update_point == GeometricNonlinearityUpdatePoint::AcceptedNonlinearState ||
           update_point == GeometricNonlinearityUpdatePoint::AcceptedTimeStep ||
           update_point == GeometricNonlinearityUpdatePoint::AcceptedRemeshOrRezoneState;
}

[[nodiscard]] constexpr bool stateFieldIsGeometryOwned(
    GeometricNonlinearityStateField field) noexcept
{
    return field == GeometricNonlinearityStateField::CurrentCoordinates ||
           field == GeometricNonlinearityStateField::DeformationGradient ||
           field == GeometricNonlinearityStateField::Strain;
}

[[nodiscard]] constexpr bool requiresLineSearchRollback(
    const GeometricNonlinearityPolicy& policy,
    const GeometricNonlinearityTransactionEvent& event) noexcept
{
    return policy.enabled &&
           event.line_search_rejected &&
           event.geometry_state == GeometryTransactionState::Trial &&
           policy.rollback_geometry_on_line_search_reject;
}

[[nodiscard]] constexpr bool requiresCoupledStateRollback(
    const GeometricNonlinearityPolicy& policy,
    const GeometricNonlinearityTransactionEvent& event) noexcept
{
    return requiresLineSearchRollback(policy, event) &&
           policy.rollback_coupled_state_on_line_search_reject;
}

[[nodiscard]] constexpr bool shouldUpdateFieldAtPoint(
    const GeometricNonlinearityPolicy& policy,
    GeometricNonlinearityStateField field,
    GeometricNonlinearityUpdatePoint update_point) noexcept
{
    if (!policy.enabled) {
        return false;
    }
    if (update_point == GeometricNonlinearityUpdatePoint::RolledBackTrial) {
        return false;
    }
    if (update_point == GeometricNonlinearityUpdatePoint::TrialIterate) {
        if (field == GeometricNonlinearityStateField::CurrentCoordinates) {
            return policy.update_current_coordinates_on_trial;
        }
        if (field == GeometricNonlinearityStateField::Strain ||
            field == GeometricNonlinearityStateField::Stress) {
            return policy.update_strain_and_stress_on_trial;
        }
        if (field == GeometricNonlinearityStateField::HistoryState) {
            return !policy.history_updates_are_trial_until_commit;
        }
        return true;
    }
    return isAcceptedGeometricState(update_point);
}

[[nodiscard]] constexpr const char* geometricNonlinearityUpdatePointName(
    GeometricNonlinearityUpdatePoint update_point) noexcept
{
    switch (update_point) {
        case GeometricNonlinearityUpdatePoint::TrialIterate: return "TrialIterate";
        case GeometricNonlinearityUpdatePoint::AcceptedNonlinearState:
            return "AcceptedNonlinearState";
        case GeometricNonlinearityUpdatePoint::AcceptedTimeStep:
            return "AcceptedTimeStep";
        case GeometricNonlinearityUpdatePoint::RolledBackTrial:
            return "RolledBackTrial";
        case GeometricNonlinearityUpdatePoint::AcceptedRemeshOrRezoneState:
            return "AcceptedRemeshOrRezoneState";
    }
    return "Unknown";
}

[[nodiscard]] constexpr const char* geometricNonlinearityStateFieldName(
    GeometricNonlinearityStateField field) noexcept
{
    switch (field) {
        case GeometricNonlinearityStateField::Displacement: return "Displacement";
        case GeometricNonlinearityStateField::CurrentCoordinates:
            return "CurrentCoordinates";
        case GeometricNonlinearityStateField::DeformationGradient:
            return "DeformationGradient";
        case GeometricNonlinearityStateField::Strain: return "Strain";
        case GeometricNonlinearityStateField::Stress: return "Stress";
        case GeometricNonlinearityStateField::HistoryState: return "HistoryState";
    }
    return "Unknown";
}

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_GEOMETRIC_NONLINEARITY_H
