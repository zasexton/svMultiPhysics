/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingTemporalRequirements.h"

#include "Systems/SystemState.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace coupling {

const char* toString(CouplingTemporalQuantity quantity) noexcept
{
    switch (quantity) {
    case CouplingTemporalQuantity::Time:
        return "time";
    case CouplingTemporalQuantity::TimeStep:
        return "time_step";
    case CouplingTemporalQuantity::EffectiveTimeStep:
        return "effective_time_step";
    case CouplingTemporalQuantity::FieldDerivative:
        return "field_derivative";
    case CouplingTemporalQuantity::FieldHistoryValue:
        return "field_history_value";
    case CouplingTemporalQuantity::MeshVelocity:
        return "mesh_velocity";
    case CouplingTemporalQuantity::MeshAcceleration:
        return "mesh_acceleration";
    case CouplingTemporalQuantity::PreviousMeshVelocity:
        return "previous_mesh_velocity";
    case CouplingTemporalQuantity::PredictedMeshVelocity:
        return "predicted_mesh_velocity";
    }
    return "unknown";
}

CouplingValidationResult validateTemporalRequirement(
    const CouplingTemporalRequirement& requirement)
{
    CouplingValidationResult result;
    if (requirement.quantity == CouplingTemporalQuantity::FieldDerivative &&
        requirement.derivative_order <= 0) {
        result.addError("field derivative requirements need a positive derivative order");
    }
    if (requirement.quantity == CouplingTemporalQuantity::FieldHistoryValue &&
        requirement.history_index <= 0) {
        result.addError("field history requirements need a positive logical history index");
    }
    return result;
}

CouplingTemporalAvailability temporalAvailabilityFromSystemState(
    const systems::SystemStateView& state,
    int max_derivative_order)
{
    CouplingTemporalAvailability availability;
    availability.max_derivative_order = max_derivative_order;
    availability.history_depth = static_cast<int>(state.u_history.size());
    if (!state.u_prev.empty()) {
        availability.history_depth = std::max(availability.history_depth, 1);
    }
    if (!state.u_prev2.empty()) {
        availability.history_depth = std::max(availability.history_depth, 2);
    }
    availability.provides_time = true;
    availability.provides_time_step = true;
    availability.provides_effective_time_step = true;
    return availability;
}

CouplingTemporalRequirementSummary summarizeTemporalRequirements(
    std::span<const CouplingTemporalRequirement> requirements)
{
    CouplingTemporalRequirementSummary summary;
    for (const auto& requirement : requirements) {
        switch (requirement.quantity) {
        case CouplingTemporalQuantity::Time:
            summary.requires_time = true;
            break;
        case CouplingTemporalQuantity::TimeStep:
            summary.requires_time_step = true;
            break;
        case CouplingTemporalQuantity::EffectiveTimeStep:
            summary.requires_effective_time_step = true;
            break;
        case CouplingTemporalQuantity::FieldDerivative:
            summary.max_derivative_order =
                std::max(summary.max_derivative_order, requirement.derivative_order);
            summary.field_temporal_requirements.push_back(requirement);
            break;
        case CouplingTemporalQuantity::FieldHistoryValue:
            summary.max_history_index =
                std::max(summary.max_history_index, requirement.history_index);
            summary.field_temporal_requirements.push_back(requirement);
            break;
        case CouplingTemporalQuantity::MeshVelocity:
        case CouplingTemporalQuantity::MeshAcceleration:
        case CouplingTemporalQuantity::PreviousMeshVelocity:
        case CouplingTemporalQuantity::PredictedMeshVelocity:
            summary.mesh_temporal_requirements.push_back(requirement);
            break;
        }
    }
    return summary;
}

CouplingValidationResult validateTemporalRequirements(
    std::span<const CouplingTemporalRequirement> requirements,
    const CouplingTemporalAvailability& availability)
{
    CouplingValidationResult result;
    for (const auto& requirement : requirements) {
        result.append(validateTemporalRequirement(requirement));
        if (requirement.requirement == CouplingRequirement::Optional) {
            continue;
        }

        switch (requirement.quantity) {
        case CouplingTemporalQuantity::Time:
            if (!availability.provides_time) {
                result.addError("required coupling time symbol is unavailable");
            }
            break;
        case CouplingTemporalQuantity::TimeStep:
            if (!availability.provides_time_step) {
                result.addError("required coupling time-step symbol is unavailable");
            }
            break;
        case CouplingTemporalQuantity::EffectiveTimeStep:
            if (!availability.provides_effective_time_step) {
                result.addError("required coupling effective-time-step symbol is unavailable");
            }
            break;
        case CouplingTemporalQuantity::FieldDerivative:
            if (requirement.derivative_order > availability.max_derivative_order) {
                result.addError("coupling field derivative requirement exceeds the temporal policy");
            }
            break;
        case CouplingTemporalQuantity::FieldHistoryValue:
            if (requirement.history_index > availability.history_depth) {
                result.addError("coupling field history requirement exceeds available state history");
            }
            break;
        case CouplingTemporalQuantity::MeshVelocity:
        case CouplingTemporalQuantity::MeshAcceleration:
        case CouplingTemporalQuantity::PreviousMeshVelocity:
        case CouplingTemporalQuantity::PredictedMeshVelocity:
            break;
        }
    }
    return result;
}

} // namespace coupling
} // namespace FE
} // namespace svmp
