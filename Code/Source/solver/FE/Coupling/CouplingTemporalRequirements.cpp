/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Coupling/CouplingTemporalRequirements.h"

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

} // namespace coupling
} // namespace FE
} // namespace svmp
