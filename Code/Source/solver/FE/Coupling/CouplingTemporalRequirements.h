/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_COUPLING_COUPLINGTEMPORALREQUIREMENTS_H
#define SVMP_FE_COUPLING_COUPLINGTEMPORALREQUIREMENTS_H

/**
 * @file CouplingTemporalRequirements.h
 * @brief Coupling-owned declarations for temporal data requirements.
 */

#include "Coupling/CouplingTypes.h"
#include "Coupling/CouplingGeometryRequirements.h"

#include <optional>
#include <string>

namespace svmp {
namespace FE {

namespace systems {
enum class MeshMotionFieldRole : std::uint8_t;
}

namespace coupling {

enum class CouplingTemporalQuantity : std::uint8_t {
    Time,
    TimeStep,
    EffectiveTimeStep,
    FieldDerivative,
    FieldHistoryValue,
    MeshVelocity,
    MeshAcceleration,
    PreviousMeshVelocity,
    PredictedMeshVelocity,
};

struct CouplingTemporalRequirement {
    CouplingTemporalQuantity quantity{CouplingTemporalQuantity::Time};
    std::optional<CouplingFieldUse> field;
    std::optional<CouplingGeometryTerminalScope> mesh_motion_scope;
    std::optional<systems::MeshMotionFieldRole> mesh_motion_role;
    int derivative_order{0};
    int history_index{0};
    CouplingRequirement requirement{CouplingRequirement::Required};
};

[[nodiscard]] const char* toString(CouplingTemporalQuantity quantity) noexcept;
[[nodiscard]] CouplingValidationResult validateTemporalRequirement(
    const CouplingTemporalRequirement& requirement);

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGTEMPORALREQUIREMENTS_H
