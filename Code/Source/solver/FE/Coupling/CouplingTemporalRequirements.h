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
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {

namespace systems {
enum class MeshMotionFieldRole : std::uint8_t;
struct SystemStateView;
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

struct CouplingTemporalAvailability {
    int max_derivative_order{0};
    int history_depth{0};
    bool provides_time{true};
    bool provides_time_step{true};
    bool provides_effective_time_step{true};
};

struct CouplingTemporalRequirementSummary {
    int max_derivative_order{0};
    int max_history_index{0};
    bool requires_time{false};
    bool requires_time_step{false};
    bool requires_effective_time_step{false};
    std::vector<CouplingTemporalRequirement> field_temporal_requirements;
    std::vector<CouplingTemporalRequirement> mesh_temporal_requirements;
};

[[nodiscard]] const char* toString(CouplingTemporalQuantity quantity) noexcept;
[[nodiscard]] CouplingValidationResult validateTemporalRequirement(
    const CouplingTemporalRequirement& requirement);
[[nodiscard]] CouplingTemporalAvailability temporalAvailabilityFromSystemState(
    const systems::SystemStateView& state,
    int max_derivative_order);
[[nodiscard]] CouplingTemporalRequirementSummary summarizeTemporalRequirements(
    std::span<const CouplingTemporalRequirement> requirements);
[[nodiscard]] CouplingValidationResult validateTemporalRequirements(
    std::span<const CouplingTemporalRequirement> requirements,
    const CouplingTemporalAvailability& availability);

} // namespace coupling
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_COUPLING_COUPLINGTEMPORALREQUIREMENTS_H
