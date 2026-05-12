#ifndef SVMP_PHYSICS_FORMULATIONS_LEVELSET_REINITIALIZATION_H
#define SVMP_PHYSICS_FORMULATIONS_LEVELSET_REINITIALIZATION_H

/**
 * @file LevelSetReinitialization.h
 * @brief Signed-distance repair helpers for level-set fields.
 */

#include "Core/Types.h"
#include "FE/Assembly/Assembler.h"
#include "FE/Dofs/DofHandler.h"
#include "FE/Systems/FESystem.h"
#include "Physics/Formulations/LevelSet/LevelSetTransportModule.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace level_set {

struct LevelSetSignedDistanceRepairResult {
    bool success{false};
    LevelSetReinitializationMethod method{LevelSetReinitializationMethod::Projection};
    std::size_t repaired_dofs{0};
    std::size_t interface_fragments{0};
    std::size_t cut_cells{0};
    FE::Real max_abs_update{0.0};
    FE::Real max_distance{0.0};
    std::string diagnostic{};
};

[[nodiscard]] LevelSetSignedDistanceRepairResult
repairLevelSetSignedDistanceByProjection(
    const FE::assembly::IMeshAccess& mesh,
    const FE::dofs::DofHandler& level_set_dofs,
    const LevelSetReinitializationOptions& options,
    std::span<const FE::Real> input_coefficients,
    std::vector<FE::Real>& repaired_coefficients);

[[nodiscard]] LevelSetSignedDistanceRepairResult
repairLevelSetSignedDistanceByProjection(
    const FE::systems::FESystem& system,
    FE::FieldId level_set_field,
    const LevelSetReinitializationOptions& options,
    std::span<const FE::Real> input_solution,
    std::vector<FE::Real>& repaired_solution);

} // namespace level_set
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_LEVELSET_REINITIALIZATION_H
