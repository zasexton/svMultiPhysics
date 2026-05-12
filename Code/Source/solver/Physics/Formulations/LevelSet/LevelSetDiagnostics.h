#ifndef SVMP_PHYSICS_FORMULATIONS_LEVELSET_DIAGNOSTICS_H
#define SVMP_PHYSICS_FORMULATIONS_LEVELSET_DIAGNOSTICS_H

/**
 * @file LevelSetDiagnostics.h
 * @brief Scalar diagnostics for level-set free-surface tracking.
 */

#include "Core/Types.h"
#include "FE/Assembly/Assembler.h"
#include "FE/Dofs/DofHandler.h"
#include "FE/Systems/FESystem.h"
#include "Physics/Formulations/LevelSet/LevelSetReinitialization.h"
#include "Physics/Formulations/LevelSet/LevelSetVolume.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace level_set {

struct LevelSetScalarDiagnostic {
    std::string name{};
    FE::Real value{0.0};
};

struct LevelSetOutputDiagnosticsOptions {
    LevelSetVolumeOptions volume{};
    LevelSetReinitializationOptions signed_distance{};
    bool compute_signed_distance_error{true};
    bool has_reference_negative_volume{false};
    FE::Real reference_negative_volume{0.0};
};

struct LevelSetOutputDiagnostics {
    bool success{false};
    LevelSetVolumeResult volume{};
    LevelSetSignedDistanceRepairResult signed_distance{};
    std::size_t signed_distance_samples{0};
    FE::Real signed_distance_max_error{0.0};
    FE::Real signed_distance_l2_error{0.0};
    FE::Real negative_volume_loss{0.0};
    FE::Real relative_negative_volume_loss{0.0};
    std::vector<LevelSetScalarDiagnostic> scalars{};
    std::string diagnostic{};
};

[[nodiscard]] LevelSetOutputDiagnostics computeLevelSetOutputDiagnostics(
    const FE::assembly::IMeshAccess& mesh,
    const FE::dofs::DofHandler& level_set_dofs,
    const LevelSetOutputDiagnosticsOptions& options,
    std::span<const FE::Real> coefficients);

[[nodiscard]] LevelSetOutputDiagnostics computeLevelSetOutputDiagnostics(
    const FE::systems::FESystem& system,
    FE::FieldId level_set_field,
    const LevelSetOutputDiagnosticsOptions& options,
    std::span<const FE::Real> solution);

} // namespace level_set
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_LEVELSET_DIAGNOSTICS_H
