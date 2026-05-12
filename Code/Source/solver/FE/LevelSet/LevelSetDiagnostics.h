#pragma once

#include "Assembly/Assembler.h"
#include "Core/Types.h"
#include "Dofs/DofHandler.h"
#include "LevelSet/LevelSetReinitialization.h"
#include "LevelSet/LevelSetVolume.h"
#include "Systems/FESystem.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace svmp::FE::level_set {

struct LevelSetScalarDiagnostic {
    std::string name{};
    Real value{0.0};
};

struct LevelSetOutputDiagnosticsOptions {
    LevelSetVolumeOptions volume{};
    LevelSetReinitializationOptions signed_distance{};
    bool compute_signed_distance_error{true};
    bool has_reference_negative_volume{false};
    Real reference_negative_volume{0.0};
};

struct LevelSetOutputDiagnostics {
    bool success{false};
    LevelSetVolumeResult volume{};
    LevelSetSignedDistanceRepairResult signed_distance{};
    std::size_t signed_distance_samples{0};
    Real signed_distance_max_error{0.0};
    Real signed_distance_l2_error{0.0};
    Real negative_volume_loss{0.0};
    Real relative_negative_volume_loss{0.0};
    std::vector<LevelSetScalarDiagnostic> scalars{};
    std::string diagnostic{};
};

[[nodiscard]] LevelSetOutputDiagnostics computeLevelSetOutputDiagnostics(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const LevelSetOutputDiagnosticsOptions& options,
    std::span<const Real> coefficients);

[[nodiscard]] LevelSetOutputDiagnostics computeLevelSetOutputDiagnostics(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetOutputDiagnosticsOptions& options,
    std::span<const Real> solution);

} // namespace svmp::FE::level_set
