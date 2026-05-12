#ifndef SVMP_PHYSICS_FORMULATIONS_LEVELSET_VOLUME_H
#define SVMP_PHYSICS_FORMULATIONS_LEVELSET_VOLUME_H

/**
 * @file LevelSetVolume.h
 * @brief Cut-cell volume diagnostics for level-set fields.
 */

#include "Core/Types.h"
#include "FE/Assembly/Assembler.h"
#include "FE/Dofs/DofHandler.h"
#include "FE/Systems/FESystem.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace level_set {

struct LevelSetVolumeOptions {
    FE::Real isovalue{0.0};
    FE::Real tolerance{1.0e-12};
};

struct LevelSetVolumeResult {
    bool success{false};
    std::size_t cells{0};
    std::size_t cut_cells{0};
    std::size_t full_negative_cells{0};
    std::size_t full_positive_cells{0};
    FE::Real total_volume{0.0};
    FE::Real negative_volume{0.0};
    FE::Real positive_volume{0.0};
    std::string diagnostic{};
};

struct LevelSetGlobalShiftCorrectionOptions {
    FE::Real target_negative_volume{0.0};
    FE::Real volume_tolerance{1.0e-10};
    int max_iterations{50};
};

struct LevelSetGlobalShiftCorrectionResult {
    bool success{false};
    int iterations{0};
    FE::Real applied_shift{0.0};
    FE::Real target_negative_volume{0.0};
    FE::Real initial_negative_volume{0.0};
    FE::Real corrected_negative_volume{0.0};
    FE::Real volume_error{0.0};
    LevelSetVolumeResult initial_volume{};
    LevelSetVolumeResult corrected_volume{};
    std::string diagnostic{};
};

[[nodiscard]] LevelSetVolumeResult computeLevelSetCutCellVolume(
    const FE::assembly::IMeshAccess& mesh,
    const FE::dofs::DofHandler& level_set_dofs,
    const LevelSetVolumeOptions& options,
    std::span<const FE::Real> coefficients);

[[nodiscard]] LevelSetVolumeResult computeLevelSetCutCellVolume(
    const FE::systems::FESystem& system,
    FE::FieldId level_set_field,
    const LevelSetVolumeOptions& options,
    std::span<const FE::Real> solution);

[[nodiscard]] LevelSetGlobalShiftCorrectionResult applyGlobalLevelSetShiftCorrection(
    const FE::assembly::IMeshAccess& mesh,
    const FE::dofs::DofHandler& level_set_dofs,
    const LevelSetVolumeOptions& volume_options,
    const LevelSetGlobalShiftCorrectionOptions& correction_options,
    std::span<const FE::Real> coefficients,
    std::vector<FE::Real>& corrected_coefficients);

[[nodiscard]] LevelSetGlobalShiftCorrectionResult applyGlobalLevelSetShiftCorrection(
    const FE::systems::FESystem& system,
    FE::FieldId level_set_field,
    const LevelSetVolumeOptions& volume_options,
    const LevelSetGlobalShiftCorrectionOptions& correction_options,
    std::span<const FE::Real> solution,
    std::vector<FE::Real>& corrected_solution);

} // namespace level_set
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_LEVELSET_VOLUME_H
