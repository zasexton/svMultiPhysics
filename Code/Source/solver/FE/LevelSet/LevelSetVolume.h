#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Cut-cell volume measurement and global-shift correction utilities.
 */

#include "Assembly/Assembler.h"
#include "Core/Types.h"
#include "Dofs/DofHandler.h"
#include "Systems/FESystem.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace svmp::FE::level_set {

struct LevelSetVolumeOptions {
    Real isovalue{0.0};
    Real tolerance{1.0e-12};
};

struct LevelSetVolumeResult {
    bool success{false};
    std::size_t cells{0};
    std::size_t cut_cells{0};
    std::size_t full_negative_cells{0};
    std::size_t full_positive_cells{0};
    Real total_volume{0.0};
    Real negative_volume{0.0};
    Real positive_volume{0.0};
    std::string diagnostic{};
};

struct LevelSetGlobalShiftCorrectionOptions {
    Real target_negative_volume{0.0};
    Real volume_tolerance{1.0e-10};
    int max_iterations{50};
};

struct LevelSetGlobalShiftCorrectionResult {
    bool success{false};
    int iterations{0};
    Real applied_shift{0.0};
    Real target_negative_volume{0.0};
    Real initial_negative_volume{0.0};
    Real corrected_negative_volume{0.0};
    Real volume_error{0.0};
    LevelSetVolumeResult initial_volume{};
    LevelSetVolumeResult corrected_volume{};
    std::string diagnostic{};
};

[[nodiscard]] LevelSetVolumeResult computeLevelSetCutCellVolume(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const LevelSetVolumeOptions& options,
    std::span<const Real> coefficients);

[[nodiscard]] LevelSetVolumeResult computeLevelSetCutCellVolume(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetVolumeOptions& options,
    std::span<const Real> solution);

[[nodiscard]] LevelSetGlobalShiftCorrectionResult applyGlobalLevelSetShiftCorrection(
    const assembly::IMeshAccess& mesh,
    const dofs::DofHandler& level_set_dofs,
    const LevelSetVolumeOptions& volume_options,
    const LevelSetGlobalShiftCorrectionOptions& correction_options,
    std::span<const Real> coefficients,
    std::vector<Real>& corrected_coefficients);

[[nodiscard]] LevelSetGlobalShiftCorrectionResult applyGlobalLevelSetShiftCorrection(
    const systems::FESystem& system,
    FieldId level_set_field,
    const LevelSetVolumeOptions& volume_options,
    const LevelSetGlobalShiftCorrectionOptions& correction_options,
    std::span<const Real> solution,
    std::vector<Real>& corrected_solution);

} // namespace svmp::FE::level_set
