#include "LevelSet/LevelSetTransport.h"

namespace svmp::FE::level_set {

bool shouldReinitializeLevelSet(
    const LevelSetReinitializationOptions& options,
    int completed_step_index) noexcept
{
    return options.enabled &&
           options.cadence_steps > 0 &&
           completed_step_index > 0 &&
           completed_step_index % options.cadence_steps == 0;
}

bool shouldApplyLevelSetVolumeCorrection(
    const LevelSetVolumeCorrectionOptions& options,
    int completed_step_index) noexcept
{
    return options.enabled &&
           options.cadence_steps > 0 &&
           completed_step_index > 0 &&
           completed_step_index % options.cadence_steps == 0;
}

} // namespace svmp::FE::level_set
