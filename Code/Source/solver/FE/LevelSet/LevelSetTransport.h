#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief FE-system installer for level-set transport residuals.
 */

#include "LevelSet/LevelSetOptions.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"

#include <memory>

namespace svmp::FE::level_set {

[[nodiscard]] systems::CoupledResidualKernels installLevelSetTransport(
    systems::FESystem& system,
    std::shared_ptr<const spaces::FunctionSpace> level_set_space,
    const LevelSetTransportOptions& options,
    const systems::FormInstallOptions& install_options = {});

} // namespace svmp::FE::level_set
