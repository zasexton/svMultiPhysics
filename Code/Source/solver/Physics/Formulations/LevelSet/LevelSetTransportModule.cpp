/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/LevelSet/LevelSetTransportModule.h"

#include <utility>

namespace svmp {
namespace Physics {
namespace formulations {
namespace level_set {

LevelSetTransportModule::LevelSetTransportModule(
    std::shared_ptr<const FE::spaces::FunctionSpace> level_set_space,
    LevelSetTransportOptions options,
    FE::systems::FormInstallOptions install_options)
    : level_set_space_(std::move(level_set_space))
    , options_(std::move(options))
    , install_options_(std::move(install_options))
{
}

void LevelSetTransportModule::registerOn(FE::systems::FESystem& system) const
{
    (void)FE::level_set::installLevelSetTransport(
        system,
        level_set_space_,
        options_,
        install_options_);
}

} // namespace level_set
} // namespace formulations
} // namespace Physics
} // namespace svmp
