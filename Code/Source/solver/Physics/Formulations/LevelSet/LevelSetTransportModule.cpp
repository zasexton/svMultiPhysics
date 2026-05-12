/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/LevelSet/LevelSetTransportModule.h"

#include "FE/Systems/FESystem.h"

#include <stdexcept>
#include <utility>

namespace svmp {
namespace Physics {
namespace formulations {
namespace level_set {

LevelSetTransportModule::LevelSetTransportModule(
    std::shared_ptr<const FE::spaces::FunctionSpace> level_set_space,
    LevelSetTransportOptions options)
    : level_set_space_(std::move(level_set_space))
    , options_(std::move(options))
{
}

void LevelSetTransportModule::registerOn(FE::systems::FESystem& system) const
{
    (void)system;
    if (options_.level_set.field_name.empty()) {
        throw std::invalid_argument(
            "LevelSetTransportModule::registerOn: level-set field name must be non-empty");
    }
    if (options_.level_set.auto_register_field && !level_set_space_) {
        throw std::invalid_argument(
            "LevelSetTransportModule::registerOn: auto-registering the level-set field requires a function space");
    }
    if (level_set_space_ && level_set_space_->value_dimension() != 1) {
        throw std::invalid_argument(
            "LevelSetTransportModule::registerOn: level-set field space must be scalar");
    }
}

} // namespace level_set
} // namespace formulations
} // namespace Physics
} // namespace svmp
