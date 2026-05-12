#ifndef SVMP_PHYSICS_FORMULATIONS_LEVELSET_TRANSPORT_MODULE_H
#define SVMP_PHYSICS_FORMULATIONS_LEVELSET_TRANSPORT_MODULE_H

/**
 * @file LevelSetTransportModule.h
 * @brief Physics module options for level-set free-surface advection.
 */

#include "Physics/Core/PhysicsModule.h"

#include "FE/Core/Types.h"
#include "FE/LevelSet/LevelSetTransport.h"
#include "FE/Spaces/FunctionSpace.h"
#include "FE/Systems/FormsInstaller.h"

#include <memory>

namespace svmp {
namespace Physics {
namespace formulations {
namespace level_set {

// Remove these aliases after the Physics adapter no longer exposes level-set option types.
using ScalarValue = FE::level_set::ScalarValue;
using LevelSetFieldSource = FE::level_set::LevelSetFieldSource;
using LevelSetVelocitySource = FE::level_set::LevelSetVelocitySource;
using LevelSetFieldOptions = FE::level_set::LevelSetFieldOptions;
using LevelSetVelocityOptions = FE::level_set::LevelSetVelocityOptions;
using LevelSetSUPGOptions = FE::level_set::LevelSetSUPGOptions;
using LevelSetReinitializationMethod = FE::level_set::LevelSetReinitializationMethod;
using LevelSetReinitializationOptions = FE::level_set::LevelSetReinitializationOptions;
using LevelSetVolumeCorrectionOptions = FE::level_set::LevelSetVolumeCorrectionOptions;
using LevelSetInflowBoundary = FE::level_set::LevelSetInflowBoundary;
using LevelSetOutflowBoundary = FE::level_set::LevelSetOutflowBoundary;
using LevelSetBoundaryOptions = FE::level_set::LevelSetBoundaryOptions;
using LevelSetTransportOptions = FE::level_set::LevelSetTransportOptions;
using FE::level_set::shouldApplyLevelSetVolumeCorrection;
using FE::level_set::shouldReinitializeLevelSet;

class LevelSetTransportModule final : public PhysicsModule {
public:
    explicit LevelSetTransportModule(
        std::shared_ptr<const FE::spaces::FunctionSpace> level_set_space,
        LevelSetTransportOptions options = {},
        FE::systems::FormInstallOptions install_options = {});

    void registerOn(FE::systems::FESystem& system) const override;

    [[nodiscard]] const LevelSetTransportOptions& options() const noexcept { return options_; }

private:
    std::shared_ptr<const FE::spaces::FunctionSpace> level_set_space_{};
    LevelSetTransportOptions options_{};
    FE::systems::FormInstallOptions install_options_{};
};

} // namespace level_set
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_LEVELSET_TRANSPORT_MODULE_H
