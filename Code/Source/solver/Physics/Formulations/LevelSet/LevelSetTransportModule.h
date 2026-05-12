#ifndef SVMP_PHYSICS_FORMULATIONS_LEVELSET_TRANSPORT_MODULE_H
#define SVMP_PHYSICS_FORMULATIONS_LEVELSET_TRANSPORT_MODULE_H

/**
 * @file LevelSetTransportModule.h
 * @brief Physics module options for level-set free-surface advection.
 */

#include "Physics/Core/PhysicsJITPolicy.h"
#include "Physics/Core/PhysicsModule.h"

#include "FE/Spaces/FunctionSpace.h"

#include <memory>
#include <string>

namespace svmp {
namespace Physics {
namespace formulations {
namespace level_set {

enum class LevelSetFieldSource {
    Unknown,
    PrescribedData
};

enum class LevelSetVelocitySource {
    CoupledField,
    PrescribedData
};

struct LevelSetFieldOptions {
    std::string field_name{"level_set"};
    LevelSetFieldSource source{LevelSetFieldSource::Unknown};
    bool auto_register_field{true};
};

struct LevelSetVelocityOptions {
    std::string field_name{"Velocity"};
    LevelSetVelocitySource source{LevelSetVelocitySource::CoupledField};
    bool auto_register_field{false};
};

struct LevelSetTransportOptions {
    LevelSetFieldOptions level_set{};
    LevelSetVelocityOptions velocity{};
    core::PhysicsJITPolicy jit_policy{};
};

class LevelSetTransportModule final : public PhysicsModule {
public:
    explicit LevelSetTransportModule(
        std::shared_ptr<const FE::spaces::FunctionSpace> level_set_space,
        LevelSetTransportOptions options = {});

    void registerOn(FE::systems::FESystem& system) const override;

private:
    std::shared_ptr<const FE::spaces::FunctionSpace> level_set_space_{};
    LevelSetTransportOptions options_{};
};

} // namespace level_set
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_LEVELSET_TRANSPORT_MODULE_H
