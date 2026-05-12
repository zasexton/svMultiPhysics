#ifndef SVMP_PHYSICS_CORE_PHYSICS_MODULE_H
#define SVMP_PHYSICS_CORE_PHYSICS_MODULE_H

/**
 * @file PhysicsModule.h
 * @brief Uniform interface for installing physics onto an FE::systems::FESystem
 *
 * The application/driver layer can treat different physics uniformly by
 * calling `registerOn(system)` to define fields, operators, and kernels.
 */

#include "FE/Core/Types.h"
#include "Physics/Core/PhysicsJITPolicy.h"

#include <span>

namespace svmp {
namespace FE {
namespace forms {
struct SymbolicOptions;
} // namespace forms
namespace systems {
class FESystem;
struct FormInstallOptions;
} // namespace systems
} // namespace FE

namespace Physics {

class PhysicsModule {
public:
    virtual ~PhysicsModule() = default;

    /**
     * @brief Register fields/operators/kernels onto a system (definition phase)
     *
     * Must be called before `system.setup()`.
     */
    virtual void registerOn(FE::systems::FESystem& system) const = 0;

    /**
     * @brief Optional initial condition helper
     *
     * The application owns the state vectors; this helper fills an initial
     * global vector in the system DOF ordering.
     */
    virtual void applyInitialConditions(const FE::systems::FESystem& /*system*/,
                                        std::span<FE::Real> /*u0*/) const
    {
    }

    /**
     * @brief Optional quantities-of-interest (QoI) registration hook
     */
    virtual void registerFunctionals(FE::systems::FESystem& /*system*/) const {}

protected:
    [[nodiscard]] FE::forms::SymbolicOptions
    physicsCompilerOptions(const core::PhysicsJITPolicy& policy = {}) const;

    [[nodiscard]] FE::systems::FormInstallOptions
    physicsInstallOptions(const core::PhysicsJITPolicy& policy = {}) const;

    void applyPhysicsJITOptions(FE::forms::SymbolicOptions& options,
                                const core::PhysicsJITPolicy& policy = {}) const;

    void setBoundaryReductionCompilerOptions(
        FE::systems::FESystem& system,
        FE::FieldId field,
        const core::PhysicsJITPolicy& policy = {}) const;
};

} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_CORE_PHYSICS_MODULE_H
