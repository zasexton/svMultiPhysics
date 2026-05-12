#include "Physics/Core/PhysicsModule.h"

#include "FE/Forms/FormExpr.h"
#include "FE/Forms/JIT/LLVMJITBuildInfo.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

namespace svmp {
namespace Physics {

FE::forms::SymbolicOptions
PhysicsModule::physicsCompilerOptions(const core::PhysicsJITPolicy& policy) const
{
    FE::forms::SymbolicOptions options{};
    applyPhysicsJITOptions(options, policy);
    return options;
}

FE::systems::FormInstallOptions
PhysicsModule::physicsInstallOptions(const core::PhysicsJITPolicy& policy) const
{
    FE::systems::FormInstallOptions options{};
    options.compiler_options = physicsCompilerOptions(policy);
    return options;
}

void PhysicsModule::applyPhysicsJITOptions(FE::forms::SymbolicOptions& options,
                                           const core::PhysicsJITPolicy& policy) const
{
    options.jit.enable = policy.enable && FE::forms::jit::llvmJITEnabled();
    options.jit.optimization_level = policy.optimization_level;
    options.jit.specialization.enable = policy.specialization;
    options.jit.specialization.specialize_n_qpts = policy.specialize_n_qpts;
    options.jit.specialization.specialize_dofs = policy.specialize_dofs;
}

void PhysicsModule::setBoundaryReductionCompilerOptions(
    FE::systems::FESystem& system,
    FE::FieldId field,
    const core::PhysicsJITPolicy& policy) const
{
    system.boundaryReductionService(field).setCompilerOptions(physicsCompilerOptions(policy));
}

} // namespace Physics
} // namespace svmp
