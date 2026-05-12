#ifndef SVMP_PHYSICS_CORE_PHYSICS_JIT_POLICY_H
#define SVMP_PHYSICS_CORE_PHYSICS_JIT_POLICY_H

#include "FE/Forms/JIT/LLVMJITBuildInfo.h"

namespace svmp::Physics::core {

struct PhysicsJITPolicy {
    bool enable{FE::forms::jit::llvmJITEnabled()};
    bool specialization{true};
    int optimization_level{3};
    bool specialize_n_qpts{true};
    bool specialize_dofs{true};
};

} // namespace svmp::Physics::core

namespace svmp::Physics {
using PhysicsJITPolicy = core::PhysicsJITPolicy;
} // namespace svmp::Physics

#endif // SVMP_PHYSICS_CORE_PHYSICS_JIT_POLICY_H
