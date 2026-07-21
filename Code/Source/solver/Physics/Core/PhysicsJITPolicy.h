#ifndef SVMP_PHYSICS_CORE_PHYSICS_JIT_POLICY_H
#define SVMP_PHYSICS_CORE_PHYSICS_JIT_POLICY_H

namespace svmp::Physics::core {

struct PhysicsJITPolicy {
    // This is the requested policy. Build capability is applied centrally by
    // effectivePhysicsJITEnable() so an unavailable default can fall back with
    // a diagnostic while an explicit jit=true request can fail closed.
    bool enable{true};
    bool specialization{true};
    int optimization_level{3};
    bool specialize_n_qpts{true};
    bool specialize_dofs{true};
    bool enable_was_explicitly_set{false};
};

/**
 * @brief Resolve a requested physics JIT policy against build capability.
 *
 * An explicit request for jit=true is a configuration contract and throws
 * when LLVM JIT support is unavailable. An implicit/default request falls
 * back to the interpreter with a process-once, rank-safe diagnostic.
 */
[[nodiscard]] bool effectivePhysicsJITEnable(const PhysicsJITPolicy& policy);

} // namespace svmp::Physics::core

namespace svmp::Physics {
using PhysicsJITPolicy = core::PhysicsJITPolicy;
} // namespace svmp::Physics

#endif // SVMP_PHYSICS_CORE_PHYSICS_JIT_POLICY_H
