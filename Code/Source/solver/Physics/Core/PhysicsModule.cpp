#include "Physics/Core/PhysicsModule.h"

#include "FE/Core/Logger.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Forms/JIT/LLVMJITBuildInfo.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

#include <cstdlib>
#include <mutex>
#include <stdexcept>

namespace svmp {
namespace Physics {

namespace {

[[nodiscard]] bool isPrimaryDiagnosticRank() noexcept
{
#if FE_HAS_MPI
    int initialized = 0;
    if (MPI_Initialized(&initialized) == MPI_SUCCESS && initialized != 0) {
        int finalized = 0;
        if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized != 0) {
            return false;
        }
        int rank = 0;
        return MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS && rank == 0;
    }
#endif

    // MPI launchers publish rank before MPI_Init. Avoid duplicate diagnostics
    // if policy resolution happens during pre-initialization setup.
    static constexpr const char* kRankEnvironmentKeys[] = {
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "MV2_COMM_WORLD_RANK",
        "SLURM_PROCID",
    };
    for (const char* key : kRankEnvironmentKeys) {
        if (const char* value = std::getenv(key); value != nullptr) {
            char* end = nullptr;
            const long rank = std::strtol(value, &end, 10);
            if (end != value && *end == '\0') {
                return rank == 0;
            }
        }
    }
    return true;
}

void warnImplicitJITFallbackOnce()
{
    if (!isPrimaryDiagnosticRank()) {
        return;
    }
    static std::once_flag warning_once;
    std::call_once(warning_once, [] {
        FE_LOG_WARNING(
            "Physics LLVM JIT is unavailable in this executable; continuing with "
            "the symbolic interpreter because JIT was not explicitly requested "
            "(FE_ENABLE_LLVM_JIT did not resolve to ON). "
            "diagnostic=physics_jit_unavailable_fallback");
    });
}

} // namespace

namespace core {

bool effectivePhysicsJITEnable(const PhysicsJITPolicy& policy)
{
    if (!policy.enable) {
        return false;
    }
    if (FE::forms::jit::llvmJITEnabled()) {
        return true;
    }
    if (policy.enable_was_explicitly_set) {
        throw std::runtime_error(
            "[svMultiPhysics::Physics] jit=true was explicitly requested, but this "
            "executable was built without LLVM JIT support "
            "(FE_ENABLE_LLVM_JIT did not resolve to ON). Rebuild with "
            "-DFE_ENABLE_LLVM_JIT=ON and "
            "a compatible LLVM installation, or set jit=false.");
    }

    warnImplicitJITFallbackOnce();
    return false;
}

} // namespace core

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
    options.jit.enable = core::effectivePhysicsJITEnable(policy);
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
