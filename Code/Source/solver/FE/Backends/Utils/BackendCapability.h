/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_BACKEND_CAPABILITY_H
#define SVMP_FE_BACKENDS_BACKEND_CAPABILITY_H

/**
 * @file BackendCapability.h
 * @brief Backend capability descriptors for the FE solver stack
 *
 * Backends are described by their capabilities, not by their identity.
 * This enables the FE public API to be defined by what formulations need
 * (e.g., "generic block solver", "matrix-free apply") rather than by
 * backend-specific limitations.
 *
 * Design principle: the FE model defines what it needs; the backend
 * advertises what it can do. Mismatches are caught at setup time, not
 * buried in runtime asserts or silent fallbacks.
 */

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

/**
 * @brief Individual capabilities a backend may support
 */
enum class Capability : std::uint32_t {
    /// Assemble and solve monolithic (single-block) systems
    MonolithicSolve          = 1u << 0,

    /// Assemble and solve block systems with arbitrary block count
    GenericBlockSolve        = 1u << 1,

    /// Saddle-point / Schur complement block solver (2-field only)
    SaddlePointSolve         = 1u << 2,

    /// Matrix-free operator application (single field)
    MatrixFreeApplySingle    = 1u << 3,

    /// Matrix-free operator application (multi-field / block)
    MatrixFreeApplyBlock     = 1u << 4,

    /// Field-split preconditioner support
    FieldSplitPreconditioner = 1u << 5,

    /// Arbitrary block count in preconditioner (not just 2-field)
    GenericBlockPreconditioner = 1u << 6,

    /// Parallel (MPI) distributed solve
    DistributedSolve         = 1u << 7,

    /// Direct (factorization-based) solver
    DirectSolve              = 1u << 8,

    /// AMG preconditioner
    AMGPreconditioner        = 1u << 9,

    /// Variable-size block DOF layout (different dof counts per node)
    VariableBlockLayout      = 1u << 10,
};

/**
 * @brief Bitwise OR of Capability flags
 */
using CapabilitySet = std::uint32_t;

inline constexpr CapabilitySet operator|(Capability a, Capability b) noexcept {
    return static_cast<CapabilitySet>(a) | static_cast<CapabilitySet>(b);
}

inline constexpr CapabilitySet operator|(CapabilitySet a, Capability b) noexcept {
    return a | static_cast<CapabilitySet>(b);
}

inline constexpr bool hasCapability(CapabilitySet set, Capability cap) noexcept {
    return (set & static_cast<CapabilitySet>(cap)) != 0u;
}

/**
 * @brief Backend identity + declared capabilities
 */
struct BackendDescriptor {
    std::string name;             ///< e.g., "FSILS", "PETSc", "Trilinos", "Eigen3"
    CapabilitySet capabilities{0};

    [[nodiscard]] bool supports(Capability cap) const noexcept {
        return hasCapability(capabilities, cap);
    }

    [[nodiscard]] bool supportsAll(CapabilitySet required) const noexcept {
        return (capabilities & required) == required;
    }
};

/**
 * @brief Well-known backend capability profiles
 */
namespace profiles {

/// FSILS: MPI-parallel saddle-point solver (2-field block Schur only)
inline BackendDescriptor fsils() {
    return {"FSILS",
            Capability::MonolithicSolve
          | Capability::SaddlePointSolve
          | Capability::DistributedSolve};
}

/// PETSc: full-featured parallel solver with generic block support
inline BackendDescriptor petsc() {
    return {"PETSc",
            Capability::MonolithicSolve
          | Capability::GenericBlockSolve
          | Capability::SaddlePointSolve
          | Capability::MatrixFreeApplySingle
          | Capability::MatrixFreeApplyBlock
          | Capability::FieldSplitPreconditioner
          | Capability::GenericBlockPreconditioner
          | Capability::DistributedSolve
          | Capability::DirectSolve
          | Capability::AMGPreconditioner
          | Capability::VariableBlockLayout};
}

/// Trilinos: full-featured parallel solver with generic block support
inline BackendDescriptor trilinos() {
    return {"Trilinos",
            Capability::MonolithicSolve
          | Capability::GenericBlockSolve
          | Capability::SaddlePointSolve
          | Capability::MatrixFreeApplySingle
          | Capability::MatrixFreeApplyBlock
          | Capability::FieldSplitPreconditioner
          | Capability::GenericBlockPreconditioner
          | Capability::DistributedSolve
          | Capability::DirectSolve
          | Capability::AMGPreconditioner
          | Capability::VariableBlockLayout};
}

/// Eigen3: serial direct/iterative solver, no MPI, generic block via nesting
inline BackendDescriptor eigen3() {
    return {"Eigen3",
            Capability::MonolithicSolve
          | Capability::GenericBlockSolve
          | Capability::SaddlePointSolve
          | Capability::DirectSolve
          | Capability::FieldSplitPreconditioner
          | Capability::VariableBlockLayout};
}

} // namespace profiles

/**
 * @brief Check if a backend can handle a given formulation's requirements
 *
 * @param backend Backend capability descriptor
 * @param required Required capabilities for the formulation
 * @return Empty string if compatible, otherwise a human-readable explanation
 */
[[nodiscard]] inline std::string checkCompatibility(
    const BackendDescriptor& backend,
    CapabilitySet required)
{
    const CapabilitySet missing = required & ~backend.capabilities;
    if (missing == 0u) return {};

    std::string msg = "Backend '" + backend.name + "' lacks required capabilities:";
    auto check = [&](Capability cap, const char* name) {
        if (hasCapability(missing, cap)) {
            msg += " ";
            msg += name;
        }
    };
    check(Capability::MonolithicSolve, "MonolithicSolve");
    check(Capability::GenericBlockSolve, "GenericBlockSolve");
    check(Capability::SaddlePointSolve, "SaddlePointSolve");
    check(Capability::MatrixFreeApplySingle, "MatrixFreeApplySingle");
    check(Capability::MatrixFreeApplyBlock, "MatrixFreeApplyBlock");
    check(Capability::FieldSplitPreconditioner, "FieldSplitPreconditioner");
    check(Capability::GenericBlockPreconditioner, "GenericBlockPreconditioner");
    check(Capability::DistributedSolve, "DistributedSolve");
    check(Capability::DirectSolve, "DirectSolve");
    check(Capability::AMGPreconditioner, "AMGPreconditioner");
    check(Capability::VariableBlockLayout, "VariableBlockLayout");
    return msg;
}

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_BACKEND_CAPABILITY_H
