#ifndef SVMP_FE_FORMS_JIT_KERNEL_IR_H
#define SVMP_FE_FORMS_JIT_KERNEL_IR_H

/**
 * @file KernelIR.h
 * @brief Deterministic, flat intermediate representation for FE/Forms integrands
 *
 * This IR is a pre-LLVM lowering target used to:
 * - validate that an expression can be lowered without interpreter-only features,
 * - provide a deterministic op list suitable for hashing/caching,
 * - enable CSE/hoisting passes before LLVM emission (future work).
 *
 * NOTE: This file deliberately contains no LLVM dependencies.
 */

#include "Forms/FormExpr.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace jit {

struct KernelIROp {
    FormExprType type{FormExprType::Constant};
    std::uint32_t first_child{0};
    std::uint32_t child_count{0};

    // Type-specific immediate payload. Interpretation depends on `type`.
    // Examples:
    // - Constant: imm0 is bit-cast double
    // - *Ref terminals: imm0 is slot index
    // - Component: imm0 packs (i,j)
    // - Discrete/StateField: imm0/imm1 pack SpaceSignature + FieldId
    std::uint64_t imm0{0};
    std::uint64_t imm1{0};
};

struct KernelIR {
    std::vector<KernelIROp> ops{};
    std::vector<std::uint32_t> children{};
    std::uint32_t root{0};

    [[nodiscard]] bool empty() const noexcept { return ops.empty(); }
    [[nodiscard]] std::size_t opCount() const noexcept { return ops.size(); }

    /**
     * @brief Compute a deterministic 64-bit hash of the IR
     *
     * The hash is stable for a given IR instance and does not depend on node
     * addresses. Expressions that rely on runtime callbacks (e.g. Coefficient)
     * should be treated as non-cacheable even if a hash is available.
     */
    [[nodiscard]] std::uint64_t stableHash64() const;
    [[nodiscard]] std::string dump() const;
};

struct KernelIRBuildOptions {
    bool cse{true};
    bool canonicalize_commutative{true};
};

struct KernelIRBuildResult {
    KernelIR ir{};
    bool cacheable{true};
};

/**
 * @brief Lower a scalar/vector/tensor expression tree to KernelIR
 *
 * @param integrand Expression without measures (dx/ds/dS). Coupled/parameter
 *        names must be resolved to slot-based refs before lowering.
 */
KernelIRBuildResult lowerToKernelIR(const FormExpr& integrand,
                                    const KernelIRBuildOptions& options = {});

KernelIRBuildResult lowerToKernelIR(const FormExprNode& root,
                                    const KernelIRBuildOptions& options = {});

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_KERNEL_IR_H
