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
    [[nodiscard]] std::vector<std::uint64_t> perOpStructuralHashes() const;
    [[nodiscard]] std::string dump() const;

    /**
     * @brief Optimize the IR in-place: zero propagation, constant folding, CSE, DCE
     *
     * Returns the number of ops eliminated.
     */
    std::size_t optimize();

    /**
     * @brief Compute per-op subtree cost (bottom-up additive cost model)
     *
     * Leaf ops (Constant, ParameterRef, etc.) have cost 1.
     * Internal ops accumulate children's costs + 1.
     * Reduce-sum ops (StateField, DiscreteField) get a multiplier
     * reflecting the implied DOF loop.
     */
    [[nodiscard]] std::vector<std::uint32_t> subtreeCosts() const;
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

// ============================================================================
// Term-group planning for micro-kernel splitting
// ============================================================================

/// Description of one contiguous group of terms within a block.
struct TermGroupPlan {
    std::size_t first_term{0};     ///< Index into the block's term list
    std::size_t num_terms{0};      ///< Number of terms in this group
    std::uint64_t estimated_text_bytes{0};
};

/// Split plan for a single coupled block.
struct BlockSplitPlan {
    std::size_t block_index{0};
    std::vector<TermGroupPlan> groups;
    bool needs_split{false};
    std::uint64_t total_estimated_bytes{0};
};

/// Plan contiguous term groups fitting within @p budget_bytes of .text.
///
/// @param term_op_counts  Number of KernelIR ops per term.
/// @param budget_bytes    Target .text budget per helper function.
/// @param bytes_per_op    Estimated bytes of machine code per KernelIR op.
/// @return A BlockSplitPlan with needs_split=true if the block exceeds budget.
[[nodiscard]] BlockSplitPlan planTermGroups(
    const std::vector<std::size_t>& term_op_counts,
    std::uint64_t budget_bytes,
    std::uint64_t bytes_per_op);

} // namespace jit
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_JIT_KERNEL_IR_H
