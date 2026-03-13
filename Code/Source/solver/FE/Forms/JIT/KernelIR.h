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

// ============================================================================
// Shaped-zero encoding for TypedZero imm1
// ============================================================================
// When imm1==0, inferShapes treats TypedZero as scalar (legacy behavior).
// When bit 63 is set, imm1 encodes the output shape so that the optimizer
// can collapse shape-changing ops (Gradient, OuterProduct, etc.) applied to
// zero without losing shape information.

namespace typed_zero {
    /// Sentinel bit indicating imm1 carries encoded shape data.
    constexpr std::uint64_t kShapeSentinel = 1ULL << 63;

    /// Shape kind values (mirroring LLVMGen's Shape::Kind).
    constexpr std::uint8_t kScalar  = 0;
    constexpr std::uint8_t kVector  = 1;
    constexpr std::uint8_t kMatrix  = 2;
    constexpr std::uint8_t kTensor3 = 3;
    constexpr std::uint8_t kTensor4 = 4;

    /// Encode a shape into imm1.
    /// @param kind  Shape kind (kScalar, kVector, kMatrix, ...).
    /// @param d0-d3 Dimension extents (unused dims should be 1).
    inline constexpr std::uint64_t encode(std::uint8_t kind,
                                          std::uint32_t d0 = 1, std::uint32_t d1 = 1,
                                          std::uint32_t d2 = 1, std::uint32_t d3 = 1) noexcept
    {
        return kShapeSentinel
             | (static_cast<std::uint64_t>(kind) & 0x7ULL)
             | ((static_cast<std::uint64_t>(d0) & 0x3FFULL) << 3)
             | ((static_cast<std::uint64_t>(d1) & 0x3FFULL) << 13)
             | ((static_cast<std::uint64_t>(d2) & 0x3FFULL) << 23)
             | ((static_cast<std::uint64_t>(d3) & 0x3FFULL) << 33);
    }

    /// Check whether imm1 carries encoded shape data.
    inline constexpr bool hasShape(std::uint64_t imm1) noexcept
    {
        return (imm1 & kShapeSentinel) != 0;
    }

    /// Decode shape kind from imm1.
    inline constexpr std::uint8_t kind(std::uint64_t imm1) noexcept
    {
        return static_cast<std::uint8_t>(imm1 & 0x7ULL);
    }

    /// Decode dimension @p i (0-based) from imm1.
    inline constexpr std::uint32_t dim(std::uint64_t imm1, int i) noexcept
    {
        const int shift = 3 + 10 * i;
        return static_cast<std::uint32_t>((imm1 >> shift) & 0x3FFULL);
    }
} // namespace typed_zero

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
