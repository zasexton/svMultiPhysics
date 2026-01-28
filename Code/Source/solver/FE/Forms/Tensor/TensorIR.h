#ifndef SVMP_FE_FORMS_TENSOR_TENSOR_IR_H
#define SVMP_FE_FORMS_TENSOR_TENSOR_IR_H

/**
 * @file TensorIR.h
 * @brief Deterministic loop-based IR for tensor calculus lowering (pre-LLVM)
 *
 * TensorIR is a structured lowering target for indexed tensor expressions that
 * avoids scalar-term explosion. It is intended as an intermediate representation
 * between tensor-calculus `FormExpr` trees (via `LoopNestProgram`) and LLVM IR
 * emission (via `Forms/JIT/LLVMTensorGen.*`).
 */

#include "Forms/Tensor/LoopStructure.h"
#include "Forms/Tensor/TensorAllocation.h"

#include <cstdint>
#include <string>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

/**
 * @brief Node kind tags for a future explicit TensorIR instruction stream.
 *
 * Current Phase 6 uses `LoopNestProgram` + `TensorAllocationPlan` directly for
 * compactness, but we keep these tags to make the intended IR vocabulary
 * explicit (loops, loads/stores, and scalar ops).
 */
enum class TensorIRNodeKind : std::uint8_t {
    // Control flow / structure.
    LoopNest,

    // Tensor operations.
    Contraction,
    Reduction,
    DeltaSubstitution,

    // Memory (tensor element access).
    Load,
    Store,

    // Scalar operations (minimal set for contraction loops).
    Constant,
    Add,
    Multiply,
    FMA,
};

struct TensorIRNode {
    TensorIRNodeKind kind{TensorIRNodeKind::Constant};
    std::uint32_t first_child{0};
    std::uint32_t child_count{0};
    std::uint64_t imm0{0};
    std::uint64_t imm1{0};
};

struct TensorIR {
    LoopNestProgram program{};
    TensorAllocationPlan allocation{};

    LoopStructureOptions loop_options{};
    TensorAllocationOptions alloc_options{};

    bool cacheable{true};

    /**
     * @brief Deterministic 64-bit hash of the TensorIR (for kernel caching)
     *
     * Notes:
     * - The hash is independent of node addresses for cacheable expressions.
     * - For non-cacheable expressions (external calls), the hash may include
     *   within-run identifiers to avoid collisions.
     */
    [[nodiscard]] std::uint64_t stableHash64() const;
};

struct TensorIRLoweringOptions {
    // If true, cache TensorIR lowering results in a process-local cache.
    bool enable_cache{true};

    // If true, force loop-based lowering when IndexedAccess is present,
    // bypassing the scalar-expansion threshold heuristic.
    bool force_loop_nest{false};

    // If true, log tensor-vs-scalar lowering decisions (reason/estimates).
    bool log_decisions{false};

    LoopStructureOptions loop{};
    TensorAllocationOptions alloc{};
};

struct TensorIRLoweringResult {
    bool ok{true};
    std::string message{};

    bool used_loop_nest{false};
    bool cacheable{true};

    TensorIR ir{};

    // When `used_loop_nest==false`, this contains the preferred scalar-lowered
    // expression (either the original scalar expression or an einsum expansion).
    FormExpr fallback_expr{};
};

/**
 * @brief Lower a tensor-calculus expression to TensorIR when profitable.
 *
 * This mirrors `lowerTensorExpressionIncremental` but returns an IR package
 * suitable for deterministic hashing and LLVM emission.
 */
[[nodiscard]] TensorIRLoweringResult lowerToTensorIR(const FormExpr& expr,
                                                     const TensorIRLoweringOptions& options = {});

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_TENSOR_IR_H
