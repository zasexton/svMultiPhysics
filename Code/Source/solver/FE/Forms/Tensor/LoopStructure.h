#ifndef SVMP_FE_FORMS_TENSOR_LOOP_STRUCTURE_H
#define SVMP_FE_FORMS_TENSOR_LOOP_STRUCTURE_H

/**
 * @file LoopStructure.h
 * @brief Loop-nest representation for tensor contractions (pre-TensorIR)
 *
 * This module generates a compact loop-based representation of fully- or
 * partially-contracted Einstein-summation expressions built from
 * `FormExprType::IndexedAccess` nodes. The intent is to avoid scalar-term
 * explosion from `forms::einsum()` and provide a lowering target for future
 * LLVM emission.
 */

#include "Forms/FormExpr.h"
#include "Forms/Tensor/TensorContraction.h"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

enum class TensorStorageKind : std::uint8_t {
    Dense,
    Symmetric2,
    Antisymmetric2,
    ElasticityVoigt,
    KroneckerDelta,
};

struct LoopIndex {
    int id{-1};
    std::string name{};
    int extent{0};

    // Optional lower bound constraint: loop variable starts at
    // (assignment[lower_bound_id] + lower_bound_offset) instead of 0.
    // Used for triangular/structured loops (e.g., symmetric/antisymmetric tensors).
    int lower_bound_id{-1};
    int lower_bound_offset{0};

    bool vectorize{false};
    int vector_width{1};
};

struct LoopStructureOptions {
    bool enable_symmetry_lowering{true};
    bool enable_optimal_contraction_order{true};
    bool enable_vectorization_hints{true};
    bool enable_delta_shortcuts{true};

    // Threshold heuristic for the incremental lowering strategy:
    // if scalar expansion would generate more than this many terms, prefer loops.
    std::uint64_t scalar_expansion_term_threshold{64};

    ContractionCostModel cost_model{};
};

struct TensorSpec {
    TensorStorageKind storage{TensorStorageKind::Dense};
    int rank{0};
    std::vector<int> axes{};    // index ids in storage order
    std::vector<int> extents{}; // same length as axes
    std::size_t size{0};        // number of stored scalars (compressed if applicable)

    // For input tensors, `base` is the underlying tensor-valued expression.
    // For temporaries, base is empty.
    FormExpr base{};

    // If this TensorSpec is a view into a higher-rank base tensor, this maps
    // base axes -> tensor axes (allows repeated axes for diagonal slices).
    int base_rank{0};
    std::vector<int> base_axis_to_tensor_axis{};
};

struct ContractionOp {
    enum class Kind : std::uint8_t {
        Contraction,        // out = contract(lhs, rhs) over sum_axes
        Reduction,          // out = reduce(lhs) over sum_axes (rhs unused)
        DeltaSubstitution,  // out = substitute via Kronecker delta (no loop)
    };

    Kind kind{Kind::Contraction};
    int lhs{-1};
    int rhs{-1};
    int out{-1};

    std::vector<int> out_axes{};
    std::vector<int> sum_axes{};

    std::vector<LoopIndex> loops{};

    std::uint64_t estimated_flops{0};
};

struct LoopNestProgram {
    bool ok{true};
    std::string message{};

    // Output tensor description (rank==0 => scalar).
    TensorSpec output{};

    // Loop metadata for iterating over the stored output components.
    // For dense outputs this is a rectangular loop nest over output.axes.
    // For structured outputs (e.g., symmetric/antisymmetric) this may be triangular.
    std::vector<LoopIndex> output_loops{};

    // Tensor buffers used by the program. Inputs come first, then temporaries.
    std::vector<TensorSpec> tensors{};

    // Contraction ops executed in order; each writes a new temporary tensor.
    std::vector<ContractionOp> ops{};

    struct Contribution {
        // Index into `tensors`. For scalar-only contributions to a scalar output,
        // set tensor_id = -1 and use `scalar` as the value.
        int tensor_id{-1};

        // Op index after which this contribution can be consumed. This enables
        // temporary lifetime tracking and cross-term reuse (contributions are
        // intended to be accumulated as soon as possible, not necessarily at
        // end-of-program).
        int available_after_op{-1};

        FormExpr scalar{};
    };

    // Output is computed as a sum of scaled contributions.
    std::vector<Contribution> contributions{};

    // Estimated flop count for the contraction ops only.
    std::uint64_t estimated_flops{0};

    [[nodiscard]] bool isScalar() const noexcept { return output.rank == 0; }
};

/**
 * @brief Generate a loop-nest program for a single Einstein-style expression.
 *
 * Limitations (current):
 * - IndexedAccess rank <= 4 is supported.
 * - Expression must be a sum of products of scalar-like factors; if an additive
 *   node appears inside a product chain, lowering fails and the caller should
 *   fall back to `forms::einsum()`.
 */
[[nodiscard]] LoopNestProgram generateLoopNest(const FormExpr& expr,
                                               const LoopStructureOptions& options = {});

/**
 * @brief Fuse two loop-nest programs that produce the same output shape.
 *
 * This is a conservative program-level fusion that concatenates contraction ops
 * and combines scalar prefactors. It does not attempt cross-program CSE.
 */
[[nodiscard]] LoopNestProgram fuseLoops(const LoopNestProgram& a,
                                        const LoopNestProgram& b);

/**
 * @brief Reorder reduction indices and loop metadata to improve locality.
 *
 * This uses a simple heuristic based on operand axis positions and extents.
 */
void optimizeLoopOrder(LoopNestProgram& program);

/**
 * @brief Result of the incremental lowering decision.
 */
struct TensorLoweringResult {
    bool ok{true};
    std::string message{};

    bool used_loop_nest{false};
    LoopNestProgram loop{};
    FormExpr einsum_expanded{};
};

/**
 * @brief Lower tensor calculus via loop-based IR when profitable, else via einsum.
 */
[[nodiscard]] TensorLoweringResult lowerTensorExpressionIncremental(const FormExpr& expr,
                                                                    const LoopStructureOptions& options = {});

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_LOOP_STRUCTURE_H
