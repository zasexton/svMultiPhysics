#ifndef SVMP_FE_FORMS_TENSOR_TENSOR_CONTRACTION_H
#define SVMP_FE_FORMS_TENSOR_TENSOR_CONTRACTION_H

/**
 * @file TensorContraction.h
 * @brief Analysis utilities for Einstein-style indexed expressions
 *
 * This module analyzes `FormExprType::IndexedAccess` usage patterns so that
 * index contractions can be lowered without explicit scalar expansion.
 */

#include "Forms/FormExpr.h"
#include "Forms/Tensor/TensorIndex.h"

#include <cstdint>
#include <unordered_map>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

struct ContractionAnalysis {
    struct IndexInfo {
        int id{-1};
        std::string name{};
        int extent{0};
        int count{0};
        std::size_t first_occurrence{0};
    };

    bool ok{true};
    std::string message{};

    // Indices appearing once (free) or twice (dummy/bound), in deterministic
    // order (first occurrence in a pre-order traversal).
    std::vector<IndexInfo> free_indices{};
    std::vector<IndexInfo> bound_indices{};

    [[nodiscard]] bool isFullyContractedScalar() const noexcept { return ok && free_indices.empty(); }
};

/**
 * @brief Analyze an expression for indexed contractions (`IndexedAccess`)
 *
 * Rules (current implementation contract):
 * - Each index id must appear exactly once (free) or exactly twice (bound).
 * - Each repeated id must have a consistent extent across appearances.
 *
 * The analysis is intended to be used by JIT validation and lowering stages.
 */
[[nodiscard]] ContractionAnalysis analyzeContractions(const FormExpr& expr);

// ============================================================================
// Contraction transforms and cost modeling
// ============================================================================

struct ContractionTransformResult {
    bool ok{true};
    std::string message{};
    FormExpr expr{};
};

/**
 * @brief Replace all occurrences of index id `eliminate_id` with `keep_id`
 *
 * This is a low-level transform used by simplification (e.g., δ-contraction).
 * It updates `IndexedAccess` id/extent/name metadata and (optionally) rewrites
 * variances for replaced occurrences.
 */
[[nodiscard]] ContractionTransformResult contractIndices(const FormExpr& expr,
                                                         int keep_id,
                                                         IndexVariance keep_variance,
                                                         int eliminate_id);

[[nodiscard]] inline ContractionTransformResult contractIndices(const FormExpr& expr, int keep_id, int eliminate_id)
{
    return contractIndices(expr, keep_id, IndexVariance::None, eliminate_id);
}

struct ContractionCostModel {
    // Weights for the combined cost objective.
    double flop_weight{1.0};
    double memory_weight{0.1};
    double write_weight{0.1};

    bool enable_delta_shortcuts{true};
    bool enable_symmetry_cost{true};
};

struct TensorOperand {
    std::vector<int> indices{};    ///< Unique index ids on this operand
    bool is_delta{false};          ///< Kronecker delta / metric-like operand
    bool is_symmetric{false};      ///< 2nd-order symmetric tensor
    bool is_antisymmetric{false};  ///< 2nd-order antisymmetric tensor
};

struct ContractionPlan {
    struct Step {
        int lhs{-1};
        int rhs{-1};
    };

    bool ok{true};
    std::string message{};

    std::uint64_t estimated_flops{0};
    std::uint64_t estimated_reads{0};
    std::uint64_t estimated_writes{0};

    // Steps refer to operand ids [0..N-1] and intermediate ids [N..].
    std::vector<Step> steps{};
};

/**
 * @brief Choose an (approx) optimal contraction order for a chain of operands
 *
 * This uses dynamic programming for up to 16 operands and a greedy fallback
 * beyond that. The cost model is a simple proxy intended for loop-order
 * selection and future TensorIR lowering.
 */
[[nodiscard]] ContractionPlan optimalContractionOrder(const std::vector<TensorOperand>& operands,
                                                      const std::unordered_map<int, int>& index_extents,
                                                      const ContractionCostModel& model = {});

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_TENSOR_CONTRACTION_H
