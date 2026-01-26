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

#include <cstdint>
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

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_TENSOR_CONTRACTION_H
