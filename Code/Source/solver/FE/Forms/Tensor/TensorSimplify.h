#ifndef SVMP_FE_FORMS_TENSOR_TENSOR_SIMPLIFY_H
#define SVMP_FE_FORMS_TENSOR_TENSOR_SIMPLIFY_H

/**
 * @file TensorSimplify.h
 * @brief Tensor-calculus-aware simplification for Einstein-index expressions
 *
 * This module provides a small, terminating rewrite system intended to reduce
 * indexed expressions before hashing/lowering. It is conservative: rules are
 * applied only when they are algebraically valid under the current index-usage
 * contract (each index appears at most twice).
 */

#include "Forms/FormExpr.h"

#include <cstddef>
#include <string>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

struct TensorSimplifyOptions {
    int max_passes{12};
    bool canonicalize_terms{true};
};

struct TensorSimplifyStats {
    std::size_t passes{0};

    std::size_t delta_traces{0};
    std::size_t delta_substitutions{0};
    std::size_t delta_compositions{0};

    std::size_t symmetry_zeroes{0};
    std::size_t epsilon_identities{0};
};

struct TensorSimplifyResult {
    bool ok{true};
    std::string message{};
    FormExpr expr{};
    TensorSimplifyStats stats{};
    bool changed{false};
};

/**
 * @brief Simplify a tensor/index expression with a fixed-point iteration
 */
[[nodiscard]] TensorSimplifyResult simplifyTensorExpr(const FormExpr& expr,
                                                      const TensorSimplifyOptions& options = {});

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_TENSOR_SIMPLIFY_H

