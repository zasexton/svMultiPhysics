#ifndef SVMP_FE_FORMS_TENSOR_TENSOR_CSE_H
#define SVMP_FE_FORMS_TENSOR_TENSOR_CSE_H

/**
 * @file TensorCSE.h
 * @brief Tensor-aware common subexpression analysis (CSE) for FormExpr
 *
 * This module performs a structural analysis of a FormExpr tree to identify
 * repeated subexpressions. Unlike the scalar `KernelIR` CSE, this analysis is
 * intended to work in the presence of tensor-calculus/index-notation nodes:
 * subexpressions are compared up to local dummy-index renaming using
 * `TensorCanonicalize` helpers.
 *
 * The output is a plan describing which subexpressions should be treated as
 * temporaries and an evaluation order suitable for future codegen.
 */

#include "Forms/FormExpr.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

struct TensorCSEOptions {
    // If true, use TensorCanonicalize term ordering before keying.
    bool canonicalize_commutative{true};

    // If true, treat local dummy-index renaming as equivalent (i.e., compare
    // via canonical index renumbering per subexpression).
    bool canonicalize_indices{true};

    // Minimum number of occurrences to create a temporary.
    int min_use_count{2};

    // Consider these node types "expensive" even if small.
    bool hoist_det{true};
    bool hoist_inv{true};
    bool hoist_cofactor{true};

    // For other node types, require at least this many nodes in the subtree.
    std::size_t min_subtree_nodes{8};
};

struct TensorCSETmp {
    std::size_t id{0};
    std::uint64_t key_hash{0};
    std::string key{};
    std::size_t use_count{0};
    std::size_t node_count{0};
    FormExpr expr{};
};

struct TensorCSEPlan {
    bool ok{true};
    std::string message{};

    // Temporaries in a safe evaluation order (dependencies first).
    std::vector<TensorCSETmp> temporaries{};
};

/**
 * @brief Analyze an expression for repeated tensor-calculus subexpressions.
 */
[[nodiscard]] TensorCSEPlan planTensorCSE(const FormExpr& expr,
                                          const TensorCSEOptions& options = {});

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_TENSOR_CSE_H

