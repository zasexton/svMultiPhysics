#ifndef SVMP_FE_FORMS_TENSOR_TENSOR_CANONICALIZE_H
#define SVMP_FE_FORMS_TENSOR_TENSOR_CANONICALIZE_H

/**
 * @file TensorCanonicalize.h
 * @brief Canonicalization helpers for indexed expressions
 *
 * Note: `FormExprType::IndexedAccess` currently stores index ids produced by
 * `forms::Index`'s global counter. These ids are not stable across program runs,
 * so any JIT caching must canonicalize ids to a deterministic numbering.
 *
 * This module computes a deterministic renumbering map for index ids and is used
 * by JIT lowering/caching (KernelIR) rather than mutating the original AST.
 */

#include "Forms/FormExpr.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

struct CanonicalIndexRenaming {
    std::unordered_map<int, int> old_to_canonical{};
    std::vector<int> canonical_to_old{};
};

/**
 * @brief Compute a deterministic index-id renumbering for `IndexedAccess`
 *
 * Indices are assigned canonical ids in order of first appearance in a pre-order
 * traversal, and in rank order within each `IndexedAccess` occurrence.
 */
[[nodiscard]] CanonicalIndexRenaming computeCanonicalIndexRenaming(const FormExpr& expr);

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_TENSOR_CANONICALIZE_H

