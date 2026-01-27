#ifndef SVMP_FE_FORMS_TENSOR_SYMMETRY_OPTIMIZER_H
#define SVMP_FE_FORMS_TENSOR_SYMMETRY_OPTIMIZER_H

/**
 * @file SymmetryOptimizer.h
 * @brief Symmetry-aware canonicalization and lowering helpers for tensor expressions
 *
 * This module provides:
 * - canonical mapping/sign rules for common symmetries (symmetric/antisymmetric/elasticity),
 * - a lightweight AST transform that rewrites IndexedAccess on symmetric/antisymmetric
 *   tensors to reference only independent components (with sign/zero rules).
 *
 * The primary use is to enable compact loop-based lowering and improve downstream
 * CSE/hashing by avoiding duplicate symmetric components.
 */

#include "Forms/FormExpr.h"
#include "Forms/Tensor/TensorSymmetry.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace tensor {

struct SymmetryCanonicalComponent {
    bool ok{true};
    std::string message{};

    // True if the component is structurally zero (e.g., antisymmetric diagonal).
    bool is_zero{false};

    // Sign multiplier for antisymmetric canonicalization (±1).
    int sign{1};

    // Canonicalized component indices (rank preserved).
    std::vector<int> indices{};
};

/**
 * @brief Canonicalize a concrete component index under a symmetry rule.
 *
 * Supported:
 * - 2nd-order symmetric / antisymmetric
 * - 4th-order full elasticity (minor + major symmetries)
 */
[[nodiscard]] SymmetryCanonicalComponent canonicalizeComponent(const TensorSymmetry& symmetry,
                                                               std::vector<int> indices);

/**
 * @brief Map a symmetric pair (i<=j) to a packed Voigt-like linear index.
 *
 * Ordering matches TensorSymmetry::independentComponents for Symmetric2:
 * (0,0),(0,1),...,(0,dim-1),(1,1),(1,2),...,(dim-1,dim-1)
 */
[[nodiscard]] int packedIndexSymmetricPair(int i, int j, int dim);

/**
 * @brief Map an antisymmetric pair (i<j) to a packed linear index.
 *
 * Ordering matches TensorSymmetry::independentComponents for Antisymmetric2:
 * (0,1),(0,2),...,(0,dim-1),(1,2),...,(dim-2,dim-1)
 */
[[nodiscard]] int packedIndexAntisymmetricPair(int i, int j, int dim);

/**
 * @brief Map a full-elasticity (i,j,k,l) component to a packed index using Voigt pairs.
 *
 * The returned index corresponds to the upper triangle of a symmetric (ncomp x ncomp)
 * matrix, where ncomp = dim*(dim+1)/2.
 */
[[nodiscard]] int packedIndexElasticityVoigt(int i, int j, int k, int l, int dim);

struct SymmetryLoweringResult {
    bool ok{true};
    std::string message{};
    FormExpr expr{};
};

/**
 * @brief Rewrite symmetric/antisymmetric IndexedAccess occurrences to canonical form.
 *
 * Currently recognizes:
 * - IndexedAccess(sym(A), i, j): enforces canonical index-id ordering within the access.
 * - IndexedAccess(skew(A), i, j): enforces ordering and applies sign; diagonal -> 0.
 * - IndexedAccess(I, i, j): enforces canonical ordering (no sign).
 *
 * Canonical ordering for symbolic indices uses index-id ordering (stable within an expression).
 */
[[nodiscard]] SymmetryLoweringResult lowerWithSymmetry(const FormExpr& expr);

} // namespace tensor
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_TENSOR_SYMMETRY_OPTIMIZER_H

