#ifndef SVMP_FE_FORMS_WEAKFORM_H
#define SVMP_FE_FORMS_WEAKFORM_H

/**
 * @file WeakForm.h
 * @brief Legacy bundle of a residual form with strong constraints
 *
 * @note **Prefer the canonical API instead:**
 * call `installFormulation()` for the residual and `installStrongDirichlet()`
 * for the boundary conditions separately. This container remains for backward
 * compatibility but is not part of the recommended authoring workflow.
 */

#include "Forms/BoundaryConditions.h"

#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {

/**
 * @brief Represents a complete variational problem (Weak Form + Strong Constraints).
 *
 * @deprecated Use installFormulation() + installStrongDirichlet() instead.
 * This container bundles the residual form R(u;v) with the set of strong
 * boundary conditions, but the canonical workflow now calls these separately.
 */
struct [[deprecated("Use installFormulation() + installStrongDirichlet() instead")]] WeakForm {
    /// The weak-form residual expression, e.g., (k*grad(u), grad(v)) * dx
    FormExpr residual{};

    /// Strong Dirichlet constraints (u = g on boundary) to be enforced.
    std::vector<bc::StrongDirichlet> strong_constraints{};

    /// Helper to append a strong Dirichlet constraint.
    void addDirichlet(bc::StrongDirichlet bc)
    {
        strong_constraints.push_back(std::move(bc));
    }
};

} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_WEAKFORM_H
