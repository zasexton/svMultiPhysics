#ifndef SVMP_FE_FORMS_WEAKFORM_H
#define SVMP_FE_FORMS_WEAKFORM_H

/**
 * @file WeakForm.h
 * @brief Bundle a residual form with strong constraints
 *
 * This is a lightweight "problem description" container used by FE/Systems
 * installers to ensure constraints are consistently installed alongside the
 * residual form.
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
 * This container bundles the residual form R(u;v) with the set of strong (essential)
 * boundary conditions that must be enforced algebraically on the system.
 */
struct WeakForm {
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
