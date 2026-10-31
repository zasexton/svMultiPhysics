/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_NORMAL_COMPONENT_CONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_NORMAL_COMPONENT_CONSTRAINT_H

/**
 * @file NormalComponentConstraint.h
 * @brief u . n = g strongly on a boundary, for a vector Lagrange field
 *
 * The free-slip (no-penetration) condition, and its nonzero form, as affine
 * constraint lines after deal.II's compute_nonzero_normal_flux_constraints.
 * At each support point of the field on the boundary:
 *
 *  - every boundary facet of the markers that holds the point gives its unit
 *    outward normal there, weighted by the facet's measure;
 *  - normals that differ by less than the feature angle are one direction
 *    (a smooth or faceted curved wall); the direction is the normalized
 *    weighted sum of its normals;
 *  - with k directions n_1..n_k at the point (k = 1 on a wall, 2 on a 3D
 *    edge, dim at a corner) the constraint is n_j . u = g for each j: when
 *    k < dim the k components whose submatrix of N = [n_1; ...; n_k] has the
 *    largest |determinant| are slaves of the others,
 *    u_S = N_S^{-1} (g 1 - N_R u_R), lowest component index first on a tie;
 *    when k >= dim every component is prescribed (least squares when
 *    k > dim, which is exact for g = 0).
 *
 * The lines are the same at any rank count: each facet's contribution is
 * computed by the owner of its cell from the facet alone, sent to the owner
 * of the point's degrees of freedom, and summed there in the order of the
 * global facet ids; g is evaluated at the point's coordinates as the facet
 * with the lowest id maps them.  A point any of whose components an earlier
 * constraint already holds (an inlet corner) is left to that constraint.
 * Walls that meet at an edge belong in one constraint, so that the edge
 * sees both normals.
 *
 * The value g may read the position, the time and the time step.  The
 * normals follow the active mesh: the lines are rebuilt when the geometry
 * changes.
 */

#include "Constraints/SystemConstraint.h"

#include "Forms/FormExpr.h"

#include <array>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {

struct NormalComponentOptions {
    /// Facet normals closer than this angle (degrees) are one direction.
    Real feature_angle_degrees{45.0};
};

class NormalComponentConstraint final : public ISystemConstraint {
public:
    /// u . n = @p value on the boundary faces of @p boundary_markers.
    NormalComponentConstraint(FieldId field,
                              std::vector<int> boundary_markers,
                              forms::FormExpr value = forms::FormExpr::constant(0.0),
                              NormalComponentOptions options = {});

    void apply(const systems::FESystem& system, AffineConstraints& constraints) override;

    bool updateValues(const systems::FESystem& system,
                      AffineConstraints& constraints,
                      double time,
                      double dt) override;

    [[nodiscard]] bool isTimeDependent() const noexcept override { return is_time_dependent_; }

    [[nodiscard]] ConstraintDependencyDeclaration dependencyDeclaration() const override;

    [[nodiscard]] systems::SetupStorageRequirements storageRequirements() const noexcept override;

    [[nodiscard]] std::string manifestDescription() const override;

    /// One constrained component: slave = sum(weight * master) + g(x) * value_factor.
    struct Line {
        GlobalIndex slave{-1};
        std::vector<GlobalIndex> masters{};
        std::vector<Real> weights{};
        Real value_factor{0.0};
        std::array<Real, 3> x{0.0, 0.0, 0.0};   ///< where g is evaluated
    };

    /// The lines this rank added (owned slaves), in the order added.
    [[nodiscard]] const std::vector<Line>& lines() const noexcept { return lines_; }

private:
    FieldId field_{INVALID_FIELD_ID};
    std::vector<int> boundary_markers_{};
    forms::FormExpr value_{};
    NormalComponentOptions options_{};
    bool is_time_dependent_{false};
    std::vector<Line> lines_{};
};

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_NORMAL_COMPONENT_CONSTRAINT_H
