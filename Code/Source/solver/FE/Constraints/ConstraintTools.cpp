/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "ConstraintTools.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace svmp {
namespace FE {
namespace constraints {

// ============================================================================
// Dirichlet constraint generation
// ============================================================================

void makeDirichletConstraints(
    std::span<const GlobalIndex> boundary_dofs,
    double value,
    AffineConstraints& constraints,
    [[maybe_unused]] const DirichletConstraintOptions& options)
{
    for (GlobalIndex dof : boundary_dofs) {
        constraints.addLine(dof);
        constraints.setInhomogeneity(dof, value);
    }
}

void makeDirichletConstraints(
    std::span<const GlobalIndex> boundary_dofs,
    std::span<const std::array<double, 3>> dof_coords,
    const std::function<double(double, double, double)>& value_func,
    AffineConstraints& constraints,
    [[maybe_unused]] const DirichletConstraintOptions& options)
{
    if (boundary_dofs.size() != dof_coords.size()) {
        CONSTRAINT_THROW("DOFs and coordinates must have same size");
    }

    for (std::size_t i = 0; i < boundary_dofs.size(); ++i) {
        GlobalIndex dof = boundary_dofs[i];
        const auto& coord = dof_coords[i];
        double value = value_func(coord[0], coord[1], coord[2]);

        constraints.addLine(dof);
        constraints.setInhomogeneity(dof, value);
    }
}

void makeDirichletConstraintsVector(
    std::span<const GlobalIndex> boundary_dofs,
    std::span<const std::array<double, 3>> dof_coords,
    const std::function<std::vector<double>(double, double, double)>& value_func,
    int num_components,
    AffineConstraints& constraints,
    const DirichletConstraintOptions& options)
{
    // Assumes DOFs are interleaved: [u0_x, u0_y, u0_z, u1_x, u1_y, u1_z, ...]
    std::size_t n_nodes = boundary_dofs.size() / static_cast<std::size_t>(num_components);

    if (n_nodes != dof_coords.size()) {
        CONSTRAINT_THROW("Number of nodes must match coordinate count");
    }

    for (std::size_t node = 0; node < n_nodes; ++node) {
        const auto& coord = dof_coords[node];
        std::vector<double> values = value_func(coord[0], coord[1], coord[2]);

        if (static_cast<int>(values.size()) != num_components) {
            CONSTRAINT_THROW("Value function must return num_components values");
        }

        for (int comp = 0; comp < num_components; ++comp) {
            if (options.component_mask.allSelected() || options.component_mask[static_cast<std::size_t>(comp)]) {
                GlobalIndex dof = boundary_dofs[node * static_cast<std::size_t>(num_components) + static_cast<std::size_t>(comp)];
                constraints.addLine(dof);
                constraints.setInhomogeneity(dof, values[static_cast<std::size_t>(comp)]);
            }
        }
    }
}

// ============================================================================
// Periodic constraint generation
// ============================================================================

void makePeriodicConstraints(
    std::span<const GlobalIndex> slave_dofs,
    std::span<const std::array<double, 3>> slave_coords,
    std::span<const GlobalIndex> master_dofs,
    std::span<const std::array<double, 3>> master_coords,
    const std::function<std::array<double, 3>(std::array<double, 3>)>& transform,
    AffineConstraints& constraints,
    const PeriodicConstraintOptions& options)
{
    // Find matching pairs
    auto matches = findMatchingPoints(slave_coords, master_coords, transform, options.tolerance);

    // Create periodic constraints for each match
    double sign = options.flip_sign ? -1.0 : 1.0;

    for (const auto& [slave_idx, master_idx] : matches) {
        GlobalIndex slave_dof = slave_dofs[slave_idx];
        GlobalIndex master_dof = master_dofs[master_idx];

        constraints.addLine(slave_dof);
        constraints.addEntry(slave_dof, master_dof, sign);
    }
}

void makePeriodicConstraintsTranslation(
    std::span<const GlobalIndex> slave_dofs,
    std::span<const std::array<double, 3>> slave_coords,
    std::span<const GlobalIndex> master_dofs,
    std::span<const std::array<double, 3>> master_coords,
    std::array<double, 3> translation,
    AffineConstraints& constraints,
    const PeriodicConstraintOptions& options)
{
    auto transform = [translation](std::array<double, 3> p) -> std::array<double, 3> {
        return {{p[0] + translation[0], p[1] + translation[1], p[2] + translation[2]}};
    };

    makePeriodicConstraints(slave_dofs, slave_coords, master_dofs, master_coords,
                            transform, constraints, options);
}

// ============================================================================
// Hanging node constraint generation
// ============================================================================

void makeHangingNodeConstraints1D(
    std::span<const GlobalIndex> hanging_dofs,
    std::span<const GlobalIndex> parent_dofs_1,
    std::span<const GlobalIndex> parent_dofs_2,
    AffineConstraints& constraints,
    [[maybe_unused]] const HangingNodeOptions& options)
{
    if (hanging_dofs.size() != parent_dofs_1.size() ||
        hanging_dofs.size() != parent_dofs_2.size()) {
        CONSTRAINT_THROW("Hanging and parent DOF arrays must have same size");
    }

    // For P1 elements, midpoint interpolation: u_h = 0.5*u_1 + 0.5*u_2
    for (std::size_t i = 0; i < hanging_dofs.size(); ++i) {
        constraints.addLine(hanging_dofs[i]);
        constraints.addEntry(hanging_dofs[i], parent_dofs_1[i], 0.5);
        constraints.addEntry(hanging_dofs[i], parent_dofs_2[i], 0.5);
    }
}

void makeHangingNodeConstraint2D(
    GlobalIndex hanging_dof,
    std::span<const GlobalIndex> parent_dofs,
    int order,
    AffineConstraints& constraints)
{
    if (order == 1) {
        // Linear interpolation: midpoint
        if (parent_dofs.size() != 2) {
            CONSTRAINT_THROW("P1 edge hanging node requires 2 parent DOFs");
        }
        constraints.addLine(hanging_dof);
        constraints.addEntry(hanging_dof, parent_dofs[0], 0.5);
        constraints.addEntry(hanging_dof, parent_dofs[1], 0.5);
    } else if (order == 2) {
        // Quadratic interpolation
        // For midpoint on quadratic edge with 3 DOFs:
        // u_h = 0.5*u_0 + 0.5*u_2 (vertices), midpoint is interpolated
        if (parent_dofs.size() >= 2) {
            constraints.addLine(hanging_dof);
            constraints.addEntry(hanging_dof, parent_dofs[0], 0.5);
            constraints.addEntry(hanging_dof, parent_dofs[parent_dofs.size()-1], 0.5);
        }
    } else {
        // General case: compute Lagrange interpolation weights
        // Simplified: assume midpoint interpolation
        constraints.addLine(hanging_dof);
        double weight = 1.0 / static_cast<double>(parent_dofs.size());
        for (GlobalIndex parent_dof : parent_dofs) {
            constraints.addEntry(hanging_dof, parent_dof, weight);
        }
    }
}

void makeHangingNodeConstraints3D(
    std::span<const GlobalIndex> hanging_dofs,
    std::span<const std::vector<GlobalIndex>> parent_dofs,
    std::span<const std::vector<double>> weights,
    AffineConstraints& constraints)
{
    if (hanging_dofs.size() != parent_dofs.size() ||
        hanging_dofs.size() != weights.size()) {
        CONSTRAINT_THROW("Hanging DOFs, parent DOFs, and weights must have same size");
    }

    for (std::size_t i = 0; i < hanging_dofs.size(); ++i) {
        const auto& parents = parent_dofs[i];
        const auto& w = weights[i];

        if (parents.size() != w.size()) {
            CONSTRAINT_THROW("Parent DOFs and weights must have same size for each hanging DOF");
        }

        constraints.addLine(hanging_dofs[i]);
        for (std::size_t j = 0; j < parents.size(); ++j) {
            if (std::abs(w[j]) > 1e-15) {
                constraints.addEntry(hanging_dofs[i], parents[j], w[j]);
            }
        }
    }
}

// ============================================================================
// Utility functions
// ============================================================================

std::vector<std::pair<std::size_t, std::size_t>> findMatchingPoints(
    std::span<const std::array<double, 3>> coords_a,
    std::span<const std::array<double, 3>> coords_b,
    const std::function<std::array<double, 3>(std::array<double, 3>)>& transform,
    double tolerance)
{
    std::vector<std::pair<std::size_t, std::size_t>> matches;
    matches.reserve(std::min(coords_a.size(), coords_b.size()));

    double tol_sq = tolerance * tolerance;

    for (std::size_t i = 0; i < coords_a.size(); ++i) {
        std::array<double, 3> point_a = coords_a[i];
        if (transform) {
            point_a = transform(point_a);
        }

        // Find closest point in B
        double best_dist_sq = std::numeric_limits<double>::max();
        std::size_t best_j = coords_b.size();

        for (std::size_t j = 0; j < coords_b.size(); ++j) {
            double dx = point_a[0] - coords_b[j][0];
            double dy = point_a[1] - coords_b[j][1];
            double dz = point_a[2] - coords_b[j][2];
            double dist_sq = dx*dx + dy*dy + dz*dz;

            if (dist_sq < best_dist_sq) {
                best_dist_sq = dist_sq;
                best_j = j;
            }
        }

        if (best_j < coords_b.size() && best_dist_sq < tol_sq) {
            matches.emplace_back(i, best_j);
        }
    }

    return matches;
}

std::vector<double> computeInterpolationWeights(
    std::array<double, 3> reference_point,
    ElementType cell_type,
    int order)
{
    std::vector<double> weights;

    // Simplified implementation for common cases
    double xi = reference_point[0];
    double eta = reference_point[1];
    double zeta = reference_point[2];

    switch (cell_type) {
        case ElementType::Line2:
            // Linear line: 2 nodes at xi = -1, 1
            weights = {0.5 * (1.0 - xi), 0.5 * (1.0 + xi)};
            break;

        case ElementType::Triangle3:
            // Linear triangle: barycentric coordinates
            // Assumes reference is [0,1] x [0,1] with node at (0,0), (1,0), (0,1)
            weights = {1.0 - xi - eta, xi, eta};
            break;

        case ElementType::Quad4:
            // Bilinear quad: 4 nodes at corners
            weights = {
                0.25 * (1.0 - xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 + eta),
                0.25 * (1.0 - xi) * (1.0 + eta)
            };
            break;

        case ElementType::Tetra4:
            // Linear tetrahedron: barycentric coordinates
            weights = {1.0 - xi - eta - zeta, xi, eta, zeta};
            break;

        case ElementType::Hex8:
            // Trilinear hex: 8 nodes at corners
            weights = {
                0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 - zeta),
                0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 - zeta),
                0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 - zeta),
                0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 - zeta),
                0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 + zeta),
                0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 + zeta),
                0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 + zeta),
                0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 + zeta)
            };
            break;

        default:
            // General case: uniform weights (placeholder)
            if (order > 0) {
                int n_nodes = (order + 1) * (order + 1);  // Rough estimate
                weights.resize(static_cast<std::size_t>(n_nodes), 1.0 / static_cast<double>(n_nodes));
            }
            break;
    }

    return weights;
}

std::vector<GlobalIndex> extractComponentDofs(
    std::span<const GlobalIndex> all_dofs,
    int component,
    int num_components)
{
    std::vector<GlobalIndex> result;
    result.reserve(all_dofs.size() / static_cast<std::size_t>(num_components));

    for (std::size_t i = static_cast<std::size_t>(component);
         i < all_dofs.size();
         i += static_cast<std::size_t>(num_components)) {
        result.push_back(all_dofs[i]);
    }

    return result;
}

void mergeConstraints(
    AffineConstraints& target,
    const AffineConstraints& source,
    bool overwrite)
{
    source.forEach([&](const AffineConstraints::ConstraintView& view) {
        if (target.isConstrained(view.slave_dof)) {
            if (!overwrite) {
                CONSTRAINT_THROW_DOF("Constraint conflict during merge", view.slave_dof);
            }
            // Remove existing (will be overwritten)
            // Note: AffineConstraints doesn't support removal, so this requires clear/rebuild
            // For now, just throw
            CONSTRAINT_THROW_DOF("Cannot overwrite existing constraint (not implemented)",
                                 view.slave_dof);
        }

        target.addLine(view.slave_dof);
        for (const auto& entry : view.entries) {
            target.addEntry(view.slave_dof, entry.master_dof, entry.weight);
        }
        target.setInhomogeneity(view.slave_dof, view.inhomogeneity);
    });
}

bool hasConstrainedDofs(
    std::span<const GlobalIndex> dofs,
    const AffineConstraints& constraints)
{
    return constraints.hasConstrainedDofs(dofs);
}

std::vector<GlobalIndex> filterUnconstrainedDofs(
    std::span<const GlobalIndex> dofs,
    const AffineConstraints& constraints)
{
    std::vector<GlobalIndex> result;
    result.reserve(dofs.size());

    for (GlobalIndex dof : dofs) {
        if (!constraints.isConstrained(dof)) {
            result.push_back(dof);
        }
    }

    return result;
}

std::unordered_set<GlobalIndex> getUnconstrainedDofSet(
    GlobalIndex n_dofs,
    const AffineConstraints& constraints)
{
    std::unordered_set<GlobalIndex> result;
    result.reserve(static_cast<std::size_t>(n_dofs));

    for (GlobalIndex dof = 0; dof < n_dofs; ++dof) {
        if (!constraints.isConstrained(dof)) {
            result.insert(dof);
        }
    }

    return result;
}

} // namespace constraints
} // namespace FE
} // namespace svmp
