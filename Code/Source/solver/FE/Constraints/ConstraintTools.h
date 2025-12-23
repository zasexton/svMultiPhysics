/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_CONSTRAINTTOOLS_H
#define SVMP_FE_CONSTRAINTS_CONSTRAINTTOOLS_H

/**
 * @file ConstraintTools.h
 * @brief High-level constraint generation utilities
 *
 * ConstraintTools provides convenience functions for creating common constraint
 * types from mesh markers, DOF handlers, and boundary information. It bridges
 * the gap between high-level constraint specification and the low-level
 * AffineConstraints API.
 *
 * Key functionality:
 * - makeDirichletConstraints(): Create Dirichlet BCs from boundary IDs
 * - makeHangingNodeConstraints(): Create constraints for adaptive meshes
 * - makePeriodicConstraints(): Create periodic boundary constraints
 * - extractComponentDofs(): Get DOFs for a specific vector component
 *
 * Module boundary:
 * - This module OWNS high-level constraint generation logic
 * - This module QUERIES DOF information from DofHandler/DofMap
 * - This module does NOT OWN mesh data (receives via parameters)
 *
 * @see AffineConstraints for low-level constraint storage
 * @see DirichletBC, PeriodicBC for constraint objects
 */

#include "AffineConstraints.h"
#include "Constraint.h"
#include "Core/Types.h"

#include <vector>
#include <span>
#include <functional>
#include <optional>
#include <array>
#include <unordered_set>

namespace svmp {
namespace FE {

// Forward declarations
namespace dofs {
    class DofHandler;
    class DofMap;
    class EntityDofMap;
}

namespace constraints {

/**
 * @brief Boundary specification for constraint generation
 *
 * Describes a boundary on which constraints are to be applied.
 */
struct BoundarySpec {
    int boundary_id{-1};                          ///< Mesh boundary marker ID
    std::string boundary_name;                    ///< Optional boundary name
    std::vector<GlobalIndex> boundary_dofs;       ///< DOFs on this boundary
    std::vector<std::array<double, 3>> dof_coords; ///< Coordinates of DOFs (optional)
};

/**
 * @brief Options for Dirichlet constraint generation
 */
struct DirichletConstraintOptions {
    ComponentMask component_mask{};               ///< Which components to constrain
    bool evaluate_at_dof_coords{true};            ///< Evaluate function at DOF coordinates
};

/**
 * @brief Options for periodic constraint generation
 */
struct PeriodicConstraintOptions {
    ComponentMask component_mask{};               ///< Which components to make periodic
    double tolerance{1e-10};                      ///< Tolerance for coordinate matching
    bool flip_sign{false};                        ///< For anti-periodic BCs
};

/**
 * @brief Options for hanging node constraint generation
 */
struct HangingNodeOptions {
    int polynomial_order{1};                      ///< Element polynomial order
    double tolerance{1e-12};                      ///< Tolerance for weight computation
};

/**
 * @brief Result of boundary DOF extraction
 */
struct BoundaryDofResult {
    std::vector<GlobalIndex> dofs;                ///< DOF indices on boundary
    std::vector<std::array<double, 3>> coords;    ///< Coordinates (if available)
    std::vector<int> components;                  ///< Component indices (for vector fields)
};

// ============================================================================
// Free functions for constraint generation
// ============================================================================

/**
 * @brief Create Dirichlet constraints with constant value
 *
 * @param boundary_dofs DOFs on the boundary
 * @param value Constant value to prescribe
 * @param constraints Output constraint container
 * @param options Constraint options
 */
void makeDirichletConstraints(
    std::span<const GlobalIndex> boundary_dofs,
    double value,
    AffineConstraints& constraints,
    const DirichletConstraintOptions& options = {});

/**
 * @brief Create Dirichlet constraints with function-based values
 *
 * @param boundary_dofs DOFs on the boundary
 * @param dof_coords Coordinates of each DOF
 * @param value_func Function returning value at coordinates
 * @param constraints Output constraint container
 * @param options Constraint options
 */
void makeDirichletConstraints(
    std::span<const GlobalIndex> boundary_dofs,
    std::span<const std::array<double, 3>> dof_coords,
    const std::function<double(double, double, double)>& value_func,
    AffineConstraints& constraints,
    const DirichletConstraintOptions& options = {});

/**
 * @brief Create Dirichlet constraints with vector-valued function
 *
 * @param boundary_dofs DOFs on the boundary (interleaved by component)
 * @param dof_coords Coordinates of each DOF
 * @param value_func Function returning vector value at coordinates
 * @param num_components Number of field components
 * @param constraints Output constraint container
 * @param options Constraint options
 */
void makeDirichletConstraintsVector(
    std::span<const GlobalIndex> boundary_dofs,
    std::span<const std::array<double, 3>> dof_coords,
    const std::function<std::vector<double>(double, double, double)>& value_func,
    int num_components,
    AffineConstraints& constraints,
    const DirichletConstraintOptions& options = {});

/**
 * @brief Create periodic constraints between two boundaries
 *
 * Establishes constraints u_slave = u_master for matching DOFs on
 * two periodic boundaries.
 *
 * @param slave_dofs DOFs on the slave boundary
 * @param slave_coords Coordinates of slave DOFs
 * @param master_dofs DOFs on the master boundary
 * @param master_coords Coordinates of master DOFs
 * @param transform Transform from slave to master coordinates (optional)
 * @param constraints Output constraint container
 * @param options Constraint options
 */
void makePeriodicConstraints(
    std::span<const GlobalIndex> slave_dofs,
    std::span<const std::array<double, 3>> slave_coords,
    std::span<const GlobalIndex> master_dofs,
    std::span<const std::array<double, 3>> master_coords,
    const std::function<std::array<double, 3>(std::array<double, 3>)>& transform,
    AffineConstraints& constraints,
    const PeriodicConstraintOptions& options = {});

/**
 * @brief Create periodic constraints with simple translation
 *
 * @param slave_dofs DOFs on the slave boundary
 * @param slave_coords Coordinates of slave DOFs
 * @param master_dofs DOFs on the master boundary
 * @param master_coords Coordinates of master DOFs
 * @param translation Translation vector from slave to master
 * @param constraints Output constraint container
 * @param options Constraint options
 */
void makePeriodicConstraintsTranslation(
    std::span<const GlobalIndex> slave_dofs,
    std::span<const std::array<double, 3>> slave_coords,
    std::span<const GlobalIndex> master_dofs,
    std::span<const std::array<double, 3>> master_coords,
    std::array<double, 3> translation,
    AffineConstraints& constraints,
    const PeriodicConstraintOptions& options = {});

/**
 * @brief Create hanging node constraints for 1D edges
 *
 * For a hanging node at the midpoint of an edge, creates:
 *   u_hanging = 0.5 * u_parent1 + 0.5 * u_parent2
 *
 * @param hanging_dofs Hanging node DOF indices
 * @param parent_dofs_1 First parent DOF indices (same order as hanging_dofs)
 * @param parent_dofs_2 Second parent DOF indices
 * @param constraints Output constraint container
 * @param options Constraint options
 */
void makeHangingNodeConstraints1D(
    std::span<const GlobalIndex> hanging_dofs,
    std::span<const GlobalIndex> parent_dofs_1,
    std::span<const GlobalIndex> parent_dofs_2,
    AffineConstraints& constraints,
    const HangingNodeOptions& options = {});

/**
 * @brief Create hanging node constraints for 2D edges (midpoint refinement)
 *
 * @param hanging_dof Hanging node DOF
 * @param parent_dofs Parent DOFs (vertices of the unrefined edge)
 * @param order Polynomial order (determines interpolation weights)
 * @param constraints Output constraint container
 */
void makeHangingNodeConstraint2D(
    GlobalIndex hanging_dof,
    std::span<const GlobalIndex> parent_dofs,
    int order,
    AffineConstraints& constraints);

/**
 * @brief Create hanging node constraints for 3D faces
 *
 * For hanging nodes on refined faces, creates interpolation constraints
 * based on the unrefined face shape functions.
 *
 * @param hanging_dofs Hanging node DOF indices
 * @param parent_dofs Parent DOF indices
 * @param weights Interpolation weights (from parent shape functions)
 * @param constraints Output constraint container
 */
void makeHangingNodeConstraints3D(
    std::span<const GlobalIndex> hanging_dofs,
    std::span<const std::vector<GlobalIndex>> parent_dofs,
    std::span<const std::vector<double>> weights,
    AffineConstraints& constraints);

// ============================================================================
// Utility functions
// ============================================================================

/**
 * @brief Find matching DOF pairs between two point sets
 *
 * Matches points from set A to points in set B based on coordinate proximity
 * after applying an optional transformation.
 *
 * @param coords_a Coordinates of first set
 * @param coords_b Coordinates of second set
 * @param transform Optional transform applied to coords_a before matching
 * @param tolerance Matching tolerance
 * @return Vector of (index_a, index_b) pairs
 */
[[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> findMatchingPoints(
    std::span<const std::array<double, 3>> coords_a,
    std::span<const std::array<double, 3>> coords_b,
    const std::function<std::array<double, 3>(std::array<double, 3>)>& transform = nullptr,
    double tolerance = 1e-10);

/**
 * @brief Compute interpolation weights for a point within a cell
 *
 * Given reference coordinates of a point within a cell, computes the
 * shape function values (interpolation weights) at that point.
 *
 * @param reference_point Point in reference coordinates
 * @param cell_type Cell topology type
 * @param order Polynomial order
 * @return Vector of weights (one per cell DOF)
 */
[[nodiscard]] std::vector<double> computeInterpolationWeights(
    std::array<double, 3> reference_point,
    ElementType cell_type,
    int order);

/**
 * @brief Get DOFs on a specific component of a vector field
 *
 * @param all_dofs All DOFs (interleaved by component)
 * @param component Component index (0, 1, 2, ...)
 * @param num_components Total number of components
 * @return DOFs for the specified component
 */
[[nodiscard]] std::vector<GlobalIndex> extractComponentDofs(
    std::span<const GlobalIndex> all_dofs,
    int component,
    int num_components);

/**
 * @brief Merge two constraint sets
 *
 * Combines constraints from two AffineConstraints objects into one.
 * Handles conflicts according to the specified mode.
 *
 * @param target Target constraint container
 * @param source Source constraints to merge
 * @param overwrite If true, source overwrites conflicts; else throws
 */
void mergeConstraints(
    AffineConstraints& target,
    const AffineConstraints& source,
    bool overwrite = false);

/**
 * @brief Check if a DOF set contains any constrained DOFs
 *
 * @param dofs DOF indices to check
 * @param constraints Constraint container to check against
 * @return true if any DOF is constrained
 */
[[nodiscard]] bool hasConstrainedDofs(
    std::span<const GlobalIndex> dofs,
    const AffineConstraints& constraints);

/**
 * @brief Filter out constrained DOFs from a set
 *
 * @param dofs Input DOF set
 * @param constraints Constraint container
 * @return DOFs that are not constrained
 */
[[nodiscard]] std::vector<GlobalIndex> filterUnconstrainedDofs(
    std::span<const GlobalIndex> dofs,
    const AffineConstraints& constraints);

/**
 * @brief Get the unconstrained DOF index set
 *
 * @param n_dofs Total number of DOFs
 * @param constraints Constraint container
 * @return Set of unconstrained DOF indices
 */
[[nodiscard]] std::unordered_set<GlobalIndex> getUnconstrainedDofSet(
    GlobalIndex n_dofs,
    const AffineConstraints& constraints);

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_CONSTRAINTTOOLS_H
