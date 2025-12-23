/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_HANGINGNODECONSTRAINT_H
#define SVMP_FE_CONSTRAINTS_HANGINGNODECONSTRAINT_H

/**
 * @file HangingNodeConstraint.h
 * @brief Constraints from mesh adaptivity (hanging nodes)
 *
 * HangingNodeConstraint handles the algebraic constraints that arise from
 * local mesh refinement, where new nodes (hanging nodes) are created at
 * the interface between refined and unrefined elements.
 *
 * For continuity across the interface, hanging node DOFs must be expressed
 * as interpolations of the parent element DOFs:
 *
 *   u_hanging = sum_i N_i(xi_h) * u_parent_i
 *
 * where N_i are the parent element shape functions evaluated at the hanging
 * node's reference coordinate xi_h.
 *
 * Features:
 * - 1D edge hanging nodes (midpoint interpolation)
 * - 2D face hanging nodes
 * - Support for various element types and polynomial orders
 * - Automatic weight computation from shape functions
 *
 * This is an ALGEBRAIC constraint that enforces continuity.
 *
 * @see Mesh/Adaptivity for the refinement logic that creates hanging nodes
 */

#include "Constraint.h"
#include "AffineConstraints.h"
#include "Core/Types.h"

#include <vector>
#include <span>
#include <memory>
#include <array>

namespace svmp {
namespace FE {
namespace constraints {

/**
 * @brief A single hanging node with its parent DOFs and interpolation weights
 */
struct HangingNodeData {
    GlobalIndex hanging_dof;                  ///< The hanging node DOF index
    std::vector<GlobalIndex> parent_dofs;     ///< Parent DOFs (on unrefined element)
    std::vector<double> weights;              ///< Interpolation weights
    int dimension{0};                         ///< Topological dimension (1=edge, 2=face)

    /**
     * @brief Check if this is a valid hanging node constraint
     */
    [[nodiscard]] bool isValid() const {
        return hanging_dof >= 0 &&
               !parent_dofs.empty() &&
               parent_dofs.size() == weights.size();
    }
};

/**
 * @brief Options for hanging node constraint generation
 */
struct HangingNodeConstraintOptions {
    int polynomial_order{1};                  ///< Element polynomial order
    double weight_tolerance{1e-15};           ///< Tolerance for zero weights
    bool validate_weights{true};              ///< Check weights sum to 1
};

/**
 * @brief Hanging node constraint from mesh adaptivity
 *
 * HangingNodeConstraint manages constraints for DOFs at hanging nodes created
 * during local mesh refinement. These constraints ensure solution continuity
 * across refinement interfaces.
 *
 * Usage:
 * @code
 *   // After mesh refinement, collect hanging nodes
 *   std::vector<HangingNodeData> hanging_nodes;
 *   for (auto& node : mesh.getHangingNodes()) {
 *       HangingNodeData data;
 *       data.hanging_dof = dof_handler.getDof(node.vertex_id);
 *       data.parent_dofs = getParentDofs(node);
 *       data.weights = computeInterpolationWeights(node);
 *       hanging_nodes.push_back(data);
 *   }
 *
 *   // Create constraint
 *   HangingNodeConstraint constraint(hanging_nodes);
 *
 *   // Apply to constraint manager
 *   AffineConstraints constraints;
 *   constraint.apply(constraints);
 *   constraints.close();
 * @endcode
 */
class HangingNodeConstraint : public Constraint {
public:
    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Default constructor (empty constraint set)
     */
    HangingNodeConstraint();

    /**
     * @brief Construct with hanging node data
     *
     * @param hanging_nodes Vector of hanging node specifications
     * @param options Constraint options
     */
    explicit HangingNodeConstraint(std::vector<HangingNodeData> hanging_nodes,
                                    const HangingNodeConstraintOptions& options = {});

    /**
     * @brief Construct for simple 1D hanging nodes (midpoint interpolation)
     *
     * @param hanging_dofs Hanging node DOF indices
     * @param parent_dofs_1 First parent DOF indices
     * @param parent_dofs_2 Second parent DOF indices
     */
    HangingNodeConstraint(std::vector<GlobalIndex> hanging_dofs,
                          std::vector<GlobalIndex> parent_dofs_1,
                          std::vector<GlobalIndex> parent_dofs_2);

    /**
     * @brief Destructor
     */
    ~HangingNodeConstraint() override = default;

    /**
     * @brief Copy constructor
     */
    HangingNodeConstraint(const HangingNodeConstraint& other);

    /**
     * @brief Move constructor
     */
    HangingNodeConstraint(HangingNodeConstraint&& other) noexcept;

    /**
     * @brief Copy assignment
     */
    HangingNodeConstraint& operator=(const HangingNodeConstraint& other);

    /**
     * @brief Move assignment
     */
    HangingNodeConstraint& operator=(HangingNodeConstraint&& other) noexcept;

    // =========================================================================
    // Constraint interface
    // =========================================================================

    /**
     * @brief Apply hanging node constraints to AffineConstraints
     */
    void apply(AffineConstraints& constraints) const override;

    /**
     * @brief Get constraint type
     */
    [[nodiscard]] ConstraintType getType() const noexcept override {
        return ConstraintType::HangingNode;
    }

    /**
     * @brief Get constraint information
     */
    [[nodiscard]] ConstraintInfo getInfo() const override;

    /**
     * @brief Clone this constraint
     */
    [[nodiscard]] std::unique_ptr<Constraint> clone() const override {
        return std::make_unique<HangingNodeConstraint>(*this);
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /**
     * @brief Get hanging node data
     */
    [[nodiscard]] const std::vector<HangingNodeData>& getHangingNodes() const noexcept {
        return hanging_nodes_;
    }

    /**
     * @brief Get number of hanging nodes
     */
    [[nodiscard]] std::size_t numHangingNodes() const noexcept {
        return hanging_nodes_.size();
    }

    /**
     * @brief Get options
     */
    [[nodiscard]] const HangingNodeConstraintOptions& getOptions() const noexcept {
        return options_;
    }

    // =========================================================================
    // Modification
    // =========================================================================

    /**
     * @brief Add a hanging node constraint
     */
    void addHangingNode(const HangingNodeData& node);

    /**
     * @brief Add a simple 1D hanging node (midpoint)
     */
    void addHangingNode1D(GlobalIndex hanging_dof,
                          GlobalIndex parent_dof_1,
                          GlobalIndex parent_dof_2);

    /**
     * @brief Add hanging nodes from raw data
     */
    void addHangingNodes(std::span<const GlobalIndex> hanging_dofs,
                         std::span<const std::vector<GlobalIndex>> parent_dofs,
                         std::span<const std::vector<double>> weights);

    /**
     * @brief Clear all hanging nodes
     */
    void clear();

    // =========================================================================
    // Validation
    // =========================================================================

    /**
     * @brief Validate hanging node constraints
     *
     * Checks:
     * - All weights sum to 1 (partition of unity)
     * - No duplicate hanging DOFs
     * - Parent DOFs exist and are not hanging
     *
     * @return Error message, or empty string if valid
     */
    [[nodiscard]] std::string validate() const;

    // =========================================================================
    // Static factory methods for common cases
    // =========================================================================

    /**
     * @brief Create constraints for P1 edge midpoints
     *
     * For linear elements, midpoint hanging nodes have weights (0.5, 0.5).
     *
     * @param hanging_dofs Hanging node DOFs
     * @param edge_endpoints For each hanging node: {parent_dof_1, parent_dof_2}
     * @return HangingNodeConstraint
     */
    static HangingNodeConstraint forP1Edges(
        std::span<const GlobalIndex> hanging_dofs,
        std::span<const std::array<GlobalIndex, 2>> edge_endpoints);

    /**
     * @brief Create constraints for P2 edge midpoints
     *
     * For quadratic elements, midpoint on a quadratic edge uses vertex DOFs.
     */
    static HangingNodeConstraint forP2Edges(
        std::span<const GlobalIndex> hanging_dofs,
        std::span<const std::array<GlobalIndex, 3>> edge_dofs);

private:
    std::vector<HangingNodeData> hanging_nodes_;
    HangingNodeConstraintOptions options_;
};

// ============================================================================
// Utility functions for hanging node weight computation
// ============================================================================

/**
 * @brief Compute P1 (linear) interpolation weights for edge midpoint
 *
 * @return {0.5, 0.5}
 */
[[nodiscard]] inline std::vector<double> computeP1EdgeWeights() {
    return {0.5, 0.5};
}

/**
 * @brief Compute P2 (quadratic) interpolation weights for edge 1/4 point
 *
 * @param parametric_coord Parametric coordinate along edge [0, 1]
 * @return Weights for 3 DOFs on quadratic edge
 */
[[nodiscard]] std::vector<double> computeP2EdgeWeights(double parametric_coord);

/**
 * @brief Compute bilinear interpolation weights for face point
 *
 * @param xi, eta Parametric coordinates on face [-1, 1] x [-1, 1]
 * @return Weights for 4 corner DOFs
 */
[[nodiscard]] std::vector<double> computeQ1FaceWeights(double xi, double eta);

/**
 * @brief Compute trilinear interpolation weights
 *
 * @param xi, eta, zeta Parametric coordinates [-1, 1]^3
 * @return Weights for 8 corner DOFs
 */
[[nodiscard]] std::vector<double> computeQ1VolumeWeights(double xi, double eta, double zeta);

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_HANGINGNODECONSTRAINT_H
