/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "HangingNodeConstraint.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>
#include <sstream>

namespace svmp {
namespace FE {
namespace constraints {

// ============================================================================
// Construction
// ============================================================================

HangingNodeConstraint::HangingNodeConstraint() = default;

HangingNodeConstraint::HangingNodeConstraint(std::vector<HangingNodeData> hanging_nodes,
                                              const HangingNodeConstraintOptions& options)
    : hanging_nodes_(std::move(hanging_nodes)), options_(options) {}

HangingNodeConstraint::HangingNodeConstraint(std::vector<GlobalIndex> hanging_dofs,
                                              std::vector<GlobalIndex> parent_dofs_1,
                                              std::vector<GlobalIndex> parent_dofs_2)
{
    if (hanging_dofs.size() != parent_dofs_1.size() ||
        hanging_dofs.size() != parent_dofs_2.size()) {
        CONSTRAINT_THROW("Hanging and parent DOF arrays must have same size");
    }

    hanging_nodes_.reserve(hanging_dofs.size());
    for (std::size_t i = 0; i < hanging_dofs.size(); ++i) {
        HangingNodeData node;
        node.hanging_dof = hanging_dofs[i];
        node.parent_dofs = {parent_dofs_1[i], parent_dofs_2[i]};
        node.weights = {0.5, 0.5};
        node.dimension = 1;
        hanging_nodes_.push_back(std::move(node));
    }
}

// ============================================================================
// Copy/Move
// ============================================================================

HangingNodeConstraint::HangingNodeConstraint(const HangingNodeConstraint& other) = default;
HangingNodeConstraint::HangingNodeConstraint(HangingNodeConstraint&& other) noexcept = default;
HangingNodeConstraint& HangingNodeConstraint::operator=(const HangingNodeConstraint& other) = default;
HangingNodeConstraint& HangingNodeConstraint::operator=(HangingNodeConstraint&& other) noexcept = default;

// ============================================================================
// Constraint interface
// ============================================================================

void HangingNodeConstraint::apply(AffineConstraints& constraints) const {
    for (const auto& node : hanging_nodes_) {
        if (!node.isValid()) continue;

        constraints.addLine(node.hanging_dof);
        for (std::size_t i = 0; i < node.parent_dofs.size(); ++i) {
            if (std::abs(node.weights[i]) > options_.weight_tolerance) {
                constraints.addEntry(node.hanging_dof, node.parent_dofs[i], node.weights[i]);
            }
        }
    }
}

ConstraintInfo HangingNodeConstraint::getInfo() const {
    ConstraintInfo info;
    info.name = "HangingNodeConstraint";
    info.type = ConstraintType::HangingNode;
    info.num_constrained_dofs = hanging_nodes_.size();
    info.is_time_dependent = false;
    info.is_homogeneous = true;  // Hanging nodes have no inhomogeneity
    return info;
}

// ============================================================================
// Modification
// ============================================================================

void HangingNodeConstraint::addHangingNode(const HangingNodeData& node) {
    hanging_nodes_.push_back(node);
}

void HangingNodeConstraint::addHangingNode1D(GlobalIndex hanging_dof,
                                              GlobalIndex parent_dof_1,
                                              GlobalIndex parent_dof_2) {
    HangingNodeData node;
    node.hanging_dof = hanging_dof;
    node.parent_dofs = {parent_dof_1, parent_dof_2};
    node.weights = {0.5, 0.5};
    node.dimension = 1;
    hanging_nodes_.push_back(std::move(node));
}

void HangingNodeConstraint::addHangingNodes(
    std::span<const GlobalIndex> hanging_dofs,
    std::span<const std::vector<GlobalIndex>> parent_dofs,
    std::span<const std::vector<double>> weights)
{
    if (hanging_dofs.size() != parent_dofs.size() ||
        hanging_dofs.size() != weights.size()) {
        CONSTRAINT_THROW("All arrays must have same size");
    }

    for (std::size_t i = 0; i < hanging_dofs.size(); ++i) {
        HangingNodeData node;
        node.hanging_dof = hanging_dofs[i];
        node.parent_dofs = parent_dofs[i];
        node.weights = weights[i];
        hanging_nodes_.push_back(std::move(node));
    }
}

void HangingNodeConstraint::clear() {
    hanging_nodes_.clear();
}

// ============================================================================
// Validation
// ============================================================================

std::string HangingNodeConstraint::validate() const {
    std::unordered_set<GlobalIndex> hanging_set;

    for (const auto& node : hanging_nodes_) {
        // Check basic validity
        if (!node.isValid()) {
            return "Invalid hanging node data for DOF " + std::to_string(node.hanging_dof);
        }

        // Check for duplicate hanging DOFs
        if (hanging_set.find(node.hanging_dof) != hanging_set.end()) {
            return "Duplicate hanging DOF: " + std::to_string(node.hanging_dof);
        }
        hanging_set.insert(node.hanging_dof);

        // Check parent DOFs are not hanging
        for (GlobalIndex parent : node.parent_dofs) {
            if (hanging_set.find(parent) != hanging_set.end()) {
                return "Parent DOF " + std::to_string(parent) +
                       " is also a hanging DOF (chain detected)";
            }
        }

        // Check weights sum to 1 (partition of unity)
        if (options_.validate_weights) {
            double sum = std::accumulate(node.weights.begin(), node.weights.end(), 0.0);
            if (std::abs(sum - 1.0) > 1e-10) {
                std::ostringstream oss;
                oss << "Weights for hanging DOF " << node.hanging_dof
                    << " sum to " << sum << " (expected 1.0)";
                return oss.str();
            }
        }
    }

    return "";  // Valid
}

// ============================================================================
// Factory methods
// ============================================================================

HangingNodeConstraint HangingNodeConstraint::forP1Edges(
    std::span<const GlobalIndex> hanging_dofs,
    std::span<const std::array<GlobalIndex, 2>> edge_endpoints)
{
    if (hanging_dofs.size() != edge_endpoints.size()) {
        CONSTRAINT_THROW("Hanging DOFs and edge endpoints must have same size");
    }

    std::vector<HangingNodeData> nodes;
    nodes.reserve(hanging_dofs.size());

    for (std::size_t i = 0; i < hanging_dofs.size(); ++i) {
        HangingNodeData node;
        node.hanging_dof = hanging_dofs[i];
        node.parent_dofs = {edge_endpoints[i][0], edge_endpoints[i][1]};
        node.weights = {0.5, 0.5};
        node.dimension = 1;
        nodes.push_back(std::move(node));
    }

    return HangingNodeConstraint(std::move(nodes));
}

HangingNodeConstraint HangingNodeConstraint::forP2Edges(
    std::span<const GlobalIndex> hanging_dofs,
    std::span<const std::array<GlobalIndex, 3>> edge_dofs)
{
    if (hanging_dofs.size() != edge_dofs.size()) {
        CONSTRAINT_THROW("Hanging DOFs and edge DOFs must have same size");
    }

    std::vector<HangingNodeData> nodes;
    nodes.reserve(hanging_dofs.size());

    for (std::size_t i = 0; i < hanging_dofs.size(); ++i) {
        HangingNodeData node;
        node.hanging_dof = hanging_dofs[i];
        // For P2 edge midpoint: only vertex contributions (midpoint node cancels)
        node.parent_dofs = {edge_dofs[i][0], edge_dofs[i][2]};  // Vertices only
        node.weights = {0.5, 0.5};
        node.dimension = 1;
        nodes.push_back(std::move(node));
    }

    return HangingNodeConstraint(std::move(nodes));
}

// ============================================================================
// Weight computation utilities
// ============================================================================

std::vector<double> computeP2EdgeWeights(double parametric_coord) {
    // P2 Lagrange shape functions on edge [0, 1]
    // N0 = (1-t)(1-2t), N1 = 4t(1-t), N2 = t(2t-1)
    double t = parametric_coord;
    return {
        (1.0 - t) * (1.0 - 2.0 * t),
        4.0 * t * (1.0 - t),
        t * (2.0 * t - 1.0)
    };
}

std::vector<double> computeQ1FaceWeights(double xi, double eta) {
    // Bilinear shape functions on [-1,1]^2
    return {
        0.25 * (1.0 - xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 + eta),
        0.25 * (1.0 - xi) * (1.0 + eta)
    };
}

std::vector<double> computeQ1VolumeWeights(double xi, double eta, double zeta) {
    // Trilinear shape functions on [-1,1]^3
    return {
        0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 - zeta),
        0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 - zeta),
        0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 - zeta),
        0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 - zeta),
        0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 + zeta),
        0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 + zeta),
        0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 + zeta),
        0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 + zeta)
    };
}

} // namespace constraints
} // namespace FE
} // namespace svmp
