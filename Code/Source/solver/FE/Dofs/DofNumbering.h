/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_DOFS_DOFNUMBERING_H
#define SVMP_FE_DOFS_DOFNUMBERING_H

/**
 * @file DofNumbering.h
 * @brief DOF numbering strategies for finite element systems
 *
 * This module provides various strategies for assigning global DOF indices
 * to mesh entities. The numbering strategy significantly impacts:
 *  - Matrix bandwidth and fill-in
 *  - Cache efficiency during assembly
 *  - Parallel load balance
 *  - Convergence of iterative solvers
 *
 * Supported strategies:
 *  - SequentialNumbering: Simple, predictable numbering
 *  - InterleavedNumbering: Components interleaved (u0,v0,w0,u1,v1,w1,...)
 *  - BlockNumbering: Components grouped (u0,u1,...,v0,v1,...,w0,w1,...)
 *  - HierarchicalNumbering: Vertices, then edges, then faces, then cells
 *  - CuthillMcKeeNumbering: Bandwidth minimization
 *  - NestedDissectionNumbering: Fill-in minimization
 */

#include "DofMap.h"
#include "Core/Types.h"
#include "Core/FEException.h"

#include <vector>
#include <span>
#include <memory>
#include <functional>
#include <optional>

namespace svmp {
namespace FE {
namespace dofs {

// Forward declarations
class DofGraph;

/**
 * @brief Abstract base class for DOF numbering strategies
 */
class NumberingStrategy {
public:
    virtual ~NumberingStrategy() = default;

    /**
     * @brief Compute DOF numbering
     *
     * @param n_dofs Total number of DOFs to number
     * @param adjacency DOF adjacency graph (CSR format offsets)
     * @param adj_indices DOF adjacency graph (CSR format indices)
     * @return Permutation vector: new_dof = perm[old_dof]
     */
    [[nodiscard]] virtual std::vector<GlobalIndex> computeNumbering(
        GlobalIndex n_dofs,
        std::span<const GlobalIndex> adjacency,
        std::span<const GlobalIndex> adj_indices) const = 0;

    /**
     * @brief Get human-readable name of the strategy
     */
    [[nodiscard]] virtual std::string name() const = 0;

    /**
     * @brief Check if strategy requires adjacency information
     */
    [[nodiscard]] virtual bool requiresAdjacency() const noexcept { return false; }
};

/**
 * @brief Sequential numbering (identity permutation)
 *
 * DOFs are numbered in the order they are encountered.
 * Fast to compute, but does not optimize bandwidth.
 */
class SequentialNumbering : public NumberingStrategy {
public:
    [[nodiscard]] std::vector<GlobalIndex> computeNumbering(
        GlobalIndex n_dofs,
        std::span<const GlobalIndex> adjacency,
        std::span<const GlobalIndex> adj_indices) const override;

    [[nodiscard]] std::string name() const override { return "Sequential"; }
    [[nodiscard]] bool requiresAdjacency() const noexcept override { return false; }
};

/**
 * @brief Interleaved numbering for vector-valued fields
 *
 * Components are interleaved: (u0,v0,w0), (u1,v1,w1), ...
 * Good for point-block solvers and cache locality.
 */
class InterleavedNumbering : public NumberingStrategy {
public:
    /**
     * @brief Construct interleaved numbering
     * @param n_components Number of components per node
     */
    explicit InterleavedNumbering(LocalIndex n_components = 3);

    [[nodiscard]] std::vector<GlobalIndex> computeNumbering(
        GlobalIndex n_dofs,
        std::span<const GlobalIndex> adjacency,
        std::span<const GlobalIndex> adj_indices) const override;

    [[nodiscard]] std::string name() const override { return "Interleaved"; }
    [[nodiscard]] bool requiresAdjacency() const noexcept override { return false; }

    [[nodiscard]] LocalIndex numComponents() const noexcept { return n_components_; }

private:
    LocalIndex n_components_;
};

/**
 * @brief Block numbering for vector-valued fields
 *
 * Components are grouped: (u0,u1,...), (v0,v1,...), (w0,w1,...)
 * Good for segregated solvers and block preconditioners.
 */
class BlockNumbering : public NumberingStrategy {
public:
    /**
     * @brief Construct block numbering
     * @param block_sizes Number of DOFs in each block
     */
    explicit BlockNumbering(std::vector<GlobalIndex> block_sizes);

    [[nodiscard]] std::vector<GlobalIndex> computeNumbering(
        GlobalIndex n_dofs,
        std::span<const GlobalIndex> adjacency,
        std::span<const GlobalIndex> adj_indices) const override;

    [[nodiscard]] std::string name() const override { return "Block"; }
    [[nodiscard]] bool requiresAdjacency() const noexcept override { return false; }

    [[nodiscard]] std::span<const GlobalIndex> blockSizes() const noexcept {
        return block_sizes_;
    }

private:
    std::vector<GlobalIndex> block_sizes_;
};

/**
 * @brief Hierarchical numbering: vertices, edges, faces, cells
 *
 * DOFs are numbered by entity dimension:
 * 1. Vertex DOFs
 * 2. Edge DOFs
 * 3. Face DOFs
 * 4. Cell interior DOFs
 *
 * Good for static condensation and hierarchical preconditioners.
 */
class HierarchicalNumbering : public NumberingStrategy {
public:
    /**
     * @brief Construct hierarchical numbering
     * @param n_vertex_dofs Total vertex DOFs
     * @param n_edge_dofs Total edge DOFs
     * @param n_face_dofs Total face DOFs
     * @param n_cell_dofs Total cell interior DOFs
     */
    HierarchicalNumbering(GlobalIndex n_vertex_dofs, GlobalIndex n_edge_dofs,
                          GlobalIndex n_face_dofs, GlobalIndex n_cell_dofs);

    [[nodiscard]] std::vector<GlobalIndex> computeNumbering(
        GlobalIndex n_dofs,
        std::span<const GlobalIndex> adjacency,
        std::span<const GlobalIndex> adj_indices) const override;

    [[nodiscard]] std::string name() const override { return "Hierarchical"; }
    [[nodiscard]] bool requiresAdjacency() const noexcept override { return false; }

private:
    GlobalIndex n_vertex_dofs_;
    GlobalIndex n_edge_dofs_;
    GlobalIndex n_face_dofs_;
    GlobalIndex n_cell_dofs_;
};

/**
 * @brief Cuthill-McKee numbering for bandwidth reduction
 *
 * Reduces matrix bandwidth using BFS-based reordering.
 * The reverse Cuthill-McKee (RCM) variant often performs better.
 *
 * Reference: Cuthill & McKee (1969) "Reducing the bandwidth of
 *           sparse symmetric matrices"
 */
class CuthillMcKeeNumbering : public NumberingStrategy {
public:
    /**
     * @brief Construct Cuthill-McKee numbering
     * @param reverse Use reverse Cuthill-McKee (usually better)
     * @param start_vertex Starting vertex (0 = auto-select peripheral)
     */
    explicit CuthillMcKeeNumbering(bool reverse = true,
                                    std::optional<GlobalIndex> start_vertex = std::nullopt);

    [[nodiscard]] std::vector<GlobalIndex> computeNumbering(
        GlobalIndex n_dofs,
        std::span<const GlobalIndex> adjacency,
        std::span<const GlobalIndex> adj_indices) const override;

    [[nodiscard]] std::string name() const override {
        return reverse_ ? "Reverse Cuthill-McKee" : "Cuthill-McKee";
    }
    [[nodiscard]] bool requiresAdjacency() const noexcept override { return true; }

    [[nodiscard]] bool isReverse() const noexcept { return reverse_; }

private:
    bool reverse_;
    std::optional<GlobalIndex> start_vertex_;

    // Find a peripheral vertex (pseudo-peripheral node)
    GlobalIndex findPeripheralVertex(
        GlobalIndex n_dofs,
        std::span<const GlobalIndex> adjacency,
        std::span<const GlobalIndex> adj_indices) const;
};

/**
 * @brief Nested dissection numbering for fill-in minimization
 *
 * Recursively bisects the graph and numbers separators last.
 * Excellent for sparse direct solvers (Cholesky, LU).
 *
 * Reference: George (1973) "Nested dissection of a regular
 *           finite element mesh"
 */
class NestedDissectionNumbering : public NumberingStrategy {
public:
    /**
     * @brief Construct nested dissection numbering
     * @param min_partition_size Stop recursion when partition < this size
     */
    explicit NestedDissectionNumbering(GlobalIndex min_partition_size = 64);

    [[nodiscard]] std::vector<GlobalIndex> computeNumbering(
        GlobalIndex n_dofs,
        std::span<const GlobalIndex> adjacency,
        std::span<const GlobalIndex> adj_indices) const override;

    [[nodiscard]] std::string name() const override { return "Nested Dissection"; }
    [[nodiscard]] bool requiresAdjacency() const noexcept override { return true; }

private:
    GlobalIndex min_partition_size_;

    // Recursive dissection helper
    void dissect(
        std::span<GlobalIndex> partition,
        std::span<const GlobalIndex> adjacency,
        std::span<const GlobalIndex> adj_indices,
        GlobalIndex& next_number,
        std::vector<GlobalIndex>& numbering) const;
};

/**
 * @brief Space-filling curve numbering for cache locality
 *
 * Orders DOFs along a space-filling curve (Hilbert, Morton/Z-order)
 * to improve cache locality for assembly.
 */
class SpaceFillingCurveNumbering : public NumberingStrategy {
public:
    enum class CurveType {
        Morton,     ///< Z-order curve (simpler, faster)
        Hilbert     ///< Hilbert curve (better locality)
    };

    /**
     * @brief Construct space-filling curve numbering
     * @param type Type of space-filling curve
     * @param dim Spatial dimension (2 or 3)
     */
    explicit SpaceFillingCurveNumbering(CurveType type = CurveType::Morton, int dim = 3);

    /**
     * @brief Set vertex coordinates for computing curve indices
     * @param coords Vertex coordinates [n_vertices * dim]
     * @param dim Spatial dimension
     */
    void setCoordinates(std::span<const double> coords, int dim);

    [[nodiscard]] std::vector<GlobalIndex> computeNumbering(
        GlobalIndex n_dofs,
        std::span<const GlobalIndex> adjacency,
        std::span<const GlobalIndex> adj_indices) const override;

    [[nodiscard]] std::string name() const override;
    [[nodiscard]] bool requiresAdjacency() const noexcept override { return false; }

private:
    CurveType type_;
    int dim_;
    std::vector<double> coords_;

    // Compute Morton code for a point
    uint64_t mortonCode(double x, double y, double z) const;

    // Compute Hilbert index for a point
    uint64_t hilbertIndex(double x, double y, double z) const;
};

// =============================================================================
// Numbering Utilities
// =============================================================================

/**
 * @brief Apply a numbering permutation to a DOF map
 *
 * @param dof_map The DOF map to renumber (modified in place)
 * @param permutation new_dof = permutation[old_dof]
 */
void applyNumbering(DofMap& dof_map, std::span<const GlobalIndex> permutation);

/**
 * @brief Compute inverse permutation
 *
 * @param permutation Forward permutation
 * @return Inverse permutation: old_dof = inverse[new_dof]
 */
std::vector<GlobalIndex> invertPermutation(std::span<const GlobalIndex> permutation);

/**
 * @brief Compose two permutations
 *
 * @param first First permutation to apply
 * @param second Second permutation to apply
 * @return Composed permutation: composed[x] = second[first[x]]
 */
std::vector<GlobalIndex> composePermutations(
    std::span<const GlobalIndex> first,
    std::span<const GlobalIndex> second);

/**
 * @brief Compute matrix bandwidth after permutation
 *
 * @param permutation The permutation
 * @param adjacency DOF adjacency (CSR offsets)
 * @param adj_indices DOF adjacency (CSR indices)
 * @return Matrix bandwidth
 */
GlobalIndex computeBandwidth(
    std::span<const GlobalIndex> permutation,
    std::span<const GlobalIndex> adjacency,
    std::span<const GlobalIndex> adj_indices);

/**
 * @brief Compute profile/envelope after permutation
 *
 * Profile is the sum of row bandwidths, related to fill-in.
 *
 * @param permutation The permutation
 * @param adjacency DOF adjacency (CSR offsets)
 * @param adj_indices DOF adjacency (CSR indices)
 * @return Matrix profile
 */
GlobalIndex computeProfile(
    std::span<const GlobalIndex> permutation,
    std::span<const GlobalIndex> adjacency,
    std::span<const GlobalIndex> adj_indices);

/**
 * @brief Statistics about a DOF numbering
 */
struct NumberingStats {
    GlobalIndex bandwidth{0};           ///< Matrix bandwidth
    GlobalIndex profile{0};             ///< Matrix profile/envelope
    GlobalIndex max_row_nnz{0};         ///< Maximum nonzeros per row
    double avg_row_nnz{0.0};            ///< Average nonzeros per row
    std::string strategy_name;          ///< Name of numbering strategy
};

/**
 * @brief Compute numbering statistics
 *
 * @param permutation The permutation
 * @param adjacency DOF adjacency (CSR offsets)
 * @param adj_indices DOF adjacency (CSR indices)
 * @param strategy_name Name of strategy (optional)
 * @return Numbering statistics
 */
NumberingStats computeNumberingStats(
    std::span<const GlobalIndex> permutation,
    std::span<const GlobalIndex> adjacency,
    std::span<const GlobalIndex> adj_indices,
    const std::string& strategy_name = "");

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * @brief Create a numbering strategy by name
 *
 * Supported names:
 * - "sequential" or "identity"
 * - "interleaved" (default 3 components)
 * - "block"
 * - "hierarchical"
 * - "rcm" or "cuthill-mckee"
 * - "nested-dissection"
 * - "morton" or "z-order"
 * - "hilbert"
 *
 * @param name Strategy name (case-insensitive)
 * @return Unique pointer to strategy, or nullptr if unknown
 */
std::unique_ptr<NumberingStrategy> createNumberingStrategy(const std::string& name);

} // namespace dofs
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_DOFS_DOFNUMBERING_H
