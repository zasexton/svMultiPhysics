/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_BACKENDS_FSILS_SHARED_H
#define SVMP_FE_BACKENDS_FSILS_SHARED_H

#include "Core/Types.h"

#include "Backends/Interfaces/DofPermutation.h"
#include "Backends/FSILS/liner_solver/fils_struct.hpp"

#include <algorithm>
#include <memory>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {

/**
 * @brief Shared FSILS layout/communication metadata for a matrix/vector pair.
 *
 * FSILS uses a node-based overlap model: each rank stores a local node set
 * (owned + ghost), and shared nodes appear on multiple ranks. The FSILS
 * `lhs` object contains the MPI communication plans and the internal
 * permutation (`lhs.map`) used by the solver.
 *
 * FE stores vectors in the "old" local node ordering (before FSILS reorders
 * nodes for overlap bookkeeping). FSILS internally maps to its ordering as
 * needed.
 */
struct FsilsShared final {
    GlobalIndex global_dofs{0}; ///< Global DOF count (= dof * gnNo)
    int dof{1};                 ///< DOFs per node (block size)
    int gnNo{0};                ///< Global node count

    // Old local node ordering (before FSILS reordering).
    int owned_node_start{0};   ///< First owned global node ID (inclusive)
    int owned_node_count{0};   ///< Number of owned nodes
    // Optional explicit owned-node list. When non-empty, owned nodes are not representable as a single contiguous
    // global-node range (owned_node_start/owned_node_count). The list must be sorted unique and its size must equal
    // owned_node_count.
    std::vector<int> owned_nodes{};
    std::vector<int> ghost_nodes{}; ///< Sorted global node IDs for ghost nodes

    // Inverse permutation for convenience: old_of_internal[internal] = old.
    std::vector<int> old_of_internal{};

    // Optional global DOF permutation: FE ordering -> FSILS node-block ordering.
    std::shared_ptr<const DofPermutation> dof_permutation{};

    fe_fsi_linear_solver::FSILS_lhsType lhs{};

    [[nodiscard]] int localNodeCount() const noexcept { return lhs.nNo; }

    [[nodiscard]] int globalNodeToOld(int global_node) const noexcept
    {
        if (!owned_nodes.empty()) {
            const auto it = std::lower_bound(owned_nodes.begin(), owned_nodes.end(), global_node);
            if (it != owned_nodes.end() && *it == global_node) {
                return static_cast<int>(it - owned_nodes.begin());
            }
        } else if (global_node >= owned_node_start && global_node < owned_node_start + owned_node_count) {
            return global_node - owned_node_start;
        }

        const auto it = std::lower_bound(ghost_nodes.begin(), ghost_nodes.end(), global_node);
        if (it == ghost_nodes.end() || *it != global_node) {
            return -1;
        }
        return owned_node_count + static_cast<int>(it - ghost_nodes.begin());
    }

    [[nodiscard]] int oldToGlobalNode(int old) const noexcept
    {
        if (old < 0) {
            return -1;
        }
        if (old < owned_node_count) {
            if (!owned_nodes.empty()) {
                const auto idx = static_cast<std::size_t>(old);
                if (idx >= owned_nodes.size()) {
                    return -1;
                }
                return owned_nodes[idx];
            }
            return owned_node_start + old;
        }

        const auto ghost_idx = static_cast<std::size_t>(old - owned_node_count);
        if (ghost_idx >= ghost_nodes.size()) {
            return -1;
        }
        return ghost_nodes[ghost_idx];
    }
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_FSILS_SHARED_H
