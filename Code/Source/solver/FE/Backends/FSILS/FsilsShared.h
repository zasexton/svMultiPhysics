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
    std::vector<int> ghost_nodes{}; ///< Sorted global node IDs for ghost nodes

    // Inverse permutation for convenience: old_of_internal[internal] = old.
    std::vector<int> old_of_internal{};

    // Optional global DOF permutation: FE ordering -> FSILS node-block ordering.
    std::shared_ptr<const DofPermutation> dof_permutation{};

    fsi_linear_solver::FSILS_lhsType lhs{};

    [[nodiscard]] int localNodeCount() const noexcept { return lhs.nNo; }

    [[nodiscard]] int globalNodeToOld(int global_node) const noexcept
    {
        if (global_node < owned_node_start) {
            // Fall through to ghost search.
        } else if (global_node < owned_node_start + owned_node_count) {
            return global_node - owned_node_start;
        }

        const auto it = std::lower_bound(ghost_nodes.begin(), ghost_nodes.end(), global_node);
        if (it == ghost_nodes.end() || *it != global_node) {
            return -1;
        }
        return owned_node_count + static_cast<int>(it - ghost_nodes.begin());
    }
};

} // namespace backends
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BACKENDS_FSILS_SHARED_H
