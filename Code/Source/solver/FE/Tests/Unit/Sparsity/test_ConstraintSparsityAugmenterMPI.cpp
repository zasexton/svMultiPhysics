/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyKernel.h"
#include "Assembly/Assembler.h"
#include "Backends/Interfaces/DofPermutation.h"
#include "Constraints/SystemConstraint.h"
#include "Sparsity/ConstraintSparsityAugmenter.h"
#include "Sparsity/ParallelSparsity.h"
#include "Dofs/DofHandler.h"
#include "Dofs/DofMap.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::assembly;
using namespace svmp::FE::dofs;
using namespace svmp::FE::sparsity;

namespace {

DofMap makeDofMap(GlobalIndex n_dofs, const std::vector<std::vector<GlobalIndex>>& cell_dofs) {
    FE_CHECK_ARG(!cell_dofs.empty(), "cell_dofs must not be empty");

    const GlobalIndex n_cells = static_cast<GlobalIndex>(cell_dofs.size());
    const LocalIndex dofs_per_cell = static_cast<LocalIndex>(cell_dofs.front().size());
    FE_CHECK_ARG(dofs_per_cell > 0, "Each cell must have at least one DOF");

    DofMap map(n_cells, n_dofs, dofs_per_cell);
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        map.setCellDofs(c, cell_dofs[static_cast<std::size_t>(c)]);
    }
    map.setNumDofs(n_dofs);
    return map;
}

DistributedSparsityPattern rebuildActiveFromBase(
    const DistributedSparsityPattern& base,
    const std::shared_ptr<SimpleConstraintSet>& constraints)
{
    DistributedSparsityPattern active(base.ownedRows(),
                                      base.ownedCols(),
                                      base.globalRows(),
                                      base.globalCols());
    active.setDofIndexing(base.dofIndexing());
    const auto owned = base.ownedRows();
    for (GlobalIndex row = owned.first; row < owned.last; ++row) {
        const auto cols = base.getOwnedRowGlobalCols(row);
        active.addEntries(row, std::span<const GlobalIndex>(cols.data(), cols.size()));
    }

    ConstraintSparsityAugmenter augmenter(constraints);
    augmenter.augment(active, AugmentationMode::EliminationFill);
    active.finalize();
    return active;
}

class FourTetraMeshAccess final : public IMeshAccess {
public:
    FourTetraMeshAccess(std::vector<int> cell_owner_ranks, int my_rank)
        : cell_owner_ranks_(std::move(cell_owner_ranks)), my_rank_(my_rank)
    {
        nodes_ = {
            std::array<Real, 3>{0.0, 0.0, 0.0},
            std::array<Real, 3>{1.0, 0.0, 0.0},
            std::array<Real, 3>{0.0, 1.0, 0.0},
            std::array<Real, 3>{0.0, 0.0, 1.0},
            std::array<Real, 3>{1.0, 1.0, 0.0},
            std::array<Real, 3>{1.0, 0.2, 1.0},
        };
        cells_ = {
            std::array<GlobalIndex, 4>{0, 1, 2, 3},
            std::array<GlobalIndex, 4>{1, 2, 3, 4},
            std::array<GlobalIndex, 4>{1, 3, 4, 5},
            std::array<GlobalIndex, 4>{2, 3, 4, 5},
        };

        for (GlobalIndex cell = 0; cell < numCells(); ++cell) {
            if (isOwnedCell(cell)) {
                owned_cells_.push_back(cell);
            }
        }
    }

    [[nodiscard]] GlobalIndex numCells() const override
    {
        return static_cast<GlobalIndex>(cells_.size());
    }

    [[nodiscard]] GlobalIndex numOwnedCells() const override
    {
        return static_cast<GlobalIndex>(owned_cells_.size());
    }

    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override
    {
        return cell_owner_ranks_.at(static_cast<std::size_t>(cell_id)) == my_rank_;
    }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override
    {
        return ElementType::Tetra4;
    }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(cell.size());
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex cell = 0; cell < numCells(); ++cell) {
            callback(cell);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (const auto cell : owned_cells_) {
            callback(cell);
        }
    }

    void forEachBoundaryFace(
        int /*marker*/,
        std::function<void(GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    std::vector<std::array<Real, 3>> nodes_{};
    std::vector<std::array<GlobalIndex, 4>> cells_{};
    std::vector<int> cell_owner_ranks_{};
    int my_rank_{0};
    std::vector<GlobalIndex> owned_cells_{};
};

class MutableMasterConstraint final : public svmp::FE::constraints::ISystemConstraint {
public:
    explicit MutableMasterConstraint(const GlobalIndex* master) : master_(master) {}

    void apply(const svmp::FE::systems::FESystem& /*system*/,
               svmp::FE::constraints::AffineConstraints& constraints) override
    {
        constraints.addLine(1);
        constraints.addEntry(1, *master_, 1.0);
    }

    bool updateValues(const svmp::FE::systems::FESystem& /*system*/,
                      svmp::FE::constraints::AffineConstraints& /*constraints*/,
                      double /*time*/,
                      double /*dt*/) override
    {
        return false;
    }

    [[nodiscard]] bool isTimeDependent() const noexcept override { return false; }

    [[nodiscard]] svmp::FE::systems::SetupStorageRequirements
    storageRequirements() const noexcept override
    {
        return {};
    }

private:
    const GlobalIndex* master_{nullptr};
};

svmp::FE::systems::SetupOptions mpiSetupOptions(MPI_Comm comm, int rank, int size)
{
    svmp::FE::systems::SetupOptions opts;
    opts.assembly_options.deterministic = true;
    opts.assembly_options.overlap_communication = false;
    opts.use_backend_row_ownership_for_assembly = true;
    opts.dof_options.global_numbering = GlobalNumberingMode::GlobalIds;
    opts.dof_options.ownership = OwnershipStrategy::VertexGID;
    opts.dof_options.my_rank = rank;
    opts.dof_options.world_size = size;
    opts.dof_options.mpi_comm = comm;
    return opts;
}

std::vector<int> allOtherRanks(int rank, int size)
{
    std::vector<int> neighbors;
    for (int other = 0; other < size; ++other) {
        if (other != rank) {
            neighbors.push_back(other);
        }
    }
    return neighbors;
}

MeshTopologyInfo fourTetraTopology(std::span<const int> cell_owner_ranks,
                                   int rank,
                                   int size)
{
    MeshTopologyInfo topo;
    topo.n_cells = 4;
    topo.n_vertices = 6;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4, 8, 12, 16};
    topo.cell2vertex_data = {
        0, 1, 2, 3,
        1, 2, 3, 4,
        1, 3, 4, 5,
        2, 3, 4, 5,
    };
    topo.vertex_gids = {0, 1, 2, 3, 4, 5};
    topo.cell_gids = {0, 1, 2, 3};
    topo.cell_owner_ranks.assign(cell_owner_ranks.begin(), cell_owner_ranks.end());
    topo.neighbor_ranks = allOtherRanks(rank, size);
    return topo;
}

} // namespace

TEST(ConstraintSparsityAugmenterMPITest, BuildReducedDistributedPatternDirichlet) {
    int my_rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    if (n_ranks < 2) {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    const GlobalIndex n_global = static_cast<GlobalIndex>(3 * n_ranks);
    const GlobalIndex owned_first = static_cast<GlobalIndex>(3 * my_rank);
    const GlobalIndex owned_last = owned_first + 3;

    std::vector<std::vector<GlobalIndex>> cell_dofs;
    cell_dofs.push_back({owned_first, owned_first + 1});
    cell_dofs.push_back({owned_first + 1, owned_first + 2});
    if (my_rank < n_ranks - 1) {
        cell_dofs.push_back({owned_first + 2, owned_first + 3});
    }

    auto dof_map = makeDofMap(n_global, cell_dofs);
    dof_map.finalize();

    ParallelSparsityManager mgr(MPI_COMM_WORLD);
    mgr.setBlockOwnership(n_global);
    mgr.setRowDofMap(dof_map);

    SparsityBuildOptions opts;
    opts.ensure_diagonal = true;
    opts.ensure_non_empty_rows = true;
    opts.include_ghost_rows = true;
    mgr.setOptions(opts);

    const auto full_pattern = mgr.build();
    EXPECT_TRUE(full_pattern.isFinalized());
    EXPECT_TRUE(full_pattern.validate());

    // Constrain the middle DOF on each rank: this removes one vertex per rank
    // but preserves cross-rank couplings between (3r+2) and (3(r+1)).
    auto constraints = std::make_shared<SimpleConstraintSet>();
    for (int r = 0; r < n_ranks; ++r) {
        constraints->addDirichlet(static_cast<GlobalIndex>(3 * r + 1));
    }

    ConstraintSparsityAugmenter augmenter(constraints);
    const auto reduced = augmenter.buildReducedDistributedPattern(full_pattern, MPI_COMM_WORLD);

    const GlobalIndex expected_global_reduced = static_cast<GlobalIndex>(2 * n_ranks);
    EXPECT_EQ(reduced.global_reduced_size, expected_global_reduced);
    EXPECT_EQ(reduced.owned_reduced_range.first, static_cast<GlobalIndex>(2 * my_rank));
    EXPECT_EQ(reduced.owned_reduced_range.last, static_cast<GlobalIndex>(2 * my_rank + 2));

    EXPECT_TRUE(reduced.pattern.isFinalized());
    EXPECT_TRUE(reduced.pattern.validate());
    EXPECT_EQ(reduced.pattern.globalRows(), expected_global_reduced);
    EXPECT_EQ(reduced.pattern.globalCols(), expected_global_reduced);

    ASSERT_EQ(reduced.full_to_reduced_owned.size(), static_cast<std::size_t>(owned_last - owned_first));
    ASSERT_EQ(reduced.reduced_to_full_owned.size(), static_cast<std::size_t>(reduced.owned_reduced_range.size()));

    // Local full->reduced mapping: [3r, 3r+1, 3r+2] -> [2r, -1, 2r+1]
    EXPECT_EQ(reduced.full_to_reduced_owned[0], static_cast<GlobalIndex>(2 * my_rank));
    EXPECT_EQ(reduced.full_to_reduced_owned[1], -1);
    EXPECT_EQ(reduced.full_to_reduced_owned[2], static_cast<GlobalIndex>(2 * my_rank + 1));

    // Local reduced->full mapping: [2r, 2r+1] -> [3r, 3r+2]
    EXPECT_EQ(reduced.reduced_to_full_owned[0], owned_first);
    EXPECT_EQ(reduced.reduced_to_full_owned[1], owned_first + 2);

    // Reduced pattern rows should at least contain diagonal entries.
    for (GlobalIndex row = reduced.owned_reduced_range.first; row < reduced.owned_reduced_range.last; ++row) {
        EXPECT_TRUE(reduced.pattern.hasEntry(row, row));
    }

    // Cross-rank couplings should survive in the reduced system:
    // A_r (2r) <-> B_{r-1} (2r-1) and B_r (2r+1) <-> A_{r+1} (2r+2).
    const GlobalIndex a_r = static_cast<GlobalIndex>(2 * my_rank);
    const GlobalIndex b_r = static_cast<GlobalIndex>(2 * my_rank + 1);
    if (my_rank > 0) {
        EXPECT_TRUE(reduced.pattern.hasEntry(a_r, a_r - 1));
    }
    if (my_rank < n_ranks - 1) {
        EXPECT_TRUE(reduced.pattern.hasEntry(b_r, b_r + 1));
    }
}

TEST(ConstraintSparsityAugmenterMPITest, DistributedRefreshFromBaseUsesCurrentMasterTopology) {
    int my_rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    if (n_ranks != 2) {
        GTEST_SKIP() << "Requires exactly 2 MPI ranks";
    }

    const GlobalIndex n_global = 4;
    const IndexRange owned{static_cast<GlobalIndex>(2 * my_rank),
                           static_cast<GlobalIndex>(2 * my_rank + 2)};

    DistributedSparsityPattern base(owned, owned, n_global, n_global);
    base.ensureDiagonal();
    if (my_rank == 0) {
        // Row 0 couples to slave column 1. Elimination fill must add the
        // current master for DOF 1, and a refresh must use the new master
        // rather than accumulating fill from the previous active pattern.
        base.addEntry(0, 1);
    }
    base.finalize();

    auto initial_constraints = std::make_shared<SimpleConstraintSet>();
    initial_constraints->addConstraint(1, 2);
    auto initial = rebuildActiveFromBase(base, initial_constraints);

    auto refreshed_constraints = std::make_shared<SimpleConstraintSet>();
    refreshed_constraints->addConstraint(1, 3);
    auto refreshed = rebuildActiveFromBase(base, refreshed_constraints);

    if (my_rank == 0) {
        EXPECT_TRUE(initial.hasEntry(0, 2));
        EXPECT_FALSE(initial.hasEntry(0, 3));

        EXPECT_TRUE(refreshed.hasEntry(0, 3));
        EXPECT_FALSE(refreshed.hasEntry(0, 2));
    }
}

TEST(FESystemSparsityRefreshMPITest, RebuildConstraintStateRefreshesDistributedPattern) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int my_rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &n_ranks);

    if (n_ranks != 2) {
        GTEST_SKIP() << "Requires exactly 2 MPI ranks";
    }

    const std::vector<int> cell_owners{0, 0, 1, 1};
    auto mesh = std::make_shared<FourTetraMeshAccess>(cell_owners, my_rank);
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(ElementType::Tetra4, 1);

    svmp::FE::systems::FESystem system(mesh);
    const auto field = system.addField(
        svmp::FE::systems::FieldSpec{.name = "u", .space = space, .components = 1});
    system.addOperator("op");
    system.addCellKernel("op", field, field, std::make_shared<MassKernel>(1.0));

    GlobalIndex current_master = 4;
    system.addSystemConstraint(std::make_unique<MutableMasterConstraint>(&current_master));

    svmp::FE::systems::SetupInputs inputs;
    inputs.topology_override = fourTetraTopology(cell_owners, my_rank, n_ranks);
    ASSERT_NO_THROW(system.setup(mpiSetupOptions(comm, my_rank, n_ranks), inputs));

    const auto initial_revision = system.sparsityPatternRevision();
    const auto& serial_initial = system.sparsity("op");
    const auto* dist_initial = system.distributedSparsityIfAvailable("op");
    ASSERT_NE(dist_initial, nullptr);
    EXPECT_EQ(dist_initial->dofIndexing(),
              DistributedSparsityPattern::DofIndexing::NodalInterleaved);
    const auto perm = system.dofPermutation();
    ASSERT_NE(perm, nullptr);
    ASSERT_GT(perm->forward.size(), 5u);
    const auto backend_row = perm->forward[0];
    const auto backend_initial_master = perm->forward[4];
    const auto backend_refreshed_master = perm->forward[5];
    const auto ghost_row_has_col =
        [](const DistributedSparsityPattern& pattern,
           GlobalIndex row,
           GlobalIndex col) {
            const auto local_row = pattern.globalToGhostRow(row);
            if (local_row < 0) {
                return false;
            }
            const auto cols = pattern.getGhostRowCols(local_row);
            return std::find(cols.begin(), cols.end(), col) != cols.end();
        };

    if (my_rank == 0) {
        EXPECT_TRUE(serial_initial.hasEntry(0, 4));
        EXPECT_FALSE(serial_initial.hasEntry(0, 5));
        EXPECT_TRUE(dist_initial->hasEntry(backend_row, backend_initial_master));
        EXPECT_FALSE(dist_initial->hasEntry(backend_row, backend_refreshed_master));
    }

    current_master = 5;
    ASSERT_NO_THROW(system.rebuildConstraintState());
    EXPECT_EQ(system.sparsityPatternRevision(), initial_revision + 1u);

    const auto& serial_refreshed = system.sparsity("op");
    const auto* dist_refreshed = system.distributedSparsityIfAvailable("op");
    ASSERT_NE(dist_refreshed, nullptr);
    EXPECT_EQ(dist_refreshed->dofIndexing(),
              DistributedSparsityPattern::DofIndexing::NodalInterleaved);

    if (my_rank == 0) {
        EXPECT_FALSE(serial_refreshed.hasEntry(0, 4));
        EXPECT_TRUE(serial_refreshed.hasEntry(0, 5));
        EXPECT_FALSE(dist_refreshed->hasEntry(backend_row, backend_initial_master));
        EXPECT_TRUE(dist_refreshed->hasEntry(backend_row, backend_refreshed_master));
        if (!dist_refreshed->ownedRows().contains(backend_refreshed_master)) {
            EXPECT_GE(dist_refreshed->globalToGhostRow(backend_refreshed_master), 0);
            EXPECT_TRUE(ghost_row_has_col(*dist_refreshed,
                                          backend_refreshed_master,
                                          backend_refreshed_master));
        }
    }

    const auto no_change_revision = system.sparsityPatternRevision();
    ASSERT_NO_THROW(system.rebuildConstraintState());
    EXPECT_EQ(system.sparsityPatternRevision(), no_change_revision);
}
