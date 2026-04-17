/**
 * @file test_HDivNormalConstraintMPI.cpp
 * @brief MPI regression tests for H(div) normal-trace ownership filtering
 */

#include <gtest/gtest.h>

#include "Assembly/Assembler.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/HDivNormalConstraint.h"
#include "Core/FEException.h"
#include "Forms/FormExpr.h"
#include "Spaces/HDivSpace.h"
#include "Systems/FESystem.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

namespace {

std::vector<int> neighborRanks(int my_rank, int world_size)
{
    std::vector<int> neighbors;
    neighbors.reserve(static_cast<std::size_t>(std::max(0, world_size - 1)));
    for (int r = 0; r < world_size; ++r) {
        if (r != my_rank) {
            neighbors.push_back(r);
        }
    }
    return neighbors;
}

class TwoCellStripBoundaryMeshAccess final : public assembly::IMeshAccess {
public:
    TwoCellStripBoundaryMeshAccess(int boundary_marker, int my_rank)
        : boundary_marker_(boundary_marker)
        , my_rank_(my_rank)
    {
        nodes_ = {
            {0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {1.0, 0.0, 0.0},
            {1.0, 1.0, 0.0},
            {2.0, 0.0, 0.0},
            {2.0, 1.0, 0.0}
        };
        cells_ = {
            std::array<GlobalIndex, 4>{0, 2, 3, 1},
            std::array<GlobalIndex, 4>{2, 4, 5, 3}
        };
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 1; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override
    {
        return static_cast<int>(cell_id) == my_rank_;
    }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override
    {
        return ElementType::Quad4;
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

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex face_id,
                                               GlobalIndex cell_id) const override
    {
        FE_THROW_IF(face_id != 0 || cell_id != 0,
                    InvalidArgumentException,
                    "TwoCellStripBoundaryMeshAccess: only left boundary face is defined");
        return 3;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        FE_THROW_IF(face_id != 0,
                    InvalidArgumentException,
                    "TwoCellStripBoundaryMeshAccess: invalid boundary face");
        return boundary_marker_;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex face_id) const override
    {
        FE_THROW_IF(face_id != 0,
                    InvalidArgumentException,
                    "TwoCellStripBoundaryMeshAccess: invalid interior face query");
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
        callback(1);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(static_cast<GlobalIndex>(my_rank_));
    }

    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (marker < 0 || marker == boundary_marker_) {
            callback(0, 0);
        }
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    int boundary_marker_{-1};
    int my_rank_{0};
    std::vector<std::array<Real, 3>> nodes_{};
    std::array<std::array<GlobalIndex, 4>, 2> cells_{};
};

dofs::MeshTopologyInfo twoCellStripTopology(int my_rank, int world_size)
{
    dofs::MeshTopologyInfo topo;
    topo.dim = 2;
    topo.n_cells = 2;
    topo.n_vertices = 6;
    topo.n_edges = 7;

    topo.cell2vertex_offsets = {0, 4, 8};
    topo.cell2vertex_data = {
        0, 2, 3, 1,
        2, 4, 5, 3
    };
    topo.vertex_gids = {0, 1, 2, 3, 4, 5};
    topo.vertex_coords = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
        2.0, 0.0,
        2.0, 1.0
    };

    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks = {0, 1};

    topo.cell2edge_offsets = {0, 4, 8};
    topo.cell2edge_data = {
        1, 2, 3, 0,
        4, 5, 6, 2
    };
    topo.edge2vertex_data = {
        0, 1,
        0, 2,
        2, 3,
        1, 3,
        2, 4,
        4, 5,
        3, 5
    };
    topo.edge_gids = {0, 1, 2, 3, 4, 5, 6};
    topo.neighbor_ranks = neighborRanks(my_rank, world_size);

    return topo;
}

} // namespace

TEST(HDivNormalConstraintMPITest, GhostReplicasDoNotInsertOwnedConstraintLines)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int my_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &world_size);

    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly 2 MPI ranks";
    }

    constexpr int marker = 11;

    auto mesh = std::make_shared<TwoCellStripBoundaryMeshAccess>(marker, my_rank);
    auto space = std::make_shared<spaces::HDivSpace>(ElementType::Quad4, /*order=*/0);

    systems::FESystem system(mesh);
    const auto flux = system.addField(
        systems::FieldSpec{.name = "q", .space = space, .components = space->value_dimension()});

    systems::SetupOptions setup_opts;
    setup_opts.dof_options.my_rank = my_rank;
    setup_opts.dof_options.world_size = world_size;
    setup_opts.dof_options.mpi_comm = comm;

    systems::SetupInputs inputs;
    inputs.topology_override = twoCellStripTopology(my_rank, world_size);
    system.setup(setup_opts, inputs);

    ASSERT_GT(system.dofHandler().getNumDofs(), 0);
    ASSERT_GT(system.dofHandler().getPartition().localOwnedSize(), 0);
    ASSERT_TRUE(system.dofHandler().getPartition().locallyRelevant().contains(0));

    HDivNormalConstraint constraint(flux, marker, forms::FormExpr::constant(2.0));
    AffineConstraints local_constraints;
    constraint.apply(system, local_constraints);
    local_constraints.close();

    const int local_count = static_cast<int>(local_constraints.numConstraints());
    int global_count = 0;
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, comm);

    EXPECT_EQ(global_count, 1);
    if (my_rank == 0) {
        ASSERT_EQ(local_constraints.numConstraints(), 1u);
        const auto constrained = local_constraints.getConstrainedDofs();
        ASSERT_EQ(constrained.size(), 1u);
        EXPECT_TRUE(local_constraints.isConstrained(constrained.front()));
    } else {
        EXPECT_EQ(local_constraints.numConstraints(), 0u);
    }
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
