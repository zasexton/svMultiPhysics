/**
 * @file test_BoundaryDofOwnerCompletionMPI.cpp
 * @brief Strong boundary constraints reach DOF owners that do not own the
 *        boundary face's cell
 */

#include <gtest/gtest.h>

#include "Assembly/Assembler.h"
#include "Constraints/AffineConstraints.h"
#include "Constraints/StrongDirichletConstraint.h"
#include "Core/FEException.h"
#include "Forms/FormExpr.h"
#include "Spaces/H1Space.h"
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

// Two unit squares side by side, cell c owned by rank c.  The bottom face of
// each cell is a boundary face with its own marker.  The vertex shared by the
// two bottom faces belongs to both cells, so for one of the two markers its
// DOF owner is the rank that holds the face's cell only as a ghost.  Like
// assembly::MeshAccess, forEachBoundaryFace visits faces of owned cells only.
class TwoCellBottomFacesMeshAccess final : public assembly::IMeshAccess {
public:
    TwoCellBottomFacesMeshAccess(std::array<int, 2> markers, int my_rank)
        : markers_(markers)
        , my_rank_(my_rank)
    {
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 2; }
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
        FE_THROW_IF(face_id != cell_id || face_id < 0 || face_id > 1,
                    InvalidArgumentException,
                    "TwoCellBottomFacesMeshAccess: face f is the bottom face of cell f");
        return 0; // reference face 0 joins Quad4 nodes 0 and 1 (the bottom edge)
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex face_id) const override
    {
        return markers_.at(static_cast<std::size_t>(face_id));
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex face_id) const override
    {
        FE_THROW_IF(face_id >= 0, InvalidArgumentException,
                    "TwoCellBottomFacesMeshAccess: no interior faces");
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
        for (GlobalIndex f = 0; f < 2; ++f) {
            if (!isOwnedCell(f)) {
                continue;
            }
            if (marker < 0 || marker == markers_[static_cast<std::size_t>(f)]) {
                callback(f, f);
            }
        }
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    std::array<int, 2> markers_{};
    int my_rank_{0};
    std::vector<std::array<Real, 3>> nodes_{
        {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0},
        {1.0, 1.0, 0.0}, {2.0, 0.0, 0.0}, {2.0, 1.0, 0.0}};
    std::array<std::array<GlobalIndex, 4>, 2> cells_{
        std::array<GlobalIndex, 4>{0, 2, 3, 1},
        std::array<GlobalIndex, 4>{2, 4, 5, 3}};
};

dofs::MeshTopologyInfo twoCellTopology(int my_rank, int world_size)
{
    dofs::MeshTopologyInfo topo;
    topo.dim = 2;
    topo.n_cells = 2;
    topo.n_vertices = 6;
    topo.n_edges = 7;
    topo.cell2vertex_offsets = {0, 4, 8};
    topo.cell2vertex_data = {0, 2, 3, 1, 2, 4, 5, 3};
    topo.vertex_gids = {0, 1, 2, 3, 4, 5};
    topo.vertex_coords = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 0.0, 2.0, 1.0};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks = {0, 1};
    topo.cell2edge_offsets = {0, 4, 8};
    topo.cell2edge_data = {1, 2, 3, 0, 4, 5, 6, 2};
    topo.edge2vertex_data = {0, 1, 0, 2, 2, 3, 1, 3, 2, 4, 4, 5, 3, 5};
    topo.edge_gids = {0, 1, 2, 3, 4, 5, 6};
    for (int r = 0; r < world_size; ++r) {
        if (r != my_rank) {
            topo.neighbor_ranks.push_back(r);
        }
    }
    return topo;
}

std::vector<GlobalIndex> allGatherSorted(const std::vector<GlobalIndex>& local, MPI_Comm comm)
{
    int size = 1;
    MPI_Comm_size(comm, &size);
    std::vector<long long> mine(local.begin(), local.end());
    int n = static_cast<int>(mine.size());
    std::vector<int> counts(static_cast<std::size_t>(size), 0);
    MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    std::vector<int> displs(static_cast<std::size_t>(size), 0);
    for (int r = 1; r < size; ++r) {
        displs[static_cast<std::size_t>(r)] =
            displs[static_cast<std::size_t>(r - 1)] + counts[static_cast<std::size_t>(r - 1)];
    }
    std::vector<long long> all(static_cast<std::size_t>(displs.back() + counts.back()));
    MPI_Allgatherv(mine.data(), n, MPI_LONG_LONG, all.data(), counts.data(), displs.data(),
                   MPI_LONG_LONG, comm);
    std::vector<GlobalIndex> out(all.begin(), all.end());
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace

TEST(BoundaryDofOwnerCompletionMPITest, StrongDirichletReachesOwnersOfGhostFaceDofs)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int my_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly 2 MPI ranks";
    }

    constexpr std::array<int, 2> markers{21, 22};
    auto mesh = std::make_shared<TwoCellBottomFacesMeshAccess>(markers, my_rank);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto u = system.addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});

    systems::SetupOptions setup_opts;
    setup_opts.dof_options.my_rank = my_rank;
    setup_opts.dof_options.world_size = world_size;
    setup_opts.dof_options.mpi_comm = comm;
    systems::SetupInputs inputs;
    inputs.topology_override = twoCellTopology(my_rank, world_size);
    system.setup(setup_opts, inputs);

    const auto& owned = system.dofHandler().getPartition().locallyOwned();
    const auto& dof_map = system.fieldDofHandler(u).getDofMap();
    const auto offset = system.fieldDofOffset(u);

    bool some_owner_lacks_the_face_cell = false;
    for (int face = 0; face < 2; ++face) {
        // The serial set: the DOFs of the face's two vertices (reference
        // nodes 0 and 1 of the face's cell).
        const auto cell_dofs = dof_map.getCellDofs(face);
        ASSERT_GE(cell_dofs.size(), 2u);
        std::vector<GlobalIndex> expected{cell_dofs[0] + offset, cell_dofs[1] + offset};
        std::sort(expected.begin(), expected.end());
        for (const auto dof : expected) {
            const int owner = system.dofHandler().getDofMap().getDofOwner(dof);
            some_owner_lacks_the_face_cell = some_owner_lacks_the_face_cell || owner != face;
        }

        StrongDirichletConstraint constraint(u, markers[static_cast<std::size_t>(face)],
                                             forms::FormExpr::constant(0.0));
        AffineConstraints local_constraints;
        constraint.apply(system, local_constraints);
        local_constraints.close();

        std::vector<GlobalIndex> local;
        for (const auto dof : local_constraints.getConstrainedDofs()) {
            EXPECT_TRUE(owned.contains(dof)) << "rank " << my_rank << " constrained non-owned DOF " << dof;
            local.push_back(dof);
        }
        EXPECT_EQ(allGatherSorted(local, comm), expected) << "marker " << markers[static_cast<std::size_t>(face)];
    }
    // The configuration must exercise a DOF owned by the rank that holds the
    // face's cell only as a ghost; otherwise the test proves nothing.
    EXPECT_TRUE(some_owner_lacks_the_face_cell);
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
