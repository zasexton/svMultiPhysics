#include "LevelSet/LevelSetSignDefinitePatchBounds.h"

#include "Assembly/Assembler.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/H1Space.h"

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

// Structured triangle mesh of the unit square with a block partition of the
// cells over the ranks.  Every rank sees the whole mesh; only the owned cells
// differ, which is the replicated level-set layout used in production.
class StructuredTrianglePartition final : public FE::assembly::IMeshAccess {
public:
    StructuredTrianglePartition(int n, int rank, int size)
        : rank_(rank), size_(size)
    {
        const int extent = n + 1;
        for (int j = 0; j <= n; ++j) {
            for (int i = 0; i <= n; ++i) {
                coordinates_.push_back({{static_cast<FE::Real>(i) / n,
                                         static_cast<FE::Real>(j) / n, 0.0}});
            }
        }
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const FE::GlobalIndex a = j * extent + i;
                const FE::GlobalIndex b = a + 1;
                const FE::GlobalIndex d = a + extent;
                const FE::GlobalIndex c = d + 1;
                if ((i + j) % 2 == 0) {
                    cells_.push_back({{a, b, c}});
                    cells_.push_back({{a, c, d}});
                } else {
                    cells_.push_back({{a, b, d}});
                    cells_.push_back({{b, c, d}});
                }
            }
        }
    }

    [[nodiscard]] int owner(FE::GlobalIndex cell) const
    {
        return static_cast<int>((cell * size_) / static_cast<FE::GlobalIndex>(cells_.size()));
    }

    [[nodiscard]] FE::dofs::MeshTopologyInfo topology() const
    {
        FE::dofs::MeshTopologyInfo info;
        info.n_cells = static_cast<FE::GlobalIndex>(cells_.size());
        info.n_vertices = static_cast<FE::GlobalIndex>(coordinates_.size());
        info.dim = 2;
        info.cell2vertex_offsets.push_back(0);
        for (std::size_t c = 0; c < cells_.size(); ++c) {
            for (const auto v : cells_[c]) {
                info.cell2vertex_data.push_back(static_cast<FE::MeshIndex>(v));
            }
            info.cell2vertex_offsets.push_back(
                static_cast<FE::MeshOffset>(info.cell2vertex_data.size()));
            info.cell_gids.push_back(static_cast<FE::MeshGlobalId>(c));
            info.cell_owner_ranks.push_back(owner(static_cast<FE::GlobalIndex>(c)));
        }
        for (std::size_t v = 0; v < coordinates_.size(); ++v) {
            info.vertex_gids.push_back(static_cast<FE::MeshGlobalId>(v));
        }
        return info;
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override
    {
        return static_cast<FE::GlobalIndex>(cells_.size());
    }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override
    {
        FE::GlobalIndex count = 0;
        for (FE::GlobalIndex c = 0; c < numCells(); ++c) {
            count += owner(c) == rank_ ? 1 : 0;
        }
        return count;
    }
    [[nodiscard]] FE::GlobalIndex numVertices() const override
    {
        return static_cast<FE::GlobalIndex>(coordinates_.size());
    }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool cellIdsAreDense() const override { return true; }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex cell) const override
    {
        return owner(cell) == rank_;
    }
    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex) const override
    {
        return FE::ElementType::Triangle3;
    }
    void getCellNodes(FE::GlobalIndex cell, std::vector<FE::GlobalIndex>& nodes) const override
    {
        const auto& c = cells_.at(static_cast<std::size_t>(cell));
        nodes.assign(c.begin(), c.end());
    }
    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(FE::GlobalIndex node) const override
    {
        return coordinates_.at(static_cast<std::size_t>(node));
    }
    void getCellCoordinates(FE::GlobalIndex cell,
                            std::vector<std::array<FE::Real, 3>>& coordinates) const override
    {
        coordinates.clear();
        for (const auto v : cells_.at(static_cast<std::size_t>(cell))) {
            coordinates.push_back(getNodeCoordinates(v));
        }
    }
    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(FE::GlobalIndex, FE::GlobalIndex) const override
    {
        return 0;
    }
    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex) const override { return 0; }
    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex> getInteriorFaceCells(
        FE::GlobalIndex) const override
    {
        return {0, 0};
    }
    void forEachCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        for (FE::GlobalIndex c = 0; c < numCells(); ++c) {
            callback(c);
        }
    }
    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        for (FE::GlobalIndex c = 0; c < numCells(); ++c) {
            if (owner(c) == rank_) {
                callback(c);
            }
        }
    }
    void forEachBoundaryFace(int,
                             std::function<void(FE::GlobalIndex, FE::GlobalIndex)>) const override
    {
    }
    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)>) const override
    {
    }

private:
    int rank_{0};
    int size_{1};
    std::vector<std::array<FE::Real, 3>> coordinates_;
    std::vector<std::array<FE::GlobalIndex, 3>> cells_;
};

[[nodiscard]] FE::dofs::DofDistributionOptions dofOptions(MPI_Comm communicator, int rank, int size)
{
    FE::dofs::DofDistributionOptions options;
    options.global_numbering = FE::dofs::GlobalNumberingMode::GlobalIds;
    options.ownership = FE::dofs::OwnershipStrategy::LowestRank;
    options.my_rank = rank;
    options.world_size = size;
    options.mpi_comm = communicator;
    return options;
}

struct Layout {
    StructuredTrianglePartition mesh;
    FE::dofs::DofHandler phi;

    Layout(int n, MPI_Comm communicator, int rank, int size)
        : mesh(n, rank, size)
    {
        FE::spaces::H1Space scalar(FE::ElementType::Triangle3, /*order=*/1);
        phi.distributeDofs(mesh.topology(), scalar, dofOptions(communicator, rank, size));
        phi.finalize();
    }

    [[nodiscard]] std::vector<FE::Real> scalar(
        const std::function<FE::Real(const std::array<FE::Real, 3>&)>& f) const
    {
        std::vector<FE::Real> out(static_cast<std::size_t>(phi.getNumDofs()), 0.0);
        const auto* map = phi.getEntityDofMap();
        for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
            out[static_cast<std::size_t>(map->getVertexDofs(v).front())] =
                f(mesh.getNodeCoordinates(v));
        }
        return out;
    }

    // Values by vertex (independent of the DOF numbering).
    [[nodiscard]] std::vector<FE::Real> byVertex(const std::vector<FE::Real>& values) const
    {
        std::vector<FE::Real> out(static_cast<std::size_t>(mesh.numVertices()), 0.0);
        const auto* map = phi.getEntityDofMap();
        for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
            out[static_cast<std::size_t>(v)] =
                values[static_cast<std::size_t>(map->getVertexDofs(v).front())];
        }
        return out;
    }
};

FE::Real circle(const std::array<FE::Real, 3>& x)
{
    return std::hypot(x[0] - 0.5031, x[1] - 0.4687) - 0.27;
}

} // namespace

TEST(LevelSetSignDefinitePatchBoundsMPI, PartitionedResultMatchesSerialReference)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    constexpr int n = 16;
    const Layout parallel(n, MPI_COMM_WORLD, rank, size);
    const Layout serial(n, MPI_COMM_SELF, 0, 1);

    // A candidate with mesh-scale wiggles away from the interface, one of
    // them crossing the isovalue, on patches split by the cell partition.
    const auto candidate_field = [](const std::array<FE::Real, 3>& x) {
        const FE::Real phi = circle(x);
        if (std::abs(x[0] - 0.875) + std::abs(x[1] - 0.5) < 1.0e-12) {
            return FE::Real{-1.0e-3};
        }
        return std::abs(phi) < 0.1 ? phi
                                   : phi + 0.02 * std::sin(97.0 * x[0]) * std::cos(89.0 * x[1]);
    };
    const auto run = [&](const Layout& layout, std::vector<FE::Real>& bounded) {
        return level_set::boundLevelSetOnSignDefinitePatches(
            layout.mesh, layout.phi, 0.0, 1.0e-12,
            layout.scalar(circle), layout.scalar(candidate_field), bounded);
    };
    std::vector<FE::Real> parallel_values;
    std::vector<FE::Real> serial_values;
    const auto p = run(parallel, parallel_values);
    const auto s = run(serial, serial_values);
    ASSERT_TRUE(p.success) << p.diagnostic;
    ASSERT_TRUE(s.success) << s.diagnostic;
    EXPECT_TRUE(p.applied);
    EXPECT_EQ(p.sign_definite_dofs, s.sign_definite_dofs);
    EXPECT_EQ(p.bounded_dofs, s.bounded_dofs);
    EXPECT_EQ(p.sign_changes_prevented, 1u);
    EXPECT_EQ(s.sign_changes_prevented, 1u);
    EXPECT_EQ(p.max_abs_correction, s.max_abs_correction);

    const auto pv = parallel.byVertex(parallel_values);
    const auto sv = serial.byVertex(serial_values);
    ASSERT_EQ(pv.size(), sv.size());
    for (std::size_t v = 0; v < pv.size(); ++v) {
        EXPECT_EQ(pv[v], sv[v]) << "vertex " << v;
    }
    // Every rank must hold the same replicated result.
    std::vector<FE::Real> maximum(pv);
    std::vector<FE::Real> minimum(pv);
    MPI_Allreduce(MPI_IN_PLACE, maximum.data(), static_cast<int>(maximum.size()),
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, minimum.data(), static_cast<int>(minimum.size()),
                  MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    for (std::size_t v = 0; v < pv.size(); ++v) {
        EXPECT_EQ(maximum[v], minimum[v]) << "vertex " << v;
    }
}
