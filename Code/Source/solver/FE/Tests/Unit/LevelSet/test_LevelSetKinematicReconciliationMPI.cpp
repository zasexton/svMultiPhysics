#include "LevelSet/LevelSetKinematicReconciliation.h"

#include "Assembly/Assembler.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/H1Space.h"
#include "Spaces/SpaceFactory.h"

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
    FE::dofs::DofHandler velocity;

    Layout(int n, MPI_Comm communicator, int rank, int size)
        : mesh(n, rank, size)
    {
        FE::spaces::H1Space scalar(FE::ElementType::Triangle3, /*order=*/1);
        phi.distributeDofs(mesh.topology(), scalar, dofOptions(communicator, rank, size));
        phi.finalize();
        const auto vector_space = FE::spaces::VectorSpace(
            FE::spaces::SpaceType::H1, FE::ElementType::Triangle3, /*order=*/1, /*components=*/2);
        velocity.distributeDofs(mesh.topology(), *vector_space,
                                dofOptions(communicator, rank, size));
        velocity.finalize();
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

    [[nodiscard]] std::vector<FE::Real> vector(
        const std::function<std::array<FE::Real, 2>(const std::array<FE::Real, 3>&)>& f) const
    {
        std::vector<FE::Real> out(static_cast<std::size_t>(velocity.getNumDofs()), 0.0);
        const auto* map = velocity.getEntityDofMap();
        for (FE::GlobalIndex v = 0; v < mesh.numVertices(); ++v) {
            const auto value = f(mesh.getNodeCoordinates(v));
            const auto dofs = map->getVertexDofs(v);
            out[static_cast<std::size_t>(dofs[0])] = value[0];
            out[static_cast<std::size_t>(dofs[1])] = value[1];
        }
        return out;
    }

    // Reconciled values by vertex (independent of the DOF numbering).
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

FE::Real ellipse(const std::array<FE::Real, 3>& x)
{
    const FE::Real dx = (x[0] - 0.513) / 0.31;
    const FE::Real dy = (x[1] - 0.472) / 0.22;
    return 0.25 * (std::sqrt(dx * dx + dy * dy) - 1.0);
}

} // namespace

TEST(LevelSetKinematicReconciliationMPI, PartitionedResultMatchesSerialReference)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    constexpr int n = 12;
    const Layout parallel(n, MPI_COMM_WORLD, rank, size);
    const Layout serial(n, MPI_COMM_SELF, 0, 1);

    const FE::Real dt = 2.0e-3;
    const auto rotation = [](const std::array<FE::Real, 3>& x) {
        return std::array<FE::Real, 2>{-1.3 * (x[1] - 0.5), 1.3 * (x[0] - 0.5)};
    };
    // A transported state with a local, sign-preserving inconsistency.
    const auto transported_field = [](const std::array<FE::Real, 3>& x) {
        return ellipse(x) + (x[0] > 0.6 ? 0.002 : 0.0) - 0.001 * x[1];
    };

    const auto run = [&](const Layout& layout, std::vector<FE::Real>& reconciled) {
        return level_set::reconcileLevelSetWithKinematicFlux(
            layout.mesh, layout.phi, layout.velocity, 0.0, 1.0e-12, dt,
            layout.scalar(ellipse), layout.scalar(transported_field),
            layout.vector(rotation), layout.vector(rotation), reconciled);
    };
    std::vector<FE::Real> parallel_values;
    std::vector<FE::Real> serial_values;
    const auto p = run(parallel, parallel_values);
    const auto s = run(serial, serial_values);
    ASSERT_TRUE(p.success) << p.diagnostic;
    ASSERT_TRUE(s.success) << s.diagnostic;
    EXPECT_TRUE(p.applied);
    EXPECT_EQ(p.iterations, s.iterations);
    EXPECT_EQ(p.previous_interface_cells, s.previous_interface_cells);
    EXPECT_EQ(p.current_interface_cells, s.current_interface_cells);
    EXPECT_EQ(p.corrected_dofs, s.corrected_dofs);
    EXPECT_EQ(p.sign_preserving_skipped_dofs, s.sign_preserving_skipped_dofs);
    EXPECT_EQ(p.sign_preserving_redistributed_dofs, s.sign_preserving_redistributed_dofs);
    EXPECT_NEAR(p.previous_negative_volume, s.previous_negative_volume, 1.0e-14);
    EXPECT_NEAR(p.transported_negative_volume, s.transported_negative_volume, 1.0e-14);
    EXPECT_NEAR(p.reconciled_negative_volume, s.reconciled_negative_volume, 1.0e-14);
    EXPECT_NEAR(p.kinematic_volume_change, s.kinematic_volume_change, 1.0e-15);
    EXPECT_NEAR(p.reconciled_volume_error, s.reconciled_volume_error, 1.0e-14);
    EXPECT_LT(std::abs(s.reconciled_volume_error),
              1.0e-3 * std::abs(s.transported_volume_error));

    const auto pv = parallel.byVertex(parallel_values);
    const auto sv = serial.byVertex(serial_values);
    ASSERT_EQ(pv.size(), sv.size());
    for (std::size_t v = 0; v < pv.size(); ++v) {
        EXPECT_NEAR(pv[v], sv[v], 1.0e-14) << "vertex " << v;
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

TEST(LevelSetKinematicReconciliationMPI, BlockedNodeRedistributionMatchesSerialReference)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    constexpr int n = 12;
    const Layout parallel(n, MPI_COMM_WORLD, rank, size);
    const Layout serial(n, MPI_COMM_SELF, 0, 1);
    const auto zero = [](const std::array<FE::Real, 3>&) {
        return std::array<FE::Real, 2>{0.0, 0.0};
    };
    // The column x = 0.5 changes sign in the transported state, so its share
    // is redistributed to the cells around it, across the partition.
    const auto run = [&](const Layout& layout, std::vector<FE::Real>& reconciled) {
        return level_set::reconcileLevelSetWithKinematicFlux(
            layout.mesh, layout.phi, layout.velocity, 0.0, 1.0e-12, 1.0,
            layout.scalar([](const auto& x) { return x[0] - 0.51; }),
            layout.scalar([](const auto& x) { return x[0] - 0.49; }),
            layout.vector(zero), layout.vector(zero), reconciled);
    };
    std::vector<FE::Real> parallel_values;
    std::vector<FE::Real> serial_values;
    const auto p = run(parallel, parallel_values);
    const auto s = run(serial, serial_values);
    ASSERT_TRUE(p.success) << p.diagnostic;
    ASSERT_TRUE(s.success) << s.diagnostic;
    EXPECT_EQ(s.sign_preserving_redistributed_dofs, static_cast<std::size_t>(n + 1));
    EXPECT_EQ(p.sign_preserving_redistributed_dofs, s.sign_preserving_redistributed_dofs);
    EXPECT_EQ(p.sign_preserving_skipped_dofs, s.sign_preserving_skipped_dofs);
    EXPECT_EQ(p.iterations, s.iterations);
    EXPECT_NEAR(p.reconciled_negative_volume, s.reconciled_negative_volume, 1.0e-14);
    const auto pv = parallel.byVertex(parallel_values);
    const auto sv = serial.byVertex(serial_values);
    for (std::size_t v = 0; v < pv.size(); ++v) {
        EXPECT_NEAR(pv[v], sv[v], 1.0e-14) << "vertex " << v;
    }
}
