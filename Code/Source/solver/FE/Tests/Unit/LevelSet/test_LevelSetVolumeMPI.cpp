#include "LevelSet/LevelSetVolume.h"

#include "Assembly/Assembler.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/H1Space.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

class TwoTrianglePartitionMeshAccess final : public FE::assembly::IMeshAccess {
public:
    TwoTrianglePartitionMeshAccess(int rank, bool distributed)
        : rank_(rank)
        , distributed_(distributed)
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 2; }

    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override
    {
        return distributed_ ? (rank_ < 2 ? 1 : 0) : 2;
    }

    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 4; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 4; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 1; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool cellIdsAreDense() const override { return true; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override
    {
        return static_cast<std::uint64_t>(distributed_ ? 31 + rank_ : 31);
    }
    [[nodiscard]] std::uint64_t numberingRevision() const override { return 13; }
    [[nodiscard]] std::uint64_t fieldLayoutRevision() const override { return 17; }

    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex cell) const override
    {
        return !distributed_ || (rank_ < 2 && cell == rank_);
    }

    [[nodiscard]] FE::ElementType getCellType(
        FE::GlobalIndex /*cell*/) const override
    {
        return FE::ElementType::Triangle3;
    }

    void getCellNodes(FE::GlobalIndex cell,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        nodes = cell == 0 ? std::vector<FE::GlobalIndex>{0, 1, 2}
                          : std::vector<FE::GlobalIndex>{0, 2, 3};
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node) const override
    {
        return coordinates_.at(static_cast<std::size_t>(node));
    }

    void getCellCoordinates(
        FE::GlobalIndex cell,
        std::vector<std::array<FE::Real, 3>>& coordinates) const override
    {
        std::vector<FE::GlobalIndex> nodes;
        getCellNodes(cell, nodes);
        coordinates.clear();
        coordinates.reserve(nodes.size());
        for (const auto node : nodes) {
            coordinates.push_back(getNodeCoordinates(node));
        }
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex face,
        FE::GlobalIndex cell) const override
    {
        for (const auto& boundary : boundary_faces_) {
            if (boundary.face == face && boundary.cell == cell) {
                return boundary.local_face;
            }
        }
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(
        FE::GlobalIndex /*face*/) const override
    {
        return 1;
    }

    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex /*face*/) const override
    {
        return {0, 1};
    }

    void forEachCell(
        std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
        callback(1);
    }

    void forEachOwnedCell(
        std::function<void(FE::GlobalIndex)> callback) const override
    {
        if (distributed_) {
            if (rank_ < 2) {
                callback(rank_);
            }
            return;
        }
        callback(0);
        callback(1);
    }

    void forEachBoundaryFace(
        int marker,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> callback) const override
    {
        if (marker != -1 && marker != 1) {
            return;
        }
        for (const auto& boundary : boundary_faces_) {
            callback(boundary.face, boundary.cell);
        }
    }

    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex,
                           FE::GlobalIndex,
                           FE::GlobalIndex)> callback) const override
    {
        callback(4, 0, 1);
    }

private:
    struct BoundaryFace {
        FE::GlobalIndex face;
        FE::GlobalIndex cell;
        FE::LocalIndex local_face;
    };

    int rank_{0};
    bool distributed_{false};
    std::array<std::array<FE::Real, 3>, 4> coordinates_{{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}},
    }};
    std::array<BoundaryFace, 4> boundary_faces_{{
        {0, 0, 0},
        {1, 0, 1},
        {2, 1, 1},
        {3, 1, 2},
    }};
};

[[nodiscard]] FE::dofs::MeshTopologyInfo twoTriangleTopology(bool distributed)
{
    FE::dofs::MeshTopologyInfo topology;
    topology.n_cells = 2;
    topology.n_vertices = 4;
    topology.dim = 2;
    topology.cell2vertex_offsets = {0, 3, 6};
    topology.cell2vertex_data = {0, 1, 2, 0, 2, 3};
    topology.vertex_gids = {0, 1, 2, 3};
    topology.cell_gids = {10, 11};
    topology.cell_owner_ranks = distributed ? std::vector<int>{0, 1}
                                            : std::vector<int>{0, 0};
    return topology;
}

[[nodiscard]] FE::dofs::DofDistributionOptions dofOptions(
    MPI_Comm communicator,
    int rank,
    int size)
{
    FE::dofs::DofDistributionOptions options;
    options.global_numbering = FE::dofs::GlobalNumberingMode::GlobalIds;
    options.ownership = FE::dofs::OwnershipStrategy::LowestRank;
    options.my_rank = rank;
    options.world_size = size;
    options.mpi_comm = communicator;
    return options;
}

[[nodiscard]] std::vector<FE::Real> planeCoefficients(
    const FE::assembly::IMeshAccess& mesh,
    const FE::dofs::DofHandler& dofs)
{
    std::vector<FE::Real> coefficients(
        static_cast<std::size_t>(dofs.getNumDofs()), FE::Real{0.0});
    const auto* entity_map = dofs.getEntityDofMap();
    if (entity_map == nullptr) {
        return {};
    }
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        const auto vertex_dofs = entity_map->getVertexDofs(vertex);
        if (vertex_dofs.size() != 1u || vertex_dofs.front() < 0) {
            return {};
        }
        const auto x = mesh.getNodeCoordinates(vertex);
        coefficients[static_cast<std::size_t>(vertex_dofs.front())] =
            x[0] + FE::Real{0.5} * x[1] - FE::Real{0.75};
    }
    return coefficients;
}

void expectVolumeMatches(
    const level_set::LevelSetVolumeResult& parallel,
    const level_set::LevelSetVolumeResult& serial)
{
    ASSERT_TRUE(parallel.success) << parallel.diagnostic;
    ASSERT_TRUE(serial.success) << serial.diagnostic;
    EXPECT_EQ(parallel.cells, serial.cells);
    EXPECT_EQ(parallel.cut_cells, serial.cut_cells);
    EXPECT_EQ(parallel.full_negative_cells, serial.full_negative_cells);
    EXPECT_EQ(parallel.full_positive_cells, serial.full_positive_cells);
    EXPECT_NEAR(parallel.total_volume, serial.total_volume, 1.0e-13);
    EXPECT_NEAR(parallel.negative_volume, serial.negative_volume, 1.0e-13);
    EXPECT_NEAR(parallel.positive_volume, serial.positive_volume, 1.0e-13);
}

struct SystemFixture {
    std::shared_ptr<TwoTrianglePartitionMeshAccess> mesh;
    FE::systems::FESystem system;
    FE::FieldId phi{FE::INVALID_FIELD_ID};
    std::vector<FE::Real> solution;

    SystemFixture(int rank,
                  bool distributed,
                  MPI_Comm communicator,
                  int communicator_rank,
                  int communicator_size)
        : mesh(std::make_shared<TwoTrianglePartitionMeshAccess>(rank, distributed))
        , system(mesh)
    {
        auto space = FE::spaces::Space(FE::spaces::SpaceType::H1,
                                       mesh,
                                       /*order=*/1,
                                       /*components=*/1);
        phi = system.addField(FE::systems::FieldSpec{
            .name = "phi",
            .space = space,
            .components = 1,
        });
        FE::systems::SetupOptions setup_options;
        setup_options.dof_options = dofOptions(
            communicator, communicator_rank, communicator_size);
        FE::systems::SetupInputs setup_inputs;
        setup_inputs.topology_override = twoTriangleTopology(distributed);
        system.setup(setup_options, setup_inputs);

        solution.assign(
            static_cast<std::size_t>(system.dofHandler().getNumDofs()),
            FE::Real{0.0});
        const auto coefficients = planeCoefficients(
            *mesh, system.fieldDofHandler(phi));
        const auto offset = static_cast<std::size_t>(system.fieldDofOffset(phi));
        std::copy(coefficients.begin(),
                  coefficients.end(),
                  solution.begin() + static_cast<std::ptrdiff_t>(offset));
    }
};

} // namespace

TEST(LevelSetVolumeMPI, CornerLinearizedVolumeMatchesSerialReference)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        GTEST_SKIP() << "This partition-equivalence fixture requires exactly two ranks.";
    }

    const TwoTrianglePartitionMeshAccess parallel_mesh(rank, true);
    const TwoTrianglePartitionMeshAccess serial_mesh(0, false);
    FE::spaces::H1Space space(FE::ElementType::Triangle3, /*order=*/1);

    FE::dofs::DofHandler parallel_dofs;
    parallel_dofs.distributeDofs(twoTriangleTopology(true),
                                space,
                                dofOptions(MPI_COMM_WORLD, rank, size));
    parallel_dofs.finalize();

    FE::dofs::DofHandler serial_dofs;
    serial_dofs.distributeDofs(twoTriangleTopology(false),
                              space,
                              dofOptions(MPI_COMM_SELF, 0, 1));
    serial_dofs.finalize();

    const auto parallel_coefficients =
        planeCoefficients(parallel_mesh, parallel_dofs);
    const auto serial_coefficients = planeCoefficients(serial_mesh, serial_dofs);
    ASSERT_EQ(parallel_coefficients.size(), serial_coefficients.size());
    ASSERT_FALSE(parallel_coefficients.empty());

    const level_set::LevelSetVolumeOptions volume_options{};
    const auto parallel_volume = level_set::computeLevelSetCutCellVolume(
        parallel_mesh, parallel_dofs, volume_options, parallel_coefficients);
    const auto serial_volume = level_set::computeLevelSetCutCellVolume(
        serial_mesh, serial_dofs, volume_options, serial_coefficients);
    expectVolumeMatches(parallel_volume, serial_volume);
}

TEST(LevelSetVolumeMPI, GeneratedQuadratureVolumeMatchesSerialReference)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        GTEST_SKIP() << "This partition-equivalence fixture requires exactly two ranks.";
    }

    SystemFixture parallel(rank, true, MPI_COMM_WORLD, rank, size);
    SystemFixture serial(0, false, MPI_COMM_SELF, 0, 1);

    level_set::LevelSetVolumeOptions parallel_volume_options;
    parallel_volume_options.use_generated_interface_quadrature = true;
    parallel_volume_options.level_set_field_name = "phi";
    parallel_volume_options.generated_domain_id = "mpi_partition_volume";
    parallel_volume_options.requested_interface_marker = 913;
    parallel_volume_options.interface_quadrature_order = 1;
    parallel_volume_options.volume_quadrature_order = 1;

    auto serial_volume_options = parallel_volume_options;
    serial_volume_options.generated_domain_id = "serial_reference_volume";
    serial_volume_options.requested_interface_marker = 914;

    const auto parallel_volume = level_set::computeLevelSetCutCellVolume(
        parallel.system,
        parallel.phi,
        parallel_volume_options,
        parallel.solution);
    const auto serial_volume = level_set::computeLevelSetCutCellVolume(
        serial.system,
        serial.phi,
        serial_volume_options,
        serial.solution);
    expectVolumeMatches(parallel_volume, serial_volume);
}

