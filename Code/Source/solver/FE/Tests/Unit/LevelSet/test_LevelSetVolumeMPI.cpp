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

[[nodiscard]] std::vector<FE::Real> disconnectedCoefficients(
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
        coefficients[static_cast<std::size_t>(vertex_dofs.front())] =
            vertex == 1 || vertex == 3 ? FE::Real{-1.0}
                                       : FE::Real{1.0};
    }
    return coefficients;
}

[[nodiscard]] level_set::LevelSetGlobalShiftCorrectionOptions
correctionOptions()
{
    level_set::LevelSetGlobalShiftCorrectionOptions options;
    options.target_negative_volume = FE::Real{0.46};
    options.volume_tolerance = FE::Real{1.0e-12};
    options.max_iterations = 80;
    options.minimum_relative_volume_error = FE::Real{0.01};
    options.maximum_interface_displacement_fraction = FE::Real{0.2};
    return options;
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

void expectCorrectionMatches(
    const level_set::LevelSetGlobalShiftCorrectionResult& parallel,
    const level_set::LevelSetGlobalShiftCorrectionResult& serial)
{
    ASSERT_TRUE(parallel.success) << parallel.diagnostic;
    ASSERT_TRUE(serial.success) << serial.diagnostic;
    EXPECT_EQ(parallel.correction_triggered, serial.correction_triggered);
    EXPECT_EQ(parallel.correction_applied, serial.correction_applied);
    EXPECT_EQ(parallel.target_reached, serial.target_reached);
    EXPECT_EQ(parallel.limited_by_displacement_bound,
              serial.limited_by_displacement_bound);
    EXPECT_EQ(parallel.iterations, serial.iterations);
    EXPECT_NEAR(parallel.applied_shift, serial.applied_shift, 1.0e-13);
    EXPECT_NEAR(parallel.initial_negative_volume,
                serial.initial_negative_volume,
                1.0e-13);
    EXPECT_NEAR(parallel.corrected_negative_volume,
                serial.corrected_negative_volume,
                1.0e-13);
    EXPECT_NEAR(parallel.volume_error, serial.volume_error, 1.0e-13);
    EXPECT_NEAR(parallel.trigger_volume_error,
                serial.trigger_volume_error,
                1.0e-13);
    EXPECT_NEAR(parallel.minimum_edge_length,
                serial.minimum_edge_length,
                1.0e-13);
    EXPECT_NEAR(parallel.maximum_allowed_interface_displacement,
                serial.maximum_allowed_interface_displacement,
                1.0e-13);
    EXPECT_NEAR(parallel.maximum_topology_stable_shift,
                serial.maximum_topology_stable_shift,
                1.0e-13);
    EXPECT_NEAR(parallel.max_interface_displacement,
                serial.max_interface_displacement,
                1.0e-13);
    EXPECT_NEAR(parallel.max_contact_line_displacement,
                serial.max_contact_line_displacement,
                1.0e-13);
    EXPECT_NEAR(parallel.max_contact_angle_change_radians,
                serial.max_contact_angle_change_radians,
                1.0e-13);
    EXPECT_EQ(parallel.negative_component_topology_preserved,
              serial.negative_component_topology_preserved);
    ASSERT_EQ(parallel.negative_component_volume_transfers.size(),
              serial.negative_component_volume_transfers.size());
    for (std::size_t component = 0;
         component < parallel.negative_component_volume_transfers.size();
         ++component) {
        const auto& parallel_component =
            parallel.negative_component_volume_transfers[component];
        const auto& serial_component =
            serial.negative_component_volume_transfers[component];
        EXPECT_EQ(parallel_component.component_global_vertex_id,
                  serial_component.component_global_vertex_id);
        EXPECT_NEAR(parallel_component.initial_negative_volume,
                    serial_component.initial_negative_volume,
                    1.0e-13);
        EXPECT_NEAR(parallel_component.corrected_negative_volume,
                    serial_component.corrected_negative_volume,
                    1.0e-13);
        EXPECT_NEAR(parallel_component.volume_transfer,
                    serial_component.volume_transfer,
                    1.0e-13);
    }
    EXPECT_NEAR(parallel.total_component_volume_transfer,
                serial.total_component_volume_transfer,
                1.0e-13);
    EXPECT_NEAR(parallel.total_absolute_component_volume_transfer,
                serial.total_absolute_component_volume_transfer,
                1.0e-13);
    EXPECT_NEAR(parallel.maximum_absolute_component_volume_transfer,
                serial.maximum_absolute_component_volume_transfer,
                1.0e-13);
    expectVolumeMatches(parallel.initial_volume, serial.initial_volume);
    expectVolumeMatches(parallel.corrected_volume, serial.corrected_volume);

    EXPECT_TRUE(parallel.correction_triggered);
    EXPECT_TRUE(parallel.correction_applied);
    EXPECT_TRUE(parallel.target_reached);
    EXPECT_FALSE(parallel.limited_by_displacement_bound);
    EXPECT_NEAR(parallel.initial_negative_volume, 0.5, 1.0e-12);
    EXPECT_NEAR(parallel.applied_shift, 0.04, 1.0e-10);
    EXPECT_NEAR(parallel.corrected_negative_volume, 0.46, 1.0e-12);
    EXPECT_NEAR(parallel.minimum_edge_length, 1.0, 1.0e-13);
    EXPECT_NEAR(parallel.maximum_allowed_interface_displacement,
                0.2,
                1.0e-13);
    EXPECT_NEAR(parallel.max_interface_displacement, 0.04, 1.0e-10);
    EXPECT_NEAR(parallel.max_contact_line_displacement, 0.04, 1.0e-10);
    EXPECT_DOUBLE_EQ(parallel.max_contact_angle_change_radians, 0.0);
    EXPECT_TRUE(parallel.negative_component_topology_preserved);
    ASSERT_EQ(parallel.negative_component_volume_transfers.size(), 1u);
    EXPECT_EQ(parallel.negative_component_volume_transfers.front()
                  .component_global_vertex_id,
              0);
    EXPECT_NEAR(parallel.total_component_volume_transfer, -0.04, 1.0e-12);
    EXPECT_NEAR(parallel.total_absolute_component_volume_transfer,
                0.04,
                1.0e-12);
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

TEST(LevelSetVolumeMPI, CornerLinearizedCorrectionMatchesSerialReference)
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

    // The global error is below the relative fallback trigger even though the
    // two local partition errors are not.  This specifically proves that the
    // correction/retention decision uses global physical volume.
    auto retain_options = correctionOptions();
    retain_options.target_negative_volume = FE::Real{0.49};
    retain_options.minimum_relative_volume_error = FE::Real{0.02};
    std::vector<FE::Real> parallel_retained;
    std::vector<FE::Real> serial_retained;
    const auto parallel_retention = level_set::applyGlobalLevelSetShiftCorrection(
        parallel_mesh,
        parallel_dofs,
        volume_options,
        retain_options,
        parallel_coefficients,
        parallel_retained);
    const auto serial_retention = level_set::applyGlobalLevelSetShiftCorrection(
        serial_mesh,
        serial_dofs,
        volume_options,
        retain_options,
        serial_coefficients,
        serial_retained);
    ASSERT_TRUE(parallel_retention.success) << parallel_retention.diagnostic;
    ASSERT_TRUE(serial_retention.success) << serial_retention.diagnostic;
    EXPECT_EQ(parallel_retention.correction_triggered,
              serial_retention.correction_triggered);
    EXPECT_EQ(parallel_retention.correction_applied,
              serial_retention.correction_applied);
    EXPECT_FALSE(parallel_retention.correction_triggered);
    EXPECT_FALSE(parallel_retention.correction_applied);
    EXPECT_DOUBLE_EQ(parallel_retention.applied_shift, 0.0);
    EXPECT_NEAR(parallel_retention.trigger_volume_error,
                serial_retention.trigger_volume_error,
                1.0e-13);
    EXPECT_NEAR(parallel_retention.initial_negative_volume,
                serial_retention.initial_negative_volume,
                1.0e-13);
    EXPECT_EQ(parallel_retained, parallel_coefficients);
    EXPECT_EQ(serial_retained, serial_coefficients);

    std::vector<FE::Real> parallel_corrected;
    std::vector<FE::Real> serial_corrected;
    const auto parallel_correction = level_set::applyGlobalLevelSetShiftCorrection(
        parallel_mesh,
        parallel_dofs,
        volume_options,
        correctionOptions(),
        parallel_coefficients,
        parallel_corrected);
    const auto serial_correction = level_set::applyGlobalLevelSetShiftCorrection(
        serial_mesh,
        serial_dofs,
        volume_options,
        correctionOptions(),
        serial_coefficients,
        serial_corrected);
    expectCorrectionMatches(parallel_correction, serial_correction);
    ASSERT_EQ(parallel_corrected.size(), serial_corrected.size());
    for (std::size_t i = 0; i < parallel_corrected.size(); ++i) {
        EXPECT_NEAR(parallel_corrected[i], serial_corrected[i], 1.0e-13);
    }
}

TEST(LevelSetVolumeMPI, GeneratedQuadratureCorrectionMatchesSerialReference)
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

    std::vector<FE::Real> parallel_corrected;
    std::vector<FE::Real> serial_corrected;
    const auto parallel_correction = level_set::applyGlobalLevelSetShiftCorrection(
        parallel.system,
        parallel.phi,
        parallel_volume_options,
        correctionOptions(),
        parallel.solution,
        parallel_corrected);
    const auto serial_correction = level_set::applyGlobalLevelSetShiftCorrection(
        serial.system,
        serial.phi,
        serial_volume_options,
        correctionOptions(),
        serial.solution,
        serial_corrected);
    expectCorrectionMatches(parallel_correction, serial_correction);
    ASSERT_EQ(parallel_corrected.size(), serial_corrected.size());
    for (std::size_t i = 0; i < parallel_corrected.size(); ++i) {
        EXPECT_NEAR(parallel_corrected[i], serial_corrected[i], 1.0e-13);
    }
}

TEST(LevelSetVolumeMPI,
     DisconnectedComponentTransferLedgerMatchesSerialReference)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        GTEST_SKIP()
            << "This partition-equivalence fixture requires exactly two ranks.";
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
        disconnectedCoefficients(parallel_mesh, parallel_dofs);
    const auto serial_coefficients =
        disconnectedCoefficients(serial_mesh, serial_dofs);
    ASSERT_EQ(parallel_coefficients.size(), serial_coefficients.size());
    ASSERT_FALSE(parallel_coefficients.empty());
    auto target_coefficients = serial_coefficients;
    for (auto& value : target_coefficients) {
        value += FE::Real{0.2};
    }
    const level_set::LevelSetVolumeOptions volume_options{};
    const auto target = level_set::computeLevelSetCutCellVolume(
        serial_mesh, serial_dofs, volume_options, target_coefficients);
    ASSERT_TRUE(target.success) << target.diagnostic;
    auto options = correctionOptions();
    options.target_negative_volume = target.negative_volume;
    options.minimum_relative_volume_error = FE::Real{0.0};

    std::vector<FE::Real> parallel_corrected;
    std::vector<FE::Real> serial_corrected;
    const auto parallel = level_set::applyGlobalLevelSetShiftCorrection(
        parallel_mesh,
        parallel_dofs,
        volume_options,
        options,
        parallel_coefficients,
        parallel_corrected);
    const auto serial = level_set::applyGlobalLevelSetShiftCorrection(
        serial_mesh,
        serial_dofs,
        volume_options,
        options,
        serial_coefficients,
        serial_corrected);

    ASSERT_TRUE(parallel.success) << parallel.diagnostic;
    ASSERT_TRUE(serial.success) << serial.diagnostic;
    EXPECT_NEAR(parallel.applied_shift, 0.2, 1.0e-10);
    EXPECT_NEAR(parallel.applied_shift, serial.applied_shift, 1.0e-13);
    EXPECT_TRUE(parallel.negative_component_topology_preserved);
    EXPECT_TRUE(serial.negative_component_topology_preserved);
    ASSERT_EQ(parallel.negative_component_volume_transfers.size(), 2u);
    ASSERT_EQ(parallel.negative_component_volume_transfers.size(),
              serial.negative_component_volume_transfers.size());
    for (std::size_t component = 0;
         component < parallel.negative_component_volume_transfers.size();
         ++component) {
        const auto& parallel_component =
            parallel.negative_component_volume_transfers[component];
        const auto& serial_component =
            serial.negative_component_volume_transfers[component];
        EXPECT_EQ(parallel_component.component_global_vertex_id,
                  component == 0u ? 1 : 3);
        EXPECT_EQ(parallel_component.component_global_vertex_id,
                  serial_component.component_global_vertex_id);
        EXPECT_NEAR(parallel_component.initial_negative_volume,
                    serial_component.initial_negative_volume,
                    1.0e-13);
        EXPECT_NEAR(parallel_component.corrected_negative_volume,
                    serial_component.corrected_negative_volume,
                    1.0e-13);
        EXPECT_NEAR(parallel_component.volume_transfer,
                    serial_component.volume_transfer,
                    1.0e-13);
    }
    EXPECT_NEAR(parallel.total_component_volume_transfer,
                serial.total_component_volume_transfer,
                1.0e-13);
    EXPECT_NEAR(parallel.total_absolute_component_volume_transfer,
                serial.total_absolute_component_volume_transfer,
                1.0e-13);
    EXPECT_NEAR(parallel.maximum_absolute_component_volume_transfer,
                serial.maximum_absolute_component_volume_transfer,
                1.0e-13);
    ASSERT_EQ(parallel_corrected.size(), serial_corrected.size());
    for (std::size_t dof = 0; dof < parallel_corrected.size(); ++dof) {
        EXPECT_NEAR(parallel_corrected[dof], serial_corrected[dof], 1.0e-13);
    }
}
