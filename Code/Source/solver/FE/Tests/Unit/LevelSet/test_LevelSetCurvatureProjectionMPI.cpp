#include "LevelSet/LevelSetCurvatureProjection.h"

#include "Assembly/Assembler.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/H1Space.h"
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
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

class PartitionedTriangleCurvatureMeshAccess final
    : public FE::assembly::IMeshAccess {
public:
    struct Cell {
        FE::GlobalIndex global_id{-1};
        int owner_rank{0};
        std::array<FE::GlobalIndex, 3> nodes{};
    };

    struct BoundaryFace {
        FE::GlobalIndex global_id{-1};
        FE::GlobalIndex parent_cell{-1};
        FE::LocalIndex local_face{FE::INVALID_LOCAL_INDEX};
        int marker{-1};
    };

    PartitionedTriangleCurvatureMeshAccess(
        int subdivisions,
        int rank,
        int size,
        bool reverse_local_numbering)
        : subdivisions_(subdivisions), rank_(rank), size_(size)
    {
        if (subdivisions_ < 2 || rank_ < 0 || size_ < 1 ||
            rank_ >= size_) {
            throw std::invalid_argument(
                "invalid partitioned curvature test mesh options");
        }
        const auto side = subdivisions_ + 1;
        const auto vertex_count = side * side;
        coordinates_.resize(static_cast<std::size_t>(vertex_count));
        vertex_global_ids_.resize(static_cast<std::size_t>(vertex_count));
        const FE::Real minimum{-0.70};
        const FE::Real h = FE::Real{1.40} /
                           static_cast<FE::Real>(subdivisions_);
        const auto local_vertex = [&](FE::GlobalIndex global_vertex) {
            return reverse_local_numbering
                ? static_cast<FE::GlobalIndex>(vertex_count - 1) -
                      global_vertex
                : global_vertex;
        };
        for (int j = 0; j <= subdivisions_; ++j) {
            for (int i = 0; i <= subdivisions_; ++i) {
                const auto global_vertex = static_cast<FE::GlobalIndex>(
                    j * side + i);
                const auto local = local_vertex(global_vertex);
                vertex_global_ids_[static_cast<std::size_t>(local)] =
                    global_vertex;
                coordinates_[static_cast<std::size_t>(local)] = {{
                    minimum + static_cast<FE::Real>(i) * h,
                    minimum + static_cast<FE::Real>(j) * h,
                    FE::Real{0.0}}};
            }
        }

        std::vector<Cell> canonical_cells;
        canonical_cells.reserve(static_cast<std::size_t>(
            2 * subdivisions_ * subdivisions_));
        const auto global_vertex = [side](int i, int j) {
            return static_cast<FE::GlobalIndex>(j * side + i);
        };
        for (int j = 0; j < subdivisions_; ++j) {
            for (int i = 0; i < subdivisions_; ++i) {
                const auto v0 = global_vertex(i, j);
                const auto v1 = global_vertex(i + 1, j);
                const auto v2 = global_vertex(i + 1, j + 1);
                const auto v3 = global_vertex(i, j + 1);
                const auto lower_global = static_cast<FE::GlobalIndex>(
                    canonical_cells.size());
                canonical_cells.push_back({
                    lower_global,
                    size_ == 1
                        ? 0
                        : static_cast<int>(lower_global % size_),
                    {{local_vertex(v0),
                      local_vertex(v1),
                      local_vertex(v2)}}});
                const auto upper_global = static_cast<FE::GlobalIndex>(
                    canonical_cells.size());
                canonical_cells.push_back({
                    upper_global,
                    size_ == 1
                        ? 0
                        : static_cast<int>(upper_global % size_),
                    {{local_vertex(v0),
                      local_vertex(v2),
                      local_vertex(v3)}}});
            }
        }
        cells_ = std::move(canonical_cells);
        if (reverse_local_numbering) {
            std::reverse(cells_.begin(), cells_.end());
        }
        global_cell_to_local_.assign(cells_.size(), FE::GlobalIndex{-1});
        for (std::size_t cell = 0; cell < cells_.size(); ++cell) {
            global_cell_to_local_[
                static_cast<std::size_t>(cells_[cell].global_id)] =
                static_cast<FE::GlobalIndex>(cell);
        }

        for (int j = 0; j < subdivisions_; ++j) {
            for (int i = 0; i < subdivisions_; ++i) {
                const auto lower_global = static_cast<FE::GlobalIndex>(
                    2 * (j * subdivisions_ + i));
                const auto upper_global = lower_global + 1;
                if (j == 0) {
                    boundary_faces_.push_back({
                        static_cast<FE::GlobalIndex>(i),
                        global_cell_to_local_[
                            static_cast<std::size_t>(lower_global)],
                        FE::LocalIndex{0},
                        1});
                }
                if (i + 1 == subdivisions_) {
                    boundary_faces_.push_back({
                        static_cast<FE::GlobalIndex>(subdivisions_ + j),
                        global_cell_to_local_[
                            static_cast<std::size_t>(lower_global)],
                        FE::LocalIndex{1},
                        2});
                }
                if (j + 1 == subdivisions_) {
                    boundary_faces_.push_back({
                        static_cast<FE::GlobalIndex>(
                            2 * subdivisions_ + i),
                        global_cell_to_local_[
                            static_cast<std::size_t>(upper_global)],
                        FE::LocalIndex{1},
                        3});
                }
                if (i == 0) {
                    boundary_faces_.push_back({
                        static_cast<FE::GlobalIndex>(
                            3 * subdivisions_ + j),
                        global_cell_to_local_[
                            static_cast<std::size_t>(upper_global)],
                        FE::LocalIndex{2},
                        4});
                }
            }
        }
        if (reverse_local_numbering) {
            std::reverse(boundary_faces_.begin(), boundary_faces_.end());
        }
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override
    {
        return static_cast<FE::GlobalIndex>(cells_.size());
    }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override
    {
        return static_cast<FE::GlobalIndex>(std::count_if(
            cells_.begin(), cells_.end(), [&](const Cell& cell) {
                return cell.owner_rank == rank_;
            }));
    }
    [[nodiscard]] FE::GlobalIndex numVertices() const override
    {
        return static_cast<FE::GlobalIndex>(coordinates_.size());
    }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override
    {
        return static_cast<FE::GlobalIndex>(boundary_faces_.size());
    }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override
    {
        return 0;
    }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] int parallelRank() const override { return rank_; }
    [[nodiscard]] int parallelSize() const override { return size_; }
    [[nodiscard]] bool revisionTrackingAvailable() const override
    {
        return true;
    }
    [[nodiscard]] std::uint64_t geometryRevision() const override
    {
        return 7u;
    }
    [[nodiscard]] std::uint64_t topologyRevision() const override
    {
        return 11u;
    }
    [[nodiscard]] std::uint64_t ownershipRevision() const override
    {
        return 13u;
    }
    [[nodiscard]] std::uint64_t numberingRevision() const override
    {
        return 17u;
    }
    [[nodiscard]] bool cellIdsAreDense() const override { return true; }
    [[nodiscard]] bool globalEntityIdsAvailable() const override
    {
        return true;
    }
    [[nodiscard]] FE::GlobalIndex getCellGlobalId(
        FE::GlobalIndex cell) const override
    {
        return cells_.at(static_cast<std::size_t>(cell)).global_id;
    }
    [[nodiscard]] FE::GlobalIndex getBoundaryFaceGlobalId(
        FE::GlobalIndex face) const override
    {
        return boundary_faces_.at(static_cast<std::size_t>(face)).global_id;
    }
    [[nodiscard]] int getCellOwnerRank(
        FE::GlobalIndex cell) const override
    {
        return cells_.at(static_cast<std::size_t>(cell)).owner_rank;
    }
    [[nodiscard]] int getBoundaryFaceOwnerRank(
        FE::GlobalIndex face, FE::GlobalIndex) const override
    {
        const auto parent = boundary_faces_.at(
            static_cast<std::size_t>(face)).parent_cell;
        return getCellOwnerRank(parent);
    }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex cell) const override
    {
        return getCellOwnerRank(cell) == rank_;
    }
    [[nodiscard]] FE::ElementType getCellType(
        FE::GlobalIndex) const override
    {
        return FE::ElementType::Triangle3;
    }
    void getCellNodes(FE::GlobalIndex cell,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        const auto& record = cells_.at(static_cast<std::size_t>(cell));
        nodes.assign(record.nodes.begin(), record.nodes.end());
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
        for (const auto node : nodes) {
            coordinates.push_back(getNodeCoordinates(node));
        }
    }
    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex face, FE::GlobalIndex cell) const override
    {
        const auto& record = boundary_faces_.at(
            static_cast<std::size_t>(face));
        return record.parent_cell == cell ? record.local_face
                                          : FE::INVALID_LOCAL_INDEX;
    }
    [[nodiscard]] int getBoundaryFaceMarker(
        FE::GlobalIndex face) const override
    {
        return boundary_faces_.at(static_cast<std::size_t>(face)).marker;
    }
    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex) const override
    {
        return {-1, -1};
    }
    void forEachCell(
        std::function<void(FE::GlobalIndex)> callback) const override
    {
        for (FE::GlobalIndex cell = 0; cell < numCells(); ++cell) {
            callback(cell);
        }
    }
    void forEachOwnedCell(
        std::function<void(FE::GlobalIndex)> callback) const override
    {
        for (FE::GlobalIndex cell = 0; cell < numCells(); ++cell) {
            if (isOwnedCell(cell)) {
                callback(cell);
            }
        }
    }
    void forEachBoundaryFace(
        int marker,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> callback)
        const override
    {
        for (std::size_t face = 0; face < boundary_faces_.size(); ++face) {
            const auto& record = boundary_faces_[face];
            if (isOwnedCell(record.parent_cell) &&
                (marker < 0 || marker == record.marker)) {
                callback(static_cast<FE::GlobalIndex>(face),
                         record.parent_cell);
            }
        }
    }
    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex,
                           FE::GlobalIndex)>) const override
    {
    }

    [[nodiscard]] FE::GlobalIndex vertexGlobalId(
        FE::GlobalIndex local_vertex) const
    {
        return vertex_global_ids_.at(
            static_cast<std::size_t>(local_vertex));
    }

private:
    int subdivisions_{0};
    int rank_{0};
    int size_{1};
    std::vector<std::array<FE::Real, 3>> coordinates_{};
    std::vector<FE::GlobalIndex> vertex_global_ids_{};
    std::vector<Cell> cells_{};
    std::vector<FE::GlobalIndex> global_cell_to_local_{};
    std::vector<BoundaryFace> boundary_faces_{};
};

[[nodiscard]] FE::systems::SetupInputs curvatureSetupInputs(
    const PartitionedTriangleCurvatureMeshAccess& mesh)
{
    FE::dofs::MeshTopologyInfo topology;
    topology.n_cells = mesh.numCells();
    topology.n_vertices = mesh.numVertices();
    topology.dim = 2;
    topology.cell2vertex_offsets.reserve(
        static_cast<std::size_t>(mesh.numCells() + 1));
    topology.cell2vertex_offsets.push_back(0);
    for (FE::GlobalIndex cell = 0; cell < mesh.numCells(); ++cell) {
        std::vector<FE::GlobalIndex> nodes;
        mesh.getCellNodes(cell, nodes);
        topology.cell2vertex_data.insert(
            topology.cell2vertex_data.end(), nodes.begin(), nodes.end());
        topology.cell2vertex_offsets.push_back(
            static_cast<FE::MeshOffset>(
                topology.cell2vertex_data.size()));
        topology.cell_gids.push_back(mesh.getCellGlobalId(cell));
        topology.cell_owner_ranks.push_back(
            mesh.getCellOwnerRank(cell));
    }
    for (FE::GlobalIndex vertex = 0; vertex < mesh.numVertices(); ++vertex) {
        topology.vertex_gids.push_back(mesh.vertexGlobalId(vertex));
    }
    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topology);
    return inputs;
}

[[nodiscard]] FE::systems::SetupOptions curvatureSetupOptions(
    int rank, int size, MPI_Comm communicator)
{
    FE::systems::SetupOptions options;
    options.dof_options.global_numbering =
        FE::dofs::GlobalNumberingMode::GlobalIds;
    options.dof_options.ownership =
        FE::dofs::OwnershipStrategy::LowestRank;
    options.dof_options.my_rank = rank;
    options.dof_options.world_size = size;
    options.dof_options.mpi_comm = communicator;
    return options;
}

struct CurvatureSystem {
    std::shared_ptr<PartitionedTriangleCurvatureMeshAccess> mesh{};
    std::unique_ptr<FE::systems::FESystem> system{};
    FE::FieldId phi_field{FE::INVALID_FIELD_ID};
    std::vector<FE::Real> phi{};
};

[[nodiscard]] CurvatureSystem makeCurvatureSystem(
    int subdivisions,
    int rank,
    int size,
    MPI_Comm communicator,
    bool reverse_local_numbering,
    FE::Real radius,
    FE::Real contact_angle)
{
    CurvatureSystem result;
    result.mesh =
        std::make_shared<PartitionedTriangleCurvatureMeshAccess>(
            subdivisions, rank, size, reverse_local_numbering);
    result.system =
        std::make_unique<FE::systems::FESystem>(result.mesh);
    result.phi_field = result.system->addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = std::make_shared<FE::spaces::H1Space>(
            FE::ElementType::Triangle3, /*order=*/1),
        .components = 1,
    });
    result.system->setup(
        curvatureSetupOptions(rank, size, communicator),
        curvatureSetupInputs(*result.mesh));
    result.phi.resize(
        static_cast<std::size_t>(result.mesh->numVertices()));
    const FE::Real wall_coordinate{-0.70};
    const FE::Real center_y =
        wall_coordinate - radius * std::cos(contact_angle);
    for (FE::GlobalIndex vertex = 0;
         vertex < result.mesh->numVertices();
         ++vertex) {
        const auto point = result.mesh->getNodeCoordinates(vertex);
        result.phi[static_cast<std::size_t>(vertex)] =
            std::hypot(point[0], point[1] - center_y) - radius;
    }
    return result;
}

[[nodiscard]] std::vector<FE::Real> curvatureByGlobalVertex(
    const PartitionedTriangleCurvatureMeshAccess& mesh,
    const std::vector<FE::Real>& local_curvature)
{
    std::vector<FE::Real> global(local_curvature.size(), FE::Real{0.0});
    for (FE::GlobalIndex local = 0; local < mesh.numVertices(); ++local) {
        global[static_cast<std::size_t>(mesh.vertexGlobalId(local))] =
            local_curvature[static_cast<std::size_t>(local)];
    }
    return global;
}

TEST(LevelSetCurvatureProjectionMPI,
     MatchesSerialAcrossOwnershipAndLocalNumberingPermutations)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_EQ(size, 2) << "This test requires exactly two ranks.";

    constexpr int subdivisions{32};
    constexpr FE::Real radius{0.347};
    const FE::Real pi = std::acos(FE::Real{-1.0});
    const FE::Real contact_angle = pi / FE::Real{3.0};
    auto distributed = makeCurvatureSystem(
        subdivisions,
        rank,
        size,
        MPI_COMM_WORLD,
        /*reverse_local_numbering=*/rank == 1,
        radius,
        contact_angle);
    auto serial = makeCurvatureSystem(
        subdivisions,
        /*rank=*/0,
        /*size=*/1,
        MPI_COMM_SELF,
        /*reverse_local_numbering=*/false,
        radius,
        contact_angle);

    level_set::LevelSetCurvatureProjectionOptions options;
    options.recovery_mode =
        level_set::LevelSetCurvatureRecoveryMode::KinematicAreaGradient;
    options.kinematic_area_gradient_young_walls.push_back(
        {1, contact_angle});
    std::vector<FE::Real> distributed_curvature;
    const auto distributed_result =
        level_set::projectLevelSetMeanCurvatureToVertices(
            *distributed.system,
            distributed.phi_field,
            distributed.phi,
            std::span<const level_set::LevelSetCurvatureProjectionSample>{},
            options,
            distributed_curvature);
    std::vector<FE::Real> serial_curvature;
    const auto serial_result =
        level_set::projectLevelSetMeanCurvatureToVertices(
            *serial.system,
            serial.phi_field,
            serial.phi,
            std::span<const level_set::LevelSetCurvatureProjectionSample>{},
            options,
            serial_curvature);
    ASSERT_TRUE(distributed_result.success)
        << distributed_result.diagnostic;
    ASSERT_TRUE(serial_result.success) << serial_result.diagnostic;
    EXPECT_TRUE(distributed_result
                    .kinematic_area_gradient_derivatives_global_dof_order);
    EXPECT_TRUE(serial_result
                    .kinematic_area_gradient_derivatives_global_dof_order);
    ASSERT_EQ(
        distributed_result
            .kinematic_area_gradient_total_energy_derivative.size(),
        serial_result
            .kinematic_area_gradient_total_energy_derivative.size());
    ASSERT_EQ(
        distributed_result
            .kinematic_area_gradient_liquid_volume_derivative.size(),
        serial_result
            .kinematic_area_gradient_liquid_volume_derivative.size());
    EXPECT_EQ(
        distributed_result
            .kinematic_area_gradient_total_energy_derivative,
        serial_result
            .kinematic_area_gradient_total_energy_derivative);
    EXPECT_EQ(
        distributed_result
            .kinematic_area_gradient_liquid_volume_derivative,
        serial_result
            .kinematic_area_gradient_liquid_volume_derivative);
    EXPECT_TRUE(distributed_result
                    .kinematic_area_gradient_collective_replication);
    EXPECT_EQ(distributed_result.kinematic_area_gradient_parallel_size, 2);
    EXPECT_EQ(
        distributed_result.kinematic_area_gradient_gathered_owned_cells,
        static_cast<std::size_t>(2 * subdivisions * subdivisions));
    EXPECT_EQ(distributed_result
                  .kinematic_area_gradient_gathered_owned_boundary_faces,
              static_cast<std::size_t>(4 * subdivisions));
    EXPECT_EQ(distributed_result
                  .kinematic_area_gradient_young_wall_boundary_faces,
              static_cast<std::size_t>(subdivisions));
    EXPECT_EQ(distributed_result
                  .kinematic_area_gradient_young_wall_cut_faces,
              2u);
    EXPECT_EQ(distributed_result
                  .kinematic_area_gradient_young_wall_measure_evaluations,
              24u);

    const auto distributed_global = curvatureByGlobalVertex(
        *distributed.mesh, distributed_curvature);
    const auto serial_global = curvatureByGlobalVertex(
        *serial.mesh, serial_curvature);
    ASSERT_EQ(distributed_global.size(), serial_global.size());
    FE::Real maximum_serial_difference{0.0};
    for (std::size_t vertex = 0; vertex < serial_global.size(); ++vertex) {
        maximum_serial_difference = std::max(
            maximum_serial_difference,
            std::abs(distributed_global[vertex] -
                     serial_global[vertex]));
    }
    EXPECT_DOUBLE_EQ(maximum_serial_difference, FE::Real{0.0});

    const FE::Real exact = FE::Real{1.0} / radius;
    const FE::Real weighted_mean_error =
        distributed_result
            .kinematic_area_gradient_mass_weighted_mean_curvature -
        exact;
    const FE::Real weighted_rms_error = std::sqrt(
        weighted_mean_error * weighted_mean_error +
        distributed_result
                .kinematic_area_gradient_mass_weighted_rms_deviation *
            distributed_result
                .kinematic_area_gradient_mass_weighted_rms_deviation);
    RecordProperty("kinematic_collective_sessile_60_rms_error_n32",
                   weighted_rms_error);
    RecordProperty("kinematic_collective_max_serial_difference",
                   maximum_serial_difference);
    EXPECT_LT(weighted_rms_error, FE::Real{0.08});

    FE::Real minimum_mean = distributed_result
        .kinematic_area_gradient_mass_weighted_mean_curvature;
    FE::Real maximum_mean = minimum_mean;
    MPI_Allreduce(MPI_IN_PLACE, &minimum_mean, 1,
                  MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_mean, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_DOUBLE_EQ(minimum_mean, maximum_mean);
    std::vector<FE::Real> minimum_curvature(distributed_global.size());
    std::vector<FE::Real> maximum_curvature(distributed_global.size());
    MPI_Allreduce(distributed_global.data(), minimum_curvature.data(),
                  static_cast<int>(distributed_global.size()),
                  MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(distributed_global.data(), maximum_curvature.data(),
                  static_cast<int>(distributed_global.size()),
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(minimum_curvature, maximum_curvature);
}

TEST(LevelSetCurvatureProjectionMPI,
     RejectsRankAsymmetricRecoveryOptionsCollectively)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_EQ(size, 2) << "This test requires exactly two ranks.";
    const FE::Real pi = std::acos(FE::Real{-1.0});
    auto distributed = makeCurvatureSystem(
        /*subdivisions=*/12,
        rank,
        size,
        MPI_COMM_WORLD,
        /*reverse_local_numbering=*/rank == 1,
        FE::Real{0.347},
        pi / FE::Real{3.0});
    level_set::LevelSetCurvatureProjectionOptions options;
    options.recovery_mode =
        level_set::LevelSetCurvatureRecoveryMode::KinematicAreaGradient;
    options.kinematic_area_gradient_filter_coefficient =
        rank == 0 ? FE::Real{1.0} : FE::Real{0.75};
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        *distributed.system,
        distributed.phi_field,
        distributed.phi,
        std::span<const level_set::LevelSetCurvatureProjectionSample>{},
        options,
        curvature);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("identical options"),
              std::string::npos)
        << result.diagnostic;
    int diagnostic_length = static_cast<int>(result.diagnostic.size());
    int minimum_length = diagnostic_length;
    int maximum_length = diagnostic_length;
    MPI_Allreduce(MPI_IN_PLACE, &minimum_length, 1,
                  MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_length, 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(minimum_length, maximum_length);
    RecordProperty("kinematic_collective_asymmetric_option_rejection_count",
                   1);
    RecordProperty("kinematic_collective_asymmetric_option_diagnostic_length",
                   diagnostic_length);
}

TEST(LevelSetCurvatureProjectionMPI,
     RejectsRankAsymmetricSharedLevelSetValuesCollectively)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_EQ(size, 2) << "This test requires exactly two ranks.";
    constexpr int subdivisions{12};
    const FE::Real pi = std::acos(FE::Real{-1.0});
    auto distributed = makeCurvatureSystem(
        subdivisions,
        rank,
        size,
        MPI_COMM_WORLD,
        /*reverse_local_numbering=*/rank == 1,
        FE::Real{0.347},
        pi / FE::Real{3.0});
    if (rank == 1) {
        const auto side = subdivisions + 1;
        const auto center_global = static_cast<FE::GlobalIndex>(
            (subdivisions / 2) * side + subdivisions / 2);
        for (FE::GlobalIndex local = 0;
             local < distributed.mesh->numVertices();
             ++local) {
            if (distributed.mesh->vertexGlobalId(local) == center_global) {
                distributed.phi[static_cast<std::size_t>(local)] +=
                    FE::Real{1.0e-4};
                break;
            }
        }
    }

    level_set::LevelSetCurvatureProjectionOptions options;
    options.recovery_mode =
        level_set::LevelSetCurvatureRecoveryMode::KinematicAreaGradient;
    std::vector<FE::Real> curvature;
    const auto result = level_set::projectLevelSetMeanCurvatureToVertices(
        *distributed.system,
        distributed.phi_field,
        distributed.phi,
        std::span<const level_set::LevelSetCurvatureProjectionSample>{},
        options,
        curvature);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("inconsistent shared-node values"),
              std::string::npos)
        << result.diagnostic;
    int diagnostic_length = static_cast<int>(result.diagnostic.size());
    int minimum_length = diagnostic_length;
    int maximum_length = diagnostic_length;
    MPI_Allreduce(MPI_IN_PLACE, &minimum_length, 1,
                  MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_length, 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(minimum_length, maximum_length);
    RecordProperty(
        "kinematic_collective_asymmetric_level_set_rejection_count", 1);
    RecordProperty(
        "kinematic_collective_asymmetric_level_set_diagnostic_length",
        diagnostic_length);
}

} // namespace
