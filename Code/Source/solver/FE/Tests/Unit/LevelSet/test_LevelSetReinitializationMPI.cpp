#include "LevelSet/LevelSetReinitialization.h"

#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

[[nodiscard]] std::shared_ptr<svmp::Mesh> makeTwoRankQuadPartition(
    int rank,
    MPI_Comm communicator)
{
    int communicator_size = 1;
    MPI_Comm_size(communicator, &communicator_size);
    auto base = std::make_shared<svmp::MeshBase>();
    const svmp::real_t x0 = static_cast<svmp::real_t>(rank);
    const svmp::real_t x1 = static_cast<svmp::real_t>(rank + 1);
    base->build_from_arrays(
        /*spatial_dim=*/2,
        std::vector<svmp::real_t>{x0, 0.0, x1, 0.0,
                                  x1, 1.0, x0, 1.0},
        std::vector<svmp::offset_t>{0, 4},
        std::vector<svmp::index_t>{0, 1, 2, 3},
        std::vector<svmp::CellShape>{svmp::CellShape{
            .family = svmp::CellFamily::Quad,
            .num_corners = 4,
            .order = 1}});
    base->set_vertex_gids(
        std::vector<svmp::gid_t>{
            static_cast<svmp::gid_t>(rank),
            static_cast<svmp::gid_t>(rank + 1),
            static_cast<svmp::gid_t>(rank + communicator_size + 2),
            static_cast<svmp::gid_t>(rank + communicator_size + 1)});
    base->set_cell_gids({static_cast<svmp::gid_t>(10 + rank)});
    base->finalize();
    return svmp::create_mesh(std::move(base), svmp::MeshComm(communicator));
}

[[nodiscard]] FE::systems::SetupOptions mpiSetupOptions(
    int rank,
    int size,
    MPI_Comm communicator)
{
    FE::systems::SetupOptions options;
    options.dof_options.global_numbering =
        FE::dofs::GlobalNumberingMode::OwnerContiguous;
    options.dof_options.ownership = FE::dofs::OwnershipStrategy::LowestRank;
    options.dof_options.my_rank = rank;
    options.dof_options.world_size = size;
    options.dof_options.mpi_comm = communicator;
    return options;
}

TEST(LevelSetReinitializationMPI,
     ProjectionUsesGlobalOwnerStateAndInterfaceSnapshot)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_GE(size, 2) << "This partition test requires at least two ranks.";

    auto mesh = makeTwoRankQuadPartition(rank, MPI_COMM_WORLD);
    auto space = std::make_shared<FE::spaces::H1Space>(
        FE::ElementType::Quad4, /*order=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = space, .components = 1});
    system.setup(mpiSetupOptions(rank, size, MPI_COMM_WORLD));

    const auto count =
        static_cast<std::size_t>(system.dofHandler().getNumDofs());
    ASSERT_GE(count, 4u);
    std::vector<FE::Real> local_x_sum(count, 0.0);
    std::vector<int> local_x_count(count, 0);
    const auto& mesh_access = system.meshAccess();
    const auto& field_dofs = system.fieldDofHandler(phi);
    std::vector<std::array<FE::Real, 3>> coordinates;
    mesh_access.forEachCell([&](FE::GlobalIndex cell) {
        mesh_access.getCellCoordinates(cell, coordinates);
        const auto dofs = field_dofs.getCellDofs(cell);
        ASSERT_EQ(dofs.size(), coordinates.size());
        for (std::size_t local = 0; local < dofs.size(); ++local) {
            const auto dof = static_cast<std::size_t>(dofs[local]);
            ASSERT_LT(dof, count);
            local_x_sum[dof] += coordinates[local][0];
            ++local_x_count[dof];
        }
    });

    std::vector<FE::Real> global_x_sum(count, 0.0);
    std::vector<int> global_x_count(count, 0);
    MPI_Allreduce(local_x_sum.data(),
                  global_x_sum.data(),
                  static_cast<int>(count),
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(local_x_count.data(),
                  global_x_count.data(),
                  static_cast<int>(count),
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    std::vector<FE::Real> expected_distance(count, 0.0);
    std::vector<FE::Real> accepted_state(count, 0.0);
    for (std::size_t i = 0; i < count; ++i) {
        ASSERT_GT(global_x_count[i], 0);
        const auto x = global_x_sum[i] /
                       static_cast<FE::Real>(global_x_count[i]);
        expected_distance[i] = x - FE::Real{0.5};
        accepted_state[i] = FE::Real{2.0} * expected_distance[i];
    }
    const auto accepted_state_before = accepted_state;
    std::vector<FE::Real> candidate{FE::Real{91.0},
                                    FE::Real{-17.0},
                                    FE::Real{42.0}};

    level_set::LevelSetReinitializationOptions options;
    options.max_iterations = 2;
    options.pseudo_time_step_scale = 1.0;
    options.interface_band_width = 4.0;
    options.signed_distance_tolerance = 1.0e-12;
    options.max_zero_set_displacement = 1.0e-12;
    std::vector<level_set::LevelSetWallContactConstraint>
        local_wall_constraints;
    if (rank == 0) {
        local_wall_constraints.push_back(
            level_set::LevelSetWallContactConstraint{
                .kind = level_set::LevelSetWallContactConstraintKind::
                    AcceptedDynamicAngle,
                .interface_marker = 81,
                .boundary_marker = 6,
                .parent_cell_global_id = 10,
                .geometry_revision = 27u,
            });
    }
    const auto result = level_set::repairLevelSetSignedDistanceByProjection(
        system,
        phi,
        options,
        accepted_state,
        candidate,
        local_wall_constraints);

    EXPECT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.converged) << result.diagnostic;
    EXPECT_TRUE(result.zero_set_bound_satisfied);
    EXPECT_EQ(result.cut_cells, 1u);
    EXPECT_EQ(result.interface_fragments, 1u);
    EXPECT_EQ(result.wall_contact_constraints, 1u);
    EXPECT_EQ(result.wall_contact_cells, 1u);
    EXPECT_EQ(result.wall_contact_dofs, 4u);
    EXPECT_TRUE(result.wall_contact_constraints_satisfied);
    EXPECT_DOUBLE_EQ(result.max_contact_line_displacement, 0.0);
    EXPECT_DOUBLE_EQ(result.max_contact_angle_change_radians, 0.0);
    EXPECT_EQ(accepted_state, accepted_state_before);
    ASSERT_EQ(candidate.size(), expected_distance.size());
    for (std::size_t i = 0; i < candidate.size(); ++i) {
        EXPECT_NEAR(candidate[i], expected_distance[i], 1.0e-12);
    }

    const auto local_max_error = [&]() {
        FE::Real value = 0.0;
        for (std::size_t i = 0; i < candidate.size(); ++i) {
            value = std::max(value,
                             std::abs(candidate[i] - expected_distance[i]));
        }
        return value;
    }();
    FE::Real global_max_error = 0.0;
    MPI_Allreduce(&local_max_error,
                  &global_max_error,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  MPI_COMM_WORLD);
    EXPECT_LE(global_max_error, 1.0e-12);
}

TEST(LevelSetReinitializationMPI,
     PrescribedAngleProjectionIsPartitionInvariant)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_GE(size, 2) << "This partition test requires at least two ranks.";

    auto mesh = makeTwoRankQuadPartition(rank, MPI_COMM_WORLD);
    auto space = std::make_shared<FE::spaces::H1Space>(
        FE::ElementType::Quad4, /*order=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = space, .components = 1});
    system.setup(mpiSetupOptions(rank, size, MPI_COMM_WORLD));

    const auto count =
        static_cast<std::size_t>(system.dofHandler().getNumDofs());
    const auto& mesh_access = system.meshAccess();
    const auto& field_dofs = system.fieldDofHandler(phi);
    std::vector<FE::Real> local_coordinate_sum(2u * count, 0.0);
    std::vector<int> local_coordinate_count(count, 0);
    std::vector<std::array<FE::Real, 3>> coordinates;
    mesh_access.forEachCell([&](FE::GlobalIndex cell) {
        mesh_access.getCellCoordinates(cell, coordinates);
        const auto dofs = field_dofs.getCellDofs(cell);
        ASSERT_EQ(dofs.size(), coordinates.size());
        for (std::size_t local = 0; local < dofs.size(); ++local) {
            const auto dof = static_cast<std::size_t>(dofs[local]);
            ASSERT_LT(dof, count);
            local_coordinate_sum[2u * dof] += coordinates[local][0];
            local_coordinate_sum[2u * dof + 1u] +=
                coordinates[local][1];
            ++local_coordinate_count[dof];
        }
    });
    std::vector<FE::Real> global_coordinate_sum(
        local_coordinate_sum.size(), 0.0);
    std::vector<int> global_coordinate_count(count, 0);
    MPI_Allreduce(local_coordinate_sum.data(),
                  global_coordinate_sum.data(),
                  static_cast<int>(global_coordinate_sum.size()),
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(local_coordinate_count.data(),
                  global_coordinate_count.data(),
                  static_cast<int>(count),
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    const FE::Real pi = std::acos(FE::Real{-1.0});
    const FE::Real target_angle = pi / FE::Real{3.0};
    const FE::Real initial_angle = pi / FE::Real{6.0};
    constexpr FE::Real contact_x = 0.35;
    std::vector<FE::Real> accepted_state(count, 0.0);
    std::vector<std::array<FE::Real, 2>> global_coordinates(count);
    for (std::size_t i = 0; i < count; ++i) {
        ASSERT_GT(global_coordinate_count[i], 0);
        const FE::Real inverse_count =
            FE::Real{1.0} /
            static_cast<FE::Real>(global_coordinate_count[i]);
        global_coordinates[i] = {{
            global_coordinate_sum[2u * i] * inverse_count,
            global_coordinate_sum[2u * i + 1u] * inverse_count}};
        accepted_state[i] = FE::Real{5.0} *
                            (std::sin(initial_angle) *
                                 (global_coordinates[i][0] - contact_x) +
                             std::cos(initial_angle) *
                                 global_coordinates[i][1]);
    }

    level_set::LevelSetReinitializationOptions options;
    options.max_iterations = 2;
    options.pseudo_time_step_scale = 1.0;
    options.interface_band_width = 4.0;
    options.signed_distance_tolerance = 1.0e-12;
    options.max_zero_set_displacement = 1.0;
    std::vector<level_set::LevelSetWallContactConstraint>
        local_wall_constraints;
    if (rank == 0) {
        local_wall_constraints.push_back(
            level_set::LevelSetWallContactConstraint{
                .kind = level_set::LevelSetWallContactConstraintKind::
                    PrescribedAngle,
                .interface_marker = 82,
                .boundary_marker = 7,
                .parent_cell_global_id = 10,
                .geometry_revision = 28u,
                .target_angle_radians = target_angle,
                .physical_wall_normal = {{0.0, -1.0, 0.0}},
                .accepted_contact_point = {{contact_x, 0.0, 0.0}},
                .accepted_contact_line_tangent = {{0.0, 0.0, 1.0}},
            });
    }
    std::vector<FE::Real> candidate;
    const auto result = level_set::repairLevelSetSignedDistanceByProjection(
        system,
        phi,
        options,
        accepted_state,
        candidate,
        local_wall_constraints);

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_TRUE(result.converged) << result.diagnostic;
    EXPECT_TRUE(result.wall_contact_constraints_satisfied);
    EXPECT_EQ(result.wall_contact_constraints, 1u);
    ASSERT_EQ(candidate.size(), count);

    std::vector<FE::Real> coefficient_min(count, 0.0);
    std::vector<FE::Real> coefficient_max(count, 0.0);
    MPI_Allreduce(candidate.data(),
                  coefficient_min.data(),
                  static_cast<int>(count),
                  MPI_DOUBLE,
                  MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(candidate.data(),
                  coefficient_max.data(),
                  static_cast<int>(count),
                  MPI_DOUBLE,
                  MPI_MAX,
                  MPI_COMM_WORLD);
    FE::Real maximum_coefficient_difference = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        maximum_coefficient_difference = std::max(
            maximum_coefficient_difference,
            std::abs(coefficient_max[i] - coefficient_min[i]));
    }
    EXPECT_LE(maximum_coefficient_difference, 1.0e-12);

    const auto coefficient_at = [&](FE::Real x, FE::Real y) {
        for (std::size_t i = 0; i < count; ++i) {
            if (std::abs(global_coordinates[i][0] - x) <= 1.0e-12 &&
                std::abs(global_coordinates[i][1] - y) <= 1.0e-12) {
                return candidate[i];
            }
        }
        throw std::runtime_error(
            "partitioned prescribed-angle test could not find a coordinate");
    };
    const FE::Real origin = coefficient_at(0.0, 0.0);
    const FE::Real gx = coefficient_at(1.0, 0.0) - origin;
    const FE::Real gy = coefficient_at(0.0, 1.0) - origin;
    const FE::Real gradient_norm = std::hypot(gx, gy);
    ASSERT_GT(gradient_norm, 0.0);
    EXPECT_NEAR(-gy / gradient_norm,
                -std::cos(target_angle),
                1.0e-12);
    EXPECT_NEAR(-origin / gx, contact_x, 1.0e-12);
    RecordProperty("prescribed_target_mpi_max_coefficient_difference",
                   ::testing::PrintToString(
                       maximum_coefficient_difference));
}

} // namespace
