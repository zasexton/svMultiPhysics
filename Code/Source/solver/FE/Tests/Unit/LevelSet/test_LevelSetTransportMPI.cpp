#include "LevelSet/LevelSetTransport.h"

#include "Dofs/EntityDofMap.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;
constexpr FE::GlobalIndex kInvalidDof{-1};

FE::systems::SetupOptions mpiSetupOptions(
    int rank,
    int size,
    MPI_Comm communicator);

std::shared_ptr<svmp::Mesh> makeTwoRankQuadPartition(
    int rank,
    MPI_Comm communicator)
{
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
        rank == 0 ? std::vector<svmp::gid_t>{0, 1, 4, 3}
                  : std::vector<svmp::gid_t>{1, 2, 5, 4});
    base->set_cell_gids({static_cast<svmp::gid_t>(10 + rank)});
    base->finalize();
    const auto marker = static_cast<svmp::label_t>(rank == 0 ? 41 : 17);
    for (svmp::index_t face = 0;
         face < static_cast<svmp::index_t>(base->n_faces()); ++face) {
        base->set_boundary_label(face, marker);
    }
    return svmp::create_mesh(std::move(base), svmp::MeshComm(communicator));
}

TEST(LevelSetTransportMPI,
     WallSafetyReportsSameWorstBoundaryMarkerOnEveryRank)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_EQ(size, 2) << "This communicator test requires exactly two ranks.";

    auto mesh = makeTwoRankQuadPartition(rank, MPI_COMM_WORLD);
    auto space = std::make_shared<FE::spaces::H1Space>(
        FE::ElementType::Quad4, /*order=*/1);
    FE::systems::FESystem system(mesh);
    system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = space, .components = 1});
    system.setup(mpiSetupOptions(rank, size, MPI_COMM_WORLD));

    level_set::LevelSetVelocityOptions velocity;
    velocity.source = level_set::LevelSetVelocitySource::ConstantVector;
    velocity.constant_value = {0.0, 0.5, 0.0};
    level_set::LevelSetBoundPreservingOptions options;
    options.enabled = true;
    options.impermeable_normal_velocity_tolerance = 1.0e-12;
    FE::systems::SystemStateView state;
    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    state.u = solution;

    const auto safety = level_set::evaluateLevelSetTransportSafety(
        system,
        velocity,
        /*boundaries=*/{},
        options,
        state,
        /*dt=*/0.25);
    EXPECT_FALSE(safety.success);
    EXPECT_TRUE(safety.courant_satisfied);
    EXPECT_FALSE(safety.impermeable_boundaries_satisfied);
    EXPECT_EQ(safety.worst_boundary_marker, 17);
    EXPECT_NEAR(safety.maximum_boundary_normal_velocity, 0.5, 1.0e-13);

    int marker_min = 0;
    int marker_max = 0;
    MPI_Allreduce(&safety.worst_boundary_marker, &marker_min, 1,
                  MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&safety.worst_boundary_marker, &marker_max, 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(marker_min, marker_max);
}

FE::systems::SetupOptions mpiSetupOptions(
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

FE::GlobalIndex localVertexDofForGid(
    const svmp::Mesh& mesh,
    const FE::dofs::DofHandler& dofs,
    svmp::gid_t gid)
{
    const auto local = mesh.local_mesh().global_to_local_vertex(gid);
    if (local == svmp::INVALID_INDEX) {
        return kInvalidDof;
    }
    const auto* entity = dofs.getEntityDofMap();
    if (entity == nullptr) {
        return kInvalidDof;
    }
    const auto vertex_dofs = entity->getVertexDofs(
        static_cast<FE::GlobalIndex>(local));
    return vertex_dofs.size() == 1u ? vertex_dofs.front()
                                    : kInvalidDof;
}

} // namespace

TEST(LevelSetTransportMPI,
     BoundLimiterUsesGlobalFieldIndexingAndSynchronizesSharedPatchBounds)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ASSERT_EQ(size, 2) << "This partition test requires exactly two ranks.";

    auto mesh = makeTwoRankQuadPartition(rank, MPI_COMM_WORLD);
    auto space = std::make_shared<FE::spaces::H1Space>(
        FE::ElementType::Quad4, /*order=*/1);
    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi", .space = space, .components = 1});
    system.setup(mpiSetupOptions(rank, size, MPI_COMM_WORLD));

    const auto& dofs = system.fieldDofHandler(phi);
    ASSERT_EQ(dofs.getNumDofs(), 6);
    const auto count = static_cast<std::size_t>(dofs.getNumDofs());
    std::vector<FE::Real> previous(count, FE::Real{0.0});
    std::vector<FE::Real> local_previous(count, FE::Real{0.0});
    const auto& gids = mesh->local_mesh().vertex_gids();
    const auto& coordinates = mesh->X_ref();
    for (std::size_t vertex = 0; vertex < mesh->n_vertices(); ++vertex) {
        const auto dof = localVertexDofForGid(*mesh, dofs, gids[vertex]);
        ASSERT_NE(dof, kInvalidDof);
        if (dofs.getDofMap().isOwnedDof(dof)) {
            local_previous[static_cast<std::size_t>(dof)] =
                FE::Real{1.0} + coordinates[2u * vertex];
        }
    }
    MPI_Allreduce(local_previous.data(), previous.data(),
                  static_cast<int>(count), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    auto candidate = previous;
    FE::GlobalIndex local_shared_dof =
        localVertexDofForGid(*mesh, dofs, /*gid=*/1);
    FE::GlobalIndex shared_dof = kInvalidDof;
    MPI_Allreduce(&local_shared_dof, &shared_dof, 1, MPI_INT64_T,
                  MPI_MAX, MPI_COMM_WORLD);
    ASSERT_NE(shared_dof, kInvalidDof);
    candidate[static_cast<std::size_t>(shared_dof)] = FE::Real{9.0};

    FE::GlobalIndex local_rank_one_dof =
        localVertexDofForGid(*mesh, dofs, /*gid=*/2);
    FE::GlobalIndex rank_one_dof = kInvalidDof;
    MPI_Allreduce(&local_rank_one_dof, &rank_one_dof, 1, MPI_INT64_T,
                  MPI_MAX, MPI_COMM_WORLD);
    ASSERT_NE(rank_one_dof, kInvalidDof);
    candidate[static_cast<std::size_t>(rank_one_dof)] = FE::Real{-4.0};

    level_set::LevelSetBoundPreservingOptions options;
    options.enabled = true;
    options.bound_tolerance = 0.0;
    options.enforce_courant_limit = true;
    options.maximum_courant = 1.0;
    std::vector<FE::Real> limited;
    const auto result = level_set::applyLevelSetBoundPreservingLimiter(
        system,
        phi,
        /*boundaries=*/{},
        options,
        std::span<const FE::Real>(previous),
        std::span<const FE::Real>(candidate),
        /*observed_courant=*/0.5,
        limited);

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.applied);
    EXPECT_EQ(result.field_dofs, 6u);
    EXPECT_EQ(result.limited_dofs, 2u);
    EXPECT_EQ(result.positive_patch_sign_flips_prevented, 1u);
    EXPECT_NEAR(limited[static_cast<std::size_t>(shared_dof)], 3.0, 1.0e-13);
    EXPECT_NEAR(limited[static_cast<std::size_t>(rank_one_dof)], 2.0, 1.0e-13);

    std::vector<FE::Real> global_min(count, 0.0);
    std::vector<FE::Real> global_max(count, 0.0);
    MPI_Allreduce(limited.data(), global_min.data(), static_cast<int>(count),
                  MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(limited.data(), global_max.data(), static_cast<int>(count),
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    for (std::size_t i = 0; i < count; ++i) {
        EXPECT_NEAR(global_min[i], global_max[i], 1.0e-14);
    }
}
