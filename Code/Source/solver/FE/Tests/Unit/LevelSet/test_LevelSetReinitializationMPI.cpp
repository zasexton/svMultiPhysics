#include "LevelSet/LevelSetReinitialization.h"

#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/SystemSetup.h"

#include <gtest/gtest.h>

#include <mpi.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
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
     ProjectionFailsClosedBeforeMutatingStateOrCandidate)
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
    std::vector<FE::Real> accepted_state(count, 0.0);
    for (std::size_t i = 0; i < accepted_state.size(); ++i) {
        accepted_state[i] = FE::Real{-0.35} +
                            FE::Real{0.2} * static_cast<FE::Real>(i);
    }
    const auto accepted_state_before = accepted_state;
    std::vector<FE::Real> candidate{FE::Real{91.0},
                                    FE::Real{-17.0},
                                    FE::Real{42.0}};
    const auto candidate_before = candidate;

    std::string diagnostic;
    bool rejected = false;
    try {
        (void)level_set::repairLevelSetSignedDistanceByProjection(
            system,
            phi,
            level_set::LevelSetReinitializationOptions{},
            accepted_state,
            candidate);
    } catch (const std::invalid_argument& error) {
        rejected = true;
        diagnostic = error.what();
    }

    EXPECT_TRUE(rejected);
    EXPECT_NE(diagnostic.find(
                  "unsupported on MPI communicators with more than one rank"),
              std::string::npos);
    EXPECT_NE(diagnostic.find(
                  "interface primitive construction and coefficient binding "
                  "are currently rank-local"),
              std::string::npos);
    EXPECT_EQ(accepted_state, accepted_state_before);
    EXPECT_EQ(candidate, candidate_before);

    int local_rejected = rejected ? 1 : 0;
    int all_rejected = 0;
    MPI_Allreduce(&local_rejected,
                  &all_rejected,
                  1,
                  MPI_INT,
                  MPI_MIN,
                  MPI_COMM_WORLD);
    EXPECT_EQ(all_rejected, 1);
}

} // namespace
