#include <gtest/gtest.h>

#include "MovingMesh/GeometryRegularizationBackend.h"

#include "Mesh/Mesh.h"
#include "Mesh/Motion/IMotionBackend.h"

#include <mpi.h>

#include <array>
#include <memory>
#include <vector>

namespace {

std::shared_ptr<svmp::Mesh> make_rank_local_square(MPI_Comm comm)
{
    auto base = std::make_shared<svmp::MeshBase>();
    const std::vector<svmp::real_t> coords = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
        0.5, 0.5,
    };
    const std::vector<svmp::offset_t> offsets = {0, 3, 6, 9, 12};
    const std::vector<svmp::index_t> conn = {
        0, 1, 4,
        1, 2, 4,
        2, 3, 4,
        3, 0, 4,
    };
    svmp::CellShape tri{};
    tri.family = svmp::CellFamily::Triangle;
    tri.num_corners = 3;
    tri.order = 1;
    base->build_from_arrays(2, coords, offsets, conn, {tri, tri, tri, tri});
    base->finalize();
    return svmp::create_mesh(std::move(base), svmp::MeshComm(comm));
}

} // namespace

TEST(FEMeshMotionBackendMPI, RankLocalKeySolveIsDeterministic)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto mesh = make_rank_local_square(MPI_COMM_WORLD);
    svmp::FE::moving_mesh::GeometryRegularizationMotionBackend backend;
    svmp::motion::MotionConfig config;
    std::vector<svmp::real_t> displacement(mesh->n_vertices() * 2u, 0.0);
    std::vector<svmp::real_t> velocity(mesh->n_vertices() * 2u, 0.0);
    std::vector<svmp::motion::MotionDirichletBC> bcs = {
        {svmp::INVALID_LABEL,
         [](const std::array<svmp::real_t, 3>& x, double, double) {
             return std::array<svmp::real_t, 3>{{0.2 * x[0], 0.0, 0.0}};
         },
         {{true, true, true}}},
    };

    svmp::motion::MotionSolveRequest request{
        *mesh,
        config,
        1.0,
        1.0,
        svmp::Configuration::Reference,
        {displacement.data(), mesh->n_vertices(), 2u},
        {velocity.data(), mesh->n_vertices(), 2u},
        &bcs};

    const auto result = backend.solve(request);
    ASSERT_TRUE(result.success) << "rank " << rank << ": " << result.message;
    EXPECT_NEAR(displacement[8], 0.1, 1.0e-12);
    EXPECT_NEAR(velocity[8], 0.1, 1.0e-12);
}
