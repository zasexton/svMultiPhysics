#include <gtest/gtest.h>

#include "MovingMesh/GeometryRegularizationBackend.h"

#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"
#include "Mesh/Motion/IMotionBackend.h"
#include "Mesh/Motion/MeshMotion.h"
#include "Mesh/Motion/MotionFields.h"

#include <mpi.h>

#include <array>
#include <memory>
#include <vector>

namespace {

int all_true(MPI_Comm comm, bool local)
{
    const int l = local ? 1 : 0;
    int g = 0;
    MPI_Allreduce(&l, &g, 1, MPI_INT, MPI_MIN, comm);
    return g;
}

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

svmp::index_t strip_vertex_lid(int x_plane, int y, int z)
{
    return static_cast<svmp::index_t>(x_plane * 4 + (y + 2 * z));
}

void build_hex_strip_global_arrays(int n_cells,
                                   std::vector<svmp::real_t>& coords,
                                   std::vector<svmp::offset_t>& offsets,
                                   std::vector<svmp::index_t>& conn,
                                   std::vector<svmp::CellShape>& shapes)
{
    const int n_planes = n_cells + 1;
    const int n_vertices = 4 * n_planes;

    coords.clear();
    coords.reserve(static_cast<std::size_t>(n_vertices) * 3u);
    for (int x_plane = 0; x_plane < n_planes; ++x_plane) {
        for (int z = 0; z <= 1; ++z) {
            for (int y = 0; y <= 1; ++y) {
                coords.push_back(static_cast<svmp::real_t>(x_plane));
                coords.push_back(static_cast<svmp::real_t>(y));
                coords.push_back(static_cast<svmp::real_t>(z));
            }
        }
    }

    offsets.assign(static_cast<std::size_t>(n_cells) + 1u, 0);
    conn.clear();
    conn.reserve(static_cast<std::size_t>(n_cells) * 8u);
    shapes.assign(static_cast<std::size_t>(n_cells), svmp::CellShape{svmp::CellFamily::Hex, 8, 1});

    offsets[0] = 0;
    for (int c = 0; c < n_cells; ++c) {
        const int x0 = c;
        const int x1 = c + 1;

        conn.push_back(strip_vertex_lid(x0, 0, 0));
        conn.push_back(strip_vertex_lid(x1, 0, 0));
        conn.push_back(strip_vertex_lid(x1, 1, 0));
        conn.push_back(strip_vertex_lid(x0, 1, 0));
        conn.push_back(strip_vertex_lid(x0, 0, 1));
        conn.push_back(strip_vertex_lid(x1, 0, 1));
        conn.push_back(strip_vertex_lid(x1, 1, 1));
        conn.push_back(strip_vertex_lid(x0, 1, 1));

        offsets[static_cast<std::size_t>(c) + 1u] = static_cast<svmp::offset_t>(conn.size());
    }
}

std::shared_ptr<svmp::Mesh> make_partitioned_hex_strip(int world_size)
{
    const int n_cells_global = world_size;
    std::vector<svmp::real_t> coords;
    std::vector<svmp::offset_t> offsets;
    std::vector<svmp::index_t> conn;
    std::vector<svmp::CellShape> shapes;
    build_hex_strip_global_arrays(n_cells_global, coords, offsets, conn, shapes);

    auto mesh = std::make_shared<svmp::Mesh>(svmp::MeshComm::world());
    mesh->build_from_arrays_global_and_partition(3,
                                                 coords,
                                                 offsets,
                                                 conn,
                                                 shapes,
                                                 svmp::PartitionHint::Cells,
                                                 /*ghost_layers=*/1,
                                                 {{"partition_method", "block"}});
    return mesh;
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

TEST(FEMeshMotionBackendMPI, DistributedMeshMotionLifecycleUpdatesGhostedFEBackendFields)
{
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) {
        GTEST_SKIP() << "requires at least two MPI ranks";
    }

    auto mesh = make_partitioned_hex_strip(world_size);
    ASSERT_GT(mesh->n_ghost_vertices(), 0u);

    auto backend = std::make_shared<svmp::FE::moving_mesh::GeometryRegularizationMotionBackend>();
    svmp::motion::MeshMotion motion(*mesh);
    motion.set_backend(backend);
    motion.set_dirichlet_bcs({
        svmp::motion::MotionDirichletBC{
            svmp::INVALID_LABEL,
            [](const std::array<svmp::real_t, 3>& x, double, double) {
                return std::array<svmp::real_t, 3>{{
                    static_cast<svmp::real_t>(0.05 * x[0]),
                    static_cast<svmp::real_t>(0.02 * x[1]),
                    static_cast<svmp::real_t>(0.01 * x[2]),
                }};
            },
            {{true, true, true}},
        },
    });

    const double dt = 0.5;
    const bool local_ok = motion.advance(dt);
    ASSERT_EQ(all_true(mesh->mpi_comm(), local_ok), 1);
    ASSERT_TRUE(local_ok);
    ASSERT_TRUE(mesh->has_current_coords());
    EXPECT_EQ(mesh->active_configuration(), svmp::Configuration::Current);

    const auto handles = svmp::motion::attach_motion_fields(*mesh, mesh->dim());
    const auto* disp =
        svmp::MeshFields::field_data_as<svmp::real_t>(mesh->local_mesh(), handles.displacement);
    const auto* vel =
        svmp::MeshFields::field_data_as<svmp::real_t>(mesh->local_mesh(), handles.velocity);
    ASSERT_NE(disp, nullptr);
    ASSERT_NE(vel, nullptr);

    const auto& x_ref = mesh->X_ref();
    const auto& x_cur = mesh->X_cur();
    ASSERT_EQ(x_ref.size(), x_cur.size());
    const auto components = svmp::MeshFields::field_components(mesh->local_mesh(), handles.displacement);
    ASSERT_EQ(components, 3u);

    for (svmp::index_t v = 0; v < static_cast<svmp::index_t>(mesh->n_vertices()); ++v) {
        const auto base = static_cast<std::size_t>(v) * components;
        const svmp::real_t expected[3] = {
            static_cast<svmp::real_t>(0.05 * x_ref[base + 0]),
            static_cast<svmp::real_t>(0.02 * x_ref[base + 1]),
            static_cast<svmp::real_t>(0.01 * x_ref[base + 2]),
        };
        for (std::size_t d = 0; d < components; ++d) {
            EXPECT_NEAR(disp[base + d], expected[d], 1.0e-12);
            EXPECT_NEAR(vel[base + d], expected[d] / static_cast<svmp::real_t>(dt), 1.0e-12);
            EXPECT_NEAR(x_cur[base + d], x_ref[base + d] + expected[d], 1.0e-12);
        }
    }
}
