/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Systems/FEAdaptivityTransfer.h"
#include "Systems/SystemCollectives.h"

#include <mpi.h>

namespace svmp::FE::systems {
namespace {

MeshBase segmentMesh(bool both_cells, Real first_length)
{
    MeshBase mesh;
    const CellShape segment{CellFamily::Line, 2, 1};
    if (both_cells) {
        mesh.build_from_arrays(1, {0., 1., 2.}, {0, 2, 4}, {0, 1, 1, 2}, {segment, segment});
        mesh.set_cell_gids({10, 11});
        mesh.set_vertex_gids({0, 1, 2});
        mesh.set_current_coords({0., first_length, first_length + 1.});
    } else {
        mesh.build_from_arrays(1, {0., 1.}, {0, 2}, {0, 1}, {segment});
        mesh.set_cell_gids({10});
        mesh.set_vertex_gids({0, 1});
        mesh.set_current_coords({0., first_length});
    }
    mesh.finalize();
    mesh.use_current_configuration();
    return mesh;
}

void checkTransfer(MPI_Comm native, Real stretch)
{
    const CollectiveCommunicator comm(native);
    const bool both = comm.size() == 1 || comm.rank() != 0;
    auto old_mesh = segmentMesh(both, 1.);
    auto new_mesh = segmentMesh(both, stretch);
    std::vector<Real> old{1., 1., 1., 1.};
    if (both) old.insert(old.end(), {4., -3.});
    std::vector<Real> values(old.size(), 9.);
    FEFieldTransferOptions options;
    options.method = FEFieldTransferMethod::Conservative;
    const auto result = transferNodalFieldByVertexProvenance(old_mesh, new_mesh, {}, 2,
                                                            old, values, options, comm);
    ASSERT_TRUE(result.success);
    EXPECT_LE(result.conservation_error, 1e-14);
    // The original integrals are 1 + 5/2 and 1 - 1. Counting the shared
    // first cell twice or correcting rank by rank changes both results.
    const Real scale = 3.5 / (stretch + 2.5);
    const Real shift = (1. - stretch) / (stretch + 1.);
    for (std::size_t v = 0; v < old_mesh.n_vertices(); ++v) {
        EXPECT_NEAR(values[2*v], old[2*v] * scale, 1e-14);
        EXPECT_NEAR(values[2*v + 1], old[2*v + 1] + shift, 1e-14);
    }
    const auto saved = values;
    const auto input = std::span<const Real>{old}.first(comm.rank() == 0 ? 0 : old.size());
    EXPECT_THROW(transferNodalFieldByVertexProvenance(old_mesh, new_mesh, {}, 2, input,
                                                      values, options, comm), std::exception);
    EXPECT_EQ(values, saved);
}

TEST(FEAdaptivityTransferMPI, ConservesBothSignedComponentsOnOverlappingPartitions)
{
    checkTransfer(MPI_COMM_WORLD, 2.);
}

TEST(FEAdaptivityTransferMPI, ImportsRetainedVerticesAndParentsAfterRepartition)
{
    const CollectiveCommunicator comm(MPI_COMM_WORLD);
    if (comm.size() < 2) GTEST_SKIP() << "Requires a repartition";
    const int left = comm.rank() == 0 ? 0 : 1;
    const int other = 1 - left;
    const CellShape line{CellFamily::Line, 2, 1};
    MeshBase coarse, fine;
    coarse.build_from_arrays(1, {Real(left), Real(left + 1)}, {0, 2}, {0, 1}, {line});
    coarse.set_vertex_gids({left, left + 1});
    coarse.set_cell_gids({10 + left});
    coarse.finalize();
    fine.build_from_arrays(1, {Real(other), other + .5, Real(other + 1)},
                           {0, 2, 4}, {0, 1, 1, 2}, {line, line});
    fine.set_vertex_gids({other, 3 + other, other + 1});
    fine.set_cell_gids({20 + 2 * other, 21 + 2 * other});
    fine.finalize();
    RefinementDelta delta;
    VertexProvenanceRecord midpoint;
    midpoint.new_vertex_gid = 3 + other;
    midpoint.parent_vertex_weights = {{other, .5}, {other + 1, .5}};
    delta.new_vertices.push_back(midpoint);
    const std::vector<Real> old{Real(left), Real(left + 1)};
    std::vector<Real> values(3, -1.);
    FEFieldTransferOptions options;
    options.method = FEFieldTransferMethod::Conservative;
    EXPECT_TRUE(transferNodalFieldByVertexProvenance(coarse, fine, delta, 1, old, values, options, comm).success);
    EXPECT_EQ(values, (std::vector<Real>{Real(other), other + .5, Real(other + 1)}));
}

TEST(FEAdaptivityTransferMPI, EachSubcommunicatorUsesItsOwnPhysicalMesh)
{
    int size = 1, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size < 4) GTEST_SKIP() << "Requires two distributed meshes";
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &comm);
    checkTransfer(comm, 2. + rank % 2);
    MPI_Comm_free(&comm);
}

} // namespace
} // namespace svmp::FE::systems
#endif
