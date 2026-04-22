/**
 * @file test_VertexDirichletConstraintMPI.cpp
 * @brief MPI coverage for global-vertex Dirichlet constraint ownership behavior.
 */

#include <gtest/gtest.h>

#include "Constraints/VertexDirichletConstraint.h"
#include "Dofs/DofHandler.h"
#include "Dofs/EntityDofMap.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include <mpi.h>

#include <array>
#include <memory>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

namespace {

int mpiRank(MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

int mpiSize(MPI_Comm comm)
{
    int size = 1;
    MPI_Comm_size(comm, &size);
    return size;
}

std::shared_ptr<Mesh> buildRankLocalQuadMesh(int rank, MPI_Comm comm)
{
    auto base = std::make_shared<MeshBase>();

    const real_t x0 = static_cast<real_t>(rank);
    const real_t x1 = static_cast<real_t>(rank + 1);
    const std::vector<real_t> x_ref = {
        x0, 0.0,
        x1, 0.0,
        x1, 1.0,
        x0, 1.0,
    };
    const std::vector<offset_t> cell2vertex_offsets = {0, 4};
    const std::vector<index_t> cell2vertex = {0, 1, 2, 3};

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, x_ref, cell2vertex_offsets, cell2vertex, {shape});

    const gid_t gid_base = static_cast<gid_t>(100 + 10 * rank);
    base->set_vertex_gids({gid_base + 0, gid_base + 1, gid_base + 2, gid_base + 3});
    base->set_cell_gids({static_cast<gid_t>(rank)});
    base->finalize();

    return create_mesh(std::move(base), MeshComm(comm));
}

systems::SetupOptions mpiSetupOptions(MPI_Comm comm, int rank, int world_size)
{
    systems::SetupOptions opts;
    opts.dof_options.global_numbering = dofs::GlobalNumberingMode::OwnerContiguous;
    opts.dof_options.ownership = dofs::OwnershipStrategy::VertexGID;
    opts.dof_options.my_rank = rank;
    opts.dof_options.world_size = world_size;
    opts.dof_options.mpi_comm = comm;
    return opts;
}

} // namespace

TEST(VertexDirichletConstraintMPI, InsertsOnlyOnOwningRankForReplicatedGlobalGidList)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int world_size = mpiSize(comm);
    if (world_size < 2) {
        GTEST_SKIP() << "Run with at least 2 MPI ranks";
    }

    auto mesh = buildRankLocalQuadMesh(rank, comm);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto field = system.addField(systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("mass");

    std::vector<VertexDirichletValue> values;
    values.reserve(static_cast<std::size_t>(world_size));
    for (int r = 0; r < world_size; ++r) {
        values.push_back(VertexDirichletValue{
            .vertex_id = static_cast<GlobalIndex>(100 + 10 * r + 2),
            .value = static_cast<Real>(3.0 + r),
        });
    }
    system.addSystemConstraint(std::make_unique<VertexDirichletConstraint>(
        field, std::move(values), VertexIdMode::GlobalVertexGid));

    ASSERT_NO_THROW(system.setup(mpiSetupOptions(comm, rank, world_size)));

    const auto* entity = system.fieldDofHandler(field).getEntityDofMap();
    ASSERT_NE(entity, nullptr);
    const auto& owned = system.dofHandler().getPartition().locallyOwned();

    for (int r = 0; r < world_size; ++r) {
        const auto gid = static_cast<gid_t>(100 + 10 * r + 2);
        int local_inserted = 0;

        const auto local_vertex = mesh->base().global_to_local_vertex(gid);
        if (local_vertex != INVALID_INDEX) {
            const auto vertex_dofs = entity->getVertexDofs(static_cast<GlobalIndex>(local_vertex));
            ASSERT_EQ(vertex_dofs.size(), 1u);
            const auto dof = vertex_dofs.front() + system.fieldDofOffset(field);
            if (owned.contains(dof) && system.constraints().isConstrained(dof)) {
                EXPECT_NEAR(system.constraints().getInhomogeneity(dof),
                            static_cast<Real>(3.0 + r),
                            1e-12);
                local_inserted = 1;
            }
        }

        int global_inserted = 0;
        MPI_Allreduce(&local_inserted, &global_inserted, 1, MPI_INT, MPI_SUM, comm);
        EXPECT_EQ(global_inserted, 1) << "gid=" << gid;
    }
#endif
}

TEST(VertexDirichletConstraintMPI, MissingGlobalGidFailsOnAllRanks)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration (FE_WITH_MESH=ON).";
#else
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int world_size = mpiSize(comm);
    if (world_size < 2) {
        GTEST_SKIP() << "Run with at least 2 MPI ranks";
    }

    auto mesh = buildRankLocalQuadMesh(rank, comm);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto field = system.addField(systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("mass");

    std::vector<VertexDirichletValue> values = {
        {.vertex_id = 999999, .value = 1.0},
    };
    system.addSystemConstraint(std::make_unique<VertexDirichletConstraint>(
        field, std::move(values), VertexIdMode::GlobalVertexGid));

    int local_threw = 0;
    try {
        system.setup(mpiSetupOptions(comm, rank, world_size));
    } catch (const std::invalid_argument&) {
        local_threw = 1;
    }

    int all_threw = 0;
    MPI_Allreduce(&local_threw, &all_threw, 1, MPI_INT, MPI_MIN, comm);
    EXPECT_EQ(all_threw, 1);
#endif
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
