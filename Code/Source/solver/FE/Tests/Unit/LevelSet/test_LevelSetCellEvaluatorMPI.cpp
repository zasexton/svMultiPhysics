/* Copyright (c) Stanford University, The Regents of the University of California,
 * and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_LevelSetCellEvaluatorMPI.cpp
 * @brief MPI ownership checks for generated level-set field evaluation.
 */

#include <gtest/gtest.h>

#include "Assembly/Assembler.h"
#include "LevelSet/LevelSetCellEvaluator.h"
#include "LevelSet/LevelSetInterfaceLifecycle.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"

#include <mpi.h>

#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

class SharedTriangleMeshAccess final : public FE::assembly::IMeshAccess {
public:
    explicit SharedTriangleMeshAccess(int rank)
        : rank_(rank)
    {
    }

    [[nodiscard]] FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] FE::GlobalIndex numOwnedCells() const override
    {
        return rank_ == 0 ? 1 : 0;
    }
    [[nodiscard]] FE::GlobalIndex numVertices() const override { return 3; }
    [[nodiscard]] FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }
    [[nodiscard]] bool revisionTrackingAvailable() const override { return true; }
    [[nodiscard]] std::uint64_t geometryRevision() const override { return 7; }
    [[nodiscard]] std::uint64_t topologyRevision() const override { return 11; }
    [[nodiscard]] std::uint64_t ownershipRevision() const override
    {
        return static_cast<std::uint64_t>(31 + rank_);
    }
    [[nodiscard]] bool isOwnedCell(FE::GlobalIndex /*cell_id*/) const override
    {
        return rank_ == 0;
    }

    [[nodiscard]] FE::ElementType getCellType(FE::GlobalIndex /*cell_id*/) const override
    {
        return FE::ElementType::Triangle6;
    }

    void getCellNodes(FE::GlobalIndex /*cell_id*/,
                      std::vector<FE::GlobalIndex>& nodes) const override
    {
        nodes = {0, 1, 2, 3, 4, 5};
    }

    [[nodiscard]] std::array<FE::Real, 3> getNodeCoordinates(
        FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(
        FE::GlobalIndex /*cell_id*/,
        std::vector<std::array<FE::Real, 3>>& coords) const override
    {
        coords.assign(nodes_.begin(), nodes_.end());
    }

    [[nodiscard]] FE::LocalIndex getLocalFaceIndex(
        FE::GlobalIndex /*face_id*/,
        FE::GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(FE::GlobalIndex /*face_id*/) const override
    {
        return -1;
    }

    [[nodiscard]] std::pair<FE::GlobalIndex, FE::GlobalIndex>
    getInteriorFaceCells(FE::GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(FE::GlobalIndex)> callback) const override
    {
        if (rank_ == 0) {
            callback(0);
        }
    }

    void forEachBoundaryFace(
        int /*marker*/,
        std::function<void(FE::GlobalIndex, FE::GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(FE::GlobalIndex, FE::GlobalIndex, FE::GlobalIndex)>
            /*callback*/) const override
    {
    }

private:
    int rank_{0};
    std::array<std::array<FE::Real, 3>, 6> nodes_{{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.5, 0.0, 0.0}},
        {{0.5, 0.5, 0.0}},
        {{0.0, 0.5, 0.0}},
    }};
};

[[nodiscard]] FE::systems::SetupInputs sharedTriangleSetupInputs()
{
    FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 3;
    topo.dim = 2;
    topo.cell2vertex_offsets = {0, 3};
    topo.cell2vertex_data = {0, 1, 2};
    topo.vertex_gids = {0, 1, 2};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};

    FE::systems::SetupInputs inputs;
    inputs.topology_override = std::move(topo);
    return inputs;
}

[[nodiscard]] FE::systems::SetupOptions mpiSetupOptions(
    MPI_Comm comm,
    int rank,
    int world_size)
{
    FE::systems::SetupOptions options;
    options.dof_options.global_numbering =
        FE::dofs::GlobalNumberingMode::OwnerContiguous;
    options.dof_options.ownership = FE::dofs::OwnershipStrategy::VertexGID;
    options.dof_options.my_rank = rank;
    options.dof_options.world_size = world_size;
    options.dof_options.mpi_comm = comm;
    return options;
}

[[nodiscard]] MPI_Datatype mpiRealType()
{
    if (sizeof(FE::Real) == sizeof(double)) {
        return MPI_DOUBLE;
    }
    if (sizeof(FE::Real) == sizeof(float)) {
        return MPI_FLOAT;
    }
    return MPI_LONG_DOUBLE;
}

[[nodiscard]] FE::Real allreduceMin(FE::Real value, MPI_Comm comm)
{
    FE::Real global = value;
    MPI_Allreduce(&value, &global, 1, mpiRealType(), MPI_MIN, comm);
    return global;
}

[[nodiscard]] FE::Real allreduceMax(FE::Real value, MPI_Comm comm)
{
    FE::Real global = value;
    MPI_Allreduce(&value, &global, 1, mpiRealType(), MPI_MAX, comm);
    return global;
}

[[nodiscard]] FE::Real levelSetValueAtNode(const std::array<FE::Real, 3>& x)
{
    return x[0] * x[0] + x[1] * x[1] - FE::Real{0.25};
}

} // namespace

TEST(LevelSetCellEvaluatorMPI, SharedOwnedAndGhostCellsUseDeterministicValues)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);
    if (world_size != 2) {
        GTEST_SKIP() << "Run with exactly 2 MPI ranks";
    }

    const auto mesh = std::make_shared<SharedTriangleMeshAccess>(rank);
    auto scalar_space =
        FE::spaces::Space(FE::spaces::SpaceType::H1, mesh, /*order=*/2, /*components=*/1);

    FE::systems::FESystem system(mesh);
    const auto phi = system.addField(FE::systems::FieldSpec{
        .name = "phi",
        .space = scalar_space,
        .components = 1,
    });
    ASSERT_NO_THROW(
        system.setup(mpiSetupOptions(comm, rank, world_size),
                     sharedTriangleSetupInputs()));

    std::vector<FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto& field_dofs = system.fieldDofHandler(phi);
    const auto cell_dofs = field_dofs.getCellDofs(0);
    ASSERT_GE(cell_dofs.size(), 6u);
    const auto offset = system.fieldDofOffset(phi);
    for (std::size_t i = 0; i < 6u; ++i) {
        const auto x = mesh->getNodeCoordinates(static_cast<FE::GlobalIndex>(i));
        solution[static_cast<std::size_t>(offset + cell_dofs[i])] =
            levelSetValueAtNode(x);
    }

    const auto evaluator =
        level_set::makeLevelSetCellEvaluator(system, phi, solution);
    const auto edge_midpoint = evaluator.evaluate(0, {{0.5, 0.0, 0.0}});
    EXPECT_NEAR(allreduceMin(edge_midpoint.value, comm),
                allreduceMax(edge_midpoint.value, comm),
                1.0e-14);

    level_set::LevelSetGeneratedInterfaceOptions options{};
    options.level_set_field_name = "phi";
    options.requested_interface_marker = 87;
    options.domain_id = "water-air";
    options.geometry_mode =
        level_set::GeneratedInterfaceGeometryMode::HighOrderImplicit;
    options.implicit_cut_quadrature_backend =
        level_set::ImplicitCutQuadratureBackend::HighOrderSubcell;
    options.implicit_cut_max_subdivision_depth = 5;
    options.interface_quadrature_order = 2;
    options.volume_quadrature_order = 2;

    level_set::LevelSetGeneratedInterfaceLifecycle lifecycle;
    const auto result = lifecycle.build(system, options, solution);
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_EQ(result.domain.request().ownership_revision,
              static_cast<std::uint64_t>(31 + rank));
    EXPECT_EQ(mesh->isOwnedCell(0), rank == 0);

    EXPECT_NEAR(allreduceMin(result.summary.negative_volume_measure, comm),
                allreduceMax(result.summary.negative_volume_measure, comm),
                1.0e-14);
    EXPECT_NEAR(allreduceMin(result.summary.positive_volume_measure, comm),
                allreduceMax(result.summary.positive_volume_measure, comm),
                1.0e-14);
    EXPECT_NEAR(allreduceMin(result.summary.measure, comm),
                allreduceMax(result.summary.measure, comm),
                1.0e-14);
}
