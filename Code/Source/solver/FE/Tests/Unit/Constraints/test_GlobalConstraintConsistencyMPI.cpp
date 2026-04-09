/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_GlobalConstraintConsistencyMPI.cpp
 * @brief MPI regression tests for replicated global-constraint setup invariants.
 */

#include <gtest/gtest.h>

#include "Constraints/GlobalConstraint.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <mpi.h>

#include <memory>
#include <string>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

namespace {

dofs::MeshTopologyInfo singleTetraTopology()
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {0, 1, 2, 3};
    topo.cell_gids = {0};
    topo.cell_owner_ranks = {0};
    return topo;
}

std::unique_ptr<systems::FESystem> makeScalarSystem()
{
    auto mesh = std::make_shared<forms::test::SingleTetraMeshAccess>();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Tetra4, 1);

    auto system = std::make_unique<systems::FESystem>(mesh);
    system->addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    return system;
}

systems::SetupInputs defaultSetupInputs()
{
    systems::SetupInputs inputs;
    inputs.topology_override = singleTetraTopology();
    return inputs;
}

} // namespace

TEST(GlobalConstraintConsistencyMPI, RankLocalGlobalConstraintDefinitionThrows)
{
    int world_size = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (world_size < 2) {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    auto system = makeScalarSystem();
    if (rank == 0) {
        system->addConstraint(
            std::make_unique<constraints::GlobalConstraint>(
                constraints::GlobalConstraint::pinDof(0, 0.0)));
    }

    const auto inputs = defaultSetupInputs();

    try {
        system->setup({}, inputs);
        FAIL() << "Expected setup to reject rank-local GlobalConstraint definitions";
    } catch (const FEException& ex) {
        EXPECT_NE(std::string(ex.what()).find("GlobalConstraint definitions differ"),
                  std::string::npos);
    }
}

TEST(GlobalConstraintConsistencyMPI, MatchingGlobalConstraintDefinitionsSucceed)
{
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    auto system = makeScalarSystem();
    system->addConstraint(
        std::make_unique<constraints::GlobalConstraint>(
            constraints::GlobalConstraint::pinDof(0, 0.0)));

    ASSERT_NO_THROW(system->setup({}, defaultSetupInputs()));
    EXPECT_TRUE(system->constraints().isConstrained(0));
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
