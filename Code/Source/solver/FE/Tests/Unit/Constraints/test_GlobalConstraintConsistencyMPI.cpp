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

#include "Assembly/Assembler.h"
#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryModelBuilder.h"
#include "Constraints/GlobalConstraint.h"
#include "Dofs/DofHandler.h"
#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

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

MPI_Datatype mpiRealType()
{
    if (sizeof(Real) == sizeof(double)) {
        return MPI_DOUBLE;
    }
    if (sizeof(Real) == sizeof(float)) {
        return MPI_FLOAT;
    }
    return MPI_LONG_DOUBLE;
}

Real allreduceMin(Real local_value, MPI_Comm comm)
{
    Real global_value = local_value;
    MPI_Allreduce(&local_value, &global_value, 1, mpiRealType(), MPI_MIN, comm);
    return global_value;
}

Real allreduceMax(Real local_value, MPI_Comm comm)
{
    Real global_value = local_value;
    MPI_Allreduce(&local_value, &global_value, 1, mpiRealType(), MPI_MAX, comm);
    return global_value;
}

class SingleOwnedQuadBoundaryMeshAccess final : public assembly::IMeshAccess {
public:
    explicit SingleOwnedQuadBoundaryMeshAccess(int boundary_marker)
        : boundary_marker_(boundary_marker)
    {
        nodes_ = {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {1.0, 1.0, 0.0},
            {0.0, 1.0, 0.0}
        };
        cell_ = {0, 1, 2, 3};
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 1; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex /*cell_id*/) const override { return true; }

    [[nodiscard]] ElementType getCellType(GlobalIndex /*cell_id*/) const override
    {
        return ElementType::Quad4;
    }

    void getCellNodes(GlobalIndex /*cell_id*/, std::vector<GlobalIndex>& nodes) const override
    {
        nodes.assign(cell_.begin(), cell_.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex /*cell_id*/,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        coords.assign(nodes_.begin(), nodes_.end());
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex /*face_id*/,
                                               GlobalIndex /*cell_id*/) const override
    {
        // Quad4 edge order is (0-1), (1-2), (2-3), (3-0).
        return 3;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex /*face_id*/) const override
    {
        return boundary_marker_;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex /*face_id*/) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(int marker,
                             std::function<void(GlobalIndex, GlobalIndex)> callback) const override
    {
        if (marker < 0 || marker == boundary_marker_) {
            callback(0, 0);
        }
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)> /*callback*/) const override
    {
    }

private:
    int boundary_marker_{-1};
    std::vector<std::array<Real, 3>> nodes_{};
    std::array<GlobalIndex, 4> cell_{};
};

systems::SetupInputs disjointSingleQuadInputs(int rank)
{
    systems::SetupInputs inputs;

    dofs::MeshTopologyInfo topo;
    topo.dim = 2;
    topo.n_cells = 1;
    topo.n_vertices = 4;
    topo.cell2vertex_offsets = {0, 4};
    topo.cell2vertex_data = {0, 1, 2, 3};
    topo.vertex_gids = {
        static_cast<dofs::gid_t>(4 * rank + 0),
        static_cast<dofs::gid_t>(4 * rank + 1),
        static_cast<dofs::gid_t>(4 * rank + 2),
        static_cast<dofs::gid_t>(4 * rank + 3)};
    topo.cell_gids = {static_cast<dofs::gid_t>(rank)};
    topo.cell_owner_ranks = {rank};

    inputs.topology_override = topo;
    return inputs;
}

std::unique_ptr<systems::FESystem> makeAuxiliaryDrivenDirichletSystem(MPI_Comm comm,
                                                                      int rank,
                                                                      int world_size,
                                                                      int marker)
{
    auto mesh = std::make_shared<SingleOwnedQuadBoundaryMeshAccess>(marker);
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, 1);

    auto system = std::make_unique<systems::FESystem>(mesh);
    const auto u_field = system->addField(
        systems::FieldSpec{.name = "u", .space = space, .components = 1});
    system->addOperator("op");

    const auto u_disc = forms::FormExpr::discreteField(u_field, *space, "u");
    const auto Q = system->boundaryIntegral(u_disc, marker);

    auto model = systems::AuxiliaryModelBuilder("mpi_dirichlet_bc")
        .input("Q")
        .state("x")
        .ode("x", forms::FormExpr::constant(0.0))
        .output("bc", systems::modelState("x") + systems::modelInput("Q"))
        .build();

    system->deployAuxiliaryModel(
        systems::use(model).name("mpi_dirichlet_bc_inst").global().partitioned("ForwardEuler")
            .bindBoundaryReduction("Q", Q)
            .initialize({0.25})
            .drivesStrongDirichlet(u_field, marker, "bc"));

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::FormExpr::testFunction(*space, "v");
    (void)systems::installFormulation(
        *system, "op", {u_field}, forms::inner(forms::grad(u), forms::grad(v)).dx());

    system->finalizeAuxiliaryLayout();

    systems::SetupOptions opts;
    opts.dof_options.global_numbering = dofs::GlobalNumberingMode::OwnerContiguous;
    opts.dof_options.ownership = dofs::OwnershipStrategy::VertexGID;
    opts.dof_options.my_rank = rank;
    opts.dof_options.world_size = world_size;
    opts.dof_options.mpi_comm = comm;

    system->setup(opts, disjointSingleQuadInputs(rank));
    return system;
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

TEST(GlobalConstraintConsistencyMPI,
     AuxiliaryDrivenDirichletValuesRemainRankConsistentAfterSynchronization)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int world_size = mpiSize(comm);
    if (world_size < 2) {
        GTEST_SKIP() << "Requires at least 2 MPI ranks";
    }

    constexpr int marker = 11;
    auto system = makeAuxiliaryDrivenDirichletSystem(comm, rank, world_size, marker);

    std::vector<Real> solution(
        static_cast<std::size_t>(system->dofHandler().getNumDofs()), Real{1.0});
    systems::SystemStateView state;
    state.time = 0.0;
    state.dt = 0.0;
    state.u = solution;

    system->prepareAuxiliaryForAssembly(state, false);

    system->updateConstraints(/*time=*/0.0, /*dt=*/0.0);

    const auto& owned = system->dofHandler().getPartition().locallyOwned();
    std::vector<Real> constrained_values;
    for (GlobalIndex dof = 0; dof < system->dofHandler().getNumDofs(); ++dof) {
        if (owned.contains(dof) && system->constraints().isConstrained(dof)) {
            constrained_values.push_back(
                static_cast<Real>(system->constraints().getInhomogeneity(dof)));
        }
    }

    ASSERT_EQ(constrained_values.size(), 2u);

    const Real expected_value = static_cast<Real>(world_size) + Real{0.25};
    for (const Real value : constrained_values) {
        EXPECT_NEAR(value, expected_value, 1e-12);
    }

    const auto [local_min_it, local_max_it] =
        std::minmax_element(constrained_values.begin(), constrained_values.end());
    ASSERT_NE(local_min_it, constrained_values.end());
    ASSERT_NE(local_max_it, constrained_values.end());

    EXPECT_NEAR(allreduceMin(*local_min_it, comm), expected_value, 1e-12);
    EXPECT_NEAR(allreduceMax(*local_max_it, comm), expected_value, 1e-12);
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
