/**
 * @file test_AuxiliaryScopeCompletionMPI.cpp
 * @brief MPI FESystem regressions for AuxiliaryState scope-completion contracts.
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryModelDSL.h"
#include "Auxiliary/AuxiliaryModelBuilder.h"
#include "Auxiliary/AuxiliaryStateManager.h"
#include "Backends/FSILS/FsilsFactory.h"
#include "Backends/FSILS/FsilsMatrix.h"
#include "Backends/FSILS/FsilsVector.h"
#include "Backends/Interfaces/GenericMatrix.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Backends/Utils/BackendOptions.h"
#include "Forms/Vocabulary.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <sstream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace svmp::FE::assembly::testing {
namespace {

int mpiRank(MPI_Comm comm)
{
    int r = 0;
    MPI_Comm_rank(comm, &r);
    return r;
}

int mpiSize(MPI_Comm comm)
{
    int s = 1;
    MPI_Comm_size(comm, &s);
    return s;
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

std::vector<Real> allreduceSum(std::span<const Real> local, MPI_Comm comm)
{
    std::vector<Real> global(local.size(), Real(0.0));
    MPI_Allreduce(local.data(),
                  global.data(),
                  static_cast<int>(local.size()),
                  mpiRealType(),
                  MPI_SUM,
                  comm);
    return global;
}

unsigned long long allreduceUnsigned(unsigned long long local, MPI_Comm comm)
{
    unsigned long long global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    return global;
}

Real maxAbsDiff(std::span<const Real> a, std::span<const Real> b)
{
    EXPECT_EQ(a.size(), b.size());
    Real out = Real(0.0);
    const auto n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        out = std::max(out, static_cast<Real>(std::abs(a[i] - b[i])));
    }
    return out;
}

std::vector<int> neighborRanks(int my_rank, int world_size)
{
    std::vector<int> neighbors;
    for (int r = 0; r < world_size; ++r) {
        if (r != my_rank) {
            neighbors.push_back(r);
        }
    }
    return neighbors;
}

class TwoQuadStripMeshAccess final : public IMeshAccess {
public:
    TwoQuadStripMeshAccess(std::vector<int> cell_owner_ranks, int my_rank)
        : cell_owner_ranks_(std::move(cell_owner_ranks))
        , my_rank_(my_rank)
    {
        nodes_ = {{
            {0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {1.0, 0.0, 0.0},
            {1.0, 1.0, 0.0},
            {2.0, 0.0, 0.0},
            {2.0, 1.0, 0.0},
        }};
        cells_ = {{
            std::array<GlobalIndex, 4>{0, 2, 3, 1},
            std::array<GlobalIndex, 4>{2, 4, 5, 3},
        }};
    }

    [[nodiscard]] GlobalIndex numCells() const override
    {
        return static_cast<GlobalIndex>(cells_.size());
    }

    [[nodiscard]] GlobalIndex numOwnedCells() const override
    {
        GlobalIndex owned = 0;
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            if (isOwnedCell(c)) {
                ++owned;
            }
        }
        return owned;
    }

    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override
    {
        return cell_owner_ranks_.at(static_cast<std::size_t>(cell_id)) == my_rank_;
    }

    [[nodiscard]] ElementType getCellType(GlobalIndex) const override
    {
        return ElementType::Quad4;
    }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(cell.size());
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override
    {
        return -1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex) const override
    {
        return {0, 0};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            callback(c);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            if (isOwnedCell(c)) {
                callback(c);
            }
        }
    }

    void forEachBoundaryFace(int,
                             std::function<void(GlobalIndex, GlobalIndex)>) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override
    {
    }

private:
    std::vector<int> cell_owner_ranks_{};
    int my_rank_{0};
    std::array<std::array<Real, 3>, 6> nodes_{};
    std::array<std::array<GlobalIndex, 4>, 2> cells_{};
};

class RestrictedNodeOwnershipMeshAccess final : public IMeshAccess {
public:
    explicit RestrictedNodeOwnershipMeshAccess(GlobalIndex owned_vertices)
        : owned_vertices_(owned_vertices)
    {
    }

    [[nodiscard]] GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] GlobalIndex numVertices() const override { return 6; }
    [[nodiscard]] GlobalIndex numOwnedVertices() const override { return owned_vertices_; }
    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex) const override { return true; }
    [[nodiscard]] ElementType getCellType(GlobalIndex) const override
    {
        return ElementType::Tetra4;
    }
    [[nodiscard]] int getCellDomainId(GlobalIndex) const override { return 7; }

    void getCellNodes(GlobalIndex, std::vector<GlobalIndex>& nodes) const override
    {
        nodes = {0, 1, 2, 3};
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return {static_cast<Real>(node_id), 0.0, 0.0};
    }

    void getCellCoordinates(GlobalIndex,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        coords = {{
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
        }};
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override { return -1; }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex) const override
    {
        return {-1, -1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(int,
                             std::function<void(GlobalIndex, GlobalIndex)>) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override
    {
    }

private:
    GlobalIndex owned_vertices_{0};
};

class TwoDisconnectedQuadMeshAccess final : public IMeshAccess {
public:
    TwoDisconnectedQuadMeshAccess(std::vector<int> cell_owner_ranks, int my_rank)
        : cell_owner_ranks_(std::move(cell_owner_ranks))
        , my_rank_(my_rank)
    {
        nodes_ = {{
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {1.0, 1.0, 0.0},
            {0.0, 1.0, 0.0},
            {3.0, 0.0, 0.0},
            {4.0, 0.0, 0.0},
            {4.0, 1.0, 0.0},
            {3.0, 1.0, 0.0},
        }};
        cells_ = {{
            std::array<GlobalIndex, 4>{0, 1, 2, 3},
            std::array<GlobalIndex, 4>{4, 5, 6, 7},
        }};
    }

    [[nodiscard]] GlobalIndex numCells() const override
    {
        return static_cast<GlobalIndex>(cells_.size());
    }

    [[nodiscard]] GlobalIndex numOwnedCells() const override
    {
        GlobalIndex owned = 0;
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            if (isOwnedCell(c)) {
                ++owned;
            }
        }
        return owned;
    }

    [[nodiscard]] GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 2; }

    [[nodiscard]] bool isOwnedCell(GlobalIndex cell_id) const override
    {
        return cell_owner_ranks_.at(static_cast<std::size_t>(cell_id)) == my_rank_;
    }

    [[nodiscard]] ElementType getCellType(GlobalIndex) const override
    {
        return ElementType::Quad4;
    }

    [[nodiscard]] int getCellDomainId(GlobalIndex cell_id) const override
    {
        return cell_id == 0 ? 10 : 20;
    }

    void getCellNodes(GlobalIndex cell_id, std::vector<GlobalIndex>& nodes) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        nodes.assign(cell.begin(), cell.end());
    }

    [[nodiscard]] std::array<Real, 3> getNodeCoordinates(GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(GlobalIndex cell_id,
                            std::vector<std::array<Real, 3>>& coords) const override
    {
        const auto& cell = cells_.at(static_cast<std::size_t>(cell_id));
        coords.resize(cell.size());
        for (std::size_t i = 0; i < cell.size(); ++i) {
            coords[i] = nodes_.at(static_cast<std::size_t>(cell[i]));
        }
    }

    [[nodiscard]] LocalIndex getLocalFaceIndex(GlobalIndex, GlobalIndex) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(GlobalIndex) const override
    {
        return -1;
    }

    [[nodiscard]] std::pair<GlobalIndex, GlobalIndex>
    getInteriorFaceCells(GlobalIndex) const override
    {
        return {-1, -1};
    }

    void forEachCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            callback(c);
        }
    }

    void forEachOwnedCell(std::function<void(GlobalIndex)> callback) const override
    {
        for (GlobalIndex c = 0; c < numCells(); ++c) {
            if (isOwnedCell(c)) {
                callback(c);
            }
        }
    }

    void forEachBoundaryFace(int,
                             std::function<void(GlobalIndex, GlobalIndex)>) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(GlobalIndex, GlobalIndex, GlobalIndex)>) const override
    {
    }

private:
    std::vector<int> cell_owner_ranks_{};
    int my_rank_{0};
    std::array<std::array<Real, 3>, 8> nodes_{};
    std::array<std::array<GlobalIndex, 4>, 2> cells_{};
};

dofs::MeshTopologyInfo twoQuadTopology(std::span<const int> cell_owner_ranks,
                                       int my_rank,
                                       int world_size)
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 6;
    topo.dim = 2;
    topo.cell2vertex_offsets = {0, 4, 8};
    topo.cell2vertex_data = {0, 2, 3, 1, 2, 4, 5, 3};
    topo.vertex_gids = {0, 1, 2, 3, 4, 5};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks.assign(cell_owner_ranks.begin(), cell_owner_ranks.end());
    topo.neighbor_ranks = neighborRanks(my_rank, world_size);
    return topo;
}

dofs::MeshTopologyInfo twoDisconnectedQuadTopology(std::span<const int> cell_owner_ranks,
                                                   int my_rank,
                                                   int world_size)
{
    dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 8;
    topo.dim = 2;
    topo.cell2vertex_offsets = {0, 4, 8};
    topo.cell2vertex_data = {0, 1, 2, 3, 4, 5, 6, 7};
    topo.vertex_gids = {0, 1, 2, 3, 4, 5, 6, 7};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks.assign(cell_owner_ranks.begin(), cell_owner_ranks.end());
    topo.neighbor_ranks = neighborRanks(my_rank, world_size);
    return topo;
}

systems::AuxiliaryDeploymentRegion materialRegion(int id)
{
    systems::AuxiliaryDeploymentRegion region;
    region.kind = systems::AuxiliaryRegionKind::MaterialIdSet;
    region.identity = std::to_string(id);
    return region;
}

std::unique_ptr<systems::FESystem>
buildGlobalMonolithicAuxSystem(std::shared_ptr<const IMeshAccess> mesh,
                               const dofs::MeshTopologyInfo& topo,
                               MPI_Comm comm,
                               int rank,
                               int size,
                               bool backend_owned_rows)
{
    auto space = spaces::Space(spaces::SpaceType::H1,
                               ElementType::Quad4,
                               /*order=*/1,
                               /*components=*/1);
    auto sys = std::make_unique<systems::FESystem>(std::move(mesh));
    const auto u_field =
        sys->addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys->addOperator("op");

    const auto u_disc = forms::FormExpr::discreteField(u_field, *space, "u_disc");
    auto Q = sys->domainIntegral("Q_domain", u_disc);

    auto model = systems::aux::model("global_mpi_mixed_dae", [](systems::ModelFacade& m) {
        auto Q_in = m.input("Q");
        auto x = m.state("x");
        auto z = m.state("z", systems::AuxiliaryVariableKind::Algebraic);

        m << systems::ddt(x) == Q_in - x - z;
        m << systems::alg(z) == x + z - forms::FormExpr::constant(1.0);
        m << systems::out("Y") == x;
    });

    auto inst = sys->deploy(
        systems::use(model)
            .name("global_aux")
            .global()
            .monolithic()
            .bind("Q", Q)
            .initialize({0.5, 0.25}));

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::TestFunction(*space, "v");
    const auto residual =
        (forms::inner(forms::grad(u), forms::grad(v)) +
         u * v +
         inst.output("Y") * u * v)
            .dx();
    (void)systems::installFormulation(*sys, "op", {u_field}, residual);

    systems::SetupOptions opts;
    opts.assembler_name = "StandardAssembler";
    opts.assembly_options.deterministic = true;
    opts.assembly_options.overlap_communication = false;
    opts.use_backend_row_ownership_for_assembly = backend_owned_rows;
    opts.dof_options.global_numbering = dofs::GlobalNumberingMode::DenseGlobalIds;
    opts.dof_options.ownership = dofs::OwnershipStrategy::LowestRank;
    opts.dof_options.my_rank = rank;
    opts.dof_options.world_size = size;
    opts.dof_options.mpi_comm = comm;

    systems::SetupInputs inputs;
    inputs.topology_override = topo;
    sys->setup(opts, inputs);
    sys->finalizeAuxiliaryLayout();

    auto* mgr = sys->auxiliaryStateManagerIfPresent();
    EXPECT_NE(mgr, nullptr);
    if (mgr && mgr->hasBlock("global_aux")) {
        auto& blk = mgr->getBlock("global_aux");
        if (blk.work().size() >= 2u) {
            blk.work()[0] = Real(0.6);
            blk.work()[1] = Real(0.25);
        }
    }

    return sys;
}

std::vector<Real> gatherOwnedFsilsMatrix(const backends::GenericMatrix& A,
                                         const backends::FsilsMatrix& fsils,
                                         GlobalIndex n)
{
    std::vector<Real> out(static_cast<std::size_t>(n * n), Real(0.0));
    auto view = const_cast<backends::GenericMatrix&>(A).createAssemblyView();
    for (GlobalIndex i = 0; i < n; ++i) {
        if (!fsils.ownsFeDofRow(i)) {
            continue;
        }
        for (GlobalIndex j = 0; j < n; ++j) {
            out[static_cast<std::size_t>(i * n + j)] = view->getMatrixEntry(i, j);
        }
    }
    return out;
}

std::vector<Real> gatherOwnedFsilsVector(backends::GenericVector& b,
                                         const backends::FsilsVector& fsils,
                                         GlobalIndex n)
{
    std::vector<Real> out(static_cast<std::size_t>(n), Real(0.0));
    auto view = b.createAssemblyView();
    for (GlobalIndex i = 0; i < n; ++i) {
        if (fsils.ownsFeDof(i)) {
            out[static_cast<std::size_t>(i)] = view->getVectorEntry(i);
        }
    }
    return out;
}

void expectClose(std::span<const Real> a,
                 std::span<const Real> b,
                 Real tol,
                 std::string_view label)
{
    const auto max_diff = maxAbsDiff(a, b);
    std::ostringstream details;
    details << label << "\na=[";
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (i != 0u) details << ", ";
        details << a[i];
    }
    details << "]\nb=[";
    for (std::size_t i = 0; i < b.size(); ++i) {
        if (i != 0u) details << ", ";
        details << b[i];
    }
    details << "]";
    EXPECT_LT(max_diff, tol) << details.str();
}

std::unique_ptr<systems::FESystem>
buildLocalCondensedAuxSystem(systems::AuxiliaryStateScope scope,
                             std::shared_ptr<const IMeshAccess> mesh,
                             const dofs::MeshTopologyInfo& topo,
                             MPI_Comm comm,
                             int rank,
                             int size,
                             bool backend_owned_rows,
                             bool ragged_layout = false)
{
    auto space = spaces::Space(spaces::SpaceType::H1,
                               ElementType::Quad4,
                               /*order=*/1,
                               /*components=*/1);
    auto sys = std::make_unique<systems::FESystem>(std::move(mesh));
    const auto u_field =
        sys->addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys->addOperator("op");

    auto model = systems::aux::model("local_condensed_mpi_mixed_dae",
                                     [](systems::ModelFacade& m) {
        auto x = m.state("x");
        auto z = m.state("z", systems::AuxiliaryVariableKind::Algebraic);

        m << systems::ddt(x) == forms::FormExpr::constant(1.0) - x - z;
        m << systems::alg(z) == x + z - forms::FormExpr::constant(0.5);
        m << systems::out("Y") == x + forms::FormExpr::constant(0.25) * z;
    });

    const std::string instance_name =
        ragged_layout && scope == systems::AuxiliaryStateScope::Cell
            ? "ragged_cell_aux"
        : ragged_layout && scope == systems::AuxiliaryStateScope::QuadraturePoint
            ? "ragged_qp_aux"
        : scope == systems::AuxiliaryStateScope::Cell ? "cell_aux"
        : (scope == systems::AuxiliaryStateScope::QuadraturePoint ? "qp_aux"
                                                                  : "node_aux");
    auto deployment = systems::use(model)
        .name(instance_name)
        .monolithic()
        .scope(scope);
    if (ragged_layout) {
        deployment.layoutMode(systems::AuxiliaryLayoutMode::Ragged)
            .raggedEntitySize([](const systems::AuxiliaryRaggedEntityContext&) {
                return 2u;
            });
    }
    auto inst = sys->deploy(deployment);

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::TestFunction(*space, "v");
    const auto residual =
        (forms::inner(forms::grad(u), forms::grad(v)) +
         u * v -
         inst.output("Y") * v)
            .dx();
    (void)systems::installFormulation(*sys, "op", {u_field}, residual);

    systems::SetupOptions opts;
    opts.assembler_name = "StandardAssembler";
    opts.assembly_options.deterministic = true;
    opts.assembly_options.overlap_communication = false;
    opts.use_backend_row_ownership_for_assembly = backend_owned_rows;
    opts.dof_options.global_numbering = dofs::GlobalNumberingMode::DenseGlobalIds;
    opts.dof_options.ownership = dofs::OwnershipStrategy::LowestRank;
    opts.dof_options.my_rank = rank;
    opts.dof_options.world_size = size;
    opts.dof_options.mpi_comm = comm;

    systems::SetupInputs inputs;
    inputs.topology_override = topo;
    sys->setup(opts, inputs);
    sys->finalizeAuxiliaryLayout();
    sys->beginTimeStep();

    return sys;
}

std::string localCondensedAuxBlockName(systems::AuxiliaryStateScope scope,
                                       bool ragged_layout)
{
    if (ragged_layout && scope == systems::AuxiliaryStateScope::Cell) {
        return "ragged_cell_aux";
    }
    if (ragged_layout && scope == systems::AuxiliaryStateScope::QuadraturePoint) {
        return "ragged_qp_aux";
    }
    if (scope == systems::AuxiliaryStateScope::Cell) {
        return "cell_aux";
    }
    if (scope == systems::AuxiliaryStateScope::QuadraturePoint) {
        return "qp_aux";
    }
    return "node_aux";
}

void expectUniformRaggedLocalCondensedBlock(systems::FESystem& sys,
                                           std::string_view aux_block_name,
                                           std::size_t expected_width)
{
    const auto& mgr = sys.auxiliaryStateManager();
    ASSERT_TRUE(mgr.hasBlock(std::string(aux_block_name)));
    const auto& block = mgr.getBlock(std::string(aux_block_name));
    EXPECT_EQ(block.layoutMode(), systems::AuxiliaryLayoutMode::Ragged);
    EXPECT_EQ(block.ownedEntityCount(), block.entityCount());
    const auto offsets = block.entityOffsets();
    ASSERT_EQ(offsets.size(), block.entityCount() + 1u);
    ASSERT_FALSE(offsets.empty());
    EXPECT_EQ(offsets.front(), 0u);
    EXPECT_EQ(offsets.back(), block.storageSize());
    for (std::size_t entity = 0; entity < block.entityCount(); ++entity) {
        EXPECT_EQ(offsets[entity + 1u] - offsets[entity], expected_width)
            << "entity " << entity;
        EXPECT_EQ(block.gatherEntityWork(entity).size(), expected_width)
            << "entity " << entity;
    }
}

std::unique_ptr<systems::FESystem>
buildRaggedNodeMonolithicOwnerSystem(std::shared_ptr<const IMeshAccess> mesh,
                                     const dofs::MeshTopologyInfo& topo,
                                     MPI_Comm comm,
                                     int rank,
                                     int size,
                                     bool backend_owned_rows,
                                     bool variable_width)
{
    auto space = spaces::Space(spaces::SpaceType::H1,
                               ElementType::Quad4,
                               /*order=*/1,
                               /*components=*/1);
    auto sys = std::make_unique<systems::FESystem>(std::move(mesh));
    const auto u_field =
        sys->addField(systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys->addOperator("op");

    auto model = systems::aux::model("ragged_node_owner_backed_mpi",
                                     [](systems::ModelFacade& m) {
        auto x = m.state("x");
        m << systems::ddt(x) == forms::FormExpr::constant(1.0) - x;
    });

    sys->deploy(systems::use(model)
        .name(variable_width ? "ragged_node_bad_width" : "ragged_node_aux")
        .node()
        .monolithic()
        .layoutMode(systems::AuxiliaryLayoutMode::Ragged)
        .raggedEntitySize([variable_width](const systems::AuxiliaryRaggedEntityContext& ctx) {
            return variable_width && ctx.materialized_entity_index == 0u ? 2u : 1u;
        })
        .initialize({0.25}));

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::TestFunction(*space, "v");
    (void)systems::installFormulation(
        *sys,
        "op",
        {u_field},
        (forms::inner(forms::grad(u), forms::grad(v)) + u * v).dx());

    systems::SetupOptions opts;
    opts.assembler_name = "StandardAssembler";
    opts.assembly_options.deterministic = true;
    opts.assembly_options.overlap_communication = false;
    opts.use_backend_row_ownership_for_assembly = backend_owned_rows;
    opts.dof_options.global_numbering = dofs::GlobalNumberingMode::DenseGlobalIds;
    opts.dof_options.ownership = dofs::OwnershipStrategy::LowestRank;
    opts.dof_options.my_rank = rank;
    opts.dof_options.world_size = size;
    opts.dof_options.mpi_comm = comm;

    systems::SetupInputs inputs;
    inputs.topology_override = topo;
    sys->setup(opts, inputs);

    return sys;
}

unsigned long long allreduceMinUnsigned(unsigned long long local, MPI_Comm comm)
{
    unsigned long long global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm);
    return global;
}

unsigned long long allreduceMaxUnsigned(unsigned long long local, MPI_Comm comm)
{
    unsigned long long global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
    return global;
}

std::vector<Real> reconstructReducedMatrix(const systems::FESystem& sys,
                                           GlobalIndex n_dofs,
                                           MPI_Comm comm,
                                           const backends::FsilsMatrix* fsils_matrix,
                                           const backends::FsilsVector* fsils_vector)
{
    const auto updates = sys.lastReducedFieldUpdates();
    const auto local_count = static_cast<unsigned long long>(updates.size());
    const auto min_count = allreduceMinUnsigned(local_count, comm);
    const auto max_count = allreduceMaxUnsigned(local_count, comm);
    EXPECT_EQ(min_count, max_count);

    const auto n = static_cast<std::size_t>(n_dofs);
    std::vector<Real> local_A(n * n, Real(0.0));
    for (std::size_t update_index = 0;
         update_index < static_cast<std::size_t>(max_count);
         ++update_index) {
        std::vector<Real> local_right(n, Real(0.0));
        if (update_index < updates.size()) {
            const auto& upd = updates[update_index];
            for (const auto& [dof, value] : upd.right) {
                if (fsils_vector != nullptr) {
                    EXPECT_TRUE(fsils_vector->ownsFeDof(dof))
                        << "right factor dof " << dof << " is not owned by this rank";
                }
                const auto col = static_cast<std::size_t>(dof);
                if (col >= n) {
                    ADD_FAILURE() << "right factor dof " << dof << " exceeds field size";
                    continue;
                }
                local_right[col] += value;
            }
        }

        const auto global_right = allreduceSum(local_right, comm);
        if (update_index >= updates.size()) {
            continue;
        }

        const auto& upd = updates[update_index];
        for (const auto& [dof, left_value] : upd.left) {
            if (fsils_matrix != nullptr) {
                EXPECT_TRUE(fsils_matrix->ownsFeDofRow(dof))
                    << "left factor row " << dof << " is not owned by this rank";
            }
            const auto row = static_cast<std::size_t>(dof);
            if (row >= n) {
                ADD_FAILURE() << "left factor row " << dof << " exceeds field size";
                continue;
            }
            for (std::size_t col = 0; col < n; ++col) {
                local_A[row * n + col] += upd.sigma * left_value * global_right[col];
            }
        }
    }

    return allreduceSum(local_A, comm);
}

std::vector<Real> gatherLocalCondensedShift(const systems::FESystem& sys,
                                            GlobalIndex n_dofs,
                                            MPI_Comm comm,
                                            const backends::FsilsVector* fsils_vector)
{
    const auto n = static_cast<std::size_t>(n_dofs);
    std::vector<Real> local_shift(n, Real(0.0));
    const auto shift = sys.lastLocalCondensedRhsShift();
    if (!shift.empty()) {
        EXPECT_EQ(shift.size(), n);
        const auto count = std::min(shift.size(), n);
        for (std::size_t i = 0; i < count; ++i) {
            if (std::abs(shift[i]) <= Real(1e-14)) {
                continue;
            }
            const auto dof = static_cast<GlobalIndex>(i);
            if (fsils_vector != nullptr) {
                EXPECT_TRUE(fsils_vector->ownsFeDof(dof))
                    << "RHS shift dof " << dof << " is not owned by this rank";
            }
            local_shift[i] = shift[i];
        }
    }
    return allreduceSum(local_shift, comm);
}

struct EffectiveAssembly {
    std::vector<Real> matrix{};
    std::vector<Real> vector{};
};

EffectiveAssembly assembleDenseEffective(systems::FESystem& sys,
                                         const systems::SystemStateView& state,
                                         MPI_Comm comm)
{
    const auto n_dofs = sys.dofHandler().getNumDofs();
    systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    DenseSystemView dense(n_dofs);
    dense.zero();
    const auto result = sys.assemble(req, state, &dense, &dense);
    EXPECT_TRUE(result.success) << result.error_message;

    auto A = allreduceSum(dense.matrixData(), comm);
    auto b = allreduceSum(dense.vectorData(), comm);
    const auto reduced_A =
        reconstructReducedMatrix(sys, n_dofs, comm, nullptr, nullptr);
    const auto shift =
        gatherLocalCondensedShift(sys, n_dofs, comm, nullptr);

    EXPECT_EQ(A.size(), reduced_A.size());
    EXPECT_EQ(b.size(), shift.size());
    for (std::size_t i = 0; i < std::min(A.size(), reduced_A.size()); ++i) {
        A[i] += reduced_A[i];
    }
    for (std::size_t i = 0; i < std::min(b.size(), shift.size()); ++i) {
        b[i] -= shift[i];
    }
    return EffectiveAssembly{std::move(A), std::move(b)};
}

EffectiveAssembly assembleFsilsEffective(systems::FESystem& sys,
                                         const systems::SystemStateView& state,
                                         MPI_Comm comm,
                                         int rank,
                                         std::string_view aux_block_name)
{
    const auto n_dofs = sys.dofHandler().getNumDofs();
    const auto layout = sys.augmentSolverOptions(
        backends::SolverOptions{}, static_cast<std::size_t>(n_dofs));
    if (!layout.mixed_block_layout.has_value()) {
        ADD_FAILURE() << "missing mixed block layout";
        return {};
    }
    EXPECT_EQ(layout.mixed_block_layout->auxiliary_unknowns, 0);
    EXPECT_EQ(layout.mixed_block_layout->findBlock(std::string(aux_block_name)), nullptr);
    EXPECT_TRUE(backends::validateFsilsMixedLayoutContract(
                    *layout.mixed_block_layout, /*dof_per_node=*/1)
                    .empty());

    const auto* dist_pattern = sys.distributedSparsityIfAvailable("op");
    if (dist_pattern == nullptr) {
        ADD_FAILURE() << "missing distributed sparsity pattern";
        return {};
    }
    auto perm = sys.dofPermutation();
    if (!perm || perm->empty()) {
        ADD_FAILURE() << "missing DOF permutation";
        return {};
    }

    backends::FsilsFactory factory(/*dof_per_node=*/1, perm, comm);
    auto A = factory.createMatrix(*dist_pattern);
    auto b = factory.createVector(n_dofs);
    A->zero();
    b->zero();

    auto* A_fsils = dynamic_cast<backends::FsilsMatrix*>(A.get());
    auto* b_fsils = dynamic_cast<backends::FsilsVector*>(b.get());
    if (A_fsils == nullptr || b_fsils == nullptr) {
        ADD_FAILURE() << "FSILS factory returned non-FSILS objects";
        return {};
    }
    EXPECT_TRUE(A_fsils->usesOwnedRowOperator());
    EXPECT_TRUE(b_fsils->usesOwnedRowLayout());

    systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    backends::FsilsMatrix::resetDroppedEntryCount();
    backends::FsilsMatrix::resetOffOwnerWriteCount();
    auto A_view = A->createAssemblyView();
    auto b_view = b->createAssemblyView();
    const auto result = sys.assemble(req, state, A_view.get(), b_view.get());
    EXPECT_TRUE(result.success) << result.error_message;
    A->finalizeAssembly();

    const auto off_owner = allreduceUnsigned(
        static_cast<unsigned long long>(backends::FsilsMatrix::offOwnerWriteCount()),
        comm);
    const auto dropped = allreduceUnsigned(
        static_cast<unsigned long long>(backends::FsilsMatrix::droppedEntryCount()),
        comm);
    EXPECT_EQ(off_owner, 0ULL);
    EXPECT_EQ(dropped, 0ULL);

    EXPECT_FALSE(sys.borderedCoupling().active);
    const auto any_rank_has_local_recovery = allreduceUnsigned(
        sys.hasLocalCondensedRecovery() ? 1ULL : 0ULL, comm);
    EXPECT_GT(any_rank_has_local_recovery, 0ULL);
    EXPECT_FALSE(sys.lastReducedFieldUpdates().empty());
    EXPECT_EQ(sys.lastLocalCondensedRhsShift().size(),
              static_cast<std::size_t>(n_dofs));

    for (GlobalIndex dof = 0; dof < n_dofs; ++dof) {
        const auto backend_dof = perm->forward[static_cast<std::size_t>(dof)];
        if (backend_dof < 0) {
            ADD_FAILURE() << "invalid backend dof for FE dof " << dof;
            continue;
        }
        const bool owned_here =
            perm->owner_rank[static_cast<std::size_t>(backend_dof)] == rank;
        EXPECT_EQ(A_fsils->ownsFeDofRow(dof), owned_here) << "row " << dof;
        EXPECT_EQ(b_fsils->ownsFeDof(dof), owned_here) << "dof " << dof;
    }

    auto effective_A = allreduceSum(
        gatherOwnedFsilsMatrix(*A, *A_fsils, n_dofs), comm);
    auto effective_b = allreduceSum(
        gatherOwnedFsilsVector(*b, *b_fsils, n_dofs), comm);
    const auto reduced_A =
        reconstructReducedMatrix(sys, n_dofs, comm, A_fsils, b_fsils);
    const auto shift =
        gatherLocalCondensedShift(sys, n_dofs, comm, b_fsils);

    EXPECT_EQ(effective_A.size(), reduced_A.size());
    EXPECT_EQ(effective_b.size(), shift.size());
    for (std::size_t i = 0; i < std::min(effective_A.size(), reduced_A.size()); ++i) {
        effective_A[i] += reduced_A[i];
    }
    for (std::size_t i = 0; i < std::min(effective_b.size(), shift.size()); ++i) {
        effective_b[i] -= shift[i];
    }
    return EffectiveAssembly{std::move(effective_A), std::move(effective_b)};
}

void expectLocalCondensedScopeMatchesRowOwnedDenseReference(
    systems::AuxiliaryStateScope scope,
    std::span<const int> cell_owners,
    bool ragged_layout = false)
{
#if !defined(FE_HAS_FSILS)
    GTEST_SKIP() << "FSILS backend is not enabled in this build";
#else
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "This regression uses a fixed two-cell strip; run with exactly 2 MPI ranks";
    }

    const auto parallel_topo = twoQuadTopology(cell_owners, rank, size);
    auto sys_parallel = buildLocalCondensedAuxSystem(
        scope,
        std::make_shared<TwoQuadStripMeshAccess>(
            std::vector<int>(cell_owners.begin(), cell_owners.end()), rank),
        parallel_topo,
        comm,
        rank,
        size,
        /*backend_owned_rows=*/true,
        ragged_layout);

    const auto reference_topo = twoQuadTopology(cell_owners, rank, size);
    auto sys_reference = buildLocalCondensedAuxSystem(
        scope,
        std::make_shared<TwoQuadStripMeshAccess>(
            std::vector<int>(cell_owners.begin(), cell_owners.end()), rank),
        reference_topo,
        comm,
        rank,
        size,
        /*backend_owned_rows=*/true,
        ragged_layout);

    const auto n_dofs = sys_parallel->dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, sys_reference->dofHandler().getNumDofs());
    ASSERT_EQ(n_dofs, 6);

    std::vector<Real> u(static_cast<std::size_t>(n_dofs), Real(0.0));
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        u[static_cast<std::size_t>(i)] = Real(0.1) + Real(0.03) * Real(i);
    }

    systems::SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = u;

    const auto aux_name = localCondensedAuxBlockName(scope, ragged_layout);
    if (ragged_layout) {
        ASSERT_TRUE(scope == systems::AuxiliaryStateScope::Cell ||
                    scope == systems::AuxiliaryStateScope::QuadraturePoint);
        expectUniformRaggedLocalCondensedBlock(*sys_reference, aux_name, 2u);
        expectUniformRaggedLocalCondensedBlock(*sys_parallel, aux_name, 2u);
    }

    const auto reference = assembleDenseEffective(*sys_reference, state, comm);
    const auto parallel =
        assembleFsilsEffective(*sys_parallel, state, comm, rank, aux_name);

    expectClose(reference.matrix,
                parallel.matrix,
                Real(1e-9),
                "local-condensed effective matrix mismatch");
    expectClose(reference.vector,
                parallel.vector,
                Real(1e-9),
                "local-condensed effective residual mismatch");
#endif
}

} // namespace

TEST(AuxiliaryScopeCompletionMPI,
     RestrictedNodeScopePartitionsExplicitNodeIdsByLocalOwnership)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "This regression uses rank-specific node ownership; run with exactly 2 MPI ranks";
    }

    auto model = systems::aux::model("restricted_node_partition_mpi",
                                     [](systems::ModelFacade& m) {
        auto x = m.state("x");
        m << systems::ddt(x) == forms::FormExpr::constant(1.0) - x;
    });

    systems::AuxiliaryDeploymentRegion region;
    region.kind = systems::AuxiliaryRegionKind::FormulationDefined;
    region.identity = "rank_local_nodes";
    region.explicit_entities =
        rank == 0 ? std::vector<std::size_t>{4u, 0u, 5u, 2u}
                  : std::vector<std::size_t>{3u, 1u, 5u, 0u};

    const GlobalIndex owned_vertices = rank == 0 ? 3 : 2;
    systems::FESystem system(
        std::make_shared<RestrictedNodeOwnershipMeshAccess>(owned_vertices));
    system.deployAuxiliaryModel(
        systems::use(model).name("node_subset").node().region(region)
            .partitioned("ForwardEuler").initialize({1.0}));

    system.finalizeAuxiliaryLayout();

    const auto& mgr = system.auxiliaryStateManager();
    const auto& block = mgr.getBlock("node_subset");
    const auto& indexing = mgr.getIndexing("node_subset");
    const auto& metadata = mgr.getEntityRemapMetadata("node_subset");

    const std::vector<std::size_t> expected_entities =
        rank == 0 ? std::vector<std::size_t>{0u, 2u, 4u, 5u}
                  : std::vector<std::size_t>{1u, 0u, 3u, 5u};
    const std::size_t expected_owned_count = 2u;

    EXPECT_EQ(block.scope(), systems::AuxiliaryStateScope::Node);
    EXPECT_EQ(block.entityCount(), expected_entities.size());
    EXPECT_EQ(block.ownedEntityCount(), expected_owned_count);
    EXPECT_EQ(indexing.scope(), systems::AuxiliaryStateScope::Node);
    EXPECT_EQ(indexing.totalEntityCount(), expected_entities.size());
    EXPECT_EQ(indexing.ownedEntityCount(), expected_owned_count);
    EXPECT_EQ(metadata.entity_ids, expected_entities);

    for (std::size_t i = 0; i < metadata.entity_ids.size(); ++i) {
        if (i < expected_owned_count) {
            EXPECT_LT(metadata.entity_ids[i], static_cast<std::size_t>(owned_vertices));
        } else {
            EXPECT_GE(metadata.entity_ids[i], static_cast<std::size_t>(owned_vertices));
        }
    }
}

TEST(AuxiliaryScopeCompletionMPI,
     GlobalMonolithicDAEUsesBorderedReducedFsilsContractWithoutDroppedEntries)
{
#if !defined(FE_HAS_FSILS)
    GTEST_SKIP() << "FSILS backend is not enabled in this build";
#else
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "This regression uses a fixed two-cell strip; run with exactly 2 MPI ranks";
    }

    const std::vector<int> cell_owners{0, 1};
    auto mesh_parallel = std::make_shared<TwoQuadStripMeshAccess>(cell_owners, rank);
    const auto topo = twoQuadTopology(cell_owners, rank, size);

    auto sys_parallel = buildGlobalMonolithicAuxSystem(
        mesh_parallel, topo, comm, rank, size, /*backend_owned_rows=*/true);
    auto sys_ref = buildGlobalMonolithicAuxSystem(
        std::make_shared<TwoQuadStripMeshAccess>(cell_owners, rank),
        topo, comm, rank, size, /*backend_owned_rows=*/false);

    const auto n_dofs = sys_parallel->dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, sys_ref->dofHandler().getNumDofs());
    ASSERT_EQ(n_dofs, 6);

    std::vector<Real> u(static_cast<std::size_t>(n_dofs), Real(0.0));
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        u[static_cast<std::size_t>(i)] = Real(0.05) * Real(i + 1);
    }

    systems::SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = u;

    const auto layout = sys_parallel->augmentSolverOptions(
        backends::SolverOptions{}, static_cast<std::size_t>(n_dofs));
    ASSERT_TRUE(layout.mixed_block_layout.has_value());
    const auto* aux_block = layout.mixed_block_layout->findBlock("global_aux");
    ASSERT_NE(aux_block, nullptr);
    EXPECT_EQ(aux_block->assembly_mode, backends::MixedBlockAssemblyMode::BorderedReduced);
    EXPECT_EQ(aux_block->row_ownership, backends::MixedRowOwnershipPolicy::SingleOwner);
    EXPECT_EQ(aux_block->single_owner_rank, 0);
    ASSERT_EQ(aux_block->row_owner_ranks.size(), 2u);
    EXPECT_EQ(aux_block->row_owner_ranks[0], 0);
    EXPECT_EQ(aux_block->row_owner_ranks[1], 0);
    EXPECT_TRUE(backends::validateFsilsMixedLayoutContract(
                    *layout.mixed_block_layout, /*dof_per_node=*/1)
                    .empty());

    systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;

    DenseSystemView ref_dense(n_dofs);
    ref_dense.zero();
    const auto ref_result = sys_ref->assemble(req, state, &ref_dense, &ref_dense);
    ASSERT_TRUE(ref_result.success) << ref_result.error_message;
    const auto ref_A = allreduceSum(ref_dense.matrixData(), comm);
    const auto ref_b = allreduceSum(ref_dense.vectorData(), comm);
    const auto& ref_bc = sys_ref->borderedCoupling();
    ASSERT_TRUE(ref_bc.active);
    ASSERT_TRUE(ref_bc.globally_reduced);
    ASSERT_TRUE(ref_bc.aux_self_terms_replicated);

    const auto* dist_pattern = sys_parallel->distributedSparsityIfAvailable("op");
    ASSERT_NE(dist_pattern, nullptr);
    auto perm = sys_parallel->dofPermutation();
    ASSERT_TRUE(perm);
    ASSERT_FALSE(perm->empty());

    backends::FsilsFactory factory(/*dof_per_node=*/1, perm, comm);
    auto A = factory.createMatrix(*dist_pattern);
    auto b = factory.createVector(n_dofs);
    A->zero();
    b->zero();

    auto* A_fsils = dynamic_cast<backends::FsilsMatrix*>(A.get());
    auto* b_fsils = dynamic_cast<backends::FsilsVector*>(b.get());
    ASSERT_NE(A_fsils, nullptr);
    ASSERT_NE(b_fsils, nullptr);
    ASSERT_TRUE(A_fsils->usesOwnedRowOperator());
    ASSERT_TRUE(b_fsils->usesOwnedRowLayout());

    backends::FsilsMatrix::resetDroppedEntryCount();
    backends::FsilsMatrix::resetOffOwnerWriteCount();
    auto A_view = A->createAssemblyView();
    auto b_view = b->createAssemblyView();
    const auto par_result = sys_parallel->assemble(req, state, A_view.get(), b_view.get());
    ASSERT_TRUE(par_result.success) << par_result.error_message;
    A->finalizeAssembly();

    const auto off_owner = allreduceUnsigned(
        static_cast<unsigned long long>(backends::FsilsMatrix::offOwnerWriteCount()),
        comm);
    const auto dropped = allreduceUnsigned(
        static_cast<unsigned long long>(backends::FsilsMatrix::droppedEntryCount()),
        comm);
    EXPECT_EQ(off_owner, 0ULL);
    EXPECT_EQ(dropped, 0ULL);

    for (GlobalIndex dof = 0; dof < n_dofs; ++dof) {
        const auto backend_dof = perm->forward[static_cast<std::size_t>(dof)];
        ASSERT_GE(backend_dof, 0);
        const bool owned_here =
            perm->owner_rank[static_cast<std::size_t>(backend_dof)] == rank;
        EXPECT_EQ(A_fsils->ownsFeDofRow(dof), owned_here) << "row " << dof;
        EXPECT_EQ(b_fsils->ownsFeDof(dof), owned_here) << "dof " << dof;
    }

    const auto fsils_A_local = gatherOwnedFsilsMatrix(*A, *A_fsils, n_dofs);
    const auto fsils_b_local = gatherOwnedFsilsVector(*b, *b_fsils, n_dofs);
    const auto fsils_A = allreduceSum(fsils_A_local, comm);
    const auto fsils_b = allreduceSum(fsils_b_local, comm);
    expectClose(ref_A, fsils_A, Real(1e-11), "field matrix serial/parallel mismatch");
    expectClose(ref_b, fsils_b, Real(1e-11), "field residual serial/parallel mismatch");

    const auto& par_bc = sys_parallel->borderedCoupling();
    ASSERT_TRUE(par_bc.active);
    ASSERT_TRUE(par_bc.globally_reduced);
    ASSERT_TRUE(par_bc.aux_self_terms_replicated);
    ASSERT_EQ(par_bc.n_aux, ref_bc.n_aux);
    ASSERT_EQ(par_bc.n_field_dofs, ref_bc.n_field_dofs);
    expectClose(ref_bc.D, par_bc.D, Real(1e-11), "bordered D mismatch");
    expectClose(ref_bc.g, par_bc.g, Real(1e-11), "bordered g mismatch");

    // Exact Q1 cell integrals for the two-unit-quad strip. Shared nodes must
    // include both owned-cell contributions before bordered coupling reduction.
    const std::vector<Real> expected_B{
        Real(1.0) / Real(40.0),
        Real(7.0) / Real(240.0),
        Real(1.0) / Real(12.0),
        Real(11.0) / Real(120.0),
        Real(7.0) / Real(120.0),
        Real(1.0) / Real(16.0),
        Real(0.0),
        Real(0.0),
        Real(0.0),
        Real(0.0),
        Real(0.0),
        Real(0.0),
    };
    const std::vector<Real> expected_Ct{
        -Real(1.0) / Real(4.0),
        -Real(1.0) / Real(4.0),
        -Real(1.0) / Real(2.0),
        -Real(1.0) / Real(2.0),
        -Real(1.0) / Real(4.0),
        -Real(1.0) / Real(4.0),
        Real(0.0),
        Real(0.0),
        Real(0.0),
        Real(0.0),
        Real(0.0),
        Real(0.0),
    };
    ASSERT_EQ(par_bc.B.size(), expected_B.size());
    ASSERT_EQ(par_bc.Ct.size(), expected_Ct.size());
    expectClose(expected_B, par_bc.B, Real(1e-11), "bordered B analytic mismatch");
    expectClose(expected_Ct, par_bc.Ct, Real(1e-11), "bordered Ct analytic mismatch");
    EXPECT_TRUE(std::any_of(par_bc.Ct.begin(), par_bc.Ct.end(), [](Real value) {
        return std::abs(value) > Real(1e-14);
    }));
    EXPECT_TRUE(std::any_of(par_bc.B.begin(), par_bc.B.end(), [](Real value) {
        return std::abs(value) > Real(1e-14);
    }));
#endif
}

TEST(AuxiliaryScopeCompletionMPI,
     CellLocalCondensationUsesOwnedFsilsRowsWithoutDroppedEntries)
{
    const std::vector<int> cell_owners{0, 1};
    expectLocalCondensedScopeMatchesRowOwnedDenseReference(
        systems::AuxiliaryStateScope::Cell, cell_owners);
}

TEST(AuxiliaryScopeCompletionMPI,
     QuadraturePointLocalCondensationUsesOwnedFsilsRowsWithoutDroppedEntries)
{
    const std::vector<int> cell_owners{0, 1};
    expectLocalCondensedScopeMatchesRowOwnedDenseReference(
        systems::AuxiliaryStateScope::QuadraturePoint, cell_owners);
}

TEST(AuxiliaryScopeCompletionMPI,
     RaggedCellLocalCondensationMatchesDenseReferenceWithoutDroppedEntries)
{
    const std::vector<int> cell_owners{0, 1};
    expectLocalCondensedScopeMatchesRowOwnedDenseReference(
        systems::AuxiliaryStateScope::Cell, cell_owners, /*ragged_layout=*/true);
}

TEST(AuxiliaryScopeCompletionMPI,
     RaggedQuadraturePointLocalCondensationMatchesDenseReferenceWithoutDroppedEntries)
{
    const std::vector<int> cell_owners{0, 1};
    expectLocalCondensedScopeMatchesRowOwnedDenseReference(
        systems::AuxiliaryStateScope::QuadraturePoint,
        cell_owners,
        /*ragged_layout=*/true);
}

TEST(AuxiliaryScopeCompletionMPI,
     RaggedNodeMonolithicRequiresBackendOwnerMap)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "This regression uses a fixed two-cell strip; run with exactly 2 MPI ranks";
    }

    const std::vector<int> cell_owners{0, 1};
    const auto topo = twoQuadTopology(cell_owners, rank, size);
    auto sys = buildRaggedNodeMonolithicOwnerSystem(
        std::make_shared<TwoQuadStripMeshAccess>(cell_owners, rank),
        topo,
        comm,
        rank,
        size,
        /*backend_owned_rows=*/false,
        /*variable_width=*/false);

    try {
        sys->finalizeAuxiliaryLayout();
        ADD_FAILURE() << "Expected ragged Node monolithic finalize to require backend owners";
    } catch (const systems::InvalidStateException& ex) {
        EXPECT_NE(std::string(ex.what()).find("backend row ownership metadata"),
                  std::string::npos)
            << ex.what();
    }
}

TEST(AuxiliaryScopeCompletionMPI,
     RaggedNodeMonolithicUsesOwnerBackedLocalCondensation)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "This regression uses a fixed two-cell strip; run with exactly 2 MPI ranks";
    }

    const std::vector<int> cell_owners{0, 1};
    const auto topo = twoQuadTopology(cell_owners, rank, size);
    auto sys = buildRaggedNodeMonolithicOwnerSystem(
        std::make_shared<TwoQuadStripMeshAccess>(cell_owners, rank),
        topo,
        comm,
        rank,
        size,
        /*backend_owned_rows=*/true,
        /*variable_width=*/false);

    ASSERT_NO_THROW(sys->finalizeAuxiliaryLayout());
    ASSERT_NE(sys->dofPermutation(), nullptr);
    ASSERT_FALSE(sys->dofPermutation()->owner_rank.empty());

    const auto& block = sys->auxiliaryStateManager().getBlock("ragged_node_aux");
    EXPECT_EQ(block.layoutMode(), systems::AuxiliaryLayoutMode::Ragged);
    EXPECT_EQ(block.storageSize(), block.entityCount());

    const auto layout = sys->augmentSolverOptions(
        backends::SolverOptions{},
        static_cast<std::size_t>(sys->dofHandler().getNumDofs()));
    ASSERT_TRUE(layout.mixed_block_layout.has_value());
    EXPECT_EQ(layout.mixed_block_layout->auxiliary_unknowns, 0);
    EXPECT_EQ(layout.mixed_block_layout->findBlock("ragged_node_aux"), nullptr);
    EXPECT_TRUE(backends::validateFsilsMixedLayoutContract(
                    *layout.mixed_block_layout, /*dof_per_node=*/1)
                    .empty());
}

TEST(AuxiliaryScopeCompletionMPI,
     RaggedNodeMonolithicRejectsVariableWidthOwnerBackedSlices)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "This regression uses a fixed two-cell strip; run with exactly 2 MPI ranks";
    }

    const std::vector<int> cell_owners{0, 1};
    const auto topo = twoQuadTopology(cell_owners, rank, size);
    auto sys = buildRaggedNodeMonolithicOwnerSystem(
        std::make_shared<TwoQuadStripMeshAccess>(cell_owners, rank),
        topo,
        comm,
        rank,
        size,
        /*backend_owned_rows=*/true,
        /*variable_width=*/true);

    try {
        sys->finalizeAuxiliaryLayout();
        ADD_FAILURE() << "Expected variable-width ragged Node monolithic finalize to fail";
    } catch (const InvalidArgumentException& ex) {
        EXPECT_NE(std::string(ex.what()).find("local-condensation contract"),
                  std::string::npos)
            << ex.what();
    }
}

TEST(AuxiliaryScopeCompletionMPI,
     NodeLocalCondensationUsesOwnerBackedSparseFsilsRowsWithoutDroppedEntries)
{
    const std::vector<int> cell_owners{0, 1};
    expectLocalCondensedScopeMatchesRowOwnedDenseReference(
        systems::AuxiliaryStateScope::Node, cell_owners);
}

TEST(AuxiliaryScopeCompletionMPI,
     CellLocalCondensationHandlesRanksWithNoOwnedEntities)
{
    const std::vector<int> cell_owners{0, 0};
    expectLocalCondensedScopeMatchesRowOwnedDenseReference(
        systems::AuxiliaryStateScope::Cell, cell_owners);
}

TEST(AuxiliaryScopeCompletionMPI,
     QuadraturePointLocalCondensationHandlesRanksWithNoOwnedEntities)
{
    const std::vector<int> cell_owners{0, 0};
    expectLocalCondensedScopeMatchesRowOwnedDenseReference(
        systems::AuxiliaryStateScope::QuadraturePoint, cell_owners);
}

TEST(AuxiliaryScopeCompletionMPI,
     RegionIdentityRestrictedDeploymentAndOwnerMapAreRankStable)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "This regression uses two disconnected cells; run with exactly 2 MPI ranks";
    }

    const std::vector<int> cell_owners{1, 0};
    auto mesh = std::make_shared<TwoDisconnectedQuadMeshAccess>(cell_owners, rank);
    const auto topo = twoDisconnectedQuadTopology(cell_owners, rank, size);

    auto model = systems::aux::model("region_owner_mpi",
                                     [](systems::ModelFacade& m) {
        auto x = m.state("x");
        m << systems::ddt(x) ==
            forms::FormExpr::constant(-1.0) * x;
    });

    systems::FESystem sys(mesh);
    auto space = spaces::Space(spaces::SpaceType::H1,
                               ElementType::Quad4,
                               /*order=*/1,
                               /*components=*/1);
    (void)sys.addField(
        systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.deployAuxiliaryModel(
        systems::use(model).name("region_mono")
            .scope(systems::AuxiliaryStateScope::Region)
            .monolithic()
            .initialize({1.0}));
    sys.deployAuxiliaryModel(
        systems::use(model).name("region_mat20_mono")
            .scope(systems::AuxiliaryStateScope::Region)
            .region(materialRegion(20))
            .monolithic()
            .initialize({2.0}));
    sys.deployAuxiliaryModel(
        systems::use(model).name("region_mat20_part")
            .scope(systems::AuxiliaryStateScope::Region)
            .region(materialRegion(20))
            .partitioned("ForwardEuler")
            .initialize({3.0}));

    systems::SetupOptions opts;
    opts.dof_options.global_numbering = dofs::GlobalNumberingMode::DenseGlobalIds;
    opts.dof_options.ownership = dofs::OwnershipStrategy::LowestRank;
    opts.dof_options.my_rank = rank;
    opts.dof_options.world_size = size;
    opts.dof_options.mpi_comm = comm;

    systems::SetupInputs inputs;
    inputs.topology_override = topo;
    sys.setup(opts, inputs);
    sys.finalizeAuxiliaryLayout();

    const auto& mgr = sys.auxiliaryStateManager();
    EXPECT_EQ(mgr.getBlock("region_mono").entityCount(), 2u);
    EXPECT_EQ(mgr.getEntityRemapMetadata("region_mat20_mono").entity_ids,
              (std::vector<std::size_t>{1u}));
    EXPECT_EQ(mgr.getEntityRemapMetadata("region_mat20_part").entity_ids,
              (std::vector<std::size_t>{1u}));

    const auto* registry = sys.auxiliaryOperatorRegistryIfPresent();
    ASSERT_NE(registry, nullptr);
    const auto* all_regions = registry->findLayoutBlock("region_mono");
    ASSERT_NE(all_regions, nullptr);
    EXPECT_EQ(all_regions->row_ownership,
              backends::MixedRowOwnershipPolicy::RegionOwner);
    EXPECT_EQ(all_regions->row_owner_ranks, (std::vector<int>{1, 0}));

    const auto* restricted = registry->findLayoutBlock("region_mat20_mono");
    ASSERT_NE(restricted, nullptr);
    EXPECT_EQ(restricted->row_owner_ranks, (std::vector<int>{0}));

    const int local_owned_region_rows =
        static_cast<int>(std::count(all_regions->row_owner_ranks.begin(),
                                    all_regions->row_owner_ranks.end(),
                                    rank));
    int global_owned_region_rows = 0;
    MPI_Allreduce(&local_owned_region_rows,
                  &global_owned_region_rows,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  comm);
    EXPECT_EQ(global_owned_region_rows, 2);
}

TEST(AuxiliaryScopeCompletionMPI,
     RegionMonolithicBorderedReducedUsesOwnerRoutedRowsWithoutDroppedEntries)
{
#if !defined(FE_HAS_FSILS)
    GTEST_SKIP() << "FSILS backend is not enabled in this build";
#else
    MPI_Comm comm = MPI_COMM_WORLD;
    const int rank = mpiRank(comm);
    const int size = mpiSize(comm);
    if (size != 2) {
        GTEST_SKIP() << "This regression uses two disconnected cells; run with exactly 2 MPI ranks";
    }

    const std::vector<int> cell_owners{1, 0};
    auto mesh = std::make_shared<TwoDisconnectedQuadMeshAccess>(cell_owners, rank);
    const auto topo = twoDisconnectedQuadTopology(cell_owners, rank, size);

    systems::FESystem sys(mesh);
    auto space = spaces::Space(spaces::SpaceType::H1,
                               ElementType::Quad4,
                               /*order=*/1,
                               /*components=*/1);
    const auto u_field = sys.addField(
        systems::FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("op");

    const auto u_disc = forms::FormExpr::discreteField(u_field, *space, "u");
    const auto region_average = sys.regionAverage("region_mono_avg_mpi", u_disc);

    auto model = systems::AuxiliaryModelBuilder("region_monolithic_mpi")
        .input("avg")
        .state("x")
        .ode("x", systems::modelInput("avg") - systems::modelState("x"))
        .build();
    sys.deployAuxiliaryModel(
        systems::use(model).name("region_mono_coupled")
            .scope(systems::AuxiliaryStateScope::Region)
            .monolithic()
            .bind("avg", region_average)
            .initialize({0.0}));

    const auto u = forms::FormExpr::stateField(u_field, *space, "u");
    const auto v = forms::TestFunction(*space, "v");
    const auto residual =
        (forms::inner(forms::grad(u), forms::grad(v)) + u * v).dx();
    (void)systems::installFormulation(sys, "op", {u_field}, residual);

    systems::SetupOptions opts;
    opts.assembler_name = "StandardAssembler";
    opts.assembly_options.deterministic = true;
    opts.assembly_options.overlap_communication = false;
    opts.use_backend_row_ownership_for_assembly = true;
    opts.dof_options.global_numbering = dofs::GlobalNumberingMode::DenseGlobalIds;
    opts.dof_options.ownership = dofs::OwnershipStrategy::LowestRank;
    opts.dof_options.my_rank = rank;
    opts.dof_options.world_size = size;
    opts.dof_options.mpi_comm = comm;

    systems::SetupInputs inputs;
    inputs.topology_override = topo;
    sys.setup(opts, inputs);
    sys.finalizeAuxiliaryLayout();
    sys.beginTimeStep();

    const auto n_dofs = sys.dofHandler().getNumDofs();
    ASSERT_EQ(n_dofs, 8);

    const auto layout = sys.augmentSolverOptions(
        backends::SolverOptions{}, static_cast<std::size_t>(n_dofs));
    ASSERT_TRUE(layout.mixed_block_layout.has_value());
    const auto* aux_block =
        layout.mixed_block_layout->findBlock("region_mono_coupled");
    ASSERT_NE(aux_block, nullptr);
    EXPECT_EQ(aux_block->assembly_mode,
              backends::MixedBlockAssemblyMode::BorderedReduced);
    EXPECT_EQ(aux_block->row_ownership,
              backends::MixedRowOwnershipPolicy::RegionOwner);
    EXPECT_EQ(aux_block->row_owner_ranks, (std::vector<int>{1, 0}));
    EXPECT_TRUE(backends::validateFsilsMixedLayoutContract(
                    *layout.mixed_block_layout, /*dof_per_node=*/1)
                    .empty());

    const auto* dist_pattern = sys.distributedSparsityIfAvailable("op");
    ASSERT_NE(dist_pattern, nullptr);
    auto perm = sys.dofPermutation();
    ASSERT_TRUE(perm);
    ASSERT_FALSE(perm->empty());

    backends::FsilsFactory factory(/*dof_per_node=*/1, perm, comm);
    auto A = factory.createMatrix(*dist_pattern);
    auto b = factory.createVector(n_dofs);
    A->zero();
    b->zero();

    auto* A_fsils = dynamic_cast<backends::FsilsMatrix*>(A.get());
    auto* b_fsils = dynamic_cast<backends::FsilsVector*>(b.get());
    ASSERT_NE(A_fsils, nullptr);
    ASSERT_NE(b_fsils, nullptr);
    EXPECT_TRUE(A_fsils->usesOwnedRowOperator());
    EXPECT_TRUE(b_fsils->usesOwnedRowLayout());

    std::vector<Real> u_values(static_cast<std::size_t>(n_dofs), Real(0.0));
    for (GlobalIndex i = 0; i < n_dofs; ++i) {
        u_values[static_cast<std::size_t>(i)] =
            Real(0.1) + Real(0.05) * Real(i);
    }

    systems::SystemStateView state;
    state.time = 0.0;
    state.dt = 0.1;
    state.u = u_values;

    systems::AssemblyRequest req;
    req.op = "op";
    req.want_matrix = true;
    req.want_vector = true;
    req.is_nonlinear_iteration = true;

    backends::FsilsMatrix::resetDroppedEntryCount();
    backends::FsilsMatrix::resetOffOwnerWriteCount();
    auto A_view = A->createAssemblyView();
    auto b_view = b->createAssemblyView();
    const auto result = sys.assemble(req, state, A_view.get(), b_view.get());
    ASSERT_TRUE(result.success) << result.error_message;
    A->finalizeAssembly();

    const auto off_owner = allreduceUnsigned(
        static_cast<unsigned long long>(backends::FsilsMatrix::offOwnerWriteCount()),
        comm);
    const auto dropped = allreduceUnsigned(
        static_cast<unsigned long long>(backends::FsilsMatrix::droppedEntryCount()),
        comm);
    EXPECT_EQ(off_owner, 0ULL);
    EXPECT_EQ(dropped, 0ULL);

    const auto& bc = sys.borderedCoupling();
    ASSERT_TRUE(bc.active);
    ASSERT_TRUE(bc.globally_reduced);
    EXPECT_FALSE(bc.aux_self_terms_replicated);
    ASSERT_EQ(bc.n_aux, 2);
    EXPECT_EQ(bc.aux_row_owner_ranks, (std::vector<int>{1, 0}));
    EXPECT_EQ(bc.aux_row_owner_routed, (std::vector<char>{char{1}, char{1}}));
    EXPECT_EQ(bc.aux_row_global_contributor_counts, (std::vector<int>{1, 1}));

    ASSERT_EQ(bc.D.size(), 4u);
    EXPECT_NEAR(bc.D[0], 11.0, 1e-11);
    EXPECT_NEAR(bc.D[1], 0.0, 1e-11);
    EXPECT_NEAR(bc.D[2], 0.0, 1e-11);
    EXPECT_NEAR(bc.D[3], 11.0, 1e-11);
    ASSERT_EQ(bc.g.size(), 2u);
    EXPECT_NEAR(bc.g[0], -0.175, 1e-11);
    EXPECT_NEAR(bc.g[1], -0.375, 1e-11);

    ASSERT_EQ(bc.Ct.size(), static_cast<std::size_t>(2 * n_dofs));
    for (GlobalIndex dof = 0; dof < n_dofs; ++dof) {
        const auto col = static_cast<std::size_t>(dof);
        const auto expected_r0 = dof < 4 ? Real(-0.25) : Real(0.0);
        const auto expected_r1 = dof >= 4 ? Real(-0.25) : Real(0.0);
        EXPECT_NEAR(bc.Ct[col], expected_r0, 1e-11)
            << "Region 0 Ct dof " << dof;
        EXPECT_NEAR(bc.Ct[static_cast<std::size_t>(n_dofs) + col],
                    expected_r1,
                    1e-11)
            << "Region 1 Ct dof " << dof;
    }
#endif
}

} // namespace svmp::FE::assembly::testing
