/**
 * @file test_AuxiliaryScopeResolution.cpp
 * @brief Focused tests for FESystem auxiliary scope deployment resolution.
 */

#include <gtest/gtest.h>

#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryModelBuilder.h"
#include "Auxiliary/AuxiliaryStateManager.h"
#include "Backends/Utils/BackendOptions.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace svmp::FE::systems;

namespace {

std::shared_ptr<BuiltAuxiliaryModel> buildScalarDecayModel()
{
    return AuxiliaryModelBuilder("scope_decay")
        .state("x")
        .ode("x", -modelState("x"))
        .build();
}

AuxiliaryDeploymentRegion materialRegion(int id)
{
    AuxiliaryDeploymentRegion region;
    region.kind = AuxiliaryRegionKind::MaterialIdSet;
    region.identity = std::to_string(id);
    return region;
}

class NonsmoothScalarModel final : public AuxiliaryStateModel {
public:
    [[nodiscard]] std::string modelName() const override { return "NonsmoothScalarModel"; }
    [[nodiscard]] int dimension() const override { return 1; }

    [[nodiscard]] AuxiliaryStructuralMetadata structuralMetadata() const override
    {
        AuxiliaryStructuralMetadata meta;
        meta.variable_kinds = {AuxiliaryVariableKind::Algebraic};
        meta.has_nonsmooth = true;
        return meta;
    }

    void evaluateResidual(const AuxiliaryLocalContext& ctx,
                          AuxiliaryResidualRequest& request) const override
    {
        request.residual[0] = ctx.x[0];
    }

    [[nodiscard]] bool hasNonsmoothHooks() const override { return true; }

    std::vector<svmp::FE::Real>
    evaluateComplementarity(const AuxiliaryLocalContext& ctx) const override
    {
        return {ctx.x[0]};
    }
};

class OwnedSubsetNodeMeshAccess final : public svmp::FE::assembly::IMeshAccess {
public:
    [[nodiscard]] svmp::FE::GlobalIndex numCells() const override { return 1; }
    [[nodiscard]] svmp::FE::GlobalIndex numOwnedCells() const override { return 1; }
    [[nodiscard]] svmp::FE::GlobalIndex numVertices() const override { return 5; }
    [[nodiscard]] svmp::FE::GlobalIndex numOwnedVertices() const override { return 3; }
    [[nodiscard]] svmp::FE::GlobalIndex numBoundaryFaces() const override { return 0; }
    [[nodiscard]] svmp::FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(svmp::FE::GlobalIndex /*cell_id*/) const override
    {
        return true;
    }

    [[nodiscard]] svmp::FE::ElementType getCellType(svmp::FE::GlobalIndex /*cell_id*/) const override
    {
        return svmp::FE::ElementType::Tetra4;
    }

    [[nodiscard]] int getCellDomainId(svmp::FE::GlobalIndex /*cell_id*/) const override
    {
        return 7;
    }

    void getCellNodes(svmp::FE::GlobalIndex /*cell_id*/,
                      std::vector<svmp::FE::GlobalIndex>& nodes) const override
    {
        nodes = {1, 2, 3, 4};
    }

    [[nodiscard]] std::array<svmp::FE::Real, 3>
    getNodeCoordinates(svmp::FE::GlobalIndex node_id) const override
    {
        return {static_cast<svmp::FE::Real>(node_id), 0.0, 0.0};
    }

    void getCellCoordinates(
        svmp::FE::GlobalIndex cell_id,
        std::vector<std::array<svmp::FE::Real, 3>>& coords) const override
    {
        std::vector<svmp::FE::GlobalIndex> nodes;
        getCellNodes(cell_id, nodes);
        coords.clear();
        coords.reserve(nodes.size());
        for (const auto node : nodes) {
            coords.push_back(getNodeCoordinates(node));
        }
    }

    [[nodiscard]] svmp::FE::LocalIndex
    getLocalFaceIndex(svmp::FE::GlobalIndex /*face_id*/,
                      svmp::FE::GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(svmp::FE::GlobalIndex /*face_id*/) const override
    {
        return -1;
    }

    [[nodiscard]] std::pair<svmp::FE::GlobalIndex, svmp::FE::GlobalIndex>
    getInteriorFaceCells(svmp::FE::GlobalIndex /*face_id*/) const override
    {
        return {-1, -1};
    }

    void forEachCell(std::function<void(svmp::FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachOwnedCell(std::function<void(svmp::FE::GlobalIndex)> callback) const override
    {
        callback(0);
    }

    void forEachBoundaryFace(
        int /*marker*/,
        std::function<void(svmp::FE::GlobalIndex, svmp::FE::GlobalIndex)> /*callback*/) const override
    {
    }

    void forEachInteriorFace(
        std::function<void(svmp::FE::GlobalIndex,
                           svmp::FE::GlobalIndex,
                           svmp::FE::GlobalIndex)> /*callback*/) const override
    {
    }
};

class TwoDisconnectedTetraMeshAccess final : public svmp::FE::assembly::IMeshAccess {
public:
    TwoDisconnectedTetraMeshAccess()
    {
        nodes_ = {
            std::array<svmp::FE::Real, 3>{0.0, 0.0, 0.0},
            std::array<svmp::FE::Real, 3>{1.0, 0.0, 0.0},
            std::array<svmp::FE::Real, 3>{0.0, 1.0, 0.0},
            std::array<svmp::FE::Real, 3>{0.0, 0.0, 1.0},
            std::array<svmp::FE::Real, 3>{3.0, 0.0, 0.0},
            std::array<svmp::FE::Real, 3>{4.0, 0.0, 0.0},
            std::array<svmp::FE::Real, 3>{3.0, 1.0, 0.0},
            std::array<svmp::FE::Real, 3>{3.0, 0.0, 1.0},
        };
    }

    [[nodiscard]] svmp::FE::GlobalIndex numCells() const override { return 2; }
    [[nodiscard]] svmp::FE::GlobalIndex numOwnedCells() const override { return 2; }
    [[nodiscard]] svmp::FE::GlobalIndex numBoundaryFaces() const override { return 2; }
    [[nodiscard]] svmp::FE::GlobalIndex numInteriorFaces() const override { return 0; }
    [[nodiscard]] int dimension() const override { return 3; }

    [[nodiscard]] bool isOwnedCell(svmp::FE::GlobalIndex /*cell_id*/) const override
    {
        return true;
    }

    [[nodiscard]] svmp::FE::ElementType getCellType(svmp::FE::GlobalIndex /*cell_id*/) const override
    {
        return svmp::FE::ElementType::Tetra4;
    }

    [[nodiscard]] int getCellDomainId(svmp::FE::GlobalIndex cell_id) const override
    {
        return cell_id == 0 ? 10 : 20;
    }

    void getCellNodes(svmp::FE::GlobalIndex cell_id,
                      std::vector<svmp::FE::GlobalIndex>& nodes) const override
    {
        if (cell_id == 0) {
            nodes = {0, 1, 2, 3};
        } else {
            nodes = {4, 5, 6, 7};
        }
    }

    [[nodiscard]] std::array<svmp::FE::Real, 3>
    getNodeCoordinates(svmp::FE::GlobalIndex node_id) const override
    {
        return nodes_.at(static_cast<std::size_t>(node_id));
    }

    void getCellCoordinates(
        svmp::FE::GlobalIndex cell_id,
        std::vector<std::array<svmp::FE::Real, 3>>& coords) const override
    {
        std::vector<svmp::FE::GlobalIndex> nodes;
        getCellNodes(cell_id, nodes);
        coords.clear();
        coords.reserve(nodes.size());
        for (const auto node : nodes) {
            coords.push_back(getNodeCoordinates(node));
        }
    }

    [[nodiscard]] svmp::FE::LocalIndex
    getLocalFaceIndex(svmp::FE::GlobalIndex /*face_id*/,
                      svmp::FE::GlobalIndex /*cell_id*/) const override
    {
        return 0;
    }

    [[nodiscard]] int getBoundaryFaceMarker(svmp::FE::GlobalIndex face_id) const override
    {
        return face_id == 0 ? 30 : 40;
    }

    [[nodiscard]] std::pair<svmp::FE::GlobalIndex, svmp::FE::GlobalIndex>
    getInteriorFaceCells(svmp::FE::GlobalIndex /*face_id*/) const override
    {
        return {-1, -1};
    }

    void forEachCell(std::function<void(svmp::FE::GlobalIndex)> callback) const override
    {
        callback(0);
        callback(1);
    }

    void forEachOwnedCell(std::function<void(svmp::FE::GlobalIndex)> callback) const override
    {
        forEachCell(std::move(callback));
    }

    void forEachBoundaryFace(
        int marker,
        std::function<void(svmp::FE::GlobalIndex, svmp::FE::GlobalIndex)> callback) const override
    {
        if (marker == 30) {
            callback(0, 0);
        } else if (marker == 40) {
            callback(1, 1);
        }
    }

    void forEachInteriorFace(
        std::function<void(svmp::FE::GlobalIndex,
                           svmp::FE::GlobalIndex,
                           svmp::FE::GlobalIndex)> /*callback*/) const override
    {
    }

private:
    std::vector<std::array<svmp::FE::Real, 3>> nodes_;
};

svmp::FE::dofs::MeshTopologyInfo twoDisconnectedTetraTopology()
{
    svmp::FE::dofs::MeshTopologyInfo topo;
    topo.n_cells = 2;
    topo.n_vertices = 8;
    topo.dim = 3;
    topo.cell2vertex_offsets = {0, 4, 8};
    topo.cell2vertex_data = {0, 1, 2, 3, 4, 5, 6, 7};
    topo.vertex_gids = {0, 1, 2, 3, 4, 5, 6, 7};
    topo.cell_gids = {0, 1};
    topo.cell_owner_ranks = {0, 0};
    return topo;
}

} // namespace

TEST(AuxiliaryScopeResolution, MaterialIdSetProjectsToCellNodeAndRegionScopes)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoCellMixedTypeMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    const auto mat1 = materialRegion(1);

    system.deployAuxiliaryModel(
        use(model).name("cell_mat1").cell().region(mat1)
            .partitioned("ForwardEuler").initialize({1.0}));
    system.deployAuxiliaryModel(
        use(model).name("node_mat1").node().region(mat1)
            .partitioned("ForwardEuler").initialize({1.0}));
    system.deployAuxiliaryModel(
        use(model).name("region_mat1").scope(AuxiliaryStateScope::Region).region(mat1)
            .partitioned("ForwardEuler").initialize({1.0}));
    system.deployAuxiliaryModel(
        use(model).name("qp_mat1").scope(AuxiliaryStateScope::QuadraturePoint).region(mat1)
            .qpOffsets({0u, 4u})
            .partitioned("ForwardEuler").initialize({1.0}));

    system.finalizeAuxiliaryLayout();

    const auto& mgr = system.auxiliaryStateManager();
    EXPECT_EQ(mgr.getBlock("cell_mat1").entityCount(), 1u);
    EXPECT_EQ(mgr.getBlock("node_mat1").entityCount(), 4u);
    EXPECT_EQ(mgr.getBlock("region_mat1").entityCount(), 1u);
    EXPECT_EQ(mgr.getBlock("qp_mat1").entityCount(), 4u);
    EXPECT_EQ(mgr.getIndexing("qp_mat1").qpOffsets().size(), 2u);
    EXPECT_EQ(mgr.getEntityRemapMetadata("cell_mat1").entity_ids,
              (std::vector<std::size_t>{0u}));
    EXPECT_EQ(mgr.getEntityRemapMetadata("node_mat1").entity_ids,
              (std::vector<std::size_t>{0u, 1u, 2u, 3u}));
    EXPECT_EQ(mgr.getEntityRemapMetadata("region_mat1").entity_ids,
              (std::vector<std::size_t>{0u}));
    EXPECT_EQ(mgr.getEntityRemapMetadata("qp_mat1").qp_cell_ids,
              (std::vector<std::size_t>{0u}));
    EXPECT_EQ(mgr.getEntityRemapMetadata("qp_mat1").qp_offsets,
              (std::vector<std::size_t>{0u, 4u}));
}

TEST(AuxiliaryScopeResolution, RestrictedNodeScopePreservesOwnedGhostSplit)
{
    auto mesh = std::make_shared<OwnedSubsetNodeMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("node_mat7").node().region(materialRegion(7))
            .partitioned("ForwardEuler").initialize({1.0}));

    system.finalizeAuxiliaryLayout();

    const auto& block = system.auxiliaryStateManager().getBlock("node_mat7");
    EXPECT_EQ(block.entityCount(), 4u);
    EXPECT_EQ(block.ownedEntityCount(), 2u);
}

TEST(AuxiliaryScopeResolution, RegionScopeMaterializesDisconnectedTopologyRegions)
{
    auto mesh = std::make_shared<TwoDisconnectedTetraMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("regions").scope(AuxiliaryStateScope::Region)
            .partitioned("ForwardEuler").initialize({1.0}));

    system.finalizeAuxiliaryLayout();

    const auto& block = system.auxiliaryStateManager().getBlock("regions");
    EXPECT_EQ(block.entityCount(), 2u);
    EXPECT_EQ(block.ownedEntityCount(), 2u);
}

TEST(AuxiliaryScopeResolution, RegionEntityMetadataUsesTopologyIdsAndMembership)
{
    auto mesh = std::make_shared<TwoDisconnectedTetraMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("regions").scope(AuxiliaryStateScope::Region)
            .partitioned("ForwardEuler").initialize({1.0}));
    system.deployAuxiliaryModel(
        use(model).name("region_mat20").scope(AuxiliaryStateScope::Region)
            .region(materialRegion(20))
            .partitioned("ForwardEuler").initialize({2.0}));

    system.finalizeAuxiliaryLayout();

    const auto& mgr = system.auxiliaryStateManager();
    const auto& all_meta = mgr.getEntityRemapMetadata("regions");
    EXPECT_EQ(all_meta.entity_ids, (std::vector<std::size_t>{0u, 1u}));
    ASSERT_EQ(all_meta.region_membership.size(), 2u);
    EXPECT_EQ(all_meta.region_membership[0].region_id, 0u);
    EXPECT_EQ(all_meta.region_membership[0].cell_ids, (std::vector<std::size_t>{0u}));
    EXPECT_EQ(all_meta.region_membership[0].node_ids,
              (std::vector<std::size_t>{0u, 1u, 2u, 3u}));
    EXPECT_EQ(all_meta.region_membership[0].boundary_markers, (std::vector<int>{30}));
    EXPECT_EQ(all_meta.region_membership[1].region_id, 1u);
    EXPECT_EQ(all_meta.region_membership[1].cell_ids, (std::vector<std::size_t>{1u}));
    EXPECT_EQ(all_meta.region_membership[1].node_ids,
              (std::vector<std::size_t>{4u, 5u, 6u, 7u}));
    EXPECT_EQ(all_meta.region_membership[1].boundary_markers, (std::vector<int>{40}));

    const auto& restricted_meta = mgr.getEntityRemapMetadata("region_mat20");
    EXPECT_EQ(restricted_meta.entity_ids, (std::vector<std::size_t>{1u}));
    ASSERT_EQ(restricted_meta.region_membership.size(), 1u);
    EXPECT_EQ(restricted_meta.region_membership[0].region_id, 1u);
    EXPECT_EQ(restricted_meta.region_membership[0].cell_ids,
              (std::vector<std::size_t>{1u}));

    const auto schema = mgr.restartSchema("region_mat20");
    EXPECT_EQ(schema.deployment_region_kind, "MaterialIdSet");
    EXPECT_EQ(schema.region_identity, "20");
    EXPECT_EQ(schema.entity_ids, (std::vector<std::size_t>{1u}));
    ASSERT_EQ(schema.region_membership.size(), 1u);
    EXPECT_EQ(schema.region_membership[0].node_ids,
              (std::vector<std::size_t>{4u, 5u, 6u, 7u}));

    auto wrong_payload = schema;
    wrong_payload.entity_ids = {0u};
    const auto validation =
        AuxiliaryTransferOperator::validateRestart(schema, wrong_payload);
    EXPECT_FALSE(validation.valid);
    EXPECT_TRUE(std::any_of(validation.errors.begin(), validation.errors.end(),
                            [](const std::string& error) {
                                return error.find("Entity map mismatch") != std::string::npos;
                            }));
}

TEST(AuxiliaryScopeResolution, RegionLocalFEAverageFeedsEachTopologyRegionEntity)
{
    auto mesh = std::make_shared<TwoDisconnectedTetraMeshAccess>();
    auto space = std::make_shared<svmp::FE::spaces::H1Space>(
        svmp::FE::ElementType::Tetra4, 1);

    FESystem system(mesh);
    const auto u_field = system.addField(
        FieldSpec{.name = "u", .space = space, .components = 1});
    system.addOperator("op");

    const auto u_disc = svmp::FE::forms::FormExpr::discreteField(
        u_field, *space, "u");
    const auto region_average = system.regionAverage("region_u_avg", u_disc);

    auto model = AuxiliaryModelBuilder("region_input_copy")
        .input("avg")
        .state("x")
        .ode("x", modelInput("avg"))
        .build();
    system.deployAuxiliaryModel(
        use(model).name("region_inputs").scope(AuxiliaryStateScope::Region)
            .partitioned("ForwardEuler")
            .bind("avg", region_average)
            .initialize({0.0}));

    const auto u_state = svmp::FE::forms::FormExpr::stateField(
        u_field, *space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(
        system,
        "op",
        {u_field},
        svmp::FE::forms::inner(svmp::FE::forms::grad(u_state),
                               svmp::FE::forms::grad(v)).dx());

    SetupInputs inputs;
    inputs.topology_override = twoDisconnectedTetraTopology();
    system.setup({}, inputs);
    system.finalizeAuxiliaryLayout();

    std::vector<svmp::FE::Real> solution(
        static_cast<std::size_t>(system.dofHandler().getNumDofs()), 0.0);
    const auto* entity_map =
        system.fieldDofHandler(u_field).getEntityDofMap();
    ASSERT_NE(entity_map, nullptr);
    const auto field_offset = system.fieldDofOffset(u_field);
    auto set_region_nodes = [&](std::span<const svmp::FE::GlobalIndex> nodes,
                                svmp::FE::Real value) {
        for (const auto node : nodes) {
            const auto dofs = entity_map->getVertexDofs(node);
            ASSERT_EQ(dofs.size(), 1u);
            const auto idx = static_cast<std::size_t>(dofs[0] + field_offset);
            ASSERT_LT(idx, solution.size());
            solution[idx] = value;
        }
    };
    const std::array<svmp::FE::GlobalIndex, 4> r0_nodes{0, 1, 2, 3};
    const std::array<svmp::FE::GlobalIndex, 4> r1_nodes{4, 5, 6, 7};
    set_region_nodes(r0_nodes, 2.0);
    set_region_nodes(r1_nodes, 5.0);

    SystemStateView state;
    state.time = 0.0;
    state.dt = 1.0;
    state.u = solution;

    system.prepareAuxiliaryForAssembly(state, false);
    auto* registry = system.auxiliaryInputRegistryIfPresent();
    ASSERT_NE(registry, nullptr);
    ASSERT_TRUE(registry->isEntityLocal("region_u_avg"));
    EXPECT_NEAR(registry->valuesOf("region_u_avg", 0)[0], 2.0, 1e-10);
    EXPECT_NEAR(registry->valuesOf("region_u_avg", 1)[0], 5.0, 1e-10);

    system.advanceAuxiliaryState(state);

    const auto& block = system.auxiliaryStateManager().getBlock("region_inputs");
    ASSERT_EQ(block.entityCount(), 2u);
    EXPECT_NEAR(block.gatherEntityWork(0)[0], 2.0, 1e-10);
    EXPECT_NEAR(block.gatherEntityWork(1)[0], 5.0, 1e-10);
}

TEST(AuxiliaryScopeResolution, RegionMonolithicLayoutHasDeterministicRowOwners)
{
    auto mesh = std::make_shared<TwoDisconnectedTetraMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("regions_mono").scope(AuxiliaryStateScope::Region)
            .monolithic().initialize({1.0}));
    system.deployAuxiliaryModel(
        use(model).name("region_mat20_mono").scope(AuxiliaryStateScope::Region)
            .region(materialRegion(20))
            .monolithic().initialize({2.0}));

    system.finalizeAuxiliaryLayout();

    auto* registry = system.auxiliaryOperatorRegistryIfPresent();
    ASSERT_NE(registry, nullptr);
    ASSERT_TRUE(registry->isLayoutFinalized());

    const auto* all_regions = registry->findLayoutBlock("regions_mono");
    ASSERT_NE(all_regions, nullptr);
    EXPECT_EQ(all_regions->scope, AuxiliaryStateScope::Region);
    EXPECT_EQ(all_regions->row_ownership,
              svmp::FE::backends::MixedRowOwnershipPolicy::RegionOwner);
    EXPECT_EQ(all_regions->row_owner_ranks, (std::vector<int>{0, 0}));

    const auto* restricted = registry->findLayoutBlock("region_mat20_mono");
    ASSERT_NE(restricted, nullptr);
    EXPECT_EQ(restricted->row_ownership,
              svmp::FE::backends::MixedRowOwnershipPolicy::RegionOwner);
    EXPECT_EQ(restricted->row_owner_ranks, (std::vector<int>{0}));

    const auto options =
        system.augmentSolverOptions(svmp::FE::backends::SolverOptions{}, 0u);
    ASSERT_TRUE(options.mixed_block_layout.has_value());
    const auto* mixed_region =
        options.mixed_block_layout->findBlock("regions_mono");
    ASSERT_NE(mixed_region, nullptr);
    EXPECT_EQ(mixed_region->row_owner_ranks, (std::vector<int>{0, 0}));
}

TEST(AuxiliaryScopeResolution, MonolithicNonsmoothModelIsRejectedAtFinalize)
{
    FESystem system(std::shared_ptr<const svmp::FE::assembly::IMeshAccess>{});
    auto model = std::make_shared<NonsmoothScalarModel>();

    system.deployAuxiliaryModel(
        use(model).name("nonsmooth_mono").global().monolithic().initialize({0.0}));

    EXPECT_THROW(system.finalizeAuxiliaryLayout(), svmp::FE::NotImplementedException);
}
