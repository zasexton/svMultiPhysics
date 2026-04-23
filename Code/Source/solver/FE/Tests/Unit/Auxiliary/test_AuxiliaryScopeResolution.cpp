/**
 * @file test_AuxiliaryScopeResolution.cpp
 * @brief Focused tests for FESystem auxiliary scope deployment resolution.
 */

#include <gtest/gtest.h>

#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryInputRegistry.h"
#include "Auxiliary/AuxiliaryModelBuilder.h"
#include "Auxiliary/AuxiliaryStateManager.h"
#include "Backends/Utils/BackendOptions.h"
#include "Dofs/EntityDofMap.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <algorithm>
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

std::shared_ptr<BuiltAuxiliaryModel> buildMixedDAEModel()
{
    return AuxiliaryModelBuilder("scope_mixed_dae")
        .state("x")
        .state("z", AuxiliaryVariableKind::Algebraic)
        .ode("x", -modelState("x") + modelState("z"))
        .algebraic("z", modelState("x") + modelState("z") -
                            svmp::FE::forms::FormExpr::constant(1.0))
        .build();
}

AuxiliaryDeploymentRegion materialRegion(int id)
{
    AuxiliaryDeploymentRegion region;
    region.kind = AuxiliaryRegionKind::MaterialIdSet;
    region.identity = std::to_string(id);
    return region;
}

AuxiliaryDeploymentRegion topologyRegion(std::size_t id)
{
    AuxiliaryDeploymentRegion region;
    region.kind = AuxiliaryRegionKind::TopologyRegion;
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

void expectRestartInvalidWithError(const AuxiliaryRestartSchema& expected,
                                   const AuxiliaryRestartSchema& payload,
                                   const std::string& expected_error_fragment)
{
    const auto validation =
        AuxiliaryTransferOperator::validateRestart(expected, payload);
    EXPECT_FALSE(validation.valid);
    EXPECT_TRUE(std::any_of(validation.errors.begin(),
                            validation.errors.end(),
                            [&](const std::string& error) {
                                return error.find(expected_error_fragment) !=
                                       std::string::npos;
                            }))
        << "Expected validation error containing '" << expected_error_fragment << "'";
}

template <typename Exception>
void expectFinalizeThrowsWithMessage(FESystem& system,
                                     const std::string& expected_fragment)
{
    bool threw_expected = false;
    try {
        system.finalizeAuxiliaryLayout();
    } catch (const Exception& ex) {
        threw_expected = true;
        EXPECT_NE(std::string(ex.what()).find(expected_fragment), std::string::npos)
            << "Expected exception message containing '" << expected_fragment
            << "', got: " << ex.what();
    } catch (const std::exception& ex) {
        ADD_FAILURE() << "Expected different exception type; got: " << ex.what();
        return;
    }
    EXPECT_TRUE(threw_expected)
        << "Expected finalizeAuxiliaryLayout() to throw";
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

    const auto cell_schema = mgr.restartSchema("cell_mat1");
    EXPECT_EQ(cell_schema.scope_name, "Cell");
    EXPECT_EQ(cell_schema.deployment_region_kind, "MaterialIdSet");
    EXPECT_EQ(cell_schema.region_identity, "1");
    EXPECT_EQ(cell_schema.entity_ids, (std::vector<std::size_t>{0u}));

    auto wrong_cell_payload = cell_schema;
    wrong_cell_payload.entity_ids = {1u};
    const auto cell_validation =
        AuxiliaryTransferOperator::validateRestart(cell_schema, wrong_cell_payload);
    EXPECT_FALSE(cell_validation.valid);
    EXPECT_TRUE(std::any_of(cell_validation.errors.begin(),
                            cell_validation.errors.end(),
                            [](const std::string& error) {
                                return error.find("Entity map mismatch") != std::string::npos;
                            }));

    const auto qp_schema = mgr.restartSchema("qp_mat1");
    EXPECT_EQ(qp_schema.scope_name, "QuadraturePoint");
    EXPECT_EQ(qp_schema.deployment_region_kind, "MaterialIdSet");
    EXPECT_EQ(qp_schema.region_identity, "1");
    EXPECT_EQ(qp_schema.entity_ids,
              (std::vector<std::size_t>{0u, 1u, 2u, 3u}));
    EXPECT_EQ(qp_schema.qp_cell_ids, (std::vector<std::size_t>{0u}));
    EXPECT_EQ(qp_schema.qp_offsets, (std::vector<std::size_t>{0u, 4u}));

    auto wrong_qp_offsets = qp_schema;
    wrong_qp_offsets.qp_offsets = {0u, 3u};
    const auto qp_offset_validation =
        AuxiliaryTransferOperator::validateRestart(qp_schema, wrong_qp_offsets);
    EXPECT_FALSE(qp_offset_validation.valid);
    EXPECT_TRUE(std::any_of(qp_offset_validation.errors.begin(),
                            qp_offset_validation.errors.end(),
                            [](const std::string& error) {
                                return error.find("QP offsets mismatch") != std::string::npos;
                            }));

    auto wrong_qp_cells = qp_schema;
    wrong_qp_cells.qp_cell_ids = {1u};
    const auto qp_cell_validation =
        AuxiliaryTransferOperator::validateRestart(qp_schema, wrong_qp_cells);
    EXPECT_FALSE(qp_cell_validation.valid);
    EXPECT_TRUE(std::any_of(qp_cell_validation.errors.begin(),
                            qp_cell_validation.errors.end(),
                            [](const std::string& error) {
                                return error.find("QP covered-cell map mismatch") !=
                                       std::string::npos;
                            }));
}

TEST(AuxiliaryScopeResolution, CellLocalCondensationRejectsEntityLocalAuxiliaryOutputInput)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::SingleTetraMeshAccess>();

    FESystem system(mesh);

    AuxiliaryInputSpec input_spec;
    input_spec.name = "neighbor_aux";
    input_spec.size = 1;
    input_spec.entity_count = 1;
    input_spec.producer = AuxiliaryInputProducer::AuxiliaryOutput;
    system.auxiliaryInputRegistry().registerEntityInput(
        input_spec,
        [](svmp::FE::Real, svmp::FE::Real, std::size_t,
           std::span<svmp::FE::Real> out) {
            out[0] = svmp::FE::Real(0.0);
        });

    auto model = AuxiliaryModelBuilder("cell_aux_output_dependency")
        .input("neighbor_aux")
        .state("x")
        .ode("x", modelInput("neighbor_aux") - modelState("x"))
        .build();

    system.deployAuxiliaryModel(
        use(model).name("cell_bad_coupling").cell().monolithic()
            .entityCount(1)
            .bind("neighbor_aux", "neighbor_aux")
            .initialize({0.0}));

    EXPECT_THROW(system.finalizeAuxiliaryLayout(), svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryScopeResolution, QuadraturePointEntityLocalBindingRequiresCoveredCellMap)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoCellMixedTypeMeshAccess>();

    FESystem system(mesh);

    AuxiliaryInputSpec input_spec;
    input_spec.name = "cell_input";
    input_spec.size = 1;
    input_spec.entity_count = 1;
    input_spec.producer = AuxiliaryInputProducer::DirectUserData;
    system.auxiliaryInputRegistry().registerEntityInput(
        input_spec,
        [](svmp::FE::Real, svmp::FE::Real, std::size_t,
           std::span<svmp::FE::Real> out) {
            out[0] = svmp::FE::Real(0.0);
        });

    auto model = AuxiliaryModelBuilder("qp_cell_input_dependency")
        .input("cell_input")
        .state("x")
        .ode("x", modelInput("cell_input") - modelState("x"))
        .build();

    system.deployAuxiliaryModel(
        use(model).name("qp_bad_cell_map").quadraturePoint()
            .region(materialRegion(2))
            .monolithic()
            .qpOffsets({0u, 1u})
            .bind("cell_input", "cell_input")
            .initialize({0.0}));

    EXPECT_THROW(system.finalizeAuxiliaryLayout(), svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryScopeResolution, RaggedRejectsEntityLocalAuxiliaryOutputCoupling)
{
    auto mesh = std::make_shared<OwnedSubsetNodeMeshAccess>();

    FESystem system(mesh);

    AuxiliaryInputSpec input_spec;
    input_spec.name = "neighbor_aux";
    input_spec.size = 1;
    input_spec.entity_count = 5;
    input_spec.producer = AuxiliaryInputProducer::AuxiliaryOutput;
    system.auxiliaryInputRegistry().registerEntityInput(
        input_spec,
        [](svmp::FE::Real, svmp::FE::Real, std::size_t,
           std::span<svmp::FE::Real> out) {
            out[0] = svmp::FE::Real(0.0);
        });

    auto model = AuxiliaryModelBuilder("ragged_aux_output_coupling")
        .input("neighbor_aux")
        .state("x")
        .ode("x", modelInput("neighbor_aux") - modelState("x"))
        .build();

    system.deployAuxiliaryModel(
        use(model).name("ragged_node_aux_output").node().region(materialRegion(7))
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedEntitySize([](const AuxiliaryRaggedEntityContext&) {
                return 1u;
            })
            .partitioned("ForwardEuler")
            .bind("neighbor_aux", "neighbor_aux")
            .initialize({0.0}));

    expectFinalizeThrowsWithMessage<svmp::FE::InvalidArgumentException>(
        system, "auxiliary-to-auxiliary coupling");
}

TEST(AuxiliaryScopeResolution, RaggedRejectsScopeMismatchedFEBackedEntityLocalBinding)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoCellMixedTypeMeshAccess>();

    FESystem system(mesh);

    AuxiliaryInputSpec input_spec;
    input_spec.name = "sampled_node_field";
    input_spec.size = 1;
    input_spec.entity_count = 2;
    input_spec.producer = AuxiliaryInputProducer::SampledStateField;
    system.auxiliaryInputRegistry().registerEntityInput(
        input_spec,
        [](svmp::FE::Real, svmp::FE::Real, std::size_t,
           std::span<svmp::FE::Real> out) {
            out[0] = svmp::FE::Real(0.0);
        });

    auto model = AuxiliaryModelBuilder("ragged_scope_mismatched_fe_input")
        .input("sampled_node_field")
        .state("x")
        .ode("x", modelInput("sampled_node_field") - modelState("x"))
        .build();

    system.deployAuxiliaryModel(
        use(model).name("ragged_cell_sampled_node_field").cell().region(materialRegion(1))
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedComponentOffsets({0u, 1u})
            .partitioned("ForwardEuler")
            .bind("sampled_node_field", "sampled_node_field")
            .initialize({0.0}));

    expectFinalizeThrowsWithMessage<svmp::FE::InvalidArgumentException>(
        system, "scope-matched quantity provider");
}

TEST(AuxiliaryScopeResolution, RaggedQuadraturePointRejectsIncompatibleQPOffsets)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoCellMixedTypeMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("ragged_qp_bad_layout").quadraturePoint().region(materialRegion(1))
            .qpOffsets({0u, 1u, 2u})
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedEntitySize([](const AuxiliaryRaggedEntityContext&) {
                return 1u;
            })
            .partitioned("ForwardEuler")
            .initialize({0.0}));

    expectFinalizeThrowsWithMessage<svmp::FE::InvalidArgumentException>(
        system, "qpOffsets().size()");
}

TEST(AuxiliaryScopeResolution, RaggedCellMonolithicLocalCondensationFinalizesUniformSlices)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoCellMixedTypeMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("ragged_cell_mono").cell().region(materialRegion(1))
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedComponentOffsets({0u, 1u})
            .monolithic()
            .initialize({1.0}));

    system.finalizeAuxiliaryLayout();

    const auto& block = system.auxiliaryStateManager().getBlock("ragged_cell_mono");
    EXPECT_EQ(block.layoutMode(), AuxiliaryLayoutMode::Ragged);
    EXPECT_EQ(block.entityCount(), 1u);
    EXPECT_EQ(block.storageSize(), 1u);

    auto* registry = system.auxiliaryOperatorRegistryIfPresent();
    ASSERT_NE(registry, nullptr);
    EXPECT_EQ(registry->findLayoutBlock("ragged_cell_mono"), nullptr);
}

TEST(AuxiliaryScopeResolution,
     RaggedQuadraturePointMonolithicLocalCondensationFinalizesUniformSlices)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoCellMixedTypeMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("ragged_qp_mono").quadraturePoint().region(materialRegion(1))
            .qpOffsets({0u, 2u})
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedEntitySize([](const AuxiliaryRaggedEntityContext&) {
                return 1u;
            })
            .monolithic()
            .initialize({2.0}));

    system.finalizeAuxiliaryLayout();

    const auto& mgr = system.auxiliaryStateManager();
    const auto& block = mgr.getBlock("ragged_qp_mono");
    EXPECT_EQ(block.layoutMode(), AuxiliaryLayoutMode::Ragged);
    EXPECT_EQ(block.entityCount(), 2u);
    EXPECT_EQ(block.storageSize(), 2u);
    const auto qp_offsets_span = mgr.getIndexing("ragged_qp_mono").qpOffsets();
    const std::vector<std::size_t> qp_offsets(qp_offsets_span.begin(),
                                              qp_offsets_span.end());
    EXPECT_EQ(qp_offsets, (std::vector<std::size_t>{0u, 2u}));

    auto* registry = system.auxiliaryOperatorRegistryIfPresent();
    ASSERT_NE(registry, nullptr);
    EXPECT_EQ(registry->findLayoutBlock("ragged_qp_mono"), nullptr);
}

TEST(AuxiliaryScopeResolution,
     RaggedCellMonolithicLocalCondensationRejectsVariableWidthSlices)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoCellMixedTypeMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("ragged_cell_mono_bad_width").cell().region(materialRegion(1))
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedComponentOffsets({0u, 2u})
            .monolithic()
            .initialize({1.0}));

    expectFinalizeThrowsWithMessage<svmp::FE::InvalidArgumentException>(
        system, "local-condensation contract");
}

TEST(AuxiliaryScopeResolution,
     RaggedCellMonolithicLocalCondensationRejectsAuxiliaryOutputCoupling)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoCellMixedTypeMeshAccess>();

    FESystem system(mesh);

    AuxiliaryInputSpec input_spec;
    input_spec.name = "neighbor_aux";
    input_spec.size = 1;
    input_spec.entity_count = 2;
    input_spec.producer = AuxiliaryInputProducer::AuxiliaryOutput;
    system.auxiliaryInputRegistry().registerEntityInput(
        input_spec,
        [](svmp::FE::Real, svmp::FE::Real, std::size_t,
           std::span<svmp::FE::Real> out) {
            out[0] = svmp::FE::Real(0.0);
        });

    auto model = AuxiliaryModelBuilder("ragged_cell_aux_output_dependency")
        .input("neighbor_aux")
        .state("x")
        .ode("x", modelInput("neighbor_aux") - modelState("x"))
        .build();

    system.deployAuxiliaryModel(
        use(model).name("ragged_cell_bad_coupling").cell().region(materialRegion(1))
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedComponentOffsets({0u, 1u})
            .monolithic()
            .bind("neighbor_aux", "neighbor_aux")
            .initialize({0.0}));

    expectFinalizeThrowsWithMessage<svmp::FE::InvalidArgumentException>(
        system, "independent local-condensation contract");
}

TEST(AuxiliaryScopeResolution, RaggedNodeMonolithicRejectsMissingBackendOwnership)
{
    auto mesh = std::make_shared<OwnedSubsetNodeMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("ragged_node_mono_no_owner").node().region(materialRegion(7))
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedEntitySize([](const AuxiliaryRaggedEntityContext&) {
                return 1u;
            })
            .monolithic()
            .initialize({1.0}));

    expectFinalizeThrowsWithMessage<svmp::FE::systems::InvalidStateException>(
        system, "backend row ownership metadata");
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

TEST(AuxiliaryScopeResolution, RaggedRestrictedNodeScopeRegistersOwnedGhostOffsets)
{
    auto mesh = std::make_shared<OwnedSubsetNodeMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("ragged_node_mat7").node().region(materialRegion(7))
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedEntitySize([](const AuxiliaryRaggedEntityContext& ctx) {
                return ctx.original_entity_id == 4u ? 3u : 1u;
            })
            .partitioned("ForwardEuler"));

    system.finalizeAuxiliaryLayout();

    const auto& mgr = system.auxiliaryStateManager();
    const auto& block = mgr.getBlock("ragged_node_mat7");
    EXPECT_EQ(block.layoutMode(), AuxiliaryLayoutMode::Ragged);
    EXPECT_EQ(block.entityCount(), 4u);
    EXPECT_EQ(block.ownedEntityCount(), 2u);
    EXPECT_EQ(block.storageSize(), 6u);

    const auto offsets_span =
        mgr.getIndexing("ragged_node_mat7").componentOffsets();
    const std::vector<std::size_t> offsets(offsets_span.begin(),
                                           offsets_span.end());
    EXPECT_EQ(offsets, (std::vector<std::size_t>{0u, 1u, 2u, 3u, 6u}));

    const auto& metadata = mgr.getEntityRemapMetadata("ragged_node_mat7");
    EXPECT_EQ(metadata.entity_ids,
              (std::vector<std::size_t>{1u, 2u, 3u, 4u}));
    EXPECT_EQ(metadata.component_offsets, offsets);

    const auto schema = mgr.restartSchema("ragged_node_mat7");
    EXPECT_EQ(schema.entity_ids,
              (std::vector<std::size_t>{1u, 2u, 3u, 4u}));
    EXPECT_EQ(schema.component_offsets, offsets);

    auto wrong_entity_ids = schema;
    wrong_entity_ids.entity_ids = {1u, 2u, 3u, 99u};
    expectRestartInvalidWithError(schema, wrong_entity_ids, "Entity map mismatch");

    auto wrong_component_offsets = schema;
    wrong_component_offsets.component_offsets = {0u, 1u, 2u, 4u, 6u};
    expectRestartInvalidWithError(
        schema, wrong_component_offsets, "Ragged component offsets mismatch");
}

TEST(AuxiliaryScopeResolution, RaggedRestrictedCellScopeRegistersExplicitOffsets)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoCellMixedTypeMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("ragged_cell_mat1").cell().region(materialRegion(1))
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedComponentOffsets({0u, 2u})
            .partitioned("ForwardEuler")
            .initialize({4.0, 5.0}));

    system.finalizeAuxiliaryLayout();

    const auto& mgr = system.auxiliaryStateManager();
    const auto& block = mgr.getBlock("ragged_cell_mat1");
    EXPECT_EQ(block.layoutMode(), AuxiliaryLayoutMode::Ragged);
    EXPECT_EQ(block.entityCount(), 1u);
    EXPECT_EQ(block.storageSize(), 2u);
    EXPECT_EQ(block.work()[0], svmp::FE::Real(4.0));
    EXPECT_EQ(block.work()[1], svmp::FE::Real(5.0));

    const auto offsets_span =
        mgr.getIndexing("ragged_cell_mat1").componentOffsets();
    const std::vector<std::size_t> offsets(offsets_span.begin(),
                                           offsets_span.end());
    EXPECT_EQ(offsets, (std::vector<std::size_t>{0u, 2u}));

    const auto& metadata = mgr.getEntityRemapMetadata("ragged_cell_mat1");
    EXPECT_EQ(metadata.entity_ids, (std::vector<std::size_t>{0u}));
    EXPECT_EQ(metadata.component_offsets, offsets);

    const auto schema = mgr.restartSchema("ragged_cell_mat1");
    EXPECT_EQ(schema.entity_ids, (std::vector<std::size_t>{0u}));
    EXPECT_EQ(schema.component_offsets, offsets);

    auto wrong_entity_ids = schema;
    wrong_entity_ids.entity_ids = {1u};
    expectRestartInvalidWithError(schema, wrong_entity_ids, "Entity map mismatch");

    auto wrong_component_offsets = schema;
    wrong_component_offsets.component_offsets = {0u, 1u};
    expectRestartInvalidWithError(
        schema, wrong_component_offsets, "Ragged component offsets mismatch");
}

TEST(AuxiliaryScopeResolution, RaggedRestrictedQuadraturePointScopeRegistersOffsets)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoCellMixedTypeMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("ragged_qp_mat1").quadraturePoint().region(materialRegion(1))
            .qpOffsets({0u, 2u})
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedEntitySize([](const AuxiliaryRaggedEntityContext& ctx) {
                return ctx.cell_id + ctx.local_qp_index + 1u;
            })
            .partitioned("ForwardEuler")
            .initialize({2.0}));

    system.finalizeAuxiliaryLayout();

    const auto& mgr = system.auxiliaryStateManager();
    const auto& block = mgr.getBlock("ragged_qp_mat1");
    EXPECT_EQ(block.layoutMode(), AuxiliaryLayoutMode::Ragged);
    EXPECT_EQ(block.entityCount(), 2u);
    EXPECT_EQ(block.storageSize(), 3u);
    EXPECT_EQ(block.work()[0], svmp::FE::Real(2.0));
    EXPECT_EQ(block.work()[1], svmp::FE::Real(2.0));
    EXPECT_EQ(block.work()[2], svmp::FE::Real(2.0));

    const auto offsets_span =
        mgr.getIndexing("ragged_qp_mat1").componentOffsets();
    const std::vector<std::size_t> offsets(offsets_span.begin(),
                                           offsets_span.end());
    EXPECT_EQ(offsets, (std::vector<std::size_t>{0u, 1u, 3u}));
    EXPECT_EQ(mgr.getIndexing("ragged_qp_mat1").qpOffsets().size(), 2u);

    const auto& metadata = mgr.getEntityRemapMetadata("ragged_qp_mat1");
    EXPECT_EQ(metadata.entity_ids, (std::vector<std::size_t>{0u, 1u}));
    EXPECT_EQ(metadata.qp_cell_ids, (std::vector<std::size_t>{0u}));
    EXPECT_EQ(metadata.qp_offsets, (std::vector<std::size_t>{0u, 2u}));
    EXPECT_EQ(metadata.component_offsets, offsets);

    const auto schema = mgr.restartSchema("ragged_qp_mat1");
    EXPECT_EQ(schema.entity_ids, (std::vector<std::size_t>{0u, 1u}));
    EXPECT_EQ(schema.qp_cell_ids, (std::vector<std::size_t>{0u}));
    EXPECT_EQ(schema.qp_offsets, (std::vector<std::size_t>{0u, 2u}));
    EXPECT_EQ(schema.component_offsets, offsets);

    auto wrong_entity_ids = schema;
    wrong_entity_ids.entity_ids = {0u, 2u};
    expectRestartInvalidWithError(schema, wrong_entity_ids, "Entity map mismatch");

    auto wrong_qp_cell_ids = schema;
    wrong_qp_cell_ids.qp_cell_ids = {1u};
    expectRestartInvalidWithError(
        schema, wrong_qp_cell_ids, "QP covered-cell map mismatch");

    auto wrong_qp_offsets = schema;
    wrong_qp_offsets.qp_offsets = {0u, 1u};
    expectRestartInvalidWithError(schema, wrong_qp_offsets, "QP offsets mismatch");

    auto wrong_component_offsets = schema;
    wrong_component_offsets.component_offsets = {0u, 1u, 2u};
    expectRestartInvalidWithError(
        schema, wrong_component_offsets, "Ragged component offsets mismatch");
}

TEST(AuxiliaryScopeResolution, RaggedPartitionedRuntimeAdvancesUniformSlices)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoCellMixedTypeMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    const auto mat1 = materialRegion(1);
    system.deployAuxiliaryModel(
        use(model).name("ragged_runtime_node").node().region(mat1)
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedEntitySize([](const AuxiliaryRaggedEntityContext&) {
                return 1u;
            })
            .partitioned("ForwardEuler")
            .initialize({1.0}));
    system.deployAuxiliaryModel(
        use(model).name("ragged_runtime_cell").cell().region(mat1)
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedComponentOffsets({0u, 1u})
            .partitioned("ForwardEuler")
            .initialize({4.0}));
    system.deployAuxiliaryModel(
        use(model).name("ragged_runtime_qp").quadraturePoint().region(mat1)
            .qpOffsets({0u, 2u})
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedEntitySize([](const AuxiliaryRaggedEntityContext&) {
                return 1u;
            })
            .partitioned("ForwardEuler")
            .initialize({2.0}));

    system.finalizeAuxiliaryLayout();
    system.advanceAuxiliaryState(0.0, 0.25);

    const auto& mgr = system.auxiliaryStateManager();
    const auto& node_block = mgr.getBlock("ragged_runtime_node");
    ASSERT_EQ(node_block.entityCount(), 4u);
    for (std::size_t e = 0; e < node_block.entityCount(); ++e) {
        const auto x = node_block.gatherEntityWork(e);
        ASSERT_EQ(x.size(), 1u);
        EXPECT_NEAR(x[0], 0.75, 1e-12);
    }

    const auto cell_x = mgr.getBlock("ragged_runtime_cell").gatherEntityWork(0);
    ASSERT_EQ(cell_x.size(), 1u);
    EXPECT_NEAR(cell_x[0], 3.0, 1e-12);

    const auto& qp_block = mgr.getBlock("ragged_runtime_qp");
    ASSERT_EQ(qp_block.entityCount(), 2u);
    for (std::size_t e = 0; e < qp_block.entityCount(); ++e) {
        const auto x = qp_block.gatherEntityWork(e);
        ASSERT_EQ(x.size(), 1u);
        EXPECT_NEAR(x[0], 1.5, 1e-12);
    }
}

TEST(AuxiliaryScopeResolution, RaggedPartitionedRuntimeRejectsVariableWidthModelMismatch)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoCellMixedTypeMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("ragged_runtime_bad_width").quadraturePoint()
            .region(materialRegion(1))
            .qpOffsets({0u, 2u})
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedEntitySize([](const AuxiliaryRaggedEntityContext& ctx) {
                return ctx.materialized_entity_index + 1u;
            })
            .partitioned("ForwardEuler")
            .initialize({2.0}));

    system.finalizeAuxiliaryLayout();

    EXPECT_THROW(system.advanceAuxiliaryState(0.0, 0.25),
                 svmp::FE::InvalidArgumentException);
}

TEST(AuxiliaryScopeResolution, RaggedPartitionedMixedDAEInitializesAndAdvancesSlices)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoCellMixedTypeMeshAccess>();
    auto model = buildMixedDAEModel();

    FESystem system(mesh);
    const auto mat1 = materialRegion(1);
    system.deployAuxiliaryModel(
        use(model).name("ragged_dae_node").node().region(mat1)
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedEntitySize([](const AuxiliaryRaggedEntityContext&) {
                return 2u;
            })
            .partitioned("BackwardEuler")
            .initialize({0.2, 0.0}));
    system.deployAuxiliaryModel(
        use(model).name("ragged_dae_cell").cell().region(mat1)
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedComponentOffsets({0u, 2u})
            .partitioned("BackwardEuler")
            .initialize({0.2, 0.0}));
    system.deployAuxiliaryModel(
        use(model).name("ragged_dae_qp").quadraturePoint().region(mat1)
            .qpOffsets({0u, 2u})
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedEntitySize([](const AuxiliaryRaggedEntityContext&) {
                return 2u;
            })
            .partitioned("BackwardEuler")
            .initialize({0.2, 0.0}));

    system.finalizeAuxiliaryLayout();
    system.advanceAuxiliaryState(0.0, 0.1);

    auto expect_block = [&](const std::string& block_name) {
        const auto& block = system.auxiliaryStateManager().getBlock(block_name);
        for (std::size_t e = 0; e < block.entityCount(); ++e) {
            const auto committed = block.gatherEntityCommitted(e);
            ASSERT_EQ(committed.size(), 2u);
            EXPECT_NEAR(committed[0], 0.2, 1e-12);
            EXPECT_NEAR(committed[1], 0.8, 1e-12);

            const auto work = block.gatherEntityWork(e);
            ASSERT_EQ(work.size(), 2u);
            EXPECT_NEAR(work[0], 0.25, 1e-12);
            EXPECT_NEAR(work[1], 0.75, 1e-12);
        }
    };

    expect_block("ragged_dae_node");
    expect_block("ragged_dae_cell");
    expect_block("ragged_dae_qp");
}

TEST(AuxiliaryScopeResolution, RaggedPartitionedMixedDAEFailurePolicyRestoresSlices)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoCellMixedTypeMeshAccess>();
    auto model = buildMixedDAEModel();

    AuxiliaryStepperSpec stepper;
    stepper.method_name = "BackwardEuler";
    stepper.max_nonlinear_iters = 0;

    AuxiliaryFailurePolicy policy;
    policy.max_local_retries = 1;
    policy.reject_timestep_on_failure = false;

    FESystem system(mesh);
    const auto mat1 = materialRegion(1);
    system.deployAuxiliaryModel(
        use(model).name("ragged_dae_fail_node").node().region(mat1)
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedEntitySize([](const AuxiliaryRaggedEntityContext&) {
                return 2u;
            })
            .partitioned("BackwardEuler")
            .stepper(stepper)
            .failurePolicy(policy)
            .initialize({0.2, 0.0}));
    system.deployAuxiliaryModel(
        use(model).name("ragged_dae_fail_cell").cell().region(mat1)
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedComponentOffsets({0u, 2u})
            .partitioned("BackwardEuler")
            .stepper(stepper)
            .failurePolicy(policy)
            .initialize({0.2, 0.0}));
    system.deployAuxiliaryModel(
        use(model).name("ragged_dae_fail_qp").quadraturePoint().region(mat1)
            .qpOffsets({0u, 2u})
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .raggedEntitySize([](const AuxiliaryRaggedEntityContext&) {
                return 2u;
            })
            .partitioned("BackwardEuler")
            .stepper(stepper)
            .failurePolicy(policy)
            .initialize({0.2, 0.0}));

    system.finalizeAuxiliaryLayout();
    EXPECT_NO_THROW(system.advanceAuxiliaryState(0.0, 0.1));

    auto expect_restored_block = [&](const std::string& block_name) {
        const auto& block = system.auxiliaryStateManager().getBlock(block_name);
        for (std::size_t e = 0; e < block.entityCount(); ++e) {
            const auto committed = block.gatherEntityCommitted(e);
            ASSERT_EQ(committed.size(), 2u);
            EXPECT_NEAR(committed[0], 0.2, 1e-12);
            EXPECT_NEAR(committed[1], 0.8, 1e-12);

            const auto work = block.gatherEntityWork(e);
            ASSERT_EQ(work.size(), 2u);
            EXPECT_NEAR(work[0], committed[0], 1e-12);
            EXPECT_NEAR(work[1], committed[1], 1e-12);
        }
    };

    expect_restored_block("ragged_dae_fail_node");
    expect_restored_block("ragged_dae_fail_cell");
    expect_restored_block("ragged_dae_fail_qp");
}

TEST(AuxiliaryScopeResolution, RaggedPartitionedMixedDAEFailurePolicyRejectsFailedSolve)
{
    auto mesh = std::make_shared<svmp::FE::forms::test::TwoCellMixedTypeMeshAccess>();
    auto model = buildMixedDAEModel();

    AuxiliaryStepperSpec stepper;
    stepper.method_name = "BackwardEuler";
    stepper.max_nonlinear_iters = 0;

    AuxiliaryFailurePolicy policy;
    policy.max_local_retries = 1;
    policy.reject_timestep_on_failure = true;

    for (const auto scope : {AuxiliaryStateScope::Node,
                             AuxiliaryStateScope::Cell,
                             AuxiliaryStateScope::QuadraturePoint}) {
        FESystem system(mesh);
        auto deployment = use(model).name("ragged_dae_reject")
            .scope(scope)
            .region(materialRegion(1))
            .layoutMode(AuxiliaryLayoutMode::Ragged)
            .partitioned("BackwardEuler")
            .stepper(stepper)
            .failurePolicy(policy)
            .initialize({0.2, 0.0});
        if (scope == AuxiliaryStateScope::Cell) {
            deployment.raggedComponentOffsets({0u, 2u});
        } else {
            deployment.raggedEntitySize([](const AuxiliaryRaggedEntityContext&) {
                return 2u;
            });
        }
        if (scope == AuxiliaryStateScope::QuadraturePoint) {
            deployment.qpOffsets({0u, 2u});
        }
        system.deployAuxiliaryModel(std::move(deployment));
        system.finalizeAuxiliaryLayout();

        EXPECT_THROW(system.advanceAuxiliaryState(0.0, 0.1),
                     svmp::FE::systems::InvalidStateException);
    }
}

TEST(AuxiliaryScopeResolution, BoundarySetProjectsNodeScopeToBoundaryFaceNodes)
{
    auto mesh = std::make_shared<TwoDisconnectedTetraMeshAccess>();
    auto model = buildScalarDecayModel();

    AuxiliaryDeploymentRegion boundary30;
    boundary30.kind = AuxiliaryRegionKind::BoundarySet;
    boundary30.identity = "30";

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("node_boundary30").node().region(boundary30)
            .partitioned("ForwardEuler").initialize({1.0}));

    system.finalizeAuxiliaryLayout();

    const auto& block = system.auxiliaryStateManager().getBlock("node_boundary30");
    EXPECT_EQ(block.entityCount(), 3u);
    EXPECT_EQ(block.ownedEntityCount(), 3u);
    EXPECT_EQ(system.auxiliaryStateManager()
                  .getEntityRemapMetadata("node_boundary30")
                  .entity_ids,
	              (std::vector<std::size_t>{0u, 1u, 2u}));
}

TEST(AuxiliaryScopeResolution, BoundaryScopeMaterializesSingleBoundaryCollection)
{
    auto mesh = std::make_shared<TwoDisconnectedTetraMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("boundary30").boundary(30)
            .partitioned("ForwardEuler").initialize({1.0}));

    system.finalizeAuxiliaryLayout();

    const auto& mgr = system.auxiliaryStateManager();
    const auto& block = mgr.getBlock("boundary30");
    EXPECT_EQ(block.scope(), AuxiliaryStateScope::Boundary);
    EXPECT_EQ(block.entityCount(), 1u);
    EXPECT_EQ(block.ownedEntityCount(), 1u);

    const auto& indexing = mgr.getIndexing("boundary30");
    EXPECT_EQ(indexing.scope(), AuxiliaryStateScope::Boundary);
    EXPECT_EQ(indexing.totalEntityCount(), 1u);

    const auto& metadata = mgr.getEntityRemapMetadata("boundary30");
    EXPECT_EQ(metadata.scope, AuxiliaryStateScope::Boundary);
    EXPECT_EQ(metadata.deployment_region.kind, AuxiliaryRegionKind::BoundarySet);
    EXPECT_EQ(metadata.deployment_region.identity, "30");
    EXPECT_EQ(metadata.entity_ids, (std::vector<std::size_t>{0u}));

    const auto schema = mgr.restartSchema("boundary30");
    EXPECT_EQ(schema.scope_name, "Boundary");
    EXPECT_EQ(schema.deployment_region_kind, "BoundarySet");
    EXPECT_EQ(schema.region_identity, "30");
    EXPECT_EQ(schema.entity_ids, (std::vector<std::size_t>{0u}));

    auto wrong_marker = schema;
    wrong_marker.region_identity = "40";
    expectRestartInvalidWithError(schema, wrong_marker, "Region mismatch");

    auto wrong_entity_ids = schema;
    wrong_entity_ids.entity_ids = {1u};
    expectRestartInvalidWithError(schema, wrong_entity_ids, "Entity map mismatch");
}

TEST(AuxiliaryScopeResolution, BoundaryScopeAllowsExternalMarkerMetadata)
{
    auto mesh = std::make_shared<TwoDisconnectedTetraMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("boundary999").boundary(999)
            .partitioned("ForwardEuler").initialize({1.0}));

    system.finalizeAuxiliaryLayout();

    const auto& mgr = system.auxiliaryStateManager();
    EXPECT_EQ(mgr.getBlock("boundary999").scope(), AuxiliaryStateScope::Boundary);
    EXPECT_EQ(mgr.getBlock("boundary999").entityCount(), 1u);

    const auto schema = mgr.restartSchema("boundary999");
    EXPECT_EQ(schema.scope_name, "Boundary");
    EXPECT_EQ(schema.deployment_region_kind, "BoundarySet");
    EXPECT_EQ(schema.region_identity, "999");
    EXPECT_EQ(schema.entity_ids, (std::vector<std::size_t>{0u}));
}

TEST(AuxiliaryScopeResolution, BoundarySetProjectsFacetScopeToBoundaryFaces)
{
    auto mesh = std::make_shared<TwoDisconnectedTetraMeshAccess>();
    auto model = buildScalarDecayModel();

    AuxiliaryDeploymentRegion boundary40;
    boundary40.kind = AuxiliaryRegionKind::BoundarySet;
    boundary40.identity = "40";

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("facet_boundary40").facet().region(boundary40)
            .partitioned("ForwardEuler").initialize({1.0}));

    system.finalizeAuxiliaryLayout();

    const auto& mgr = system.auxiliaryStateManager();
    const auto& block = mgr.getBlock("facet_boundary40");
    EXPECT_EQ(block.scope(), AuxiliaryStateScope::Facet);
    EXPECT_EQ(block.entityCount(), 1u);
    EXPECT_EQ(block.ownedEntityCount(), 1u);

    const auto& indexing = mgr.getIndexing("facet_boundary40");
    EXPECT_EQ(indexing.scope(), AuxiliaryStateScope::Facet);
    EXPECT_EQ(indexing.totalEntityCount(), 1u);

    const auto& metadata = mgr.getEntityRemapMetadata("facet_boundary40");
    EXPECT_EQ(metadata.scope, AuxiliaryStateScope::Facet);
    EXPECT_EQ(metadata.deployment_region.kind, AuxiliaryRegionKind::BoundarySet);
    EXPECT_EQ(metadata.deployment_region.identity, "40");
    EXPECT_EQ(metadata.entity_ids, (std::vector<std::size_t>{1u}));

    const auto schema = mgr.restartSchema("facet_boundary40");
    EXPECT_EQ(schema.scope_name, "Facet");
    EXPECT_EQ(schema.deployment_region_kind, "BoundarySet");
    EXPECT_EQ(schema.region_identity, "40");
    EXPECT_EQ(schema.entity_ids, (std::vector<std::size_t>{1u}));

    auto wrong_marker = schema;
    wrong_marker.region_identity = "30";
    expectRestartInvalidWithError(schema, wrong_marker, "Region mismatch");

    auto wrong_entity_ids = schema;
    wrong_entity_ids.entity_ids = {0u};
    expectRestartInvalidWithError(schema, wrong_entity_ids, "Entity map mismatch");
}

TEST(AuxiliaryScopeResolution, MissingBoundarySetRejectsFacetScope)
{
    auto mesh = std::make_shared<TwoDisconnectedTetraMeshAccess>();
    auto model = buildScalarDecayModel();

    AuxiliaryDeploymentRegion boundary999;
    boundary999.kind = AuxiliaryRegionKind::BoundarySet;
    boundary999.identity = "999";

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("facet_boundary999").facet().region(boundary999)
            .partitioned("ForwardEuler").initialize({1.0}));

    expectFinalizeThrowsWithMessage<svmp::FE::systems::InvalidStateException>(
        system, "expanded to 0 entities");
}

TEST(AuxiliaryScopeResolution, TopologyRegionProjectsNodeScopeToRegionNodes)
{
    auto mesh = std::make_shared<TwoDisconnectedTetraMeshAccess>();
    auto model = buildScalarDecayModel();

    FESystem system(mesh);
    system.deployAuxiliaryModel(
        use(model).name("node_topology_region1").node().region(topologyRegion(1))
            .partitioned("ForwardEuler").initialize({1.0}));

    system.finalizeAuxiliaryLayout();

    const auto& mgr = system.auxiliaryStateManager();
    const auto& block = mgr.getBlock("node_topology_region1");
    EXPECT_EQ(block.entityCount(), 4u);
    EXPECT_EQ(block.ownedEntityCount(), 4u);

    const auto& metadata =
        mgr.getEntityRemapMetadata("node_topology_region1");
    EXPECT_EQ(metadata.entity_ids,
              (std::vector<std::size_t>{4u, 5u, 6u, 7u}));

    const auto schema = mgr.restartSchema("node_topology_region1");
    EXPECT_EQ(schema.deployment_region_kind, "TopologyRegion");
    EXPECT_EQ(schema.region_identity, "1");
    EXPECT_EQ(schema.entity_ids,
              (std::vector<std::size_t>{4u, 5u, 6u, 7u}));

    auto wrong_payload = schema;
    wrong_payload.entity_ids = {0u, 1u, 2u, 3u};
    const auto validation =
        AuxiliaryTransferOperator::validateRestart(schema, wrong_payload);
    EXPECT_FALSE(validation.valid);
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
        (svmp::FE::forms::inner(svmp::FE::forms::grad(u_state),
                                svmp::FE::forms::grad(v)) +
         u_state * v).dx());

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

TEST(AuxiliaryScopeResolution, RegionMonolithicFEAverageAssemblesDenseReference)
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
    const auto region_average = system.regionAverage("region_mono_avg", u_disc);

    auto model = AuxiliaryModelBuilder("region_monolithic_average")
        .input("avg")
        .state("x")
        .ode("x", modelInput("avg") - modelState("x"))
        .build();
    system.deployAuxiliaryModel(
        use(model).name("region_mono_inputs").scope(AuxiliaryStateScope::Region)
            .monolithic()
            .bind("avg", region_average)
            .initialize({0.0}));

    const auto u_state = svmp::FE::forms::FormExpr::stateField(
        u_field, *space, "u");
    const auto v = svmp::FE::forms::FormExpr::testFunction(*space, "v");
    (void)installFormulation(
        system,
        "op",
        {u_field},
        (svmp::FE::forms::inner(svmp::FE::forms::grad(u_state),
                                svmp::FE::forms::grad(v)) +
         u_state * v).dx());

    SetupInputs inputs;
    inputs.topology_override = twoDisconnectedTetraTopology();
    system.setup({}, inputs);
    system.finalizeAuxiliaryLayout();

    const auto n_field =
        static_cast<std::size_t>(system.dofHandler().getNumDofs());
    ASSERT_EQ(n_field, 8u);

    std::vector<svmp::FE::Real> solution(n_field, 0.0);
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

    std::vector<svmp::FE::Real> residual;
    std::vector<svmp::FE::Real> matrix;
    system.assembleMixedAuxiliaryDense(state, n_field, residual, matrix);

    const auto n_total = n_field + std::size_t{2};
    ASSERT_EQ(residual.size(), n_total);
    ASSERT_EQ(matrix.size(), n_total * n_total);

    EXPECT_NEAR(residual[n_field + 0u], -2.0, 1e-10);
    EXPECT_NEAR(residual[n_field + 1u], -5.0, 1e-10);
    EXPECT_NEAR(matrix[(n_field + 0u) * n_total + (n_field + 0u)], 2.0, 1e-10);
    EXPECT_NEAR(matrix[(n_field + 1u) * n_total + (n_field + 1u)], 2.0, 1e-10);
    EXPECT_NEAR(matrix[(n_field + 0u) * n_total + (n_field + 1u)], 0.0, 1e-10);
    EXPECT_NEAR(matrix[(n_field + 1u) * n_total + (n_field + 0u)], 0.0, 1e-10);

    auto expect_ct = [&](std::size_t aux_row,
                         std::span<const svmp::FE::GlobalIndex> active_nodes) {
        for (svmp::FE::GlobalIndex node = 0; node < 8; ++node) {
            const auto dofs = entity_map->getVertexDofs(node);
            ASSERT_EQ(dofs.size(), 1u);
            const auto dof = static_cast<std::size_t>(dofs[0] + field_offset);
            const bool active =
                std::find(active_nodes.begin(), active_nodes.end(), node) !=
                active_nodes.end();
            const auto row = n_field + aux_row;
            const auto expected = active ? -0.25 : 0.0;
            EXPECT_NEAR(matrix[row * n_total + dof], expected, 1e-10)
                << "aux row " << aux_row << " field node " << node;
        }
    };
    expect_ct(0u, r0_nodes);
    expect_ct(1u, r1_nodes);
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
