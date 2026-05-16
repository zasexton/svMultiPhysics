/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/AssemblyKernel.h"
#include "Constraints/LevelSetActiveSideVertexDirichletConstraint.h"
#include "Assembly/CutIntegrationContext.h"
#include "Assembly/GlobalSystemView.h"
#include "Dofs/EntityDofMap.h"
#include "Geometry/CutQuadrature.h"
#include "Mesh/Fields/MeshFields.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"
#include "Spaces/H1Space.h"
#include "Systems/FESystem.h"

#include <memory>
#include <unordered_set>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {
namespace test {

namespace {

std::shared_ptr<Mesh> buildTwoQuadStripWithCutLeftCell()
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<real_t> x_ref = {
        0.0, 0.0,
        1.0, 0.0,
        2.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0,
    };
    const std::vector<offset_t> cell2vertex_offsets = {0, 4, 8};
    const std::vector<index_t> cell2vertex = {
        0, 1, 4, 3,
        1, 2, 5, 4,
    };

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(
        /*spatial_dim=*/2,
        x_ref,
        cell2vertex_offsets,
        cell2vertex,
        std::vector<CellShape>(2, shape));
    base->finalize();

    const auto phi_handle = MeshFields::attach_field(
        *base,
        EntityKind::Vertex,
        "phi",
        FieldScalarType::Float64,
        1);
    auto* phi = MeshFields::field_data_as<real_t>(*base, phi_handle);
    phi[0] = -1.0;
    phi[1] = 1.0;
    phi[2] = 1.0;
    phi[3] = -1.0;
    phi[4] = 1.0;
    phi[5] = 1.0;

    return create_mesh(std::move(base));
}

[[nodiscard]] GlobalIndex vertexDof(const systems::FESystem& system,
                                    FieldId field,
                                    GlobalIndex vertex)
{
    const auto* entity = system.fieldDofHandler(field).getEntityDofMap();
    EXPECT_NE(entity, nullptr);
    if (entity == nullptr) {
        return GlobalIndex{-1};
    }
    const auto dofs = entity->getVertexDofs(vertex);
    EXPECT_EQ(dofs.size(), 1u);
    if (dofs.size() != 1u) {
        return GlobalIndex{-1};
    }
    return system.fieldDofOffset(field) + dofs.front();
}

[[nodiscard]] GlobalIndex findEdge(const Mesh& mesh,
                                   index_t first_vertex,
                                   index_t second_vertex)
{
    const auto& base = mesh.local_mesh();
    for (GlobalIndex edge = 0;
         edge < static_cast<GlobalIndex>(base.n_edges());
         ++edge) {
        const auto vertices = base.edge_vertices(static_cast<index_t>(edge));
        if ((vertices[0] == first_vertex && vertices[1] == second_vertex) ||
            (vertices[0] == second_vertex && vertices[1] == first_vertex)) {
            return edge;
        }
    }
    return GlobalIndex{-1};
}

[[nodiscard]] GlobalIndex edgeDof(const systems::FESystem& system,
                                  FieldId field,
                                  GlobalIndex edge)
{
    const auto* entity = system.fieldDofHandler(field).getEntityDofMap();
    EXPECT_NE(entity, nullptr);
    if (entity == nullptr) {
        return GlobalIndex{-1};
    }
    const auto dofs = entity->getEdgeDofs(edge);
    EXPECT_EQ(dofs.size(), 1u);
    if (dofs.size() != 1u) {
        return GlobalIndex{-1};
    }
    return system.fieldDofOffset(field) + dofs.front();
}

[[nodiscard]] std::size_t expectInactiveEdgeDofsConstrained(
    const systems::FESystem& system,
    FieldId field,
    const std::unordered_set<GlobalIndex>& active_support)
{
    const auto* entity = system.fieldDofHandler(field).getEntityDofMap();
    EXPECT_NE(entity, nullptr);
    if (entity == nullptr) {
        return 0u;
    }

    std::size_t inactive_edge_dofs = 0u;
    const auto offset = system.fieldDofOffset(field);
    for (GlobalIndex edge = 0; edge < entity->numEdges(); ++edge) {
        for (const auto local_dof : entity->getEdgeDofs(edge)) {
            if (active_support.find(local_dof) != active_support.end()) {
                continue;
            }
            ++inactive_edge_dofs;
            EXPECT_TRUE(system.constraints().isConstrained(offset + local_dof));
        }
    }
    return inactive_edge_dofs;
}

void expectRowsAreIdentity(const assembly::DenseMatrixView& matrix,
                           const std::vector<GlobalIndex>& rows)
{
    for (const auto row : rows) {
        EXPECT_NEAR(matrix.getMatrixEntry(row, row), 1.0, 1.0e-12);
        for (GlobalIndex column = 0; column < matrix.numCols(); ++column) {
            if (column == row) {
                continue;
            }
            SCOPED_TRACE(::testing::Message() << "row=" << row
                                              << ", column=" << column);
            EXPECT_NEAR(matrix.getMatrixEntry(row, column), 0.0, 1.0e-12);
        }
    }
}

} // namespace

TEST(LevelSetActiveSideVertexDirichletConstraint,
     ConstrainsOnlyVerticesWithoutActiveCellSupport)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = buildTwoQuadStripWithCutLeftCell();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(
        std::make_unique<LevelSetActiveSideVertexDirichletConstraint>(
            pressure,
            "phi",
            LevelSetConstraintSide::Negative,
            Real{0.0},
            Real{0.0}));

    ASSERT_NO_THROW(system.setup());

    const auto& constraints = system.constraints();
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, pressure, 0)));
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, pressure, 1)));
    EXPECT_TRUE(constraints.isConstrained(vertexDof(system, pressure, 2)));
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, pressure, 3)));
    EXPECT_FALSE(constraints.isConstrained(vertexDof(system, pressure, 4)));
    EXPECT_TRUE(constraints.isConstrained(vertexDof(system, pressure, 5)));
#endif
}

TEST(LevelSetActiveSideVertexDirichletConstraint,
     RebuildUpdatesVerticesWithoutActiveCellSupport)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = buildTwoQuadStripWithCutLeftCell();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(
        std::make_unique<LevelSetActiveSideVertexDirichletConstraint>(
            pressure,
            "phi",
            LevelSetConstraintSide::Negative,
            Real{0.0},
            Real{0.0}));

    ASSERT_NO_THROW(system.setup());
    EXPECT_TRUE(system.constraints().isConstrained(vertexDof(system, pressure, 2)));
    EXPECT_TRUE(system.constraints().isConstrained(vertexDof(system, pressure, 5)));

    const auto phi_handle =
        MeshFields::get_field_handle(mesh->local_mesh(), EntityKind::Vertex, "phi");
    auto* phi = MeshFields::field_data_as<real_t>(mesh->local_mesh(), phi_handle);
    ASSERT_NE(phi, nullptr);
    phi[2] = -1.0;
    phi[5] = -1.0;

    ASSERT_NO_THROW(system.rebuildConstraintState());
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 0)));
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 1)));
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 2)));
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 3)));
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 4)));
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 5)));
#endif
}

TEST(LevelSetActiveSideVertexDirichletConstraint,
     RebuildUsesRetainedCutVolumeSupportWhenContextIsAvailable)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    constexpr int marker = 51;
    auto mesh = buildTwoQuadStripWithCutLeftCell();
    const auto phi_handle =
        MeshFields::get_field_handle(mesh->local_mesh(), EntityKind::Vertex, "phi");
    auto* phi = MeshFields::field_data_as<real_t>(mesh->local_mesh(), phi_handle);
    ASSERT_NE(phi, nullptr);
    phi[2] = -1.0;
    phi[5] = -1.0;

    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(
        std::make_unique<LevelSetActiveSideVertexDirichletConstraint>(
            pressure,
            "phi",
            LevelSetConstraintSide::Negative,
            Real{0.0},
            Real{0.0},
            marker));

    ASSERT_NO_THROW(system.setup());
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 2)));
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 5)));

    auto context = std::make_shared<assembly::CutIntegrationContext>();
    assembly::CutCellAssemblyMetadata metadata{};
    metadata.cell = 0;
    metadata.parent_entity = 0;
    metadata.side = geometry::CutIntegrationSide::Negative;
    metadata.volume_fraction = Real{0.25};
    geometry::CutQuadratureRule rule{};
    rule.kind = geometry::CutQuadratureKind::Volume;
    rule.side = geometry::CutIntegrationSide::Negative;
    rule.measure = Real{0.25};
    rule.parent_measure = Real{1.0};
    rule.volume_fraction = Real{0.25};
    context->addGeneratedVolumeRule(marker, metadata, rule);
    auto pruned_metadata = metadata;
    pruned_metadata.cell = 1;
    pruned_metadata.parent_entity = 1;
    pruned_metadata.volume_fraction =
        assembly::CutIntegrationContext::minGeneratedCutVolumeFraction() *
        Real{0.5};
    auto pruned_rule = rule;
    pruned_rule.measure = pruned_metadata.volume_fraction;
    pruned_rule.volume_fraction = pruned_metadata.volume_fraction;
    context->addGeneratedVolumeRule(marker, pruned_metadata, pruned_rule);
    EXPECT_EQ(context->generatedPrunedVolumeRuleCount(), 1u);

    system.setCutIntegrationContext(std::move(context));
    ASSERT_NO_THROW(system.rebuildConstraintState());

    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 0)));
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 1)));
    EXPECT_TRUE(system.constraints().isConstrained(vertexDof(system, pressure, 2)));
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 3)));
    EXPECT_FALSE(system.constraints().isConstrained(vertexDof(system, pressure, 4)));
    EXPECT_TRUE(system.constraints().isConstrained(vertexDof(system, pressure, 5)));
#endif
}

TEST(LevelSetActiveSideVertexDirichletConstraint,
     ConstrainsHigherOrderDofsWithoutActiveCellSupport)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = buildTwoQuadStripWithCutLeftCell();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/2);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(
        std::make_unique<LevelSetActiveSideVertexDirichletConstraint>(
            pressure,
            "phi",
            LevelSetConstraintSide::Negative,
            Real{0.0},
            Real{0.0}));

    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    ASSERT_NO_THROW(system.setup());
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    const auto& pressure_dofs = system.fieldDofHandler(pressure);
    const auto offset = system.fieldDofOffset(pressure);
    const auto active_cell_dofs = pressure_dofs.getCellDofs(0);
    const auto dry_cell_dofs = pressure_dofs.getCellDofs(1);
    ASSERT_FALSE(active_cell_dofs.empty());
    ASSERT_FALSE(dry_cell_dofs.empty());

    std::unordered_set<GlobalIndex> active_support(
        active_cell_dofs.begin(), active_cell_dofs.end());
    for (const auto local_dof : active_cell_dofs) {
        EXPECT_FALSE(system.constraints().isConstrained(offset + local_dof));
    }

    std::size_t dry_only_dofs = 0u;
    for (const auto local_dof : dry_cell_dofs) {
        if (active_support.find(local_dof) != active_support.end()) {
            continue;
        }
        ++dry_only_dofs;
        EXPECT_TRUE(system.constraints().isConstrained(offset + local_dof));
    }
    EXPECT_GT(dry_only_dofs, 0u);

    const auto shared_edge = findEdge(*mesh, 1, 4);
    ASSERT_GE(shared_edge, 0);
    EXPECT_FALSE(system.constraints().isConstrained(
        edgeDof(system, pressure, shared_edge)));
    EXPECT_EQ(expectInactiveEdgeDofsConstrained(system, pressure, active_support),
              3u);

    EXPECT_NE(log_output.find("active_support_edge_dofs=4"), std::string::npos);
    EXPECT_NE(log_output.find("active_support_cell_dofs=1"), std::string::npos);
    EXPECT_NE(log_output.find("inactive_edge_dofs=3"), std::string::npos);
    EXPECT_NE(log_output.find("inactive_cell_dofs=1"), std::string::npos);
    EXPECT_NE(log_output.find("constrained_owned_edge_dofs=3"), std::string::npos);
    EXPECT_NE(log_output.find("constrained_owned_cell_dofs=1"), std::string::npos);
    EXPECT_NE(log_output.find("active_support_face_dofs=0"), std::string::npos);
#endif
}

TEST(LevelSetActiveSideVertexDirichletConstraint,
     RetainedCutVolumeSupportKeepsSharedP2EdgeDofActive)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    constexpr int marker = 52;
    auto mesh = buildTwoQuadStripWithCutLeftCell();
    const auto phi_handle =
        MeshFields::get_field_handle(mesh->local_mesh(), EntityKind::Vertex, "phi");
    auto* phi = MeshFields::field_data_as<real_t>(mesh->local_mesh(), phi_handle);
    ASSERT_NE(phi, nullptr);
    phi[2] = -1.0;
    phi[5] = -1.0;

    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/2);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addSystemConstraint(
        std::make_unique<LevelSetActiveSideVertexDirichletConstraint>(
            pressure,
            "phi",
            LevelSetConstraintSide::Negative,
            Real{0.0},
            Real{0.0},
            marker));

    ASSERT_NO_THROW(system.setup());
    const auto shared_edge = findEdge(*mesh, 1, 4);
    ASSERT_GE(shared_edge, 0);
    EXPECT_FALSE(system.constraints().isConstrained(
        edgeDof(system, pressure, shared_edge)));

    auto context = std::make_shared<assembly::CutIntegrationContext>();
    assembly::CutCellAssemblyMetadata metadata{};
    metadata.cell = 0;
    metadata.parent_entity = 0;
    metadata.side = geometry::CutIntegrationSide::Negative;
    metadata.volume_fraction = Real{0.25};
    geometry::CutQuadratureRule rule{};
    rule.kind = geometry::CutQuadratureKind::Volume;
    rule.side = geometry::CutIntegrationSide::Negative;
    rule.measure = Real{0.25};
    rule.parent_measure = Real{1.0};
    rule.volume_fraction = Real{0.25};
    context->addGeneratedVolumeRule(marker, metadata, rule);
    auto pruned_metadata = metadata;
    pruned_metadata.cell = 1;
    pruned_metadata.parent_entity = 1;
    pruned_metadata.volume_fraction =
        assembly::CutIntegrationContext::minGeneratedCutVolumeFraction() *
        Real{0.5};
    auto pruned_rule = rule;
    pruned_rule.measure = pruned_metadata.volume_fraction;
    pruned_rule.volume_fraction = pruned_metadata.volume_fraction;
    context->addGeneratedVolumeRule(marker, pruned_metadata, pruned_rule);

    system.setCutIntegrationContext(std::move(context));
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    ASSERT_NO_THROW(system.rebuildConstraintState());
    auto log_output = testing::internal::GetCapturedStdout();
    log_output += testing::internal::GetCapturedStderr();

    EXPECT_FALSE(system.constraints().isConstrained(
        edgeDof(system, pressure, shared_edge)));
    const auto active_cell_dofs = system.fieldDofHandler(pressure).getCellDofs(0);
    const std::unordered_set<GlobalIndex> active_support(
        active_cell_dofs.begin(), active_cell_dofs.end());
    EXPECT_EQ(expectInactiveEdgeDofsConstrained(system, pressure, active_support),
              3u);
    EXPECT_NE(log_output.find("support_mode=retained_cut_volume"), std::string::npos);
    EXPECT_NE(log_output.find("active_support_edge_dofs=4"), std::string::npos);
    EXPECT_NE(log_output.find("inactive_edge_dofs=3"), std::string::npos);
    EXPECT_NE(log_output.find("inactive_cell_dofs=1"), std::string::npos);
#endif
}

TEST(LevelSetActiveSideVertexDirichletConstraint,
     ConstrainedHighOrderDryPressureRowsAssembleIdentityDiagonal)
{
#if !(defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH)
    GTEST_SKIP() << "Requires FE built with Mesh integration.";
#else
    auto mesh = buildTwoQuadStripWithCutLeftCell();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/2);

    systems::FESystem system(mesh);
    const auto pressure = system.addField(
        systems::FieldSpec{.name = "p", .space = space, .components = 1});
    system.addOperator("pressure");
    system.addCellKernel(
        "pressure",
        pressure,
        std::make_shared<assembly::MassKernel>(1.0));
    system.addSystemConstraint(
        std::make_unique<LevelSetActiveSideVertexDirichletConstraint>(
            pressure,
            "phi",
            LevelSetConstraintSide::Negative,
            Real{0.0},
            Real{0.0}));

    systems::SetupOptions setup_options;
    setup_options.sparsity_options.ensure_diagonal = false;
    ASSERT_NO_THROW(system.setup(setup_options));

    const auto& pressure_dofs = system.fieldDofHandler(pressure);
    const auto offset = system.fieldDofOffset(pressure);
    const auto active_cell_dofs = pressure_dofs.getCellDofs(0);
    const std::unordered_set<GlobalIndex> active_support(
        active_cell_dofs.begin(), active_cell_dofs.end());

    std::vector<GlobalIndex> constrained_dry_pressure_dofs;
    for (GlobalIndex local_dof = 0; local_dof < pressure_dofs.getNumDofs();
         ++local_dof) {
        if (active_support.find(local_dof) != active_support.end()) {
            continue;
        }
        const auto global_dof = offset + local_dof;
        ASSERT_TRUE(system.constraints().isConstrained(global_dof));
        constrained_dry_pressure_dofs.push_back(global_dof);
    }
    ASSERT_FALSE(constrained_dry_pressure_dofs.empty());

    const auto& pressure_pattern = system.sparsity("pressure");
    for (const auto global_dof : constrained_dry_pressure_dofs) {
        EXPECT_TRUE(pressure_pattern.hasDiagonal(global_dof))
            << "missing constrained dry pressure diagonal for DOF " << global_dof;
    }

    assembly::DenseMatrixView matrix(system.dofHandler().getNumDofs());
    matrix.zero();
    systems::SystemStateView state;
    systems::AssemblyRequest request;
    request.op = "pressure";
    request.want_matrix = true;
    ASSERT_NO_THROW((void)system.assemble(request, state, &matrix, nullptr));

    expectRowsAreIdentity(matrix, constrained_dry_pressure_dofs);
#endif
}

} // namespace test
} // namespace constraints
} // namespace FE
} // namespace svmp
