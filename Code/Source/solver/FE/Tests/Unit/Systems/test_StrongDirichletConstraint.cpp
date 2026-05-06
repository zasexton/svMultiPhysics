/**
 * @file test_StrongDirichletConstraint.cpp
 * @brief Unit tests for Systems strong Dirichlet lowering from Forms declarations
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Constraints/SystemConstraints.h"
#include "Constraints/VertexDirichletConstraint.h"
#include "Dofs/EntityDofMap.h"
#include "Auxiliary/AuxiliaryBindings.h"
#include "Auxiliary/AuxiliaryModelBuilder.h"
#include "Auxiliary/AuxiliaryStateManager.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"

#include "Forms/Vocabulary.h"
#include "Spaces/H1Space.h"

#include "Mesh/Mesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellShape.h"

#include <array>
#include <memory>
#include <span>
#include <string>
#include <vector>

using svmp::CellFamily;
using svmp::CellShape;
using svmp::EntityKind;
using svmp::Mesh;
using svmp::MeshBase;

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;

using svmp::FE::assembly::DenseMatrixView;
using svmp::FE::assembly::MassKernel;

using svmp::FE::forms::FormExpr;
using svmp::FE::forms::deltat;
using svmp::FE::forms::t;
using svmp::FE::forms::bc::strongDirichlet;
using svmp::FE::systems::AuxiliaryConstraintValueSource;
using svmp::FE::systems::AuxiliaryModelBuilder;
using svmp::FE::systems::use;

using svmp::FE::spaces::H1Space;

using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::SystemStateView;

namespace {

std::shared_ptr<Mesh> build_single_quad_mesh_with_left_edge_marker(int marker)
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, {shape});
    base->finalize();

    // Find left edge (vertices {0,3}) and mark it.
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base->n_faces()); ++f) {
        const auto verts = base->face_vertices(f);
        if (verts.size() != 2u) {
            continue;
        }
        const bool has0 = (verts[0] == 0 || verts[1] == 0);
        const bool has3 = (verts[0] == 3 || verts[1] == 3);
        if (has0 && has3) {
            base->set_boundary_label(f, marker);
            base->add_to_set(EntityKind::Face, "left", f);
            break;
        }
    }

    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<Mesh> build_single_quad_mesh_with_vertex_gids()
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, {shape});
    base->set_vertex_gids({10, 20, 30, 40});
    base->finalize();

    return svmp::create_mesh(std::move(base));
}

} // namespace

TEST(VertexDirichletConstraint, GlobalVertexGidConstrainsScalarH1Vertex)
{
    auto mesh = build_single_quad_mesh_with_vertex_gids();
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));

    std::vector<svmp::FE::constraints::VertexDirichletValue> values = {
        {.vertex_id = 30, .value = 3.25},
    };
    sys.addSystemConstraint(std::make_unique<svmp::FE::constraints::VertexDirichletConstraint>(
        u, std::move(values), svmp::FE::constraints::VertexIdMode::GlobalVertexGid));

    sys.setup();

    const auto* entity = sys.fieldDofHandler(u).getEntityDofMap();
    ASSERT_NE(entity, nullptr);
    const auto vertex_dofs = entity->getVertexDofs(2);
    ASSERT_EQ(vertex_dofs.size(), 1u);
    const auto dof = vertex_dofs.front() + sys.fieldDofOffset(u);

    EXPECT_TRUE(sys.constraints().isConstrained(dof));
    EXPECT_NEAR(sys.constraints().getInhomogeneity(dof), 3.25, 1e-12);
}

TEST(VertexDirichletConstraint, MissingGlobalVertexGidFailsDuringSetup)
{
    auto mesh = build_single_quad_mesh_with_vertex_gids();
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));

    std::vector<svmp::FE::constraints::VertexDirichletValue> values = {
        {.vertex_id = 999, .value = 1.0},
    };
    sys.addSystemConstraint(std::make_unique<svmp::FE::constraints::VertexDirichletConstraint>(
        u, std::move(values), svmp::FE::constraints::VertexIdMode::GlobalVertexGid));

    EXPECT_THROW(sys.setup(), std::invalid_argument);
}

TEST(StrongDirichletConstraint, InstalledBeforeSetupAffectsAssembly)
{
    const int marker = 7;
    auto mesh = build_single_quad_mesh_with_left_edge_marker(marker);
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));

    const auto bc = strongDirichlet(u, marker, FormExpr::constant(0.0), "u");
    svmp::FE::systems::installStrongDirichlet(sys, std::span<const svmp::FE::forms::bc::StrongDirichlet>(&bc, 1));

    sys.setup();

    DenseMatrixView mass(sys.dofHandler().getNumDofs());
    SystemStateView state;
    sys.assembleMass(state, mass);

    const auto constrained = svmp::FE::constraints::boundaryDofsByMarker(*mesh, sys.dofHandler(), marker);
    ASSERT_EQ(constrained.size(), 2u);
    for (auto dof : constrained) {
        EXPECT_TRUE(sys.constraints().isConstrained(dof));
        EXPECT_NEAR(mass.getMatrixEntry(dof, dof), 1.0, 1e-12);
        for (GlobalIndex j = 0; j < mass.numCols(); ++j) {
            if (j == dof) continue;
            EXPECT_NEAR(mass.getMatrixEntry(dof, j), 0.0, 1e-12);
        }
    }
}

TEST(StrongDirichletConstraint, UpdateConstraintsRefreshesTimeDependentValues)
{
    const int marker = 7;
    auto mesh = build_single_quad_mesh_with_left_edge_marker(marker);
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));

    const auto bc = strongDirichlet(u, marker, t() + deltat(), "u");
    svmp::FE::systems::installStrongDirichlet(sys, std::span<const svmp::FE::forms::bc::StrongDirichlet>(&bc, 1));

    sys.setup();

    EXPECT_TRUE(sys.hasTimeDependentConstraints());
    EXPECT_TRUE(sys.requiresTimeAdvancement());
    EXPECT_FALSE(sys.isTransient());

    const auto constrained = svmp::FE::constraints::boundaryDofsByMarker(*mesh, sys.dofHandler(), marker);
    ASSERT_EQ(constrained.size(), 2u);

    for (auto dof : constrained) {
        EXPECT_NEAR(sys.constraints().getInhomogeneity(dof), 0.0, 1e-15);
    }

    sys.updateConstraints(/*time=*/2.0, /*dt=*/0.5);
    for (auto dof : constrained) {
        EXPECT_NEAR(sys.constraints().getInhomogeneity(dof), 2.5, 1e-15);
    }
}

TEST(StrongDirichletConstraint, AuxiliaryStateBindingDrivesStrongDirichlet)
{
    const int marker = 7;
    auto mesh = build_single_quad_mesh_with_left_edge_marker(marker);
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});

    auto model = AuxiliaryModelBuilder("state_bc")
        .state("x")
        .ode("x", FormExpr::constant(0.0))
        .build();

    sys.deployAuxiliaryModel(
        use(model).name("state_bc_inst").global().partitioned("ForwardEuler")
            .initialize({1.25})
            .drivesStrongDirichlet(u, marker, "x", -1, AuxiliaryConstraintValueSource::State));
    sys.finalizeAuxiliaryLayout();
    sys.setup();

    const auto constrained = svmp::FE::constraints::boundaryDofsByMarker(*mesh, sys.dofHandler(), marker);
    ASSERT_EQ(constrained.size(), 2u);
    for (const auto dof : constrained) {
        EXPECT_TRUE(sys.constraints().isConstrained(dof));
        EXPECT_NEAR(sys.constraints().getInhomogeneity(dof), 1.25, 1e-12);
    }

    auto& blk = sys.auxiliaryStateManager().getBlock("state_bc_inst");
    auto work = blk.work();
    ASSERT_EQ(work.size(), 1u);
    work[0] = 2.5;

    sys.updateConstraints(/*time=*/0.0, /*dt=*/0.0);
    for (const auto dof : constrained) {
        EXPECT_NEAR(sys.constraints().getInhomogeneity(dof), 2.5, 1e-12);
    }
}

TEST(StrongDirichletConstraint, AuxiliaryOutputBindingDrivesStrongDirichlet)
{
    const int marker = 7;
    auto mesh = build_single_quad_mesh_with_left_edge_marker(marker);
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});

    auto model = AuxiliaryModelBuilder("output_bc")
        .state("x")
        .ode("x", FormExpr::constant(0.0))
        .output("y", svmp::FE::systems::modelState("x") * FormExpr::constant(2.0))
        .build();

    sys.deployAuxiliaryModel(
        use(model).name("output_bc_inst").global().partitioned("ForwardEuler")
            .initialize({1.0})
            .drivesStrongDirichlet(u, marker, "y"));
    sys.finalizeAuxiliaryLayout();
    sys.setup();

    const auto constrained = svmp::FE::constraints::boundaryDofsByMarker(*mesh, sys.dofHandler(), marker);
    ASSERT_EQ(constrained.size(), 2u);
    for (const auto dof : constrained) {
        EXPECT_NEAR(sys.constraints().getInhomogeneity(dof), 2.0, 1e-12);
    }

    auto& blk = sys.auxiliaryStateManager().getBlock("output_bc_inst");
    auto work = blk.work();
    ASSERT_EQ(work.size(), 1u);
    work[0] = 1.75;

    sys.updateConstraints(/*time=*/0.0, /*dt=*/0.0);
    for (const auto dof : constrained) {
        EXPECT_NEAR(sys.constraints().getInhomogeneity(dof), 3.5, 1e-12);
    }
}

TEST(StrongDirichletConstraint, AuxiliaryDrivenBindingRejectsUnknownOutput)
{
    const int marker = 7;
    auto mesh = build_single_quad_mesh_with_left_edge_marker(marker);
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});

    auto model = AuxiliaryModelBuilder("bad_output_bc")
        .state("x")
        .ode("x", FormExpr::constant(0.0))
        .output("y", svmp::FE::systems::modelState("x"))
        .build();

    EXPECT_THROW(
        sys.deployAuxiliaryModel(
            use(model).name("bad_output_bc_inst").global().partitioned("ForwardEuler")
                .initialize({1.0})
                .drivesStrongDirichlet(u, marker, "missing_output")),
        svmp::FE::InvalidArgumentException);
}
