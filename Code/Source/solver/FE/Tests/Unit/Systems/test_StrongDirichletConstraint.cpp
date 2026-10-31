/**
 * @file test_StrongDirichletConstraint.cpp
 * @brief Unit tests for Systems strong Dirichlet lowering from Forms declarations
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/FormsInstaller.h"
#include "Systems/TimeSeries.h"
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
#include <functional>
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

std::shared_ptr<Mesh> build_single_quad9_mesh_with_boundary_only_left_edge_marker(int marker)
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
        0.5, 0.0,
        1.0, 0.5,
        0.5, 1.0,
        0.0, 0.5,
        0.5, 0.5
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 9};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 2;
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, {shape});

    CellShape face_shape{};
    face_shape.family = CellFamily::Line;
    face_shape.num_corners = 2;
    face_shape.order = 1;
    base->set_faces_from_arrays({face_shape},
                                {0, 2},
                                {3, 0},
                                {std::array<svmp::index_t, 2>{0, svmp::INVALID_INDEX}},
                                svmp::MeshCodim1StorageMode::BoundaryOnly);
    base->set_boundary_label(0, marker);

    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<Mesh> build_single_tetra10_mesh_with_boundary_only_face_marker(int marker)
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.5, 0.0, 0.0,
        0.5, 0.5, 0.0,
        0.0, 0.5, 0.0,
        0.0, 0.0, 0.5,
        0.5, 0.0, 0.5,
        0.0, 0.5, 0.5,
    };
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 10};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    CellShape shape{};
    shape.family = CellFamily::Tetra;
    shape.num_corners = 4;
    shape.order = 2;
    base->build_from_arrays(/*spatial_dim=*/3, X_ref, cell2vertex_offsets, cell2vertex, {shape});

    CellShape face_shape{};
    face_shape.family = CellFamily::Triangle;
    face_shape.num_corners = 3;
    face_shape.order = 2;
    base->set_faces_from_arrays({face_shape},
                                {0, 6},
                                {0, 1, 2, 4, 5, 6},
                                {std::array<svmp::index_t, 2>{0, svmp::INVALID_INDEX}},
                                svmp::MeshCodim1StorageMode::BoundaryOnly);
    base->set_boundary_label(0, marker);

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

TEST(StrongDirichletConstraint, BoundaryOnlyQuadraticLineLabelSurvivesFullTopologySetup)
{
    const int marker = 7;
    auto mesh = build_single_quad9_mesh_with_boundary_only_left_edge_marker(marker);
    auto space = std::make_shared<H1Space>(ElementType::Quad9, /*order=*/2);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));

    const auto bc = strongDirichlet(u, marker, FormExpr::constant(0.0), "u");
    svmp::FE::systems::installStrongDirichlet(
        sys, std::span<const svmp::FE::forms::bc::StrongDirichlet>(&bc, 1));

    sys.setup();

    EXPECT_EQ(mesh->base().codim1_storage_mode(), svmp::MeshCodim1StorageMode::Full);

    std::size_t labelled_faces = 0;
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(mesh->n_faces()); ++f) {
        if (static_cast<int>(mesh->base().boundary_label(f)) == marker) {
            ++labelled_faces;
        }
    }
    EXPECT_EQ(labelled_faces, 1u);
    EXPECT_EQ(sys.constraints().numConstraints(), 3);

    const auto boundary_dofs =
        svmp::FE::constraints::boundaryDofsByMarker(*mesh, sys.dofHandler(), marker);
    EXPECT_EQ(boundary_dofs.size(), 3u);
}

TEST(StrongDirichletConstraint, BoundaryOnlyTetra10Triangle6LabelConstrainsCornerAndEdgeDofs)
{
    const int marker = 7;
    auto mesh = build_single_tetra10_mesh_with_boundary_only_face_marker(marker);
    auto space = std::make_shared<H1Space>(ElementType::Tetra10, /*order=*/2);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});

    const auto bc = strongDirichlet(u, marker, FormExpr::constant(0.0), "u");
    svmp::FE::systems::installStrongDirichlet(
        sys, std::span<const svmp::FE::forms::bc::StrongDirichlet>(&bc, 1));

    sys.setup();

    const auto boundary_dofs =
        svmp::FE::constraints::boundaryDofsByMarker(*mesh, sys.dofHandler(), marker);
    EXPECT_EQ(boundary_dofs.size(), 6u);
    EXPECT_EQ(sys.constraints().numConstraints(), 6);
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

TEST(StrongDirichletConstraint, RebuildConstraintStatePreservesLastTimeDependentValues)
{
    const int marker = 7;
    auto mesh = build_single_quad_mesh_with_left_edge_marker(marker);
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));

    const auto bc = strongDirichlet(u, marker, t() + deltat(), "u");
    svmp::FE::systems::installStrongDirichlet(
        sys,
        std::span<const svmp::FE::forms::bc::StrongDirichlet>(&bc, 1));

    sys.setup();
    sys.updateConstraints(/*time=*/2.0, /*dt=*/0.5);

    const auto constrained =
        svmp::FE::constraints::boundaryDofsByMarker(*mesh, sys.dofHandler(), marker);
    ASSERT_EQ(constrained.size(), 2u);
    for (auto dof : constrained) {
        ASSERT_NEAR(sys.constraints().getInhomogeneity(dof), 2.5, 1e-15);
    }

    sys.rebuildConstraintState();
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

namespace {

/// The system DOF of field @p field at vertex @p vertex (component 0).
GlobalIndex vertexDof(const FESystem& sys, svmp::FE::FieldId field, GlobalIndex vertex, bool field_local = false)
{
    const auto* map = sys.fieldDofHandler(field).getEntityDofMap();
    EXPECT_NE(map, nullptr);
    const auto dofs = map->getVertexDofs(vertex);
    EXPECT_FALSE(dofs.empty());
    return dofs[0] + (field_local ? 0 : sys.fieldDofOffset(field));
}

/// One scalar Q1 field "u" on the single quad, u = value on the left edge (marker 7).
struct StrongValueSystem {
    std::shared_ptr<Mesh> mesh{};
    std::shared_ptr<H1Space> space{};
    std::unique_ptr<FESystem> system{};
    svmp::FE::FieldId u{svmp::FE::INVALID_FIELD_ID};
};

StrongValueSystem buildStrongValueSystem(const std::function<FormExpr(FESystem&, const StrongValueSystem&)>& value)
{
    StrongValueSystem out;
    out.mesh = build_single_quad_mesh_with_left_edge_marker(7);
    out.space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    out.system = std::make_unique<FESystem>(out.mesh);
    out.u = out.system->addField(FieldSpec{.name = "u", .space = out.space, .components = 1});
    out.system->addOperator("mass");
    out.system->addCellKernel("mass", out.u, std::make_shared<MassKernel>(1.0));
    const auto bc = strongDirichlet(out.u, 7, value(*out.system, out), "u");
    svmp::FE::systems::installStrongDirichlet(
        *out.system, std::span<const svmp::FE::forms::bc::StrongDirichlet>(&bc, 1));
    return out;
}

} // namespace

TEST(StrongDirichletConstraint, TimeFunctionParameterGivesTheCallbackInhomogeneitiesBitForBit)
{
    using svmp::FE::Real;
    const svmp::FE::systems::TimeSeries flow({0.0, 0.25, 0.5, 1.0}, {0.0, 3.7, -1.1, 0.0},
                                             svmp::FE::systems::TimeSeriesEnd::Periodic);
    const svmp::FE::forms::ScalarCoefficient profile = [](Real x, Real y, Real) {
        return 1.25 + 0.7 * y + 0.1 * y * y + 0.3 * x;
    };

    // Q(t) w(x): a time function times a spatial coefficient, against the one
    // callback the Navier-Stokes inflow uses today, flow(t) * profile(x).
    auto parameter = buildStrongValueSystem([&](FESystem& sys, const StrongValueSystem&) {
        return svmp::FE::systems::timeFunction(sys, "Q", flow) * FormExpr::coefficient("w", profile);
    });
    auto callback = buildStrongValueSystem([&](FESystem&, const StrongValueSystem&) {
        return FormExpr::coefficient("Qw", svmp::FE::forms::TimeScalarCoefficient([flow, profile](Real x, Real y,
                                                                                                    Real z, Real t) {
                                         return flow(t) * profile(x, y, z);
                                     }));
    });
    parameter.system->setup();
    callback.system->setup();
    EXPECT_TRUE(parameter.system->hasTimeDependentConstraints());

    const auto dofs = svmp::FE::constraints::boundaryDofsByMarker(*parameter.mesh, parameter.system->dofHandler(), 7);
    ASSERT_EQ(dofs.size(), 2u);
    bool nonzero = false;
    for (const double time : {0.0, 0.13, 0.37, 0.9, 1.61, 2.25}) {
        parameter.system->updateConstraints(time, 0.01);
        callback.system->updateConstraints(time, 0.01);
        for (const auto dof : dofs) {
            const auto value = parameter.system->constraints().getInhomogeneity(dof);
            EXPECT_EQ(value, callback.system->constraints().getInhomogeneity(dof)) << "t=" << time;
            nonzero = nonzero || value != 0.0;
        }
    }
    EXPECT_TRUE(nonzero);
}

TEST(StrongDirichletConstraint, PrescribedFieldInTheValueIsReadAtTheNodes)
{
    using svmp::FE::Real;
    svmp::FE::FieldId w = svmp::FE::INVALID_FIELD_ID;
    auto s = buildStrongValueSystem([&](FESystem& sys, const StrongValueSystem& out) {
        FieldSpec spec{.name = "w", .space = out.space, .components = 1};
        spec.source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData;
        w = sys.addField(spec);
        return FormExpr::discreteField(w, *out.space, "w") * FormExpr::constant(2.0) + t();
    });
    s.system->setup();
    EXPECT_TRUE(s.system->hasTimeDependentConstraints());

    // Before the coefficients are set the value reads zeros.
    const GlobalIndex left_vertices[] = {0, 3};
    for (const auto vertex : left_vertices) {
        EXPECT_EQ(s.system->constraints().getInhomogeneity(vertexDof(*s.system, s.u, vertex)), 0.0);
    }

    const auto n_w = static_cast<std::size_t>(s.system->fieldDofHandler(w).getNumDofs());
    std::vector<Real> coefficients(n_w);
    for (std::size_t i = 0; i < n_w; ++i) {
        coefficients[i] = 10.0 + 1.5 * static_cast<Real>(i);
    }
    s.system->setPrescribedFieldCoefficients(w, coefficients);
    s.system->updateConstraints(/*time=*/1.5, /*dt=*/0.1);
    for (const auto vertex : left_vertices) {
        const auto w_dof = static_cast<std::size_t>(vertexDof(*s.system, w, vertex, /*field_local=*/true));
        EXPECT_EQ(s.system->constraints().getInhomogeneity(vertexDof(*s.system, s.u, vertex)),
                  coefficients[w_dof] * 2.0 + 1.5);
    }
}

TEST(StrongDirichletConstraint, ValueRefusesUnknownsUndeclaredParametersAndForeignElements)
{
    // An unknown, as a state symbol: refused when the constraint is built.
    EXPECT_THROW((void)buildStrongValueSystem([](FESystem&, const StrongValueSystem& out) {
                     return svmp::FE::forms::StateField(out.u, *out.space, "u");
                 }),
                 std::invalid_argument);

    // An unknown read as a discrete field: refused at setup.
    auto unknown = buildStrongValueSystem([](FESystem&, const StrongValueSystem& out) {
        return FormExpr::discreteField(out.u, *out.space, "u");
    });
    EXPECT_THROW(unknown.system->setup(), svmp::FE::InvalidArgumentException);

    // A parameter that is not a time function of the system.
    auto undeclared = buildStrongValueSystem([](FESystem&, const StrongValueSystem&) {
        return FormExpr::parameter("Q");
    });
    EXPECT_THROW(undeclared.system->setup(), svmp::FE::InvalidArgumentException);

    // A prescribed field of another order.
    auto foreign = buildStrongValueSystem([](FESystem& sys, const StrongValueSystem&) {
        auto quadratic = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/2);
        FieldSpec spec{.name = "w", .space = quadratic, .components = 1};
        spec.source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData;
        const auto w = sys.addField(spec);
        return FormExpr::discreteField(w, *quadratic, "w");
    });
    EXPECT_THROW(foreign.system->setup(), svmp::FE::InvalidArgumentException);
}
