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
#include "Assembly/CutIntegrationContext.h"
#include "Assembly/GlobalSystemView.h"

#include "Constraints/NormalComponentConstraint.h"
#include "Forms/StandardBCs.h"
#include "Forms/Vocabulary.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Basis/BernsteinBasis.h"
#include "Spaces/H1Space.h"
#include "Spaces/SpaceFactory.h"
#include "Systems/BoundaryConditionManager.h"

#include "Mesh/Mesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellShape.h"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <utility>
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
using svmp::FE::systems::AuxiliaryOutputStateView;
using svmp::FE::systems::AuxiliaryModelBuilder;
using svmp::FE::systems::use;

using svmp::FE::spaces::H1Space;

using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::SystemStateView;

namespace {

std::shared_ptr<Mesh> build_single_quad_mesh_with_left_edge_marker(int marker, double shear = 0.0)
{
    auto base = std::make_shared<MeshBase>();

    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,
        1.0, 0.0,
        1.0 + shear, 1.0,
        shear, 1.0
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

std::shared_ptr<Mesh> build_single_quad9_mesh_with_boundary_only_left_edge_marker(int marker, double left_mid_x = 0.0)
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
        left_mid_x, 0.5,
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

/// A unit quad with its left edge {0, 3} marked @p left and its bottom edge
/// {0, 1} marked @p bottom: the two share vertex 0.
std::shared_ptr<Mesh> build_single_quad_mesh_with_corner_markers(int left, int bottom)
{
    auto base = std::make_shared<MeshBase>();
    const std::vector<svmp::real_t> X_ref = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3};
    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, {shape});
    base->finalize();
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base->n_faces()); ++f) {
        const auto verts = base->face_vertices(f);
        if (verts.size() != 2u) {
            continue;
        }
        const auto has = [&](svmp::index_t vertex) { return verts[0] == vertex || verts[1] == vertex; };
        if (has(0) && has(3)) {
            base->set_boundary_label(f, left);
        } else if (has(0) && has(1)) {
            base->set_boundary_label(f, bottom);
        }
    }
    return svmp::create_mesh(std::move(base));
}

TEST(StrongDirichletConstraint, HigherPrecedenceHoldsTheSharedCorner)
{
    constexpr int left = 7;
    constexpr int bottom = 8;
    auto mesh = build_single_quad_mesh_with_corner_markers(left, bottom);
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    const auto corner_of = [&](const FESystem& sys) {
        const auto a = svmp::FE::constraints::boundaryDofsByMarker(*mesh, sys.dofHandler(), left);
        const auto b = svmp::FE::constraints::boundaryDofsByMarker(*mesh, sys.dofHandler(), bottom);
        for (const auto dof : a) {
            if (std::find(b.begin(), b.end(), dof) != b.end()) {
                return dof;
            }
        }
        return GlobalIndex{-1};
    };

    // Equal precedence: the two values disagree at the corner.
    {
        FESystem sys(mesh);
        auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
        sys.addOperator("mass");
        sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
        const std::vector bcs{strongDirichlet(u, left, FormExpr::constant(1.0), "u"),
                              strongDirichlet(u, bottom, FormExpr::constant(2.0), "u")};
        svmp::FE::systems::installStrongDirichlet(sys, bcs);
        EXPECT_ANY_THROW(sys.setup());
    }

    // The wall (bottom) holds the corner, in either order and at every update.
    for (const bool wall_first : {false, true}) {
        FESystem sys(mesh);
        auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
        sys.addOperator("mass");
        sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
        auto inlet = strongDirichlet(u, left, t() + FormExpr::constant(1.0), "u");
        auto wall = strongDirichlet(u, bottom, FormExpr::constant(2.0), "u");
        wall.precedence = 1;
        const std::vector bcs = wall_first ? std::vector{wall, inlet} : std::vector{inlet, wall};
        svmp::FE::systems::installStrongDirichlet(sys, bcs);
        ASSERT_NO_THROW(sys.setup());
        const auto corner = corner_of(sys);
        ASSERT_GE(corner, 0);
        EXPECT_EQ(sys.constraints().getInhomogeneity(corner), 2.0);
        for (const auto dof : svmp::FE::constraints::boundaryDofsByMarker(*mesh, sys.dofHandler(), left)) {
            EXPECT_TRUE(sys.constraints().isConstrained(dof));
            EXPECT_EQ(sys.constraints().getInhomogeneity(dof), dof == corner ? 2.0 : 1.0);
        }
        sys.updateConstraints(/*time=*/3.0, /*dt=*/0.5);
        EXPECT_EQ(sys.constraints().getInhomogeneity(corner), 2.0);
        for (const auto dof : svmp::FE::constraints::boundaryDofsByMarker(*mesh, sys.dofHandler(), left)) {
            EXPECT_EQ(sys.constraints().getInhomogeneity(dof), dof == corner ? 2.0 : 4.0);
        }
    }
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

TEST(StrongDirichletConstraint, BernsteinTraceReproducesTheQuadraticDatumAtSetupAndUpdate)
{
    constexpr int marker = 17;
    auto mesh = build_single_quad_mesh_with_left_edge_marker(marker);
    for (int degree : {2, 3}) {
        auto space = std::make_shared<H1Space>(ElementType::Quad4, degree, svmp::FE::BasisType::Bernstein);
        FESystem system(mesh);
        const auto u = system.addField(FieldSpec{.name = "u", .space = space, .components = 1});
        const auto y = FormExpr::coordinate().component(1);
        const auto data = 4.0 * y * (1.0 - y) + t();
        const auto bc = strongDirichlet(u, marker, data, "u");
        svmp::FE::systems::installStrongDirichlet(system, std::span(&bc, 1));
        system.setup();
        const auto dofs = system.fieldDofHandler(u).getDofMap().getCellDofs(0);
        const svmp::FE::basis::BernsteinBasis basis(ElementType::Quad4, degree);
        const auto points = basis.control_layout().points;
        ASSERT_EQ(dofs.size(), points.size());
        for (const double time : {0.0, 0.375}) {
            if (time != 0.0) system.updateConstraints(time, 0.1);
            std::vector<double> coefficients(dofs.size(), 0.0);
            for (std::size_t i = 0; i < dofs.size(); ++i) {
                const auto dof = dofs[i] + system.fieldDofOffset(u);
                const bool on_left = points[i][0] == -1.0;
                EXPECT_EQ(system.constraints().isConstrained(dof), on_left);
                if (!on_left) continue;
                coefficients[i] = system.constraints().getInhomogeneity(dof);
                const double k = (points[i][1] + 1.0) * degree / 2.0;
                EXPECT_NEAR(coefficients[i], 4.0 * k * (degree-k) / (degree*(degree-1)) + time, 2e-13);
            }
            for (const double s : {0.23, 0.5, 0.81}) {
                std::vector<double> row;
                basis.evaluate_values({-1.0, 2.0*s-1.0, 0.0}, row);
                EXPECT_NEAR(std::inner_product(row.begin(), row.end(), coefficients.begin(), 0.0),
                            4.0*s*(1.0-s) + time, 2e-13);
            }
        }
    }
}

TEST(StrongDirichletConstraint, BernsteinControlsUseTheCurvedPhysicalBoundary)
{
    constexpr int marker = 17;
    auto mesh = build_single_quad9_mesh_with_boundary_only_left_edge_marker(marker, -0.125);
    auto space = std::make_shared<H1Space>(ElementType::Quad4, 2, svmp::FE::BasisType::Bernstein);
    FESystem system(mesh);
    const auto u = system.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    const auto bc = strongDirichlet(u, marker, FormExpr::coordinate().component(0), "u");
    svmp::FE::systems::installStrongDirichlet(system, std::span(&bc, 1));
    system.setup();
    const auto dofs = system.fieldDofHandler(u).getDofMap().getCellDofs(0);
    const auto points = svmp::FE::basis::BernsteinBasis(ElementType::Quad4, 2).control_layout().points;
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (points[i][0] != -1.0) continue;
        EXPECT_NEAR(system.constraints().getInhomogeneity(dofs[i] + system.fieldDofOffset(u)),
                    points[i][1] == 0.0 ? -0.25 : 0.0, 2e-13);
    }
}

TEST(StrongDirichletConstraint, BernsteinTriangularFaceReproducesACubicBubble)
{
    constexpr int marker = 19;
    auto mesh = build_single_tetra10_mesh_with_boundary_only_face_marker(marker);
    auto space = std::make_shared<H1Space>(ElementType::Tetra4, 3, svmp::FE::BasisType::Bernstein);
    FESystem system(mesh);
    const auto u = system.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    const auto x = FormExpr::coordinate().component(0), y = FormExpr::coordinate().component(1);
    const auto bc = strongDirichlet(u, marker, 27.0*x*y*(1.0-x-y), "u");
    svmp::FE::systems::installStrongDirichlet(system, std::span(&bc, 1));
    system.setup();
    const svmp::FE::basis::BernsteinBasis basis(ElementType::Tetra4, 3);
    const auto points = basis.control_layout().points;
    const auto dofs = system.fieldDofHandler(u).getDofMap().getCellDofs(0);
    std::vector<double> coefficients(dofs.size(), 0.0);
    ASSERT_EQ(points.size(), dofs.size());
    int face_controls = 0;
    for (std::size_t i = 0; i < dofs.size(); ++i) {
        const auto dof = dofs[i] + system.fieldDofOffset(u);
        EXPECT_EQ(system.constraints().isConstrained(dof), points[i][2] == 0);
        if (points[i][2] != 0) continue;
        ++face_controls;
        coefficients[i] = system.constraints().getInhomogeneity(dof);
        const bool interior = points[i][0] > 0 && points[i][1] > 0 && points[i][0] + points[i][1] < 0.9;
        EXPECT_NEAR(coefficients[i], interior ? 4.5 : 0.0, 6e-13);
    }
    EXPECT_EQ(face_controls, 10);
    std::vector<double> values;
    basis.evaluate_values({0.2, 0.4, 0.0}, values);
    EXPECT_NEAR(std::inner_product(values.begin(), values.end(), coefficients.begin(), 0.0), 0.864, 3e-13);
}

TEST(StrongDirichletConstraint, NonlinearValueReadsBernsteinFieldAtTheInterpolationPoints)
{
    constexpr int marker = 23;
    auto mesh = build_single_quad_mesh_with_left_edge_marker(marker);
    auto space = std::make_shared<H1Space>(ElementType::Quad4, 2, svmp::FE::BasisType::Bernstein);
    FESystem system(mesh);
    const auto u = system.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    FieldSpec source{.name = "w", .space = space, .components = 1};
    source.source_kind = svmp::FE::systems::FieldSourceKind::PrescribedData;
    const auto w = system.addField(source);
    const auto value = FormExpr::discreteField(w, *space, "w");
    const auto bc = strongDirichlet(u, marker, value*value + 2.0*value, "u");
    svmp::FE::systems::installStrongDirichlet(system, std::span(&bc, 1));
    system.setup();
    const auto points = svmp::FE::basis::BernsteinBasis(ElementType::Quad4, 2).control_layout().points;
    const auto source_dofs = system.fieldDofHandler(w).getDofMap().getCellDofs(0);
    const auto target_dofs = system.fieldDofHandler(u).getDofMap().getCellDofs(0);
    std::vector<double> coefficients(system.fieldDofHandler(w).getNumDofs());
    for (std::size_t i = 0; i < points.size(); ++i)
        coefficients.at(source_dofs[i]) = points[i][1] == 0 ? 1.0 : 0.0;
    system.setPrescribedFieldCoefficients(w, coefficients);
    system.updateConstraints(0.0, 0.1);
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (points[i][0] != -1.0) continue;
        // w = 1/2 at the midpoint; w² + 2w = 1.25 there, so its middle control is 2.5.
        EXPECT_NEAR(system.constraints().getInhomogeneity(target_dofs[i] + system.fieldDofOffset(u)),
                    points[i][1] == 0 ? 2.5 : 0.0, 2e-13);
    }
}

TEST(StrongDirichletConstraint, AuxiliaryConstantSelectsExactlyTheBernsteinTraceControls)
{
    constexpr int marker = 17;
    auto mesh = build_single_quad_mesh_with_left_edge_marker(marker);
    auto space = std::make_shared<H1Space>(ElementType::Quad4, 2, svmp::FE::BasisType::Bernstein);
    FESystem system(mesh);
    const auto u = system.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    auto model = AuxiliaryModelBuilder("driver").state("x").ode("x", FormExpr::constant(0.0)).build();
    system.deployAuxiliaryModel(use(model).name("driver").global().partitioned("ForwardEuler").initialize({1.25})
                                .drivesStrongDirichlet(u, marker, "x", -1, AuxiliaryConstraintValueSource::State));
    system.finalizeAuxiliaryLayout();
    system.setup();
    const auto dofs = system.fieldDofHandler(u).getDofMap().getCellDofs(0);
    const auto points = svmp::FE::basis::BernsteinBasis(ElementType::Quad4, 2).control_layout().points;
    ASSERT_EQ(dofs.size(), points.size());
    for (std::size_t i = 0; i < dofs.size(); ++i) {
        const auto dof = dofs[i] + system.fieldDofOffset(u);
        const bool on_left = points[i][0] == -1.0;
        EXPECT_EQ(system.constraints().isConstrained(dof), on_left) << "control " << i;
        if (on_left && system.constraints().isConstrained(dof)) EXPECT_DOUBLE_EQ(system.constraints().getInhomogeneity(dof), 1.25);
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

// A live binding follows the work state whatever view it names: the stage
// view of a constraint is the stage being solved, not the accepted-stage
// record the scalar history reads.
TEST(StrongDirichletConstraint, AuxiliaryStateBindingWithTheStageViewFollowsTheWorkState)
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
            .drivesStrongDirichlet(u, marker, "x", -1, AuxiliaryConstraintValueSource::State,
                                   AuxiliaryOutputStateView::Stage));
    sys.finalizeAuxiliaryLayout();
    sys.setup();

    const auto constrained = svmp::FE::constraints::boundaryDofsByMarker(*mesh, sys.dofHandler(), marker);
    ASSERT_EQ(constrained.size(), 2u);
    for (const auto dof : constrained) {
        EXPECT_NEAR(sys.constraints().getInhomogeneity(dof), 1.25, 1e-12);
    }
    sys.auxiliaryStateManager().getBlock("state_bc_inst").work()[0] = 2.5;
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

namespace {

/// Two unit quads side by side, (0,0)-(2,1): vertices 0..2 on the floor
/// y = 0, 3..5 on top; the floor faces carry @p floor, the face x = 2 carries
/// @p right.  Records each wall face with the cell that holds it.
struct TwoQuadTank {
    std::shared_ptr<Mesh> mesh{};
    std::pair<GlobalIndex, GlobalIndex> floor_left{};    ///< face 0-1, cell 0
    std::pair<GlobalIndex, GlobalIndex> floor_right{};   ///< face 1-2, cell 1
    std::pair<GlobalIndex, GlobalIndex> right_wall{};    ///< face 2-5, cell 1
};

TwoQuadTank buildTwoQuadTank(int floor, int right)
{
    auto base = std::make_shared<MeshBase>();
    const std::vector<svmp::real_t> X_ref = {0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0};
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4, 8};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 4, 3, 1, 2, 5, 4};
    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, {shape, shape});
    base->finalize();
    TwoQuadTank out;
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base->n_faces()); ++f) {
        const auto verts = base->face_vertices(f);
        if (verts.size() != 2u) {
            continue;
        }
        const auto has = [&](svmp::index_t vertex) { return verts[0] == vertex || verts[1] == vertex; };
        if (has(0) && has(1)) {
            base->set_boundary_label(f, floor);
            out.floor_left = {f, 0};
        } else if (has(1) && has(2)) {
            base->set_boundary_label(f, floor);
            out.floor_right = {f, 1};
        } else if (has(2) && has(5)) {
            base->set_boundary_label(f, right);
            out.right_wall = {f, 1};
        }
    }
    out.mesh = svmp::create_mesh(std::move(base));
    return out;
}

/// The wetted part of one physical marker: its generated marker and the
/// (face, cell) pairs of its faces of positive wetted measure.
struct WettedWall {
    int physical{-1};
    int generated{-1};
    std::vector<std::pair<GlobalIndex, GlobalIndex>> wet_faces{};
};

/// A cut-integration context that holds the generated active boundary of
/// each wall (half of each wet face wetted).
std::shared_ptr<svmp::FE::assembly::CutIntegrationContext>
wettedWallContext(svmp::FE::FieldId level_set, const std::vector<WettedWall>& walls)
{
    namespace geometry = svmp::FE::geometry;
    namespace interfaces = svmp::FE::interfaces;
    auto context = std::make_shared<svmp::FE::assembly::CutIntegrationContext>();
    for (const auto& wall : walls) {
        interfaces::GeneratedActiveBoundaryRequest request;
        request.source = interfaces::LevelSetInterfaceSource::fromField(level_set, /*layout_revision=*/1u,
                                                                         /*value_revision=*/1u);
        request.generated_domain_id = "wetted-wall";
        request.interface_marker = 901;
        request.boundary_marker = wall.physical;
        request.active_boundary_marker = wall.generated;
        request.side = geometry::CutIntegrationSide::Negative;
        request.quadrature_order = 1;
        request.source_value_revision = 1u;
        interfaces::GeneratedActiveBoundaryDomain domain(request);
        for (const auto& [face, cell] : wall.wet_faces) {
            interfaces::GeneratedActiveBoundaryFragment fragment;
            fragment.parent_cell = static_cast<svmp::FE::MeshIndex>(cell);
            fragment.parent_face = static_cast<svmp::FE::MeshIndex>(face);
            fragment.represented_implicit_geometry_mode = "LinearCorner";
            fragment.represented_implicit_quadrature_backend = "LinearCorner";
            fragment.represented_implicit_fallback_status = "None";
            fragment.measure = 0.5;
            fragment.parent_measure = 1.0;
            fragment.achieved_quadrature_order = 1;
            fragment.topology_id = "wetted-wall-face-" + std::to_string(face);
            geometry::CutQuadraturePoint point;
            point.point = {{0.25, 0.0, 0.0}};
            point.weight = 0.5;
            point.reference_measure_factor = 0.5;
            fragment.quadrature_points.push_back(point);
            domain.addFragment(std::move(fragment));
        }
        context->addGeneratedActiveBoundaryDomain(domain);
    }
    return context;
}

} // namespace

// A strong wall condition on the wetted part of its marker holds only the
// nodes of faces with wetted measure: the dry wall nodes of a cut cell are
// released (free-surface audit A27), and a face holds its nodes once it is
// wet.  Without a generated subset in the context the whole marker holds.
TEST(StrongDirichletConstraint, WettedSubsetHoldsOnlyTheNodesOfWettedFaces)
{
    constexpr int floor = 11;
    constexpr int right = 12;
    constexpr int floor_wet = 3000111;
    constexpr int right_wet = 3000112;
    const auto tank = buildTwoQuadTank(floor, right);
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    FESystem sys(tank.mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
    auto floor_bc = strongDirichlet(u, floor, FormExpr::constant(1.0), "u");
    floor_bc.active_boundary_marker = floor_wet;
    auto right_bc = strongDirichlet(u, right, FormExpr::constant(2.0), "u");
    right_bc.active_boundary_marker = right_wet;
    right_bc.precedence = 1;   // the right wall holds the corner once it is wet
    svmp::FE::systems::installStrongDirichlet(sys, std::vector{floor_bc, right_bc});
    ASSERT_NO_THROW(sys.setup());

    // The value at a vertex, or nullopt where it is free.
    const auto held = [&](GlobalIndex vertex) -> std::optional<double> {
        const auto dof = vertexDof(sys, u, vertex);
        if (!sys.constraints().isConstrained(dof)) {
            return std::nullopt;
        }
        return sys.constraints().getInhomogeneity(dof);
    };
    const auto expect_holds = [&](const std::array<std::optional<double>, 6>& expected, const char* stage) {
        for (GlobalIndex v = 0; v < 6; ++v) {
            EXPECT_EQ(held(v), expected[static_cast<std::size_t>(v)]) << stage << ": vertex " << v;
        }
    };
    const std::optional<double> free;

    // No cut context: the whole markers hold.
    expect_holds({1.0, 1.0, 2.0, free, free, 2.0}, "no cut context");

    // The liquid wets the left floor face; the cut cell 1 touches the floor
    // and the right wall only at the dry nodes 2 and 5: no row there.
    sys.setCutIntegrationContext(
        wettedWallContext(u, {{floor, floor_wet, {tank.floor_left}}, {right, right_wet, {}}}));
    ASSERT_NO_THROW(sys.rebuildConstraintState());
    expect_holds({1.0, 1.0, free, free, free, free}, "dry corner cell");
    sys.updateConstraints(/*time=*/1.0, /*dt=*/0.5);
    expect_holds({1.0, 1.0, free, free, free, free}, "dry corner cell, updated");

    // The front reaches the floor face of the corner cell: its nodes hold.
    sys.setCutIntegrationContext(
        wettedWallContext(u, {{floor, floor_wet, {tank.floor_left, tank.floor_right}}, {right, right_wet, {}}}));
    ASSERT_NO_THROW(sys.rebuildConstraintState());
    expect_holds({1.0, 1.0, 1.0, free, free, free}, "wet floor face");

    // The right wall wets: the corner rule (the higher precedence) returns.
    sys.setCutIntegrationContext(wettedWallContext(
        u, {{floor, floor_wet, {tank.floor_left, tank.floor_right}}, {right, right_wet, {tank.right_wall}}}));
    ASSERT_NO_THROW(sys.rebuildConstraintState());
    expect_holds({1.0, 1.0, 2.0, free, free, 2.0}, "wet corner");
}

TEST(NormalComponentConstraint, AChangingNormalPreservesHistoryAtItsOwnTime)
{
    using namespace svmp::FE;
    constexpr int side = 7;
    auto mesh = build_single_quad_mesh_with_left_edge_marker(side);
    auto& geometry = mesh->local_mesh();
    geometry.set_current_coords({0, 0, 1, 0, 1, 1, 0, 1});
    FESystem system(mesh, svmp::Configuration::Current);
    auto vector = spaces::SpaceFactory::create_vector_h1(ElementType::Quad4, 1, 2);
    const auto field = system.addField({.name = "u", .space = vector, .components = 2});
    auto condition = std::make_unique<constraints::NormalComponentConstraint>(
        field, std::vector<int>{side}, FormExpr::constant(0.0));
    const auto* normal = condition.get();
    system.addSystemConstraint(std::move(condition));
    system.setup();
    ASSERT_EQ(normal->lines().size(), 2u);
    for (const auto& line : normal->lines()) {
        EXPECT_TRUE(line.masters.empty());
    }

    // The formerly vertical wall now has tangent (1/2,1). The current
    // constraint is u_x = u_y/2. At the accepted time u was (0,1).
    geometry.set_current_coords({0, 0, 1, 0, 1.5, 1, 0.5, 1});
    system.updateConstraints(1.0, 1.0);
    ASSERT_EQ(normal->lines().size(), 2u);
    std::vector<Real> accepted(system.dofHandler().getNumDofs(), 0);
    auto rate = accepted;
    for (const auto& line : normal->lines()) {
        ASSERT_EQ(line.masters.size(), 1u);
        EXPECT_DOUBLE_EQ(line.weights[0], 0.5);
        accepted[line.masters[0]] = 1;
        rate[line.slave] = 0.5;
    }
    const auto& lines = system.constraints();
    auto at_current_time = accepted;
    lines.distributeMasterBearing(at_current_time);
    const auto old_rate = rate;
    const auto old_state = accepted;
    lines.distributeMasterBearing(accepted, constraints::AffineConstraints::MasterBearingLines::TimeInvariant);
    lines.distributeMasterBearingHomogeneous(rate);
    EXPECT_EQ(accepted, old_state);
    EXPECT_EQ(rate, old_rate);
    for (const auto& line : normal->lines()) {
        EXPECT_TRUE(lines.isInstantaneous(line.slave));
        EXPECT_DOUBLE_EQ(at_current_time[line.slave], 0.5);
    }
}

TEST(NormalComponentConstraint, HomogeneousRowsCancelIntegratedFluxOnAWarpedQuadrilateral)
{
    using namespace svmp::FE;
    constexpr int top = 71, active = 72;
    auto base = std::make_shared<MeshBase>();
    // Unit cube with the top graph z = 1 + xy/2.
    const CellShape hex{.family = CellFamily::Hex, .num_corners = 8, .order = 1};
    const CellShape quad{.family = CellFamily::Quad, .num_corners = 4, .order = 1};
    base->build_from_arrays(3, {0,0,0, 1,0,0, 1,1,0, 0,1,0, 0,0,1, 1,0,1, 1,1,1.5, 0,1,1},
                            {0, 8}, {0, 1, 2, 3, 4, 5, 6, 7}, {hex});
    base->set_faces_from_arrays({quad}, {0, 4}, {4, 5, 6, 7},
        {std::array<svmp::index_t, 2>{0, svmp::INVALID_INDEX}}, svmp::MeshCodim1StorageMode::BoundaryOnly);
    base->set_boundary_label(0, top);
    FESystem system(svmp::create_mesh(std::move(base)));
    const auto field = system.addField({
        .name = "u", .space = spaces::SpaceFactory::create_vector_h1(ElementType::Hex8, 1, 3), .components = 3});
    auto condition = std::make_unique<constraints::NormalComponentConstraint>(field,
        std::vector{forms::ExteriorBoundaryMeasure::generatedActiveSubset(top, active)});
    const auto* normal = condition.get();
    system.addSystemConstraint(std::move(condition));
    system.setup();
    const auto cell = system.fieldDofHandler(field).getDofMap().getCellDofs(0);
    ASSERT_EQ(cell.size(), 24u);
    const auto pointwise = constraints::boundaryPointNormals(system, field, std::array{top});
    ASSERT_TRUE(pointwise.at(cell[4]).has_value());
    EXPECT_EQ(*pointwise.at(cell[4]), (std::array<Real, 3>{0, 0, 1}));

    const Real gauss = std::sqrt(Real{1} / 12);
    const auto verify = [&](Real height, Real expected_corner) {
        std::vector<Real> u(system.dofHandler().getNumDofs(), 0);
        u[cell[4]] = u[cell[5]] = 4;  // u_x = 4(1-y), u_y = 0 on the face.
        system.constraints().distributeMasterBearing(u);
        // Direct Gauss integration of u_h · (-y/2,-x/2,1) on [0,1] x [0,height].
        Real flux{};
        for (Real x : {.5 - gauss, .5 + gauss}) {
            for (Real y : {height * (.5 - gauss), height * (.5 + gauss)}) {
                const std::array<Real, 4> n{(1-x) * (1-y), x * (1-y), x*y, (1-x) * y};
                for (int a = 0; a < 4; ++a) {
                    flux += height * n[a] * (-.5 * y * u[cell[4 + a]] - .5 * x * u[cell[12 + a]] + u[cell[20 + a]]) / 4;
                }
            }
        }
        ASSERT_EQ(normal->lines().size(), 4u);
        EXPECT_NEAR(flux, 0, 1e-13);
        EXPECT_NEAR(u[cell[20]], expected_corner, 1e-13);
    };
    verify(1, 2.0 / 3.0);
    interfaces::GeneratedActiveBoundaryRequest request;
    request.source = interfaces::LevelSetInterfaceSource::fromField(field, 1u, 1u);
    request.generated_domain_id = "half-quadrilateral";
    request.boundary_marker = top;
    request.active_boundary_marker = active;
    request.interface_marker = 73;
    request.quadrature_order = 2;
    interfaces::GeneratedActiveBoundaryDomain domain(request);
    interfaces::GeneratedActiveBoundaryFragment fragment;
    fragment.parent_cell = 0;
    fragment.parent_face = 0;
    fragment.topology_id = "half-quadrilateral";
    fragment.represented_implicit_geometry_mode = "LinearCorner";
    fragment.represented_implicit_quadrature_backend = "LinearCorner";
    fragment.represented_implicit_fallback_status = "None";
    fragment.achieved_quadrature_order = 2;
    for (Real x : {.5 - gauss, .5 + gauss}) {
        for (Real y : {.5 * (.5 - gauss), .5 * (.5 + gauss)}) {
            geometry::CutQuadraturePoint point;
            point.parent_coordinate = {2*x - 1, 2*y - 1, 1};
            point.point = point.parent_coordinate;
            point.weight = .5;
            fragment.measure += point.weight;
            fragment.quadrature_points.push_back(point);
        }
    }
    fragment.parent_measure = 2 * fragment.measure;
    domain.addFragment(std::move(fragment));
    auto context = std::make_shared<assembly::CutIntegrationContext>();
    context->addGeneratedActiveBoundaryDomain(domain);
    system.setCutIntegrationContext(std::move(context));
    system.rebuildConstraintState();
    verify(.5, 4.0 / 9.0);
}

TEST(NormalComponentConstraint, ZeroQuadraticVertexFluxMomentsKeepTheOutwardPrescribedValue)
{
    using namespace svmp::FE;
    constexpr int wall = 74;
    FESystem system(build_single_tetra10_mesh_with_boundary_only_face_marker(wall));
    const auto field = system.addField({
        .name = "u", .space = spaces::SpaceFactory::create_vector_h1(ElementType::Tetra10, 2, 3), .components = 3});
    auto condition = std::make_unique<constraints::NormalComponentConstraint>(
        field, std::vector<int>{wall}, FormExpr::constant(1));
    const auto* normal = condition.get();
    system.addSystemConstraint(std::move(condition));
    system.setup();
    ASSERT_EQ(normal->lines().size(), 6u);
    for (const auto& line : normal->lines()) {
        EXPECT_TRUE(line.masters.empty());
        EXPECT_DOUBLE_EQ(line.value_factor, -1);  // outward normal of the z=0 face
    }
}

TEST(NormalComponentConstraint, AFittedReferenceNormalProjectsStaticHistory)
{
    using namespace svmp::FE;
    constexpr int side = 7;
    FESystem system(build_single_quad_mesh_with_left_edge_marker(side, 0.5));
    auto vector = spaces::SpaceFactory::create_vector_h1(ElementType::Quad4, 1, 2);
    const auto field = system.addField({.name = "u", .space = vector, .components = 2});
    auto condition = std::make_unique<constraints::NormalComponentConstraint>(field, std::vector<int>{side});
    const auto* normal = condition.get();
    system.addSystemConstraint(std::move(condition));
    system.setup();
    ASSERT_EQ(normal->lines().size(), 2u);
    std::vector<Real> rate(system.dofHandler().getNumDofs(), 0);
    for (const auto& line : normal->lines()) {
        ASSERT_EQ(line.masters.size(), 1u);
        EXPECT_FALSE(system.constraints().isInstantaneous(line.slave));
        rate[line.masters[0]] = 1;
    }
    system.constraints().distributeMasterBearingHomogeneous(rate);
    for (const auto& line : normal->lines()) {
        EXPECT_DOUBLE_EQ(rate[line.slave], 0.5);
    }
}

// The strong normal (slip) route: walls of one constant value join one
// NormalComponentConstraint, so a wet corner keeps both normals (u = 0) in
// either order of the conditions (audit A43; apart, the second wall was
// dropped and u = (1, 0) passed the corner), and a dry wall gives no normal
// at a node of a wet one (audit A27).
TEST(NormalComponentConstraint, JoinedWettedWallsHoldTheCornerOnlyOnWetFaces)
{
    namespace bc = svmp::FE::forms::bc;
    constexpr int floor = 11;
    constexpr int right = 12;
    constexpr int floor_wet = 3000111;
    constexpr int right_wet = 3000112;
    const auto tank = buildTwoQuadTank(floor, right);
    auto space = svmp::FE::spaces::SpaceFactory::create_vector_h1(ElementType::Quad4, /*order=*/1, /*components=*/2);
    for (const bool floor_first : {true, false}) {
        FESystem sys(tank.mesh);
        const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 2});
        sys.addOperator("mass");
        sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
        const auto slip = [](int marker, int wetted) {
            return std::make_unique<bc::Dirichlet>(bc::Boundary::of(marker, wetted), bc::Trace::Normal,
                                                   FormExpr::constant(0.0), bc::Strong{});
        };
        svmp::FE::systems::BoundaryConditionManager manager;
        manager.add(slip(floor_first ? floor : right, floor_first ? floor_wet : right_wet));
        manager.add(slip(floor_first ? right : floor, floor_first ? right_wet : floor_wet));
        manager.applyAll(sys, u);
        ASSERT_NO_THROW(sys.setup());

        const auto* map = sys.fieldDofHandler(u).getEntityDofMap();
        ASSERT_NE(map, nullptr);
        const auto offset = sys.fieldDofOffset(u);
        // Whether component c of vertex v is constrained.
        const auto held = [&](GlobalIndex vertex, int c) {
            return sys.constraints().isConstrained(offset +
                                                   map->getVertexDofs(vertex)[static_cast<std::size_t>(c)]);
        };
        const std::string order = floor_first ? "floor first" : "right wall first";

        // Whole walls: the corner (vertex 2) holds both components.
        EXPECT_TRUE(held(2, 0) && held(2, 1)) << order;
        EXPECT_TRUE(held(5, 0) && !held(5, 1)) << order;
        EXPECT_TRUE(!held(0, 0) && held(0, 1)) << order;

        // The floor face of the corner cell is wet, the right wall dry: the
        // corner keeps only the floor normal and the right wall releases 5.
        sys.setCutIntegrationContext(wettedWallContext(
            u, {{floor, floor_wet, {tank.floor_left, tank.floor_right}}, {right, right_wet, {}}}));
        ASSERT_NO_THROW(sys.rebuildConstraintState());
        EXPECT_TRUE(!held(2, 0) && held(2, 1)) << order;
        EXPECT_TRUE(!held(5, 0) && !held(5, 1)) << order;

        // Only the left floor face is wet: the dry corner node is free.
        sys.setCutIntegrationContext(
            wettedWallContext(u, {{floor, floor_wet, {tank.floor_left}}, {right, right_wet, {}}}));
        ASSERT_NO_THROW(sys.rebuildConstraintState());
        EXPECT_TRUE(!held(2, 0) && !held(2, 1)) << order;
        EXPECT_TRUE(!held(1, 0) && held(1, 1)) << order;

        // A wet corner: both normals again.
        sys.setCutIntegrationContext(wettedWallContext(
            u, {{floor, floor_wet, {tank.floor_left, tank.floor_right}}, {right, right_wet, {tank.right_wall}}}));
        ASSERT_NO_THROW(sys.rebuildConstraintState());
        EXPECT_TRUE(held(2, 0) && held(2, 1)) << order;
        EXPECT_TRUE(held(5, 0) && !held(5, 1)) << order;
    }
}

namespace {

/// One Quad8 cell (0,0)-(1,1) with its left edge labelled @p marker.
std::shared_ptr<Mesh> build_single_quad8_mesh_with_left_edge_marker(int marker)
{
    auto base = std::make_shared<MeshBase>();
    const std::vector<svmp::real_t> X_ref = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
                                             0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.5};
    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 8};
    const std::vector<svmp::index_t> cell2vertex = {0, 1, 2, 3, 4, 5, 6, 7};
    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 2;
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, {shape});
    base->finalize();
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base->n_faces()); ++f) {
        const auto verts = base->face_vertices(f);
        const auto has = [&](svmp::index_t vertex) {
            return std::find(verts.begin(), verts.end(), vertex) != verts.end();
        };
        if (has(0) && has(3)) {
            base->set_boundary_label(f, marker);
        }
    }
    return svmp::create_mesh(std::move(base));
}

} // namespace

// A wall condition builds the face degrees of freedom of the field's own
// element and maps the cell with its own geometry order (free-surface audit
// A52): P1 on a Tetra10 cell has three DOFs on the marked face (the Tetra10
// layout gave six slots), and a Quad8 cell maps with its quadratic geometry
// (an order-1 serendipity map was requested and refused).
TEST(StrongDirichletConstraint, WallDofsFollowTheFieldElementAndTheCellGeometryOrder)
{
    constexpr int marker = 7;
    {
        auto mesh = build_single_tetra10_mesh_with_boundary_only_face_marker(marker);
        auto space = std::make_shared<H1Space>(ElementType::Tetra4, /*order=*/1);
        FESystem sys(mesh);
        const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
        const auto bc = strongDirichlet(u, marker, FormExpr::constant(0.0), "u");
        svmp::FE::systems::installStrongDirichlet(sys, std::span<const svmp::FE::forms::bc::StrongDirichlet>(&bc, 1));
        ASSERT_NO_THROW(sys.setup());
        EXPECT_EQ(sys.constraints().numConstraints(), 3);
    }
    {
        auto mesh = build_single_quad8_mesh_with_left_edge_marker(marker);
        auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
        FESystem sys(mesh);
        const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
        sys.addOperator("mass");
        sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
        const auto bc = strongDirichlet(u, marker, t() + FormExpr::constant(1.0), "u");
        svmp::FE::systems::installStrongDirichlet(sys, std::span<const svmp::FE::forms::bc::StrongDirichlet>(&bc, 1));
        ASSERT_NO_THROW(sys.setup());
        EXPECT_EQ(sys.constraints().numConstraints(), 2);
        sys.updateConstraints(/*time=*/2.0, /*dt=*/0.5);
        for (const GlobalIndex vertex : {0, 3}) {
            EXPECT_EQ(sys.constraints().getInhomogeneity(vertexDof(sys, u, vertex)), 3.0);
        }
    }
}
