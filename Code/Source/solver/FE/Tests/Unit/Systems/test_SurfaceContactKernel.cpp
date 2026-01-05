/**
 * @file test_SurfaceContactKernel.cpp
 * @brief Unit tests for surface-to-surface penalty contact kernel and marker-filtered projection.
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/MeshSearchAccess.h"
#include "Systems/SurfaceContactKernel.h"

#include "Assembly/GlobalSystemView.h"

#include "Dofs/EntityDofMap.h"

#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"

#include "Mesh/Mesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellShape.h"

#include <array>
#include <cmath>
#include <memory>
#include <vector>

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Mesh;
using svmp::MeshBase;

using svmp::FE::ElementType;
using svmp::FE::FieldId;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

using svmp::FE::assembly::DenseMatrixView;
using svmp::FE::assembly::DenseVectorView;

using svmp::FE::spaces::H1Space;
using svmp::FE::spaces::ProductSpace;

using svmp::FE::systems::AssemblyRequest;
using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::MeshSearchAccess;
using svmp::FE::systems::PenaltySurfaceContactConfig;
using svmp::FE::systems::PenaltySurfaceContactKernel;
using svmp::FE::systems::SystemStateView;

namespace {

std::shared_ptr<Mesh> build_two_quad_gap_mesh(double gap, int slave_marker, int master_marker)
{
    auto base = std::make_shared<MeshBase>();

    // Two disconnected quad cells with a gap in x.
    const std::vector<svmp::real_t> X_ref = {
        0.0, 0.0,   // 0
        1.0, 0.0,   // 1  (slave face)
        1.0, 1.0,   // 2  (slave face)
        0.0, 1.0,   // 3

        1.0 + gap, 0.0,   // 4  (master face)
        2.0 + gap, 0.0,   // 5
        2.0 + gap, 1.0,   // 6
        1.0 + gap, 1.0    // 7  (master face)
    };

    const std::vector<svmp::offset_t> cell2vertex_offsets = {0, 4, 8};
    const std::vector<svmp::index_t> cell2vertex = {
        0, 1, 2, 3,
        4, 5, 6, 7
    };

    CellShape shape{};
    shape.family = CellFamily::Quad;
    shape.num_corners = 4;
    shape.order = 1;
    base->build_from_arrays(/*spatial_dim=*/2, X_ref, cell2vertex_offsets, cell2vertex, {shape, shape});
    base->finalize();

    // Label the facing edges by x-coordinate of edge center.
    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base->n_faces()); ++f) {
        auto verts = base->face_vertices(f);
        if (verts.size() != 2u) continue;
        const auto a = base->get_vertex_coords(verts[0]);
        const auto b = base->get_vertex_coords(verts[1]);
        const double cx = 0.5 * (a[0] + b[0]);
        if (std::abs(cx - 1.0) < 1e-12) {
            base->set_boundary_label(f, slave_marker);
        } else if (std::abs(cx - (1.0 + gap)) < 1e-12) {
            base->set_boundary_label(f, master_marker);
        }
    }

    return svmp::create_mesh(std::move(base));
}

std::shared_ptr<ProductSpace> build_displacement_space()
{
    auto scalar = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);
    return std::make_shared<ProductSpace>(scalar, /*components=*/2);
}

} // namespace

TEST(SearchAccess, ClosestBoundaryPointOnMarkerFiltersCorrectly)
{
    const int slave_marker = 10;
    const int master_marker = 20;
    const double gap = 0.1;

    auto mesh = build_two_quad_gap_mesh(gap, slave_marker, master_marker);
    MeshSearchAccess search(*mesh);
    search.build();

    const std::array<Real, 3> p{Real(1.02), Real(0.5), Real(0.0)};

    const auto on_slave = search.closestBoundaryPointOnMarker(slave_marker, p, /*max_distance=*/1.0);
    ASSERT_TRUE(on_slave.found);
    EXPECT_NEAR(on_slave.closest_point[0], 1.0, 1e-12);
    EXPECT_NEAR(on_slave.distance, 0.02, 1e-12);

    const auto on_master = search.closestBoundaryPointOnMarker(master_marker, p, /*max_distance=*/1.0);
    ASSERT_TRUE(on_master.found);
    EXPECT_NEAR(on_master.closest_point[0], 1.0 + gap, 1e-12);
    EXPECT_NEAR(on_master.distance, 0.08, 1e-12);
}

TEST(SurfaceContactKernel, AssemblesOppositeForces)
{
    const int slave_marker = 10;
    const int master_marker = 20;
    const double gap = 0.1;

    auto mesh = build_two_quad_gap_mesh(gap, slave_marker, master_marker);
    FESystem sys(mesh);

    FieldSpec field;
    field.name = "u";
    field.space = build_displacement_space();
    field.components = 2;
    FieldId u = sys.addField(field);

    sys.addOperator("contact");

    PenaltySurfaceContactConfig cfg;
    cfg.field = u;
    cfg.slave_marker = slave_marker;
    cfg.master_marker = master_marker;
    cfg.search_radius = 1.0;
    cfg.activation_distance = 0.2;
    cfg.penalty = 10.0;
    cfg.quadrature_order = 2;

    sys.addGlobalKernel("contact", std::make_shared<PenaltySurfaceContactKernel>(cfg));
    sys.setup();

    std::vector<Real> uvec(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    SystemStateView state;
    state.u = std::span<const Real>(uvec.data(), uvec.size());

    AssemblyRequest req;
    req.op = "contact";
    req.want_vector = true;

    DenseVectorView rhs(sys.dofHandler().getNumDofs());
    const auto rep = sys.assemble(req, state, nullptr, &rhs);
    EXPECT_TRUE(rep.success);

    const auto& dh = sys.fieldDofHandler(u);
    const auto* entity = dh.getEntityDofMap();
    ASSERT_NE(entity, nullptr);
    const auto offset = sys.fieldDofOffset(u);

    // Expected integrated traction magnitude: k * (h - d) = 10 * (0.2 - 0.1) = 1.
    // For a linear edge of length 1, nodal distribution gives half to each endpoint.
    const Real fx = -0.5;
    const Real tol = 1e-12;

    const auto v1 = entity->getVertexDofs(/*vertex=*/1);
    const auto v2 = entity->getVertexDofs(/*vertex=*/2);
    const auto v4 = entity->getVertexDofs(/*vertex=*/4);
    const auto v7 = entity->getVertexDofs(/*vertex=*/7);
    ASSERT_GE(v1.size(), 2u);
    ASSERT_GE(v2.size(), 2u);
    ASSERT_GE(v4.size(), 2u);
    ASSERT_GE(v7.size(), 2u);

    const GlobalIndex v1x = v1[0] + offset;
    const GlobalIndex v1y = v1[1] + offset;
    const GlobalIndex v2x = v2[0] + offset;
    const GlobalIndex v2y = v2[1] + offset;
    const GlobalIndex v4x = v4[0] + offset;
    const GlobalIndex v4y = v4[1] + offset;
    const GlobalIndex v7x = v7[0] + offset;
    const GlobalIndex v7y = v7[1] + offset;

    EXPECT_NEAR(rhs.getVectorEntry(v1x), fx, tol);
    EXPECT_NEAR(rhs.getVectorEntry(v1y), 0.0, tol);
    EXPECT_NEAR(rhs.getVectorEntry(v2x), fx, tol);
    EXPECT_NEAR(rhs.getVectorEntry(v2y), 0.0, tol);

    EXPECT_NEAR(rhs.getVectorEntry(v4x), -fx, tol);
    EXPECT_NEAR(rhs.getVectorEntry(v4y), 0.0, tol);
    EXPECT_NEAR(rhs.getVectorEntry(v7x), -fx, tol);
    EXPECT_NEAR(rhs.getVectorEntry(v7y), 0.0, tol);

    const Real total_fx = rhs.getVectorEntry(v1x) + rhs.getVectorEntry(v2x) +
                          rhs.getVectorEntry(v4x) + rhs.getVectorEntry(v7x);
    EXPECT_NEAR(total_fx, 0.0, tol);
}

TEST(SurfaceContactKernel, AssemblesSymmetricJacobian)
{
    const int slave_marker = 10;
    const int master_marker = 20;
    const double gap = 0.1;

    auto mesh = build_two_quad_gap_mesh(gap, slave_marker, master_marker);
    FESystem sys(mesh);

    FieldSpec field;
    field.name = "u";
    field.space = build_displacement_space();
    field.components = 2;
    FieldId u = sys.addField(field);

    sys.addOperator("contact");

    PenaltySurfaceContactConfig cfg;
    cfg.field = u;
    cfg.slave_marker = slave_marker;
    cfg.master_marker = master_marker;
    cfg.search_radius = 1.0;
    cfg.activation_distance = 0.2;
    cfg.penalty = 10.0;
    cfg.quadrature_order = 2;

    sys.addGlobalKernel("contact", std::make_shared<PenaltySurfaceContactKernel>(cfg));
    sys.setup();

    std::vector<Real> uvec(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    SystemStateView state;
    state.u = std::span<const Real>(uvec.data(), uvec.size());

    AssemblyRequest req;
    req.op = "contact";
    req.want_matrix = true;

    DenseMatrixView jac(sys.dofHandler().getNumDofs());
    const auto rep = sys.assemble(req, state, &jac, nullptr);
    EXPECT_TRUE(rep.success);
    EXPECT_TRUE(jac.isSymmetric(1e-12));

    const auto& dh = sys.fieldDofHandler(u);
    const auto* entity = dh.getEntityDofMap();
    ASSERT_NE(entity, nullptr);
    const auto offset = sys.fieldDofOffset(u);

    const auto v1 = entity->getVertexDofs(/*vertex=*/1);
    const auto v2 = entity->getVertexDofs(/*vertex=*/2);
    const auto v4 = entity->getVertexDofs(/*vertex=*/4);
    const auto v7 = entity->getVertexDofs(/*vertex=*/7);
    ASSERT_GE(v1.size(), 2u);
    ASSERT_GE(v2.size(), 2u);
    ASSERT_GE(v4.size(), 2u);
    ASSERT_GE(v7.size(), 2u);

    const GlobalIndex v1x = v1[0] + offset;
    const GlobalIndex v2x = v2[0] + offset;
    const GlobalIndex v4x = v4[0] + offset;
    const GlobalIndex v7x = v7[0] + offset;

    // For r = (-gap, 0), d = gap, active contact:
    // dF_x/dr_x = -penalty. Integrated stiffness scales with edge mass matrix:
    // M = (L/6) [[2,1],[1,2]] for L=1.
    const Real k = 10.0;
    const Real M00 = Real(2.0 / 6.0);
    const Real M01 = Real(1.0 / 6.0);
    const Real tol = 1e-10;

    EXPECT_NEAR(jac(v1x, v1x), -k * M00, tol);
    EXPECT_NEAR(jac(v1x, v2x), -k * M01, tol);
    EXPECT_NEAR(jac(v1x, v4x), +k * M00, tol);
    EXPECT_NEAR(jac(v1x, v7x), +k * M01, tol);
}

TEST(SurfaceContactKernel, ContactCountStatePersistsAcrossTimeSteps)
{
    const int slave_marker = 10;
    const int master_marker = 20;
    const double gap = 0.1;

    auto mesh = build_two_quad_gap_mesh(gap, slave_marker, master_marker);
    FESystem sys(mesh);

    FieldSpec field;
    field.name = "u";
    field.space = build_displacement_space();
    field.components = 2;
    FieldId u = sys.addField(field);

    sys.addOperator("contact");

    PenaltySurfaceContactConfig cfg;
    cfg.field = u;
    cfg.slave_marker = slave_marker;
    cfg.master_marker = master_marker;
    cfg.search_radius = 1.0;
    cfg.activation_distance = 0.2;
    cfg.penalty = 10.0;
    cfg.quadrature_order = 2;
    cfg.track_contact_count = true;
    cfg.max_state_qpts = 8;

    auto kernel = std::make_shared<PenaltySurfaceContactKernel>(cfg);
    sys.addGlobalKernel("contact", kernel);
    sys.setup();

    GlobalIndex slave_face_id = svmp::FE::INVALID_GLOBAL_INDEX;
    sys.meshAccess().forEachBoundaryFace(slave_marker, [&](GlobalIndex face_id, GlobalIndex /*cell_id*/) {
        if (slave_face_id == svmp::FE::INVALID_GLOBAL_INDEX) {
            slave_face_id = face_id;
        }
    });
    ASSERT_NE(slave_face_id, svmp::FE::INVALID_GLOBAL_INDEX);

    std::vector<Real> uvec(static_cast<std::size_t>(sys.dofHandler().getNumDofs()), 0.0);
    SystemStateView state;
    state.u = std::span<const Real>(uvec.data(), uvec.size());

    AssemblyRequest req;
    req.op = "contact";
    req.want_vector = true;

    DenseVectorView rhs(sys.dofHandler().getNumDofs());

    sys.beginTimeStep();
    (void)sys.assemble(req, state, nullptr, &rhs);

    {
        const auto view = sys.globalKernelBoundaryFaceState(*kernel, slave_face_id, /*num_qpts=*/1);
        ASSERT_TRUE(static_cast<bool>(view));
        const auto* counter = reinterpret_cast<const int*>(view.data_work);
        ASSERT_NE(counter, nullptr);
        EXPECT_EQ(*counter, 1);
    }

    sys.commitTimeStep();

    sys.beginTimeStep();
    (void)sys.assemble(req, state, nullptr, &rhs);
    {
        const auto view = sys.globalKernelBoundaryFaceState(*kernel, slave_face_id, /*num_qpts=*/1);
        ASSERT_TRUE(static_cast<bool>(view));
        const auto* counter = reinterpret_cast<const int*>(view.data_work);
        ASSERT_NE(counter, nullptr);
        EXPECT_EQ(*counter, 2);
    }
    sys.commitTimeStep();
}

