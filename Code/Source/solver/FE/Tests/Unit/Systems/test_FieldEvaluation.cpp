/**
 * @file test_FieldEvaluation.cpp
 * @brief Systems nonlocal helpers: point location + field evaluation at a point
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"

#include "Assembly/GlobalSystemView.h"

#include "Spaces/H1Space.h"

#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <vector>

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Mesh;
using svmp::MeshBase;

namespace svmp::FE::systems::test {
namespace {

std::shared_ptr<Mesh> build_single_quad_mesh()
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

    return svmp::create_mesh(std::move(base));
}

} // namespace

TEST(FieldEvaluation, LocatesPointAndEvaluatesScalarField)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    const auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.setup();

    ASSERT_NE(sys.searchAccess(), nullptr);
    sys.searchAccess()->build();

    const std::array<Real, 3> x{0.3, 0.7, 0.0};
    const auto loc = sys.locatePoint(x);
    ASSERT_TRUE(loc.found);
    EXPECT_EQ(loc.cell_id, 0);

    const auto ndofs = static_cast<std::size_t>(sys.dofHandler().getNumDofs());
    ASSERT_EQ(ndofs, 4u);

    // Assign nodal values through cell-local DOF ordering so the test stays
    // valid even when global DOF numbering is spatially reordered.
    std::vector<Real> uvec(ndofs, 0.0);
    const auto cell_dofs = sys.dofHandler().getCellDofs(0);
    ASSERT_EQ(cell_dofs.size(), 4u);
    const std::array<Real, 4> cell_local_x = {0.0, 1.0, 1.0, 0.0};
    for (std::size_t a = 0; a < cell_local_x.size(); ++a) {
        const auto dof = static_cast<std::size_t>(cell_dofs[a]);
        ASSERT_LT(dof, uvec.size());
        uvec[dof] = cell_local_x[a];
    }

    SystemStateView state;
    state.u = uvec;

    const auto v = sys.evaluateFieldAtPoint(u, state, x);
    ASSERT_TRUE(v.has_value());
    EXPECT_NEAR((*v)[0], x[0], 1e-12);
}

TEST(FieldEvaluation, ExposesNearestVertexQueries)
{
    auto mesh = build_single_quad_mesh();
    auto space = std::make_shared<spaces::H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    (void)sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.setup();

    ASSERT_NE(sys.searchAccess(), nullptr);
    sys.searchAccess()->build();

    const std::array<Real, 3> p{0.05, 0.05, 0.0};
    const auto nn = sys.searchAccess()->nearestVertex(p);
    ASSERT_TRUE(nn.found);
    EXPECT_EQ(nn.vertex_id, 0);

    const auto knn = sys.searchAccess()->kNearestVertices(p, 2u);
    ASSERT_EQ(knn.size(), 2u);
    EXPECT_EQ(knn[0].vertex_id, 0);
    EXPECT_GE(knn[1].distance, knn[0].distance);
}

} // namespace svmp::FE::systems::test
