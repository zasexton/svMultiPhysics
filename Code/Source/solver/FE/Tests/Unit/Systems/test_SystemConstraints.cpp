/**
 * @file test_SystemConstraints.cpp
 * @brief Unit tests for Systems boundary-DOF extraction helpers
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Systems/SystemConstraints.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"

#include "Spaces/H1Space.h"

#include "Mesh/Mesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellShape.h"

#include <algorithm>
#include <array>
#include <vector>

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Mesh;
using svmp::MeshBase;
using svmp::EntityKind;
using svmp::INVALID_INDEX;

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;

using svmp::FE::assembly::DenseMatrixView;
using svmp::FE::assembly::MassKernel;

using svmp::FE::spaces::H1Space;

using svmp::FE::systems::FESystem;
using svmp::FE::systems::FieldSpec;
using svmp::FE::systems::SystemStateView;

namespace {

std::shared_ptr<Mesh> build_single_quad_mesh_with_left_edge_marker(int marker, std::string set_name)
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
            base->add_to_set(EntityKind::Face, set_name, f);
            break;
        }
    }

    return svmp::create_mesh(std::move(base));
}

} // namespace

TEST(SystemConstraints, MarkerAndFaceSetExtractionAgree)
{
    const int marker = 7;
    const std::string set_name = "left";

    auto mesh = build_single_quad_mesh_with_left_edge_marker(marker, set_name);
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
    sys.setup();

    auto by_marker = svmp::FE::systems::boundaryDofsByMarker(*mesh, sys.dofHandler(), marker);
    auto by_set = svmp::FE::systems::boundaryDofsByFaceSet(*mesh, sys.dofHandler(), set_name);

    EXPECT_EQ(by_marker, by_set);
    EXPECT_EQ(by_marker.size(), 2u);  // vertices 0 and 3 for Q1
}

TEST(SystemConstraints, DirichletBCViaFaceSetAffectsAssembly)
{
    const int marker = 7;
    const std::string set_name = "left";

    auto mesh = build_single_quad_mesh_with_left_edge_marker(marker, set_name);
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));

    sys.setup();

    sys.addConstraint(svmp::FE::systems::makeDirichletConstantByFaceSet(*mesh, sys.dofHandler(),
                                                                       set_name, /*value=*/0.0));
    sys.setup();

    DenseMatrixView mass(sys.dofHandler().getNumDofs());
    SystemStateView state;
    sys.assembleMass(state, mass);

    // Constrained rows should be identity-like (diag=1, off-diagonals=0).
    const auto constrained = svmp::FE::systems::boundaryDofsByFaceSet(*mesh, sys.dofHandler(), set_name);
    ASSERT_EQ(constrained.size(), 2u);
    for (auto dof : constrained) {
        EXPECT_NEAR(mass.getMatrixEntry(dof, dof), 1.0, 1e-12);
        for (GlobalIndex j = 0; j < mass.numCols(); ++j) {
            if (j == dof) continue;
            EXPECT_NEAR(mass.getMatrixEntry(dof, j), 0.0, 1e-12);
        }
    }
}

TEST(SystemConstraints, ReturnsEmptyForNonExistentFaceSet)
{
    const int marker = 7;
    const std::string set_name = "left";

    auto mesh = build_single_quad_mesh_with_left_edge_marker(marker, set_name);
    auto space = std::make_shared<H1Space>(ElementType::Quad4, /*order=*/1);

    FESystem sys(mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.setup();

    auto result = svmp::FE::systems::boundaryDofsByFaceSet(*mesh, sys.dofHandler(), "non_existent_set");
    EXPECT_TRUE(result.empty());
}
