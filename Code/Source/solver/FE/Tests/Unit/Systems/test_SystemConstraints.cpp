/**
 * @file test_SystemConstraints.cpp
 * @brief Unit tests for Systems boundary-DOF extraction helpers
 *
 * The marker and face-set lookups are checked against the node coordinates of
 * each DOF: on structured grids of the unit square and cube, a DOF lies on a
 * boundary side exactly when its Lagrange node does.  The grids cover the
 * linear and the quadratic cells of each family with fields of order 1 to 3,
 * built both ways the solver meets a mesh: with every face at build time, and
 * with the boundary faces only, completed by FESystem::setup().
 */

#include <gtest/gtest.h>

#include "Systems/FESystem.h"
#include "Constraints/SystemConstraints.h"

#include "Assembly/AssemblyKernel.h"
#include "Assembly/GlobalSystemView.h"

#include "Basis/LagrangeBasis.h"
#include "Geometry/MappingFactory.h"
#include "Spaces/H1Space.h"

#include "Mesh/Mesh.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Topology/CellShape.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

using svmp::CellFamily;
using svmp::CellShape;
using svmp::Mesh;
using svmp::MeshBase;
using svmp::EntityKind;
using svmp::INVALID_INDEX;

using svmp::FE::ElementType;
using svmp::FE::GlobalIndex;
using svmp::FE::Real;

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

    auto by_marker = svmp::FE::constraints::boundaryDofsByMarker(*mesh, sys.dofHandler(), marker);
    auto by_set = svmp::FE::constraints::boundaryDofsByFaceSet(*mesh, sys.dofHandler(), set_name);

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

    sys.addConstraint(svmp::FE::constraints::makeDirichletConstantByFaceSet(*mesh, sys.dofHandler(),
                                                                       set_name, /*value=*/0.0));
    sys.setup();

    DenseMatrixView mass(sys.dofHandler().getNumDofs());
    SystemStateView state;
    sys.assembleMass(state, mass);

    // Constrained rows should be identity-like (diag=1, off-diagonals=0).
    const auto constrained = svmp::FE::constraints::boundaryDofsByFaceSet(*mesh, sys.dofHandler(), set_name);
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

    auto result = svmp::FE::constraints::boundaryDofsByFaceSet(*mesh, sys.dofHandler(), "non_existent_set");
    EXPECT_TRUE(result.empty());
}

namespace {

/// How a test grid gets its faces.
enum class FaceRoute {
    /// Every face at build time (MeshBase::finalize), as in a mesh read whole.
    AllFacesAtBuild,
    /// Only the labelled boundary faces, as the solver builds a mesh from its
    /// boundary face files; FESystem::setup() completes the topology.
    BoundaryFacesOnly
};

struct GridCase {
    std::string name;
    CellFamily family;
    int geometry_order;  ///< 1: linear cells; 2: quadratic cells (Triangle6, Quad9).
    int field_order;
    FaceRoute route;
};

std::ostream& operator<<(std::ostream& os, const GridCase& c) { return os << c.name; }

/// Side labels of the unit square or cube: x = 0, 1 are 1, 2; y = 0, 1 are 3, 4; z = 0, 1 are 5, 6.
int sideLabel(int axis, bool upper) { return 2 * axis + (upper ? 2 : 1); }

struct Grid {
    std::shared_ptr<Mesh> mesh;
    int dim{0};
    ElementType linear_type{ElementType::Unknown};
    std::vector<std::array<Real, 3>> points;
    std::vector<std::vector<svmp::index_t>> cell_corners;
};

/**
 * The structured grid of a case on the unit square (n x n squares) or cube
 * (n x n x n cubes), on a lattice of (g n + 1)^dim points for geometry order g.
 * Squares split into two triangles along one diagonal, cubes into six
 * tetrahedra about their main diagonal (a conforming split).  Each boundary
 * face is labelled by its side and put in the face set "side<label>".
 */
Grid buildGrid(const GridCase& c, int n)
{
    Grid grid;
    const bool solid = (c.family == CellFamily::Tetra || c.family == CellFamily::Hex);
    grid.dim = solid ? 3 : 2;
    const int g = c.geometry_order;
    const int m = g * n;
    const int nz = solid ? m : 0;
    const auto id = [&](int i, int j, int k) {
        return static_cast<svmp::index_t>((k * (m + 1) + j) * (m + 1) + i);
    };

    std::vector<svmp::real_t> X;
    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j <= m; ++j) {
            for (int i = 0; i <= m; ++i) {
                const std::array<Real, 3> x{Real(i) / Real(m), Real(j) / Real(m), solid ? Real(k) / Real(m) : Real(0)};
                grid.points.push_back(x);
                for (int d = 0; d < grid.dim; ++d) {
                    X.push_back(static_cast<svmp::real_t>(x[static_cast<std::size_t>(d)]));
                }
            }
        }
    }

    std::vector<svmp::offset_t> offsets{0};
    std::vector<svmp::index_t> conn;
    std::vector<CellShape> shapes;
    CellShape shape{};
    shape.family = c.family;
    shape.order = g;
    // Lattice index of the midpoint of two lattice points of the corner sublattice.
    const auto mid = [&](std::array<int, 3> a, std::array<int, 3> b) {
        return id((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2);
    };
    const auto at = [&](std::array<int, 3> a) { return id(a[0], a[1], a[2]); };
    const auto add_cell = [&](const std::vector<std::array<int, 3>>& corners, std::vector<svmp::index_t> extra) {
        std::vector<svmp::index_t> ids;
        for (const auto& a : corners) {
            ids.push_back(at(a));
        }
        grid.cell_corners.push_back(ids);
        conn.insert(conn.end(), ids.begin(), ids.end());
        conn.insert(conn.end(), extra.begin(), extra.end());
        offsets.push_back(static_cast<svmp::offset_t>(conn.size()));
        shapes.push_back(shape);
    };

    if (c.family == CellFamily::Triangle) {
        grid.linear_type = ElementType::Triangle3;
        shape.num_corners = 3;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const std::array<int, 3> p00{g * i, g * j, 0}, p10{g * (i + 1), g * j, 0};
                const std::array<int, 3> p11{g * (i + 1), g * (j + 1), 0}, p01{g * i, g * (j + 1), 0};
                for (const auto& tri : {std::array{p00, p10, p11}, std::array{p00, p11, p01}}) {
                    std::vector<svmp::index_t> extra;
                    if (g == 2) {
                        extra = {mid(tri[0], tri[1]), mid(tri[1], tri[2]), mid(tri[2], tri[0])};
                    }
                    add_cell({tri[0], tri[1], tri[2]}, extra);
                }
            }
        }
    } else if (c.family == CellFamily::Quad) {
        grid.linear_type = ElementType::Quad4;
        shape.num_corners = 4;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const std::array<int, 3> p00{g * i, g * j, 0}, p10{g * (i + 1), g * j, 0};
                const std::array<int, 3> p11{g * (i + 1), g * (j + 1), 0}, p01{g * i, g * (j + 1), 0};
                std::vector<svmp::index_t> extra;
                if (g == 2) {
                    extra = {mid(p00, p10), mid(p10, p11), mid(p11, p01), mid(p01, p00), mid(p00, p11)};
                }
                add_cell({p00, p10, p11, p01}, extra);
            }
        }
    } else if (c.family == CellFamily::Hex) {
        grid.linear_type = ElementType::Hex8;
        shape.num_corners = 8;
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    add_cell({{i, j, k}, {i + 1, j, k}, {i + 1, j + 1, k}, {i, j + 1, k},
                              {i, j, k + 1}, {i + 1, j, k + 1}, {i + 1, j + 1, k + 1}, {i, j + 1, k + 1}},
                             {});
                }
            }
        }
    } else {
        grid.linear_type = ElementType::Tetra4;
        shape.num_corners = 4;
        const std::array<std::array<int, 3>, 6> axis_orders{{{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
                                                             {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    for (const auto& order : axis_orders) {
                        std::vector<std::array<int, 3>> path{{i, j, k}};
                        for (const int axis : order) {
                            auto next = path.back();
                            next[static_cast<std::size_t>(axis)] += 1;
                            path.push_back(next);
                        }
                        // Keep the volume positive: an odd axis order reverses it.
                        const bool odd = (order == std::array<int, 3>{0, 2, 1} ||
                                          order == std::array<int, 3>{1, 0, 2} ||
                                          order == std::array<int, 3>{2, 1, 0});
                        if (odd) {
                            std::swap(path[1], path[2]);
                        }
                        add_cell(path, {});
                    }
                }
            }
        }
    }

    auto base = std::make_shared<MeshBase>();
    base->build_from_arrays(grid.dim, X, offsets, conn, shapes);
    svmp::MeshFinalizeOptions options{};
    if (c.route == FaceRoute::BoundaryFacesOnly) {
        options.codim1_storage = svmp::MeshCodim1StorageMode::BoundaryOnly;
        options.edge_storage = false;
    }
    base->finalize(options);

    for (svmp::index_t f = 0; f < static_cast<svmp::index_t>(base->n_faces()); ++f) {
        const auto [verts, count] = base->face_vertices_span(f);
        for (int axis = 0; axis < grid.dim; ++axis) {
            for (const bool upper : {false, true}) {
                const Real value = upper ? Real(1) : Real(0);
                bool on_side = count > 0u;
                for (std::size_t v = 0; v < count; ++v) {
                    const auto& x = grid.points[static_cast<std::size_t>(verts[v])];
                    on_side = on_side && std::abs(x[static_cast<std::size_t>(axis)] - value) < 1e-12;
                }
                if (on_side) {
                    const int label = sideLabel(axis, upper);
                    base->set_boundary_label(f, label);
                    base->add_to_set(EntityKind::Face, "side" + std::to_string(label), f);
                }
            }
        }
    }
    grid.mesh = svmp::create_mesh(std::move(base));
    return grid;
}

/// The DOFs of a scalar Lagrange field whose nodes lie on each side, by side label.
std::map<int, std::vector<GlobalIndex>> sideDofsByNodeCoordinates(const Grid& grid,
                                                                  const FESystem& sys,
                                                                  const H1Space& space)
{
    const auto* lagrange = dynamic_cast<const svmp::FE::basis::LagrangeBasis*>(&space.element().basis());
    EXPECT_NE(lagrange, nullptr);
    std::map<int, std::vector<GlobalIndex>> sides;
    if (lagrange == nullptr) {
        return sides;
    }
    const auto& nodes = lagrange->nodes();
    const auto& map = sys.dofHandler().getDofMap();
    for (std::size_t cell = 0; cell < grid.cell_corners.size(); ++cell) {
        std::vector<svmp::FE::math::Vector<Real, 3>> corners;
        for (const auto v : grid.cell_corners[cell]) {
            const auto& x = grid.points[static_cast<std::size_t>(v)];
            corners.push_back(svmp::FE::math::Vector<Real, 3>{x[0], x[1], x[2]});
        }
        svmp::FE::geometry::MappingRequest request;
        request.element_type = grid.linear_type;
        request.geometry_order = 1;
        request.use_affine = (grid.linear_type == ElementType::Triangle3 || grid.linear_type == ElementType::Tetra4);
        const auto mapping = svmp::FE::geometry::MappingFactory::create(request, corners);
        const auto dofs = map.getCellDofs(static_cast<GlobalIndex>(cell));
        EXPECT_EQ(dofs.size(), nodes.size());
        for (std::size_t k = 0; k < nodes.size() && k < dofs.size(); ++k) {
            const auto x = mapping->map_to_physical(nodes[k]);
            for (int axis = 0; axis < grid.dim; ++axis) {
                for (const bool upper : {false, true}) {
                    const Real value = upper ? Real(1) : Real(0);
                    if (std::abs(x[static_cast<std::size_t>(axis)] - value) < 1e-12) {
                        sides[sideLabel(axis, upper)].push_back(dofs[k]);
                    }
                }
            }
        }
    }
    for (auto& [label, dofs] : sides) {
        std::sort(dofs.begin(), dofs.end());
        dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
    }
    return sides;
}

class BoundaryDofsOnGrids : public ::testing::TestWithParam<GridCase> {};

} // namespace

/**
 * boundaryDofsByMarker and boundaryDofsByFaceSet return, for each side, the
 * DOFs whose nodes lie on that side: no DOF of another edge or face, and none
 * missing.  A side of k n + 1 nodes (2D) or (k n + 1)^2 nodes (3D) for order k.
 */
TEST_P(BoundaryDofsOnGrids, SideDofsAreTheDofsWithNodesOnTheSide)
{
    const auto& c = GetParam();
    const int n = (c.family == CellFamily::Tetra || c.family == CellFamily::Hex) ? 2 : 3;
    const auto grid = buildGrid(c, n);
    auto space = std::make_shared<H1Space>(grid.linear_type, c.field_order);

    FESystem sys(grid.mesh);
    auto u = sys.addField(FieldSpec{.name = "u", .space = space, .components = 1});
    sys.addOperator("mass");
    sys.addCellKernel("mass", u, std::make_shared<MassKernel>(1.0));
    sys.setup();

    const auto expected = sideDofsByNodeCoordinates(grid, sys, *space);
    ASSERT_EQ(expected.size(), static_cast<std::size_t>(2 * grid.dim));
    const auto per_line = static_cast<std::size_t>(c.field_order * n + 1);
    for (const auto& [label, dofs] : expected) {
        SCOPED_TRACE("side " + std::to_string(label));
        EXPECT_EQ(dofs.size(), grid.dim == 2 ? per_line : per_line * per_line);
        EXPECT_EQ(svmp::FE::constraints::boundaryDofsByMarker(*grid.mesh, sys.dofHandler(), label), dofs);
        EXPECT_EQ(svmp::FE::constraints::boundaryDofsByFaceSet(*grid.mesh, sys.dofHandler(),
                                                               "side" + std::to_string(label)),
                  dofs);
    }
}

INSTANTIATE_TEST_SUITE_P(
    SystemConstraints,
    BoundaryDofsOnGrids,
    ::testing::Values(
        GridCase{"Triangle3P1AllFaces", CellFamily::Triangle, 1, 1, FaceRoute::AllFacesAtBuild},
        GridCase{"Triangle3P2AllFaces", CellFamily::Triangle, 1, 2, FaceRoute::AllFacesAtBuild},
        GridCase{"Triangle3P2BoundaryFaces", CellFamily::Triangle, 1, 2, FaceRoute::BoundaryFacesOnly},
        GridCase{"Triangle3P3AllFaces", CellFamily::Triangle, 1, 3, FaceRoute::AllFacesAtBuild},
        GridCase{"Triangle3P3BoundaryFaces", CellFamily::Triangle, 1, 3, FaceRoute::BoundaryFacesOnly},
        GridCase{"Triangle6P2AllFaces", CellFamily::Triangle, 2, 2, FaceRoute::AllFacesAtBuild},
        GridCase{"Triangle6P2BoundaryFaces", CellFamily::Triangle, 2, 2, FaceRoute::BoundaryFacesOnly},
        GridCase{"Quad4Q1AllFaces", CellFamily::Quad, 1, 1, FaceRoute::AllFacesAtBuild},
        GridCase{"Quad4Q2AllFaces", CellFamily::Quad, 1, 2, FaceRoute::AllFacesAtBuild},
        GridCase{"Quad4Q2BoundaryFaces", CellFamily::Quad, 1, 2, FaceRoute::BoundaryFacesOnly},
        GridCase{"Quad4Q3AllFaces", CellFamily::Quad, 1, 3, FaceRoute::AllFacesAtBuild},
        GridCase{"Quad9Q2AllFaces", CellFamily::Quad, 2, 2, FaceRoute::AllFacesAtBuild},
        GridCase{"Quad9Q2BoundaryFaces", CellFamily::Quad, 2, 2, FaceRoute::BoundaryFacesOnly},
        GridCase{"Tetra4P2AllFaces", CellFamily::Tetra, 1, 2, FaceRoute::AllFacesAtBuild},
        GridCase{"Tetra4P2BoundaryFaces", CellFamily::Tetra, 1, 2, FaceRoute::BoundaryFacesOnly},
        GridCase{"Tetra4P3AllFaces", CellFamily::Tetra, 1, 3, FaceRoute::AllFacesAtBuild},
        GridCase{"Hex8Q2AllFaces", CellFamily::Hex, 1, 2, FaceRoute::AllFacesAtBuild},
        GridCase{"Hex8Q2BoundaryFaces", CellFamily::Hex, 1, 2, FaceRoute::BoundaryFacesOnly}),
    [](const ::testing::TestParamInfo<GridCase>& info) { return info.param.name; });
