/**
 * @file test_HigherOrderMappings.cpp
 * @brief Tests for higher-order isoparametric mappings across element types
 */

#include <gtest/gtest.h>

#include "FE/Assembly/FunctionalAssembler.h"
#include "FE/Assembly/MeshAccess.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/NodeOrderingConventions.h"
#include "FE/Basis/SerendipityBasis.h"
#include "FE/Dofs/DofMap.h"
#include "FE/Elements/ReferenceElement.h"
#include "FE/Geometry/CutQuadratureMapping.h"
#include "FE/Geometry/FrameGeometry.h"
#include "FE/Geometry/IsoparametricMapping.h"
#include "FE/Geometry/MappingFactory.h"
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Spaces/H1Space.h"
#include "Mesh/Core/MeshBase.h"
#include "Mesh/Mesh.h"
#include "Mesh/Topology/CellShape.h"

#include <array>
#include <numeric>

using namespace svmp::FE;
using namespace svmp::FE::geometry;

namespace {

svmp::MeshBase makeQuad9GeometryDofMesh()
{
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 2);
    const auto& nodes = basis->nodes();

    std::vector<svmp::real_t> x_ref;
    x_ref.reserve(nodes.size() * 3u);
    for (const auto& node : nodes) {
        x_ref.push_back(node[0]);
        x_ref.push_back(node[1]);
        x_ref.push_back(node[2]);
    }

    std::vector<svmp::offset_t> offsets = {0, static_cast<svmp::offset_t>(nodes.size())};
    std::vector<svmp::index_t> conn(nodes.size());
    std::iota(conn.begin(), conn.end(), 0);
    std::vector<svmp::CellShape> shapes = {{svmp::CellFamily::Quad, 4, 2}};

    svmp::MeshBase mesh;
    mesh.build_from_arrays(3, x_ref, offsets, conn, shapes);

    const auto bottom_edge = mesh.cell_face_geometry_dofs(0, 0);
    if (bottom_edge.size() >= 3u) {
        auto mid = mesh.geometry_dof_coords(bottom_edge[1], svmp::Configuration::Reference);
        mid[1] += 0.30;
        mid[2] += 0.10;
        mesh.set_current_geometry_dof_coords(bottom_edge[1], mid);
    }
    const auto interiors = mesh.cell_interior_geometry_dofs(0);
    if (!interiors.empty()) {
        auto center = mesh.geometry_dof_coords(interiors.front(), svmp::Configuration::Current);
        center[2] += 0.40;
        mesh.set_current_geometry_dof_coords(interiors.front(), center);
    }
    return mesh;
}

std::vector<math::Vector<Real, 3>> geometryNodesFromMesh(
    const svmp::MeshBase& mesh,
    svmp::Configuration cfg)
{
    std::vector<math::Vector<Real, 3>> nodes;
    const auto dofs = mesh.cell_geometry_dofs(0);
    nodes.reserve(dofs.size());
    for (const auto dof : dofs) {
        const auto x = mesh.geometry_dof_coords(dof, cfg);
        nodes.push_back({x[0], x[1], x[2]});
    }
    return nodes;
}

} // namespace

TEST(HigherOrderMapping, IdentityLineOrder2) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Line2, 2); // Line3
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.3), Real(0), Real(0)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-12);

    EXPECT_GT(map.jacobian_determinant(xi), 0.0);
}

TEST(HigherOrderMapping, IdentityTriangleOrder2) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Triangle3, 2); // Triangle6
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.2), Real(0.1), Real(0)}; // inside reference triangle
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-10);
    EXPECT_NEAR(x[1], xi[1], 1e-10);

    EXPECT_GT(map.jacobian_determinant(xi), 0.0);
}

TEST(HigherOrderMapping, IdentityTetraOrder2) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Tetra4, 2); // Tetra10
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.1), Real(0.2), Real(0.1)}; // sum < 1
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-10);
    EXPECT_NEAR(x[1], xi[1], 1e-10);
    EXPECT_NEAR(x[2], xi[2], 1e-10);

    EXPECT_GT(map.jacobian_determinant(xi), 0.0);
}

TEST(HigherOrderMapping, IdentityQuadOrder2) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 2); // Quad9
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.2), Real(-0.3), Real(0)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-12);
    EXPECT_NEAR(x[1], xi[1], 1e-12);

    auto det = map.jacobian_determinant(xi);
    EXPECT_NEAR(det, 1.0, 1e-10);
}

TEST(HigherOrderMapping, IdentityQuadSerendipityOrder2) {
    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Quad8, 2);
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(-1), Real(-1), Real(0)},
        {Real(1),  Real(-1), Real(0)},
        {Real(1),  Real(1),  Real(0)},
        {Real(-1), Real(1),  Real(0)},
        {Real(0),  Real(-1), Real(0)}, // mid-bottom
        {Real(1),  Real(0),  Real(0)}, // mid-right
        {Real(0),  Real(1),  Real(0)}, // mid-top
        {Real(-1), Real(0),  Real(0)}, // mid-left
    };
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.1), Real(0.2), Real(0)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-10);
    EXPECT_NEAR(x[1], xi[1], 1e-10);

    EXPECT_GT(map.jacobian_determinant(xi), 0.0);
}

TEST(HigherOrderMapping, IdentityHexOrder2) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Hex8, 2); // Hex27
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.1), Real(-0.2), Real(0.25)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-10);
    EXPECT_NEAR(x[1], xi[1], 1e-10);
    EXPECT_NEAR(x[2], xi[2], 1e-10);

    auto det = map.jacobian_determinant(xi);
    EXPECT_NEAR(det, 1.0, 1e-8);
}

TEST(HigherOrderMapping, IdentityHexSerendipityOrder2) {
    // Geometry-mode serendipity basis uses robust Hex8-style mapping
    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Hex20, 2, true);
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(-1), Real(-1), Real(-1)},
        {Real(1),  Real(-1), Real(-1)},
        {Real(1),  Real(1),  Real(-1)},
        {Real(-1), Real(1),  Real(-1)},
        {Real(-1), Real(-1), Real(1)},
        {Real(1),  Real(-1), Real(1)},
        {Real(1),  Real(1),  Real(1)},
        {Real(-1), Real(1),  Real(1)},
        // x-edges (y,z fixed)
        {Real(0), Real(-1), Real(-1)},
        {Real(0), Real(-1), Real(1)},
        {Real(0), Real(1),  Real(-1)},
        {Real(0), Real(1),  Real(1)},
        // y-edges
        {Real(-1), Real(0), Real(-1)},
        {Real(1),  Real(0), Real(-1)},
        {Real(-1), Real(0), Real(1)},
        {Real(1),  Real(0), Real(1)},
        // z-edges
        {Real(-1), Real(-1), Real(0)},
        {Real(1),  Real(-1), Real(0)},
        {Real(-1), Real(1),  Real(0)},
        {Real(1),  Real(1),  Real(0)},
    };
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.1), Real(-0.1), Real(0.2)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-10);
    EXPECT_NEAR(x[1], xi[1], 1e-10);
    EXPECT_NEAR(x[2], xi[2], 1e-10);

    EXPECT_GT(map.jacobian_determinant(xi), 0.0);
}

TEST(HigherOrderMapping, Quad8SerendipityAffineDistortion) {
    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Quad8, 2);
    // Affine skew/scale: x' = 2x + y, y' = x + 3y
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(-2-1), Real(-1-3), Real(0)},   // (-1,-1)
        {Real(2-1),  Real(1-3),  Real(0)},   // (1,-1)
        {Real(2+1),  Real(1+3),  Real(0)},   // (1,1)
        {Real(-2+1), Real(-1+3), Real(0)},   // (-1,1)
        {Real(0-1),  Real(-3),   Real(0)},   // (0,-1)
        {Real(2),    Real(1),    Real(0)},   // (1,0)
        {Real(0+1),  Real(3),    Real(0)},   // (0,1)
        {Real(-2),   Real(-1),   Real(0)},   // (-1,0)
    };
    IsoparametricMapping map(basis, nodes);
    math::Vector<Real,3> xi{Real(0.2), Real(-0.1), Real(0)};
    auto x = map.map_to_physical(xi);
    // Analytic map: x' = 2x + y, y' = x + 3y
    Real xp = 2*xi[0] + xi[1];
    Real yp = xi[0] + 3*xi[1];
    EXPECT_NEAR(x[0], xp, 1e-10);
    EXPECT_NEAR(x[1], yp, 1e-10);
    auto det = map.jacobian_determinant(xi);
    // Jacobian of affine map: [[2,1],[1,3]] det = 5
    EXPECT_NEAR(det, 5.0, 1e-10);
}

TEST(HigherOrderMapping, Hex20SerendipityAffineDistortion) {
    // Geometry-mode Hex20 mapping; edge DOFs inert for geometry
    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Hex20, 2, true);
    // Affine map: x' = 2x + y, y' = x + 3y, z' = 4z
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(-2-1), Real(-1-3), Real(-4)},   // (-1,-1,-1)
        {Real(2-1),  Real(1-3),  Real(-4)},   // (1,-1,-1)
        {Real(2+1),  Real(1+3),  Real(-4)},   // (1,1,-1)
        {Real(-2+1), Real(-1+3), Real(-4)},   // (-1,1,-1)
        {Real(-2-1), Real(-1-3), Real(4)},    // (-1,-1,1)
        {Real(2-1),  Real(1-3),  Real(4)},    // (1,-1,1)
        {Real(2+1),  Real(1+3),  Real(4)},    // (1,1,1)
        {Real(-2+1), Real(-1+3), Real(4)},    // (-1,1,1)
        {Real(0-1),  Real(-3),   Real(-4)},   // (0,-1,-1)
        {Real(0+1),  Real(3),    Real(-4)},   // (0,1,-1)
        {Real(0-1),  Real(-3),   Real(4)},    // (0,-1,1)
        {Real(0+1),  Real(3),    Real(4)},    // (0,1,1)
        {Real(-2),   Real(-1),   Real(-4)},   // (-1,0,-1)
        {Real(2),    Real(1),    Real(-4)},   // (1,0,-1)
        {Real(-2),   Real(-1),   Real(4)},    // (-1,0,1)
        {Real(2),    Real(1),    Real(4)},    // (1,0,1)
        {Real(-2-1), Real(-1-3), Real(0)},    // (-1,-1,0)
        {Real(2-1),  Real(1-3),  Real(0)},    // (1,-1,0)
        {Real(2+1),  Real(1+3),  Real(0)},    // (1,1,0)
        {Real(-2+1), Real(-1+3), Real(0)},    // (-1,1,0)
    };
    IsoparametricMapping map(basis, nodes);
    math::Vector<Real,3> xi{Real(0.2), Real(-0.1), Real(0.3)};
    auto x = map.map_to_physical(xi);
    Real xp = 2*xi[0] + xi[1];
    Real yp = xi[0] + 3*xi[1];
    Real zp = 4*xi[2];
    EXPECT_NEAR(x[0], xp, 1e-10);
    EXPECT_NEAR(x[1], yp, 1e-10);
    EXPECT_NEAR(x[2], zp, 1e-10);
    auto det = map.jacobian_determinant(xi);
    // Jacobian [[2,1,0],[1,3,0],[0,0,4]] det = 2*3*4 - 1*1*4 = 24 - 4 = 20
    EXPECT_NEAR(det, 20.0, 1e-8);
}

TEST(HigherOrderMapping, Hex20SerendipityNodalIdentityNoGeometryMode) {
    using svmp::FE::basis::ReferenceNodeLayout;

    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Hex20, 2, false);

    std::vector<math::Vector<Real,3>> nodes;
    const std::size_t nn = ReferenceNodeLayout::num_nodes(ElementType::Hex20);
    nodes.reserve(nn);
    for (std::size_t i = 0; i < nn; ++i) {
        nodes.push_back(ReferenceNodeLayout::get_node_coords(ElementType::Hex20, i));
    }

    IsoparametricMapping map(basis, nodes);

    // Nodal identity: x(xi_i) = xi_i for all reference nodes.
    for (std::size_t i = 0; i < nn; ++i) {
        const auto xi = nodes[i];
        const auto x = map.map_to_physical(xi);
        EXPECT_NEAR(x[0], xi[0], 1e-10);
        EXPECT_NEAR(x[1], xi[1], 1e-10);
        EXPECT_NEAR(x[2], xi[2], 1e-10);
    }

    const auto det = map.jacobian_determinant(math::Vector<Real,3>{Real(0), Real(0), Real(0)});
    EXPECT_NEAR(det, 1.0, 1e-8);
}

TEST(HigherOrderMapping, MeshGeometryDofsBuildReferenceAndCurrentQuadraticMappings) {
    auto mesh = makeQuad9GeometryDofMesh();
    ASSERT_EQ(mesh.geometry_order_descriptor().storage, svmp::GeometryDofStorage::VertexCoordinates);
    ASSERT_EQ(mesh.geometry_order(0), 2);
    ASSERT_TRUE(mesh.has_current_coords());

    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 2);
    const auto ref_nodes = geometryNodesFromMesh(mesh, svmp::Configuration::Reference);
    const auto cur_nodes = geometryNodesFromMesh(mesh, svmp::Configuration::Current);
    ASSERT_EQ(ref_nodes.size(), basis->size());
    ASSERT_EQ(cur_nodes.size(), basis->size());

    IsoparametricMapping ref_map(basis, ref_nodes);
    IsoparametricMapping cur_map(basis, cur_nodes);

    const math::Vector<Real,3> xi{Real(0), Real(0), Real(0)};
    const auto x_ref = ref_map.map_to_physical(xi);
    const auto x_cur = cur_map.map_to_physical(xi);
    EXPECT_NEAR(x_ref[2], 0.0, 1e-12);
    EXPECT_GT(x_cur[2], x_ref[2] + Real(0.2));
    EXPECT_GT(ref_map.jacobian_determinant(xi), 0.0);
    EXPECT_GT(cur_map.jacobian_determinant(xi), 0.0);
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
TEST(HigherOrderMapping, SerendipityMeshOrderControlsGeometryInVolumeAssembly) {
    struct Case {
        ElementType type;
        ElementType vertex_type;
        svmp::CellFamily family;
        int vertex_count;
        math::Vector<Real, 3> probe;
        Real curved_volume;
        Real vertex_volume;
    };
    const std::array cases{
        Case{ElementType::Hex20, ElementType::Hex8, svmp::CellFamily::Hex, 8, {0., 0., 1.}, 8.8, 8.},
        Case{ElementType::Wedge15, ElementType::Wedge6, svmp::CellFamily::Wedge, 6, {.5, 0., 1.}, 1.05, 1.},
        Case{ElementType::Pyramid13, ElementType::Pyramid5, svmp::CellFamily::Pyramid, 5,
             {0., 0., 0.}, 4. / 3. - .3, 4. / 3.}
    };
    for (const auto& test : cases) {
        SCOPED_TRACE(static_cast<int>(test.type));
        const auto displacement = [&](const math::Vector<Real, 3>& x) {
            const double r = x[0], s = x[1], t = x[2];
            if (test.type == ElementType::Hex20) {
                return .15 * (1. + t) * (1. - r * r);
            }
            if (test.type == ElementType::Wedge15) {
                return .6 * (1. + t) * r * (1. - r - s);
            }
            return .15 * ((1. - t) * (1. - t) - r * r);
        };
        auto nodes = basis::SerendipityBasis(test.type, 2).nodes();
        std::vector<svmp::real_t> coordinates;
        for (auto& node : nodes) {
            node[2] += displacement(node);
            coordinates.insert(coordinates.end(), {node[0], node[1], node[2]});
        }
        std::vector<svmp::index_t> ids(nodes.size());
        std::iota(ids.begin(), ids.end(), 0);
        svmp::Mesh mesh;
        mesh.build_from_arrays(3, coordinates, {0, static_cast<svmp::offset_t>(ids.size())}, ids,
                               {{test.family, test.vertex_count, 2}});
        assembly::MeshAccess access(mesh);
        ASSERT_EQ(access.getCellType(0), test.type);
        ASSERT_EQ(access.getCellGeometryOrder(0), 2);
        auto mapping = makeCutCellGeometryMapping(access, 0);
        EXPECT_EQ(mapping->geometryBasis().order(), 2);
        EXPECT_EQ(mapping->num_nodes(), nodes.size());
        EXPECT_NEAR(mapping->map_to_physical(test.probe)[2], test.probe[2] + displacement(test.probe), 1.e-12);

        std::shared_ptr<GeometryMapping> inferred;
        EXPECT_NO_THROW(inferred = MappingFactory::create(MappingRequest{test.type}, nodes));
        if (inferred) {
            EXPECT_EQ(inferred->geometryBasis().order(), 2);
            EXPECT_NEAR(inferred->map_to_physical(test.probe)[2],
                        test.probe[2] + displacement(test.probe), 1.e-12);
        }

        dofs::DofMap dofs(1, test.vertex_count, test.vertex_count);
        std::vector<GlobalIndex> vertex_ids(test.vertex_count);
        std::iota(vertex_ids.begin(), vertex_ids.end(), 0);
        dofs.setCellDofs(0, vertex_ids);
        dofs.finalize();
        spaces::H1Space space(test.vertex_type, 1);
        assembly::FunctionalAssembler assembler;
        assembler.setMesh(access);
        assembler.setDofMap(dofs);
        assembler.setSpace(space);
        EXPECT_NEAR(assembler.computeVolume(), test.curved_volume, 1.e-12);

        // Integrating div(z e_z) = 1 also exercises each fitted face's curved
        // position, outward normal and physical quadrature weight.
        const auto& topology = elements::ReferenceElement::shared(test.type);
        Real flux = 0.;
        for (std::size_t face = 0; face < topology.num_faces(); ++face) {
            const auto face_type = topology.face_nodes(face).size() == 3
                                       ? ElementType::Triangle3 : ElementType::Quad4;
            const auto rule = quadrature::QuadratureFactory::create(face_type, 5);
            const auto frame = evaluateFaceFrame(*mapping, test.type, face, face_type, *rule);
            for (std::size_t q = 0; q < rule->num_points(); ++q) {
                flux += rule->weight(q) * frame.surface_measures[q] *
                        frame.cell_geometry.points[q][2] * frame.normals[q][2];
            }
        }
        EXPECT_NEAR(flux, test.curved_volume, 1.e-12);

        MappingRequest straight{test.type, 1};
        std::shared_ptr<GeometryMapping> vertex_mapping;
        EXPECT_NO_THROW(vertex_mapping = MappingFactory::create(straight, nodes));
        if (!vertex_mapping) {
            continue;
        }
        EXPECT_EQ(vertex_mapping->geometryBasis().order(), 1);
        EXPECT_EQ(vertex_mapping->num_nodes(), static_cast<std::size_t>(test.vertex_count));
        EXPECT_NEAR(vertex_mapping->map_to_physical(test.probe)[2], test.probe[2], 1.e-12);
        for (auto& node : nodes) {
            node[2] += 2.;
        }
        vertex_mapping->resetNodes(nodes);
        EXPECT_EQ(vertex_mapping->num_nodes(), static_cast<std::size_t>(test.vertex_count));
        EXPECT_NEAR(vertex_mapping->map_to_physical(test.probe)[2], test.probe[2] + 2., 1.e-12);
        const auto quadrature = quadrature::QuadratureFactory::create(test.type, 4);
        double volume = 0.;
        for (std::size_t q = 0; q < quadrature->num_points(); ++q) {
            volume += quadrature->weight(q) * vertex_mapping->jacobian_determinant(quadrature->point(q));
        }
        EXPECT_NEAR(volume, test.vertex_volume, 1.e-12);
    }
}
#endif
