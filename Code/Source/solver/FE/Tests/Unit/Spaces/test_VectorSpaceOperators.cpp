/**
 * @file test_VectorSpaceOperators.cpp
 * @brief Unit tests for vector-space differential operators (curl/div)
 */

#include <gtest/gtest.h>

#include "FE/Basis/NodeOrderingConventions.h"
#include "FE/Basis/VectorBasis.h"
#include "FE/Elements/ReferenceElement.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/HCurlSpace.h"
#include "FE/Spaces/HDivSpace.h"
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include "FE/Quadrature/TriangleQuadrature.h"

using namespace svmp::FE;
using namespace svmp::FE::spaces;

namespace {

FunctionSpace::Value xi2(Real x, Real y) {
    FunctionSpace::Value xi{};
    xi[0] = x;
    xi[1] = y;
    xi[2] = Real(0);
    return xi;
}

FunctionSpace::Value xi3(Real x, Real y, Real z) {
    FunctionSpace::Value xi{};
    xi[0] = x;
    xi[1] = y;
    xi[2] = z;
    return xi;
}

std::vector<FunctionSpace::Value> commuting_sample_points(ElementType type) {
    switch (type) {
        case ElementType::Wedge6:
            return {
                xi3(Real(0.18), Real(0.21), Real(-0.35)),
                xi3(Real(0.22), Real(0.11), Real(0.15)),
                xi3(Real(0.08), Real(0.26), Real(0.55)),
            };
        case ElementType::Pyramid5:
            return {
                xi3(Real(0.0), Real(0.0), Real(0.2)),
                xi3(Real(0.18), Real(-0.12), Real(0.35)),
                xi3(Real(0.04), Real(0.03), Real(0.78)),
            };
        default:
            return {xi3(Real(0), Real(0), Real(0))};
    }
}

double integrate_wedge_face_flux(const std::function<FunctionSpace::Value(const FunctionSpace::Value&)>& field,
                                 int face_id,
                                 int tri_order = 8,
                                 int quad_order = 8) {
    using namespace svmp::FE::quadrature;

    double flux = 0.0;

    if (face_id == 0 || face_id == 1) {
        TriangleQuadrature tri(tri_order);
        const double z = (face_id == 0) ? -1.0 : 1.0;
        const double nz = (face_id == 0) ? -1.0 : 1.0;
        for (std::size_t q = 0; q < tri.num_points(); ++q) {
            const auto& pt = tri.point(q);
            const auto v = field(xi3(pt[0], pt[1], static_cast<Real>(z)));
            flux += static_cast<double>(tri.weight(q)) * static_cast<double>(v[2]) * nz;
        }
        return flux;
    }

    QuadrilateralQuadrature quad(quad_order, quad_order);
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const auto& pt = quad.point(q);
        const double u = pt[0];
        const double eta = pt[1];

        FunctionSpace::Value x_ref{};
        FunctionSpace::Value cross{};
        if (face_id == 2) {
            const double t = 0.5 * (u + 1.0);
            x_ref = xi3(static_cast<Real>(t), Real(0), static_cast<Real>(eta));
            cross = xi3(Real(0), Real(-0.5), Real(0));
        } else if (face_id == 3) {
            const double t = 0.5 * (u + 1.0);
            x_ref = xi3(Real(0), static_cast<Real>(t), static_cast<Real>(eta));
            cross = xi3(Real(0.5), Real(0), Real(0));
        } else {
            const double t = 0.5 * (u + 1.0);
            x_ref = xi3(static_cast<Real>(t), static_cast<Real>(1.0 - t), static_cast<Real>(eta));
            cross = xi3(Real(-0.5), Real(-0.5), Real(0));
        }

        const auto v = field(x_ref);
        flux += static_cast<double>(quad.weight(q)) *
                static_cast<double>(v[0] * cross[0] + v[1] * cross[1] + v[2] * cross[2]);
    }

    return flux;
}

double integrate_pyramid_face_flux(const std::function<FunctionSpace::Value(const FunctionSpace::Value&)>& field,
                                   int face_id,
                                   int tri_order = 8,
                                   int quad_order = 8) {
    using namespace svmp::FE::quadrature;

    const FunctionSpace::Value v0 = xi3(Real(-1), Real(-1), Real(0));
    const FunctionSpace::Value v1 = xi3(Real(1),  Real(-1), Real(0));
    const FunctionSpace::Value v2 = xi3(Real(1),  Real(1),  Real(0));
    const FunctionSpace::Value v3 = xi3(Real(-1), Real(1),  Real(0));
    const FunctionSpace::Value v4 = xi3(Real(0),  Real(0),  Real(1));

    double flux = 0.0;
    if (face_id == 0) {
        QuadrilateralQuadrature quad(quad_order, quad_order);
        for (std::size_t q = 0; q < quad.num_points(); ++q) {
            const auto& pt = quad.point(q);
            const auto v = field(xi3(pt[0], pt[1], Real(0)));
            flux += static_cast<double>(quad.weight(q)) * -static_cast<double>(v[2]);
        }
        return flux;
    }

    TriangleQuadrature tri(tri_order);
    const FunctionSpace::Value* a = nullptr;
    const FunctionSpace::Value* b = nullptr;
    const FunctionSpace::Value* c = nullptr;
    switch (face_id) {
        case 1: a = &v0; b = &v1; c = &v4; break;
        case 2: a = &v1; b = &v2; c = &v4; break;
        case 3: a = &v2; b = &v3; c = &v4; break;
        default: a = &v3; b = &v0; c = &v4; break;
    }
    const FunctionSpace::Value e1 = *b - *a;
    const FunctionSpace::Value e2 = *c - *a;
    const auto cross = e1.cross(e2);

    for (std::size_t q = 0; q < tri.num_points(); ++q) {
        const auto& pt = tri.point(q);
        const Real l0 = Real(1) - pt[0] - pt[1];
        const Real l1 = pt[0];
        const Real l2 = pt[1];
        const auto xi = (*a) * l0 + (*b) * l1 + (*c) * l2;
        const auto v = field(xi);
        flux += static_cast<double>(tri.weight(q)) *
                static_cast<double>(v[0] * cross[0] + v[1] * cross[1] + v[2] * cross[2]);
    }

    return flux;
}

double integrate_segment_tangential_line_integral(
    const std::function<FunctionSpace::Value(const FunctionSpace::Value&)>& field,
    const FunctionSpace::Value& a,
    const FunctionSpace::Value& b,
    int quad_order = 8) {
    using svmp::FE::quadrature::GaussQuadrature1D;

    const FunctionSpace::Value dx_ds = (b - a) * Real(0.5);
    GaussQuadrature1D quad(quad_order);
    double value = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real s = quad.point(q)[0];
        const Real t = (s + Real(1)) * Real(0.5);
        const auto xi = a * (Real(1) - t) + b * t;
        const auto v = field(xi);
        value += static_cast<double>(quad.weight(q)) *
                 static_cast<double>(v.dot(dx_ds));
    }
    return value;
}

double integrate_edge_tangential_line_integral(
    const std::function<FunctionSpace::Value(const FunctionSpace::Value&)>& field,
    ElementType type,
    int edge_id,
    int quad_order = 8) {
    const auto ref = elements::ReferenceElement::create(type);
    const auto& edge_nodes = ref.edge_nodes(static_cast<std::size_t>(edge_id));
    const auto a = basis::ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(edge_nodes[0]));
    const auto b = basis::ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(edge_nodes[1]));
    return integrate_segment_tangential_line_integral(field, a, b, quad_order);
}

FunctionSpace::Value reference_interior_point(ElementType type) {
    switch (type) {
        case ElementType::Wedge6:
            return xi3(Real(1) / Real(3), Real(1) / Real(3), Real(0));
        case ElementType::Pyramid5:
            return xi3(Real(0), Real(0), Real(0.25));
        default:
            return xi3(Real(0), Real(0), Real(0));
    }
}

std::vector<FunctionSpace::Value> oriented_face_vertices(ElementType type, int face_id) {
    const auto ref = elements::ReferenceElement::create(type);
    const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(face_id));

    std::vector<FunctionSpace::Value> vertices;
    vertices.reserve(face_nodes.size());
    for (const auto node : face_nodes) {
        vertices.push_back(
            basis::ReferenceNodeLayout::get_node_coords(type, static_cast<std::size_t>(node)));
    }

    FunctionSpace::Value center{};
    for (const auto& v : vertices) {
        center += v;
    }
    center /= static_cast<Real>(vertices.size());

    FunctionSpace::Value normal{};
    if (vertices.size() == 3u) {
        const FunctionSpace::Value e01 = vertices[1] - vertices[0];
        const FunctionSpace::Value e02 = vertices[2] - vertices[0];
        normal = e01.cross(e02);
    } else {
        const FunctionSpace::Value e01 = vertices[1] - vertices[0];
        const FunctionSpace::Value e03 = vertices[3] - vertices[0];
        normal = e01.cross(e03);
    }

    if (normal.dot(reference_interior_point(type) - center) > Real(0)) {
        std::reverse(vertices.begin() + 1, vertices.end());
    }

    return vertices;
}

double integrate_face_boundary_circulation(
    const std::function<FunctionSpace::Value(const FunctionSpace::Value&)>& field,
    ElementType type,
    int face_id,
    int quad_order = 8) {
    const auto vertices = oriented_face_vertices(type, face_id);
    double circulation = 0.0;
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const auto& a = vertices[i];
        const auto& b = vertices[(i + 1) % vertices.size()];
        circulation += integrate_segment_tangential_line_integral(field, a, b, quad_order);
    }
    return circulation;
}

double integrate_cell_scalar(
    const std::function<Real(const FunctionSpace::Value&)>& scalar,
    ElementType type,
    int quad_order = 8) {
    const auto quad = quadrature::QuadratureFactory::create(type, quad_order, QuadratureType::GaussLegendre, false);
    double value = 0.0;
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        value += static_cast<double>(quad->weight(q)) *
                 static_cast<double>(scalar(quad->point(q)));
    }
    return value;
}

} // namespace

TEST(VectorSpaceOperators, HDivEvaluateDivergenceMatchesBasisCombination) {
    HDivSpace space(ElementType::Quad4, 0);
    ASSERT_EQ(space.dofs_per_element(), 4u);

    // For Quad4 RT0 with moment-based normalization, each basis divergence is constant 1/4.
    std::vector<Real> coeffs = {Real(1), Real(2), Real(3), Real(4)};
    const Real expected = Real(0.25) * (coeffs[0] + coeffs[1] + coeffs[2] + coeffs[3]);

    const Real div0 = space.evaluate_divergence(xi2(Real(0), Real(0)), coeffs);
    const Real div1 = space.evaluate_divergence(xi2(Real(0.25), Real(-0.5)), coeffs);
    EXPECT_NEAR(div0, expected, 1e-12);
    EXPECT_NEAR(div1, expected, 1e-12);
}

TEST(VectorSpaceOperators, HCurlEvaluateCurlMatchesBasisCombination) {
    HCurlSpace space(ElementType::Quad4, 0);
    ASSERT_EQ(space.dofs_per_element(), 4u);

    // For Quad4 Nedelec0, each basis curl is (0,0,0.25).
    std::vector<Real> coeffs = {Real(1), Real(2), Real(3), Real(4)};
    const Real expected_z = Real(0.25) * (coeffs[0] + coeffs[1] + coeffs[2] + coeffs[3]);

    const auto curl = space.evaluate_curl(xi2(Real(0), Real(0)), coeffs);
    EXPECT_NEAR(curl[0], 0.0, 1e-12);
    EXPECT_NEAR(curl[1], 0.0, 1e-12);
    EXPECT_NEAR(curl[2], expected_z, 1e-12);
}

TEST(VectorSpaceOperators, HybridOrder3HDivEvaluateDivergenceMatchesBasisCombination) {
    HDivSpace space(ElementType::Pyramid5, 3);
    basis::RaviartThomasBasis basis(ElementType::Pyramid5, 3);
    ASSERT_EQ(space.dofs_per_element(), basis.size());

    std::vector<Real> coeffs(space.dofs_per_element());
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        coeffs[i] = Real(0.02) * Real(i + 1);
    }

    const auto point = xi3(Real(0.1), Real(-0.15), Real(0.35));
    std::vector<Real> div_basis;
    basis.evaluate_divergence(point, div_basis);
    ASSERT_EQ(div_basis.size(), coeffs.size());

    Real expected = Real(0);
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        expected += coeffs[i] * div_basis[i];
    }

    const Real actual = space.evaluate_divergence(point, coeffs);
    EXPECT_NEAR(actual, expected, 1e-10);
}

TEST(VectorSpaceOperators, HybridOrder3HCurlEvaluateCurlMatchesBasisCombination) {
    HCurlSpace space(ElementType::Wedge6, 3);
    basis::NedelecBasis basis(ElementType::Wedge6, 3);
    ASSERT_EQ(space.dofs_per_element(), basis.size());

    std::vector<Real> coeffs(space.dofs_per_element());
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        coeffs[i] = Real(-0.015) * Real(i + 1);
    }

    const auto point = xi3(Real(0.2), Real(0.3), Real(-0.1));
    std::vector<math::Vector<Real, 3>> curl_basis;
    basis.evaluate_curl(point, curl_basis);
    ASSERT_EQ(curl_basis.size(), coeffs.size());

    math::Vector<Real, 3> expected{};
    for (std::size_t i = 0; i < coeffs.size(); ++i) {
        expected[0] += coeffs[i] * curl_basis[i][0];
        expected[1] += coeffs[i] * curl_basis[i][1];
        expected[2] += coeffs[i] * curl_basis[i][2];
    }

    const auto actual = space.evaluate_curl(point, coeffs);
    EXPECT_NEAR(actual[0], expected[0], 1e-10);
    EXPECT_NEAR(actual[1], expected[1], 1e-10);
    EXPECT_NEAR(actual[2], expected[2], 1e-10);
}

TEST(VectorSpaceOperators, WedgeLowestOrderGradientEdgePotentialsCommute) {
    const auto ref = elements::ReferenceElement::create(ElementType::Wedge6);
    H1Space h1(ElementType::Wedge6, 1);

    std::vector<Real> coeffs = {Real(0.2), Real(-0.1), Real(0.35), Real(0.5), Real(-0.05), Real(0.15)};
    const auto grad_field = [&h1, &coeffs](const FunctionSpace::Value& xi) {
        FunctionSpace::Value value{};
        const auto grad = h1.evaluate_gradient(xi, coeffs);
        value[0] = grad[0];
        value[1] = grad[1];
        value[2] = grad[2];
        return value;
    };

    for (std::size_t e = 0; e < ref.num_edges(); ++e) {
        const auto& edge_nodes = ref.edge_nodes(e);
        const Real expected =
            coeffs[static_cast<std::size_t>(edge_nodes[1])] -
            coeffs[static_cast<std::size_t>(edge_nodes[0])];
        const double actual =
            integrate_edge_tangential_line_integral(grad_field, ElementType::Wedge6, static_cast<int>(e));
        EXPECT_NEAR(actual, expected, 1e-10);
    }
}

TEST(VectorSpaceOperators, PyramidLowestOrderGradientEdgePotentialsCommute) {
    const auto ref = elements::ReferenceElement::create(ElementType::Pyramid5);
    H1Space h1(ElementType::Pyramid5, 1);

    std::vector<Real> coeffs = {Real(-0.25), Real(0.15), Real(0.4), Real(0.05), Real(0.3)};
    const auto grad_field = [&h1, &coeffs](const FunctionSpace::Value& xi) {
        FunctionSpace::Value value{};
        const auto grad = h1.evaluate_gradient(xi, coeffs);
        value[0] = grad[0];
        value[1] = grad[1];
        value[2] = grad[2];
        return value;
    };

    for (std::size_t e = 0; e < ref.num_edges(); ++e) {
        const auto& edge_nodes = ref.edge_nodes(e);
        const Real expected =
            coeffs[static_cast<std::size_t>(edge_nodes[1])] -
            coeffs[static_cast<std::size_t>(edge_nodes[0])];
        const double actual =
            integrate_edge_tangential_line_integral(grad_field, ElementType::Pyramid5, static_cast<int>(e));
        EXPECT_NEAR(actual, expected, 1e-10);
    }
}
