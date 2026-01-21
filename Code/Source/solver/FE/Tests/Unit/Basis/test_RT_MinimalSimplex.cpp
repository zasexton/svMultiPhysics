/**
 * @file test_RT_MinimalSimplex.cpp
 * @brief Minimal RT0 spaces on triangles and tetrahedra: edge/face flux DOFs
 */

#include <gtest/gtest.h>
#include "FE/Basis/NodeOrderingConventions.h"
#include "FE/Basis/VectorBasis.h"
#include "FE/Elements/ReferenceElement.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/QuadratureFactory.h"

using namespace svmp::FE;
using namespace svmp::FE::basis;
using namespace svmp::FE::quadrature;

namespace {

static double integrate_triangle_edge_flux(const RaviartThomasBasis& rt,
                                          std::size_t edge_id,
                                          std::size_t func_id,
                                          int quad_order = 6) {
    using svmp::FE::elements::ReferenceElement;
    using svmp::FE::math::Vector;

    const ReferenceElement ref = ReferenceElement::create(ElementType::Triangle3);
    const auto& en = ref.edge_nodes(edge_id);
    EXPECT_EQ(en.size(), 2u);
    const Vector<Real, 3> p0 =
        NodeOrdering::get_node_coords(ElementType::Triangle3, static_cast<std::size_t>(en[0]));
    const Vector<Real, 3> p1 =
        NodeOrdering::get_node_coords(ElementType::Triangle3, static_cast<std::size_t>(en[1]));

    const Vector<Real, 3> tvec = p1 - p0;
    const Real len = tvec.norm();
    EXPECT_GT(len, Real(0));
    if (len <= Real(0)) {
        return 0.0;
    }
    const Vector<Real, 3> t = tvec / len;
    const Vector<Real, 3> nrm{t[1], -t[0], Real(0)};
    const Real J = len * Real(0.5);

    GaussQuadrature1D quad(quad_order);
    double flux = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real s = quad.point(q)[0];
        const Real tpar = (s + Real(1)) * Real(0.5);
        const Vector<Real, 3> xi = p0 * (Real(1) - tpar) + p1 * tpar;

        std::vector<Vector<Real, 3>> vals;
        rt.evaluate_vector_values(xi, vals);
        const auto& v = vals[func_id];

        flux += static_cast<double>(quad.weight(q) * (J * v.dot(nrm)));
    }
    return flux;
}

static double integrate_tetra_face_flux(const RaviartThomasBasis& rt,
                                       std::size_t face_id,
                                       std::size_t func_id,
                                       int quad_order = 4) {
    using svmp::FE::elements::ReferenceElement;
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::QuadratureFactory;

    const ReferenceElement ref = ReferenceElement::create(ElementType::Tetra4);
    const auto& fn = ref.face_nodes(face_id);
    EXPECT_EQ(fn.size(), 3u);

    const Vector<Real, 3> v0 =
        NodeOrdering::get_node_coords(ElementType::Tetra4, static_cast<std::size_t>(fn[0]));
    const Vector<Real, 3> v1 =
        NodeOrdering::get_node_coords(ElementType::Tetra4, static_cast<std::size_t>(fn[1]));
    const Vector<Real, 3> v2 =
        NodeOrdering::get_node_coords(ElementType::Tetra4, static_cast<std::size_t>(fn[2]));
    const Vector<Real, 3> e01 = v1 - v0;
    const Vector<Real, 3> e02 = v2 - v0;
    const Vector<Real, 3> cr = e01.cross(e02);
    const Real scale = cr.norm();
    EXPECT_GT(scale, Real(0));
    if (scale <= Real(0)) {
        return 0.0;
    }
    const Vector<Real, 3> nrm = cr / scale;

    const auto quad = QuadratureFactory::create(
        ElementType::Triangle3, quad_order, QuadratureType::GaussLegendre, /*use_cache=*/false);

    double flux = 0.0;
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const auto uv = quad->point(q);
        const Real u = uv[0];
        const Real v = uv[1];
        const Vector<Real, 3> xi = v0 + e01 * u + e02 * v;

        std::vector<Vector<Real, 3>> vals;
        rt.evaluate_vector_values(xi, vals);
        const auto& w = vals[func_id];

        flux += static_cast<double>(quad->weight(q) * (scale * w.dot(nrm)));
    }
    return flux;
}

} // namespace

TEST(RaviartThomasBasis, TriangleMinimalEdgeFluxDofs) {
    RaviartThomasBasis rt(ElementType::Triangle3, 0);
    using svmp::FE::elements::ReferenceElement;

    const ReferenceElement ref = ReferenceElement::create(ElementType::Triangle3);
    ASSERT_EQ(ref.num_edges(), 3u);
    ASSERT_EQ(rt.size(), ref.num_edges());

    for (std::size_t edge = 0; edge < ref.num_edges(); ++edge) {
        for (std::size_t func = 0; func < rt.size(); ++func) {
            const double flux = integrate_triangle_edge_flux(rt, edge, func);
            if (edge == func) {
                EXPECT_NEAR(flux, 1.0, 1e-12);
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-12);
            }
        }
    }
}

TEST(RaviartThomasBasis, TetraMinimalFaceFluxDofs) {
    RaviartThomasBasis rt(ElementType::Tetra4, 0);
    using svmp::FE::elements::ReferenceElement;

    const ReferenceElement ref = ReferenceElement::create(ElementType::Tetra4);
    ASSERT_EQ(ref.num_faces(), 4u);
    ASSERT_EQ(rt.size(), ref.num_faces());

    for (std::size_t face = 0; face < ref.num_faces(); ++face) {
        for (std::size_t func = 0; func < rt.size(); ++func) {
            const double flux = integrate_tetra_face_flux(rt, face, func);
            if (face == func) {
                EXPECT_NEAR(flux, 1.0, 1e-10);
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-10);
            }
        }
    }
}
