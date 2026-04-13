/**
 * @file test_HigherOrderWedgePyramid.cpp
 * @brief Tests for higher-order RT and Nedelec on Wedge and Pyramid
 *
 * This file exercises wedge/pyramid H(div)/H(curl) construction, the actual
 * face and edge functionals used by the Basis module, and basic consistency
 * identities on the reference elements.
 */

#include <gtest/gtest.h>
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/NodeOrderingConventions.h"
#include "FE/Basis/VectorBasis.h"
#include "FE/Core/FEException.h"
#include "FE/Elements/ReferenceElement.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Quadrature/TriangleQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include <array>
#include <cmath>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::basis;

// =============================================================================
// Local helpers for moment-based DOF integration
// =============================================================================

/// Integrate N_i . n_hat over a face of the reference wedge.
/// For RT(k), the DOF functional for face f is:
///   integral_f (N_i . n) * q dA = delta_{i,dof}
/// where q is a polynomial test function on the face.
/// With q=1 (mode 0), this gives the zeroth-order face moment.
static double integrate_wedge_face_flux(const RaviartThomasBasis& basis,
                                        int face_id,
                                        int func_id,
                                        int tri_order = 6,
                                        int quad_order = 6) {
    using namespace svmp::FE::quadrature;
    using svmp::FE::math::Vector;

    double flux = 0.0;

    // Faces 0,1 are triangular (bottom z=-1, top z=+1)
    if (face_id == 0 || face_id == 1) {
        TriangleQuadrature tri(tri_order);
        const double z = (face_id == 0) ? -1.0 : 1.0;
        const double nz = (face_id == 0) ? -1.0 : 1.0;
        for (std::size_t q = 0; q < tri.num_points(); ++q) {
            const auto& pt = tri.point(q);
            Vector<Real,3> xi{pt[0], pt[1], static_cast<Real>(z)};
            std::vector<Vector<Real,3>> vals;
            basis.evaluate_vector_values(xi, vals);
            const auto& v = vals[static_cast<std::size_t>(func_id)];
            const double integrand = static_cast<double>(v[2]) * nz;
            flux += static_cast<double>(tri.weight(q)) * integrand;
        }
        return flux;
    }

    // Faces 2,3,4 are quadrilateral, parameterized over [-1,1]^2
    QuadrilateralQuadrature quad(quad_order, quad_order);
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const auto& pt = quad.point(q);
        const double xi_q = pt[0];
        const double eta = pt[1];

        Vector<Real,3> x_ref{};
        Vector<Real,3> cross{};

        if (face_id == 2) {        // y=0 face: nodes 0,1,4,3
            const double t = 0.5 * (xi_q + 1.0);
            const double z = eta;
            x_ref = Vector<Real,3>{static_cast<Real>(t), Real(0), static_cast<Real>(z)};
            cross = Vector<Real,3>{Real(0), Real(-0.5), Real(0)};
        } else if (face_id == 3) { // x=0 face: nodes 1,2,5,4
            const double t = 0.5 * (xi_q + 1.0);
            const double z = eta;
            x_ref = Vector<Real,3>{Real(0), static_cast<Real>(t), static_cast<Real>(z)};
            cross = Vector<Real,3>{Real(0.5), Real(0), Real(0)};
        } else {                   // face_id == 4, x+y=1 face: nodes 2,0,3,5
            const double t = 0.5 * (xi_q + 1.0);
            const double z = eta;
            const double x = t;
            const double y = 1.0 - t;
            x_ref = Vector<Real,3>{static_cast<Real>(x), static_cast<Real>(y), static_cast<Real>(z)};
            cross = Vector<Real,3>{Real(-0.5), Real(-0.5), Real(0)};
        }

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(x_ref, vals);
        const auto& v = vals[static_cast<std::size_t>(func_id)];
        const double integrand =
            static_cast<double>(v[0]) * cross[0] +
            static_cast<double>(v[1]) * cross[1] +
            static_cast<double>(v[2]) * cross[2];
        flux += static_cast<double>(quad.weight(q)) * integrand;
    }

    return flux;
}

/// Integrate N_i . n_hat over a face of the reference pyramid.
static double integrate_pyramid_face_flux(const RaviartThomasBasis& basis,
                                          int face_id,
                                          int func_id,
                                          int tri_order = 6,
                                          int quad_order = 6) {
    using namespace svmp::FE::quadrature;
    using svmp::FE::math::Vector;

    double flux = 0.0;

    const Vector<Real,3> v0{Real(-1), Real(-1), Real(0)};
    const Vector<Real,3> v1{Real(1),  Real(-1), Real(0)};
    const Vector<Real,3> v2{Real(1),  Real(1),  Real(0)};
    const Vector<Real,3> v3{Real(-1), Real(1),  Real(0)};
    const Vector<Real,3> v4{Real(0),  Real(0),  Real(1)};

    // Face 0: quad base {0,1,2,3}, outward normal = -z
    if (face_id == 0) {
        QuadrilateralQuadrature quad(quad_order, quad_order);
        for (std::size_t q = 0; q < quad.num_points(); ++q) {
            const auto& pt = quad.point(q);
            Vector<Real,3> xi{pt[0], pt[1], Real(0)};
            std::vector<Vector<Real,3>> vals;
            basis.evaluate_vector_values(xi, vals);
            const auto& v = vals[static_cast<std::size_t>(func_id)];
            const double integrand = -static_cast<double>(v[2]); // outward -z
            flux += static_cast<double>(quad.weight(q)) * integrand;
        }
        return flux;
    }

    // Faces 1-4: triangular faces to apex
    TriangleQuadrature tri(tri_order);
    const Vector<Real,3>* a = nullptr;
    const Vector<Real,3>* b = nullptr;
    const Vector<Real,3>* c = nullptr;
    switch (face_id) {
        case 1: a = &v0; b = &v1; c = &v4; break;
        case 2: a = &v1; b = &v2; c = &v4; break;
        case 3: a = &v2; b = &v3; c = &v4; break;
        default: a = &v3; b = &v0; c = &v4; break;
    }
    Vector<Real,3> e1 = *b - *a;
    Vector<Real,3> e2 = *c - *a;
    Vector<Real,3> cross = e1.cross(e2);

    for (std::size_t q = 0; q < tri.num_points(); ++q) {
        const auto& pt = tri.point(q);
        const double r = pt[0];
        const double s = pt[1];
        const double l0 = 1.0 - r - s;
        const double l1 = r;
        const double l2 = s;
        Vector<Real,3> xi =
            (*a) * static_cast<Real>(l0) +
            (*b) * static_cast<Real>(l1) +
            (*c) * static_cast<Real>(l2);

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& v = vals[static_cast<std::size_t>(func_id)];
        const double integrand =
            static_cast<double>(v[0]) * cross[0] +
            static_cast<double>(v[1]) * cross[1] +
            static_cast<double>(v[2]) * cross[2];
        flux += static_cast<double>(tri.weight(q)) * integrand;
    }

    return flux;
}

static svmp::FE::math::Vector<Real,3> reference_interior_point(ElementType type) {
    switch (type) {
        case ElementType::Wedge6:
            return {Real(1) / Real(3), Real(1) / Real(3), Real(0)};
        case ElementType::Pyramid5:
            return {Real(0), Real(0), Real(0.25)};
        default:
            return {Real(0), Real(0), Real(0)};
    }
}

static svmp::FE::math::Vector<Real,3> bilinear_face_point(
    const std::array<svmp::FE::math::Vector<Real,3>, 4>& fv,
    Real u,
    Real v) {
    const Real N0 = Real(0.25) * (Real(1) - u) * (Real(1) - v);
    const Real N1 = Real(0.25) * (Real(1) + u) * (Real(1) - v);
    const Real N2 = Real(0.25) * (Real(1) + u) * (Real(1) + v);
    const Real N3 = Real(0.25) * (Real(1) - u) * (Real(1) + v);
    return fv[0] * N0 + fv[1] * N1 + fv[2] * N2 + fv[3] * N3;
}

static svmp::FE::math::Vector<Real,3> bilinear_face_du(
    const std::array<svmp::FE::math::Vector<Real,3>, 4>& fv,
    Real u,
    Real v) {
    const Real dN0 = Real(-0.25) * (Real(1) - v);
    const Real dN1 = Real(0.25) * (Real(1) - v);
    const Real dN2 = Real(0.25) * (Real(1) + v);
    const Real dN3 = Real(-0.25) * (Real(1) + v);
    return fv[0] * dN0 + fv[1] * dN1 + fv[2] * dN2 + fv[3] * dN3;
}

static svmp::FE::math::Vector<Real,3> bilinear_face_dv(
    const std::array<svmp::FE::math::Vector<Real,3>, 4>& fv,
    Real u,
    Real v) {
    const Real dN0 = Real(-0.25) * (Real(1) - u);
    const Real dN1 = Real(-0.25) * (Real(1) + u);
    const Real dN2 = Real(0.25) * (Real(1) + u);
    const Real dN3 = Real(0.25) * (Real(1) - u);
    return fv[0] * dN0 + fv[1] * dN1 + fv[2] * dN2 + fv[3] * dN3;
}

static std::vector<svmp::FE::math::Vector<Real,3>> oriented_face_vertices(
    ElementType elem_type,
    int face_id) {
    using svmp::FE::math::Vector;

    const auto ref = elements::ReferenceElement::create(elem_type);
    const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(face_id));
    std::vector<Vector<Real,3>> vertices;
    vertices.reserve(face_nodes.size());
    for (const auto node : face_nodes) {
        vertices.push_back(
            NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(node)));
    }

    Vector<Real,3> center{};
    for (const auto& v : vertices) {
        center += v;
    }
    center /= static_cast<Real>(vertices.size());

    const Vector<Real,3> interior = reference_interior_point(elem_type);
    Vector<Real,3> normal{};
    if (vertices.size() == 3u) {
        const Vector<Real,3> e01 = vertices[1] - vertices[0];
        const Vector<Real,3> e02 = vertices[2] - vertices[0];
        normal = e01.cross(e02);
    } else {
        const Vector<Real,3> e01 = vertices[1] - vertices[0];
        const Vector<Real,3> e03 = vertices[3] - vertices[0];
        normal = e01.cross(e03);
    }

    if (normal.dot(interior - center) > Real(0)) {
        std::reverse(vertices.begin() + 1, vertices.end());
    }

    return vertices;
}

static double integrate_rt_face_functional(const RaviartThomasBasis& basis,
                                           ElementType elem_type,
                                           int face_id,
                                           int moment_index,
                                           int func_id,
                                           int quad_order = 8) {
    using svmp::FE::math::Vector;
    using namespace svmp::FE::quadrature;

    const auto ref = elements::ReferenceElement::create(elem_type);
    const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(face_id));

    double value = 0.0;
    if (face_nodes.size() == 3u) {
        const LagrangeBasis face_basis(ElementType::Triangle3, basis.order());
        TriangleQuadrature tri(quad_order);

        const Vector<Real,3> v0 =
            NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[0]));
        const Vector<Real,3> v1 =
            NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[1]));
        const Vector<Real,3> v2 =
            NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[2]));
        const Vector<Real,3> e01 = v1 - v0;
        const Vector<Real,3> e02 = v2 - v0;
        const Vector<Real,3> cross = e01.cross(e02);

        for (std::size_t q = 0; q < tri.num_points(); ++q) {
            const auto& pt = tri.point(q);
            const Real u = pt[0];
            const Real v = pt[1];
            const Vector<Real,3> xi = v0 + e01 * u + e02 * v;

            std::vector<Real> face_vals;
            face_basis.evaluate_values(Vector<Real,3>{u, v, Real(0)}, face_vals);

            std::vector<Vector<Real,3>> values;
            basis.evaluate_vector_values(xi, values);
            const auto& basis_value = values[static_cast<std::size_t>(func_id)];

            const double flux =
                static_cast<double>(basis_value[0] * cross[0] +
                                    basis_value[1] * cross[1] +
                                    basis_value[2] * cross[2]);
            value += static_cast<double>(tri.weight(q)) *
                     static_cast<double>(face_vals[static_cast<std::size_t>(moment_index)]) *
                     flux;
        }
        return value;
    }

    const LagrangeBasis face_basis(ElementType::Quad4, basis.order());
    QuadrilateralQuadrature quad(quad_order, quad_order);
    const std::array<Vector<Real,3>, 4> fv{
        NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[0])),
        NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[1])),
        NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[2])),
        NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[3]))
    };

    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const auto& pt = quad.point(q);
        const Real u = pt[0];
        const Real v = pt[1];
        const Vector<Real,3> xi = bilinear_face_point(fv, u, v);
        const Vector<Real,3> du = bilinear_face_du(fv, u, v);
        const Vector<Real,3> dv = bilinear_face_dv(fv, u, v);
        const Vector<Real,3> cross = du.cross(dv);

        std::vector<Real> face_vals;
        face_basis.evaluate_values(Vector<Real,3>{u, v, Real(0)}, face_vals);

        std::vector<Vector<Real,3>> values;
        basis.evaluate_vector_values(xi, values);
        const auto& basis_value = values[static_cast<std::size_t>(func_id)];

        const double flux =
            static_cast<double>(basis_value[0] * cross[0] +
                                basis_value[1] * cross[1] +
                                basis_value[2] * cross[2]);
        value += static_cast<double>(quad.weight(q)) *
                 static_cast<double>(face_vals[static_cast<std::size_t>(moment_index)]) *
                 flux;
    }

    return value;
}

static double integrate_rt_face_flux_outward(const RaviartThomasBasis& basis,
                                             ElementType elem_type,
                                             int face_id,
                                             int func_id,
                                             int quad_order = 8) {
    using svmp::FE::math::Vector;
    using namespace svmp::FE::quadrature;

    const auto ref = elements::ReferenceElement::create(elem_type);
    const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(face_id));
    const Vector<Real,3> interior = reference_interior_point(elem_type);

    double flux = 0.0;
    if (face_nodes.size() == 3u) {
        TriangleQuadrature tri(quad_order);
        const Vector<Real,3> v0 =
            NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[0]));
        const Vector<Real,3> v1 =
            NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[1]));
        const Vector<Real,3> v2 =
            NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[2]));
        const Vector<Real,3> e01 = v1 - v0;
        const Vector<Real,3> e02 = v2 - v0;
        Vector<Real,3> cross = e01.cross(e02);
        const Vector<Real,3> face_mid = (v0 + v1 + v2) / Real(3);
        if (cross.dot(interior - face_mid) > Real(0)) {
            cross = cross * Real(-1);
        }

        for (std::size_t q = 0; q < tri.num_points(); ++q) {
            const auto& pt = tri.point(q);
            const Real u = pt[0];
            const Real v = pt[1];
            const Vector<Real,3> xi = v0 + e01 * u + e02 * v;

            std::vector<Vector<Real,3>> values;
            basis.evaluate_vector_values(xi, values);
            const auto& basis_value = values[static_cast<std::size_t>(func_id)];

            const double integrand =
                static_cast<double>(basis_value[0] * cross[0] +
                                    basis_value[1] * cross[1] +
                                    basis_value[2] * cross[2]);
            flux += static_cast<double>(tri.weight(q)) * integrand;
        }
        return flux;
    }

    QuadrilateralQuadrature quad(quad_order, quad_order);
    const std::array<Vector<Real,3>, 4> fv{
        NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[0])),
        NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[1])),
        NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[2])),
        NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[3]))
    };

    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const auto& pt = quad.point(q);
        const Real u = pt[0];
        const Real v = pt[1];
        const Vector<Real,3> xi = bilinear_face_point(fv, u, v);
        const Vector<Real,3> du = bilinear_face_du(fv, u, v);
        const Vector<Real,3> dv = bilinear_face_dv(fv, u, v);
        Vector<Real,3> cross = du.cross(dv);
        if (cross.dot(interior - xi) > Real(0)) {
            cross = cross * Real(-1);
        }

        std::vector<Vector<Real,3>> values;
        basis.evaluate_vector_values(xi, values);
        const auto& basis_value = values[static_cast<std::size_t>(func_id)];

        const double integrand =
            static_cast<double>(basis_value[0] * cross[0] +
                                basis_value[1] * cross[1] +
                                basis_value[2] * cross[2]);
        flux += static_cast<double>(quad.weight(q)) * integrand;
    }

    return flux;
}

static double integrate_rt_volume_divergence(const RaviartThomasBasis& basis,
                                             ElementType elem_type,
                                             int func_id,
                                             int quad_order = 8) {
    auto quad = quadrature::QuadratureFactory::create(
        elem_type, quad_order, QuadratureType::GaussLegendre, /*use_cache=*/false);

    double integral = 0.0;
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const auto& pt = quad->point(q);
        std::vector<Real> div;
        basis.evaluate_divergence(pt, div);
        integral += static_cast<double>(quad->weight(q)) *
                    static_cast<double>(div[static_cast<std::size_t>(func_id)]);
    }
    return integral;
}

static double integrate_segment_tangent(const NedelecBasis& basis,
                                        const math::Vector<Real, 3>& a,
                                        const math::Vector<Real, 3>& b,
                                        int func_id,
                                        int quad_order = 8) {
    using svmp::FE::quadrature::GaussQuadrature1D;

    const math::Vector<Real, 3> tangent = b - a;
    GaussQuadrature1D quad(quad_order);

    double value = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real tq = quad.point(q)[0];
        const Real s = (tq + Real(1)) * Real(0.5);
        const auto xi = a * (Real(1) - s) + b * s;

        std::vector<math::Vector<Real, 3>> values;
        basis.evaluate_vector_values(xi, values);
        const auto& basis_value = values[static_cast<std::size_t>(func_id)];

        value += 0.5 * static_cast<double>(quad.weight(q)) *
                 static_cast<double>(basis_value.dot(tangent));
    }
    return value;
}

static double integrate_nd_face_boundary_circulation(const NedelecBasis& basis,
                                                     ElementType elem_type,
                                                     int face_id,
                                                     int func_id,
                                                     int quad_order = 8) {
    const auto vertices = oriented_face_vertices(elem_type, face_id);

    double circulation = 0.0;
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const auto& a = vertices[i];
        const auto& b = vertices[(i + 1) % vertices.size()];
        circulation += integrate_segment_tangent(basis, a, b, func_id, quad_order);
    }

    return circulation;
}

static double integrate_nd_face_curl_flux(const NedelecBasis& basis,
                                          ElementType elem_type,
                                          int face_id,
                                          int func_id,
                                          int quad_order = 8) {
    using svmp::FE::math::Vector;
    using namespace svmp::FE::quadrature;

    const auto vertices = oriented_face_vertices(elem_type, face_id);

    double flux = 0.0;
    if (vertices.size() == 3u) {
        TriangleQuadrature tri(quad_order);
        const Vector<Real,3> e01 = vertices[1] - vertices[0];
        const Vector<Real,3> e02 = vertices[2] - vertices[0];
        const Vector<Real,3> cross = e01.cross(e02);

        for (std::size_t q = 0; q < tri.num_points(); ++q) {
            const auto& pt = tri.point(q);
            const Real u = pt[0];
            const Real v = pt[1];
            const Vector<Real,3> xi = vertices[0] + e01 * u + e02 * v;

            std::vector<Vector<Real,3>> curl;
            basis.evaluate_curl(xi, curl);
            const auto& basis_curl = curl[static_cast<std::size_t>(func_id)];

            flux += static_cast<double>(tri.weight(q)) *
                    static_cast<double>(basis_curl.dot(cross));
        }
        return flux;
    }

    const std::array<Vector<Real,3>, 4> fv{
        vertices[0], vertices[1], vertices[2], vertices[3]
    };
    QuadrilateralQuadrature quad(quad_order, quad_order);
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const auto& pt = quad.point(q);
        const Real u = pt[0];
        const Real v = pt[1];
        const Vector<Real,3> xi = bilinear_face_point(fv, u, v);
        const Vector<Real,3> du = bilinear_face_du(fv, u, v);
        const Vector<Real,3> dv = bilinear_face_dv(fv, u, v);
        const Vector<Real,3> cross = du.cross(dv);

        std::vector<Vector<Real,3>> curl;
        basis.evaluate_curl(xi, curl);
        const auto& basis_curl = curl[static_cast<std::size_t>(func_id)];

        flux += static_cast<double>(quad.weight(q)) *
                static_cast<double>(basis_curl.dot(cross));
    }

    return flux;
}

static void expect_curl_is_divergence_free(const NedelecBasis& basis,
                                           const math::Vector<Real, 3>& xc,
                                           Real h,
                                           Real tol) {
    for (std::size_t i = 0; i < basis.size(); ++i) {
        auto eval_curl = [&](const math::Vector<Real, 3>& p) {
            std::vector<math::Vector<Real, 3>> c;
            basis.evaluate_curl(p, c);
            return c[i];
        };

        auto derivative_4th = [&](int axis, int component) {
            math::Vector<Real, 3> pmm = xc;
            math::Vector<Real, 3> pm = xc;
            math::Vector<Real, 3> pp = xc;
            math::Vector<Real, 3> ppp = xc;
            pmm[static_cast<std::size_t>(axis)] -= Real(2) * h;
            pm[static_cast<std::size_t>(axis)] -= h;
            pp[static_cast<std::size_t>(axis)] += h;
            ppp[static_cast<std::size_t>(axis)] += Real(2) * h;

            const auto cmm = eval_curl(pmm);
            const auto cm = eval_curl(pm);
            const auto cp = eval_curl(pp);
            const auto cpp = eval_curl(ppp);

            return (-static_cast<double>(cpp[static_cast<std::size_t>(component)]) +
                    8.0 * static_cast<double>(cp[static_cast<std::size_t>(component)]) -
                    8.0 * static_cast<double>(cm[static_cast<std::size_t>(component)]) +
                    static_cast<double>(cmm[static_cast<std::size_t>(component)])) /
                   (12.0 * static_cast<double>(h));
        };

        const double dCx_dx = derivative_4th(/*axis=*/0, /*component=*/0);
        const double dCy_dy = derivative_4th(/*axis=*/1, /*component=*/1);
        const double dCz_dz = derivative_4th(/*axis=*/2, /*component=*/2);

        const double div_curl = dCx_dx + dCy_dy + dCz_dz;
        EXPECT_NEAR(div_curl, 0.0, static_cast<double>(tol))
            << "Basis function " << i << " curl is not divergence-free";
    }
}

/// Integrate N_i . t along an edge of the reference wedge (for Nedelec DOFs).
/// The edge tangent t = (b - a), not normalized, matching the edge DOF definition.
static double integrate_wedge_edge_tangent(const NedelecBasis& basis,
                                           int edge_id,
                                           int func_id,
                                           int quad_order = 6) {
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::GaussQuadrature1D;

    const Vector<Real,3> v0{Real(0), Real(0), Real(-1)};
    const Vector<Real,3> v1{Real(1), Real(0), Real(-1)};
    const Vector<Real,3> v2{Real(0), Real(1), Real(-1)};
    const Vector<Real,3> v3{Real(0), Real(0), Real(1)};
    const Vector<Real,3> v4{Real(1), Real(0), Real(1)};
    const Vector<Real,3> v5{Real(0), Real(1), Real(1)};

    struct Edge { Vector<Real,3> a, b; };
    const Edge edges[9] = {
        {v0, v1}, {v1, v2}, {v2, v0}, // bottom tri
        {v3, v4}, {v4, v5}, {v5, v3}, // top tri
        {v0, v3}, {v1, v4}, {v2, v5}  // vertical
    };

    const Edge& e = edges[static_cast<std::size_t>(edge_id)];
    const Vector<Real,3> t_vec = e.b - e.a;

    GaussQuadrature1D quad(quad_order);
    double dof = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real tq = quad.point(q)[0]; // in [-1,1]
        const Real s = (tq + Real(1)) * Real(0.5); // map to [0,1]
        Vector<Real,3> xi = e.a * (Real(1) - s) + e.b * s;

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];

        const double integrand = static_cast<double>(
            w[0]*t_vec[0] + w[1]*t_vec[1] + w[2]*t_vec[2]);
        dof += 0.5 * quad.weight(q) * integrand;
    }
    return dof;
}

/// Integrate N_i . t along an edge of the reference pyramid (for Nedelec DOFs).
static double integrate_pyramid_edge_tangent(const NedelecBasis& basis,
                                             int edge_id,
                                             int func_id,
                                             int quad_order = 6) {
    using svmp::FE::math::Vector;
    using svmp::FE::quadrature::GaussQuadrature1D;

    const Vector<Real,3> v0{Real(-1), Real(-1), Real(0)};
    const Vector<Real,3> v1{Real(1),  Real(-1), Real(0)};
    const Vector<Real,3> v2{Real(1),  Real(1),  Real(0)};
    const Vector<Real,3> v3{Real(-1), Real(1),  Real(0)};
    const Vector<Real,3> v4{Real(0),  Real(0),  Real(1)};

    struct Edge { Vector<Real,3> a, b; };
    const Edge edges[8] = {
        {v0, v1}, {v1, v2}, {v2, v3}, {v3, v0}, // base
        {v0, v4}, {v1, v4}, {v2, v4}, {v3, v4}  // vertical
    };

    const Edge& e = edges[static_cast<std::size_t>(edge_id)];
    const Vector<Real,3> t_vec = e.b - e.a;

    GaussQuadrature1D quad(quad_order);
    double dof = 0.0;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const Real tq = quad.point(q)[0];
        const Real s = (tq + Real(1)) * Real(0.5);
        Vector<Real,3> xi = e.a * (Real(1) - s) + e.b * s;

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];

        const double integrand = static_cast<double>(
            w[0]*t_vec[0] + w[1]*t_vec[1] + w[2]*t_vec[2]);
        dof += 0.5 * quad.weight(q) * integrand;
    }
    return dof;
}

static double integrate_nd_edge_tangential_moment(const NedelecBasis& basis,
                                                  ElementType elem_type,
                                                  int edge_id,
                                                  int mode_id,
                                                  int func_id,
                                                  int quad_order = 8) {
    using svmp::FE::basis::LagrangeBasis;
    using svmp::FE::math::Vector;

    const auto ref = elements::ReferenceElement::create(elem_type);
    const auto& edge_nodes = ref.edge_nodes(static_cast<std::size_t>(edge_id));
    EXPECT_EQ(edge_nodes.size(), 2u);
    if (edge_nodes.size() != 2u) {
        return 0.0;
    }

    const Vector<Real,3> a =
        NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(edge_nodes[0]));
    const Vector<Real,3> b =
        NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(edge_nodes[1]));
    const Vector<Real,3> tvec = b - a;
    const Real len = tvec.norm();
    EXPECT_GT(len, Real(0));
    if (len <= Real(0)) {
        return 0.0;
    }

    const Vector<Real,3> t = tvec / len;
    const Real J = len * Real(0.5);
    LagrangeBasis line(ElementType::Line2, basis.order());
    auto quad = quadrature::QuadratureFactory::create(
        ElementType::Line2,
        std::max(quad_order, 2 * basis.order() + 4),
        QuadratureType::GaussLegendre,
        /*use_cache=*/false);

    double moment = 0.0;
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const Real s = quad->point(q)[0];
        std::vector<Real> lvals;
        line.evaluate_values(Vector<Real,3>{s, Real(0), Real(0)}, lvals);
        const Real phi = lvals[static_cast<std::size_t>(mode_id)];

        const Real tpar = (s + Real(1)) * Real(0.5);
        const Vector<Real,3> xi = a * (Real(1) - tpar) + b * tpar;

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];

        moment += static_cast<double>(quad->weight(q) * (J * phi * w.dot(t)));
    }
    return moment;
}

static double integrate_nd_face_tangential_moment(const NedelecBasis& basis,
                                                  ElementType elem_type,
                                                  int face_id,
                                                  int moment_index,
                                                  int func_id,
                                                  int quad_order = 8) {
    using svmp::FE::basis::LagrangeBasis;
    using svmp::FE::math::Vector;

    const int k = basis.order();
    const auto ref = elements::ReferenceElement::create(elem_type);
    const auto& face_nodes = ref.face_nodes(static_cast<std::size_t>(face_id));

    if (face_nodes.size() == 3u) {
        const Vector<Real,3> v0 =
            NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[0]));
        const Vector<Real,3> v1 =
            NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[1]));
        const Vector<Real,3> v2 =
            NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[2]));
        const Vector<Real,3> tu = v1 - v0;
        const Vector<Real,3> tv = v2 - v0;
        const Real scale = tu.cross(tv).norm();

        const LagrangeBasis tri_basis(ElementType::Triangle3, k - 1);
        const std::size_t n_face = tri_basis.size();
        const int dir_id = (moment_index < static_cast<int>(n_face)) ? 0 : 1;
        const int mode_id = (dir_id == 0) ? moment_index : moment_index - static_cast<int>(n_face);
        auto quad = quadrature::QuadratureFactory::create(
            ElementType::Triangle3,
            std::max(quad_order, 2 * k + 4),
            QuadratureType::GaussLegendre,
            /*use_cache=*/false);

        double moment = 0.0;
        for (std::size_t q = 0; q < quad->num_points(); ++q) {
            const auto pt = quad->point(q);
            const Real u = pt[0];
            const Real v = pt[1];

            std::vector<Real> bvals;
            tri_basis.evaluate_values(Vector<Real,3>{u, v, Real(0)}, bvals);
            const Real phi = bvals[static_cast<std::size_t>(mode_id)];

            const Vector<Real,3> xi = v0 + tu * u + tv * v;
            std::vector<Vector<Real,3>> vals;
            basis.evaluate_vector_values(xi, vals);
            const auto& w = vals[static_cast<std::size_t>(func_id)];
            const Vector<Real,3>& tangent = (dir_id == 0) ? tu : tv;

            moment += static_cast<double>(quad->weight(q) * (scale * phi * w.dot(tangent)));
        }
        return moment;
    }

    const std::array<Vector<Real,3>, 4> fv{
        NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[0])),
        NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[1])),
        NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[2])),
        NodeOrdering::get_node_coords(elem_type, static_cast<std::size_t>(face_nodes[3]))
    };
    const Vector<Real,3> tu_raw = fv[1] - fv[0];
    const Vector<Real,3> tw_raw = fv[3] - fv[0];
    const Vector<Real,3> tu = tu_raw / tu_raw.norm();
    const Vector<Real,3> tw = tw_raw / tw_raw.norm();

    const LagrangeBasis u_low(ElementType::Line2, k - 1);
    const LagrangeBasis u_full(ElementType::Line2, k);
    const LagrangeBasis w_low(ElementType::Line2, k - 1);
    const LagrangeBasis w_full(ElementType::Line2, k);
    const int n_u = k * (k + 1);

    auto quad = quadrature::QuadratureFactory::create(
        ElementType::Quad4,
        std::max(quad_order, 2 * k + 4),
        QuadratureType::GaussLegendre,
        /*use_cache=*/false);

    double moment = 0.0;
    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const auto pt = quad->point(q);
        const Real u = pt[0];
        const Real v = pt[1];
        const Vector<Real,3> xi = bilinear_face_point(fv, u, v);
        const Vector<Real,3> du = bilinear_face_du(fv, u, v);
        const Vector<Real,3> dv = bilinear_face_dv(fv, u, v);
        const Real scale = du.cross(dv).norm();

        std::vector<Real> u_low_vals, u_full_vals, w_low_vals, w_full_vals;
        u_low.evaluate_values(Vector<Real,3>{u, Real(0), Real(0)}, u_low_vals);
        u_full.evaluate_values(Vector<Real,3>{u, Real(0), Real(0)}, u_full_vals);
        w_low.evaluate_values(Vector<Real,3>{v, Real(0), Real(0)}, w_low_vals);
        w_full.evaluate_values(Vector<Real,3>{v, Real(0), Real(0)}, w_full_vals);

        Real phi = Real(0);
        Vector<Real,3> tangent{};
        if (moment_index < n_u) {
            const int iu = moment_index % k;
            const int jw = moment_index / k;
            phi = u_low_vals[static_cast<std::size_t>(iu)] *
                  w_full_vals[static_cast<std::size_t>(jw)];
            tangent = tu;
        } else {
            const int local = moment_index - n_u;
            const int iu = local % (k + 1);
            const int jw = local / (k + 1);
            phi = u_full_vals[static_cast<std::size_t>(iu)] *
                  w_low_vals[static_cast<std::size_t>(jw)];
            tangent = tw;
        }

        std::vector<Vector<Real,3>> vals;
        basis.evaluate_vector_values(xi, vals);
        const auto& w = vals[static_cast<std::size_t>(func_id)];
        moment += static_cast<double>(quad->weight(q) * (scale * phi * w.dot(tangent)));
    }

    return moment;
}

static void expect_nd_edge_kronecker(const NedelecBasis& basis,
                                     ElementType elem_type,
                                     double tol) {
    const auto assoc = basis.dof_associations();
    ASSERT_EQ(assoc.size(), basis.size());

    const auto ref = elements::ReferenceElement::create(elem_type);
    for (std::size_t i = 0; i < basis.size(); ++i) {
        for (std::size_t e = 0; e < ref.num_edges(); ++e) {
            for (int m = 0; m <= basis.order(); ++m) {
                const double functional = integrate_nd_edge_tangential_moment(
                    basis,
                    elem_type,
                    static_cast<int>(e),
                    m,
                    static_cast<int>(i),
                    10);
                const bool is_own_dof =
                    assoc[i].entity_type == DofEntity::Edge &&
                    assoc[i].entity_id == static_cast<int>(e) &&
                    assoc[i].moment_index == m;
                EXPECT_NEAR(functional, is_own_dof ? 1.0 : 0.0, tol)
                    << "ND edge functional (" << e << ", " << m
                    << ") applied to basis function " << i
                    << " should be " << (is_own_dof ? 1.0 : 0.0);
            }
        }
    }
}

static void expect_nd_face_kronecker(const NedelecBasis& basis,
                                     ElementType elem_type,
                                     double tol) {
    const auto assoc = basis.dof_associations();
    ASSERT_EQ(assoc.size(), basis.size());

    const auto ref = elements::ReferenceElement::create(elem_type);
    for (std::size_t i = 0; i < basis.size(); ++i) {
        for (std::size_t f = 0; f < ref.num_faces(); ++f) {
            const auto& face_nodes = ref.face_nodes(f);
            const int dofs_per_face = (face_nodes.size() == 3u)
                ? basis.order() * (basis.order() + 1)
                : 2 * basis.order() * (basis.order() + 1);

            for (int m = 0; m < dofs_per_face; ++m) {
                const double functional = integrate_nd_face_tangential_moment(
                    basis,
                    elem_type,
                    static_cast<int>(f),
                    m,
                    static_cast<int>(i),
                    10);
                const bool is_own_dof =
                    assoc[i].entity_type == DofEntity::Face &&
                    assoc[i].entity_id == static_cast<int>(f) &&
                    assoc[i].moment_index == m;
                EXPECT_NEAR(functional, is_own_dof ? 1.0 : 0.0, tol)
                    << "ND face functional (" << f << ", " << m
                    << ") applied to basis function " << i
                    << " should be " << (is_own_dof ? 1.0 : 0.0);
            }
        }
    }
}

// =============================================================================
// RT(0) on Wedge and Pyramid - WORKING
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRT0Construction) {
    // RT(0) on Wedge should work with 5 DOFs (5 faces)
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Wedge6, 0);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 0);
        EXPECT_EQ(basis.size(), 5u);
    });
}

TEST(HigherOrderWedgePyramid, PyramidRT0Construction) {
    // RT(0) on Pyramid should work with 5 DOFs (5 faces)
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Pyramid5, 0);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 0);
        EXPECT_EQ(basis.size(), 5u);
    });
}

TEST(HigherOrderWedgePyramid, WedgeRT0Evaluation) {
    RaviartThomasBasis basis(ElementType::Wedge6, 0);

    // Test point inside the wedge
    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 5u);

    // All values should be finite
    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT0Evaluation) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 0);

    // Test point inside the pyramid
    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 5u);

    // All values should be finite
    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

// =============================================================================
// Nedelec(0) on Wedge and Pyramid - WORKING
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeND0Construction) {
    // Nedelec(0) on Wedge should work with 9 DOFs (9 edges)
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Wedge6, 0);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 0);
        EXPECT_EQ(basis.size(), 9u);
    });
}

TEST(HigherOrderWedgePyramid, PyramidND0Construction) {
    // Nedelec(0) on Pyramid should work with 8 DOFs (8 edges)
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Pyramid5, 0);
        EXPECT_EQ(basis.dimension(), 3);
        EXPECT_EQ(basis.order(), 0);
        EXPECT_EQ(basis.size(), 8u);
    });
}

TEST(HigherOrderWedgePyramid, WedgeND0Evaluation) {
    NedelecBasis basis(ElementType::Wedge6, 0);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 9u);

    // All values should be finite
    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidND0Evaluation) {
    NedelecBasis basis(ElementType::Pyramid5, 0);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 8u);

    // All values should be finite
    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

// =============================================================================
// RT(k>=1) on Wedge and Pyramid - NOW IMPLEMENTED via Direct DOF Construction
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRT1Implemented) {
    // RT(1) on Wedge is implemented via direct DOF-based construction
    // Expected DOFs: face = 6 + 12 = 18, interior = 6, total = 24
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Wedge6, 1);
        EXPECT_EQ(basis.size(), 24u);
    });
}

TEST(HigherOrderWedgePyramid, PyramidRT1Implemented) {
    // RT(1) on Pyramid is implemented via direct DOF-based construction
    // Expected DOFs: face = 4 + 12 = 16, interior = 3, total = 19
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Pyramid5, 1);
        EXPECT_EQ(basis.size(), 19u);
    });
}

// =============================================================================
// Nedelec(k>=1) on Wedge and Pyramid - NOW IMPLEMENTED via Direct DOF Construction
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeND1Implemented) {
    // Nedelec(1) on Wedge is implemented via direct DOF-based construction
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Wedge6, 1);
        EXPECT_EQ(basis.size(), 34u);
    });
}

TEST(HigherOrderWedgePyramid, PyramidND1Implemented) {
    // Nedelec(1) on Pyramid is fully constructed and evaluable
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Pyramid5, 1);
        EXPECT_EQ(basis.size(), 28u);
    });
}

// =============================================================================
// Dimension Formula Tests (Theoretical Values)
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRTDimensionFormulas) {
    // Document the correct dimension formulas for wedge RT(k)
    // Face DOFs: 2*(k+1)(k+2)/2 + 3*(k+1)^2 = (k+1)(k+2) + 3(k+1)^2
    // Interior DOFs: 3*k*(k+1)^2/2 for k >= 1

    // RT(0): 5 faces, 0 interior = 5
    {
        RaviartThomasBasis basis(ElementType::Wedge6, 0);
        EXPECT_EQ(basis.size(), 5u);
    }

    // RT(1): theoretical = 18 face + 6 interior = 24
    // (not yet working, but size() returns the expected value before construction fails)

    // RT(2): theoretical = 39 face + 27 interior = 66
}

TEST(HigherOrderWedgePyramid, PyramidRTDimensionFormulas) {
    // Document the correct dimension formulas for pyramid RT(k)
    // Face DOFs: (k+1)^2 + 4*(k+1)(k+2)/2 = (k+1)^2 + 2(k+1)(k+2)
    // Interior DOFs: 3*k^3 for k >= 1

    // RT(0): 5 faces, 0 interior = 5
    {
        RaviartThomasBasis basis(ElementType::Pyramid5, 0);
        EXPECT_EQ(basis.size(), 5u);
    }

    // RT(1): theoretical = 16 face + 3 interior = 19
    // RT(2): theoretical = 33 face + 24 interior = 57
}

TEST(HigherOrderWedgePyramid, WedgeNedelecDimensionFormulas) {
    // Document the correct dimension formulas for wedge Nedelec(k)
    // Edge DOFs: 9*(k+1)
    // Face DOFs: varies with k
    // Interior DOFs: varies with k

    // Nedelec(0): 9 edges = 9
    {
        NedelecBasis basis(ElementType::Wedge6, 0);
        EXPECT_EQ(basis.size(), 9u);
    }
}

TEST(HigherOrderWedgePyramid, PyramidNedelecDimensionFormulas) {
    // Document the correct dimension formulas for pyramid Nedelec(k)
    // Edge DOFs: 8*(k+1)
    // Face DOFs: varies with k
    // Interior DOFs: varies with k

    // Nedelec(0): 8 edges = 8
    {
        NedelecBasis basis(ElementType::Pyramid5, 0);
        EXPECT_EQ(basis.size(), 8u);
    }
}

// =============================================================================
// RT(0) Divergence Tests - Verify Constant Divergence
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRT0DivergenceIsConstant) {
    RaviartThomasBasis basis(ElementType::Wedge6, 0);

    // Two different interior points
    math::Vector<Real, 3> xi1{Real(0.2), Real(0.3), Real(-0.4)};
    math::Vector<Real, 3> xi2{Real(0.1), Real(0.1), Real(0.5)};

    std::vector<Real> div1, div2;
    basis.evaluate_divergence(xi1, div1);
    basis.evaluate_divergence(xi2, div2);

    ASSERT_EQ(div1.size(), 5u);
    ASSERT_EQ(div2.size(), 5u);

    // For RT(0), divergence should be constant over the element
    for (std::size_t i = 0; i < div1.size(); ++i) {
        EXPECT_NEAR(div1[i], div2[i], 1e-12);
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT0DivergenceIsConstant) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 0);

    // Two different interior points
    math::Vector<Real, 3> xi1{Real(0.1), Real(0.2), Real(0.3)};
    math::Vector<Real, 3> xi2{Real(-0.3), Real(0.1), Real(0.5)};

    std::vector<Real> div1, div2;
    basis.evaluate_divergence(xi1, div1);
    basis.evaluate_divergence(xi2, div2);

    ASSERT_EQ(div1.size(), 5u);
    ASSERT_EQ(div2.size(), 5u);

    // For RT(0), divergence should be constant over the element
    for (std::size_t i = 0; i < div1.size(); ++i) {
        EXPECT_NEAR(div1[i], div2[i], 1e-12);
    }
}

// =============================================================================
// Nedelec(0) Curl Tests - Verify curl(phi) is divergence-free
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeND0CurlIsDivergenceFree) {
    NedelecBasis basis(ElementType::Wedge6, 0);
    expect_curl_is_divergence_free(
        basis,
        math::Vector<Real, 3>{Real(0.25), Real(0.25), Real(0.0)},
        Real(1e-3),
        Real(1e-5));
}

TEST(HigherOrderWedgePyramid, PyramidND0CurlIsDivergenceFree) {
    NedelecBasis basis(ElementType::Pyramid5, 0);
    expect_curl_is_divergence_free(
        basis,
        math::Vector<Real, 3>{Real(0.0), Real(0.0), Real(0.25)},
        Real(1e-3),
        Real(1e-5));
}

// =============================================================================
// RT(1) and Nedelec(1) Evaluation Tests - NOW FULLY WORKING
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRT1Evaluation) {
    RaviartThomasBasis basis(ElementType::Wedge6, 1);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 24u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT1Evaluation) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 1);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 19u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, WedgeND1Evaluation) {
    NedelecBasis basis(ElementType::Wedge6, 1);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 34u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidND1Evaluation) {
    NedelecBasis basis(ElementType::Pyramid5, 1);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 28u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidND1CurlEvaluation) {
    NedelecBasis basis(ElementType::Pyramid5, 1);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<math::Vector<Real, 3>> curl;
    EXPECT_NO_THROW(basis.evaluate_curl(xi, curl));
    EXPECT_EQ(curl.size(), 28u);

    for (const auto& c : curl) {
        EXPECT_TRUE(std::isfinite(c[0]));
        EXPECT_TRUE(std::isfinite(c[1]));
        EXPECT_TRUE(std::isfinite(c[2]));
    }
}

// =============================================================================
// RT(2) and Nedelec(2) on Wedge and Pyramid - NOW IMPLEMENTED
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRT2Construction) {
    // RT(2) on Wedge: 66 DOFs
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Wedge6, 2);
        EXPECT_EQ(basis.size(), 66u);
    });
}

TEST(HigherOrderWedgePyramid, PyramidRT2Construction) {
    // RT(2) on Pyramid: 57 DOFs
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Pyramid5, 2);
        EXPECT_EQ(basis.size(), 57u);
    });
}

TEST(HigherOrderWedgePyramid, WedgeND2Construction) {
    // Nedelec(2) on Wedge: 84 DOFs
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Wedge6, 2);
        EXPECT_EQ(basis.size(), 84u);
    });
}

TEST(HigherOrderWedgePyramid, PyramidND2Construction) {
    // Nedelec(2) on Pyramid: 63 DOFs
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Pyramid5, 2);
        EXPECT_EQ(basis.size(), 63u);
    });
}

TEST(HigherOrderWedgePyramid, WedgeRT3Construction) {
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Wedge6, 3);
        EXPECT_EQ(basis.size(), 140u);
    });
}

TEST(HigherOrderWedgePyramid, PyramidRT3Construction) {
    EXPECT_NO_THROW({
        RaviartThomasBasis basis(ElementType::Pyramid5, 3);
        EXPECT_EQ(basis.size(), 137u);
    });
}

TEST(HigherOrderWedgePyramid, WedgeND3Construction) {
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Wedge6, 3);
        EXPECT_EQ(basis.size(), 168u);
    });
}

TEST(HigherOrderWedgePyramid, PyramidND3Construction) {
    EXPECT_NO_THROW({
        NedelecBasis basis(ElementType::Pyramid5, 3);
        EXPECT_EQ(basis.size(), 116u);
    });
}

TEST(HigherOrderWedgePyramid, WedgeRT2Evaluation) {
    RaviartThomasBasis basis(ElementType::Wedge6, 2);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 66u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT2Evaluation) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 2);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 57u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, WedgeND2Evaluation) {
    NedelecBasis basis(ElementType::Wedge6, 2);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 84u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidND2Evaluation) {
    NedelecBasis basis(ElementType::Pyramid5, 2);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 63u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, WedgeRT3Evaluation) {
    RaviartThomasBasis basis(ElementType::Wedge6, 3);

    math::Vector<Real, 3> xi{Real(0.2), Real(0.3), Real(-0.1)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 140u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT3Evaluation) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 3);

    math::Vector<Real, 3> xi{Real(0.1), Real(-0.15), Real(0.35)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 137u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, WedgeND3Evaluation) {
    NedelecBasis basis(ElementType::Wedge6, 3);

    math::Vector<Real, 3> xi{Real(0.2), Real(0.3), Real(-0.1)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 168u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidND3Evaluation) {
    NedelecBasis basis(ElementType::Pyramid5, 3);

    math::Vector<Real, 3> xi{Real(0.1), Real(-0.15), Real(0.35)};

    std::vector<math::Vector<Real, 3>> values;
    EXPECT_NO_THROW(basis.evaluate_vector_values(xi, values));
    EXPECT_EQ(values.size(), 116u);

    for (const auto& v : values) {
        EXPECT_TRUE(std::isfinite(v[0]));
        EXPECT_TRUE(std::isfinite(v[1]));
        EXPECT_TRUE(std::isfinite(v[2]));
    }
}

// =============================================================================
// RT(k) Divergence Tests for k=1,2
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRT1DivergenceFinite) {
    RaviartThomasBasis basis(ElementType::Wedge6, 1);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<Real> div;
    EXPECT_NO_THROW(basis.evaluate_divergence(xi, div));
    EXPECT_EQ(div.size(), 24u);

    for (const auto& d : div) {
        EXPECT_TRUE(std::isfinite(d));
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT1DivergenceFinite) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 1);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<Real> div;
    EXPECT_NO_THROW(basis.evaluate_divergence(xi, div));
    EXPECT_EQ(div.size(), 19u);

    for (const auto& d : div) {
        EXPECT_TRUE(std::isfinite(d));
    }
}

TEST(HigherOrderWedgePyramid, WedgeRT2DivergenceFinite) {
    RaviartThomasBasis basis(ElementType::Wedge6, 2);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<Real> div;
    EXPECT_NO_THROW(basis.evaluate_divergence(xi, div));
    EXPECT_EQ(div.size(), 66u);

    for (const auto& d : div) {
        EXPECT_TRUE(std::isfinite(d));
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT2DivergenceFinite) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 2);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<Real> div;
    EXPECT_NO_THROW(basis.evaluate_divergence(xi, div));
    EXPECT_EQ(div.size(), 57u);

    for (const auto& d : div) {
        EXPECT_TRUE(std::isfinite(d));
    }
}

TEST(HigherOrderWedgePyramid, WedgeRT3DivergenceFinite) {
    RaviartThomasBasis basis(ElementType::Wedge6, 3);

    math::Vector<Real, 3> xi{Real(0.2), Real(0.3), Real(-0.1)};

    std::vector<Real> div;
    EXPECT_NO_THROW(basis.evaluate_divergence(xi, div));
    EXPECT_EQ(div.size(), 140u);

    for (const auto& d : div) {
        EXPECT_TRUE(std::isfinite(d));
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT3DivergenceFinite) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 3);

    math::Vector<Real, 3> xi{Real(0.1), Real(-0.15), Real(0.35)};

    std::vector<Real> div;
    EXPECT_NO_THROW(basis.evaluate_divergence(xi, div));
    EXPECT_EQ(div.size(), 137u);

    for (const auto& d : div) {
        EXPECT_TRUE(std::isfinite(d));
    }
}

// =============================================================================
// Nedelec(k) Curl Tests for k=1,2
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeND1CurlFinite) {
    NedelecBasis basis(ElementType::Wedge6, 1);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<math::Vector<Real, 3>> curl;
    EXPECT_NO_THROW(basis.evaluate_curl(xi, curl));
    EXPECT_EQ(curl.size(), 34u);

    for (const auto& c : curl) {
        EXPECT_TRUE(std::isfinite(c[0]));
        EXPECT_TRUE(std::isfinite(c[1]));
        EXPECT_TRUE(std::isfinite(c[2]));
    }
}

TEST(HigherOrderWedgePyramid, WedgeND2CurlFinite) {
    NedelecBasis basis(ElementType::Wedge6, 2);

    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.0)};

    std::vector<math::Vector<Real, 3>> curl;
    EXPECT_NO_THROW(basis.evaluate_curl(xi, curl));
    EXPECT_EQ(curl.size(), 84u);

    for (const auto& c : curl) {
        EXPECT_TRUE(std::isfinite(c[0]));
        EXPECT_TRUE(std::isfinite(c[1]));
        EXPECT_TRUE(std::isfinite(c[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidND2CurlFinite) {
    NedelecBasis basis(ElementType::Pyramid5, 2);

    math::Vector<Real, 3> xi{Real(0.0), Real(0.0), Real(0.25)};

    std::vector<math::Vector<Real, 3>> curl;
    EXPECT_NO_THROW(basis.evaluate_curl(xi, curl));
    EXPECT_EQ(curl.size(), 63u);

    for (const auto& c : curl) {
        EXPECT_TRUE(std::isfinite(c[0]));
        EXPECT_TRUE(std::isfinite(c[1]));
        EXPECT_TRUE(std::isfinite(c[2]));
    }
}

TEST(HigherOrderWedgePyramid, WedgeND3CurlFinite) {
    NedelecBasis basis(ElementType::Wedge6, 3);

    math::Vector<Real, 3> xi{Real(0.2), Real(0.3), Real(-0.1)};

    std::vector<math::Vector<Real, 3>> curl;
    EXPECT_NO_THROW(basis.evaluate_curl(xi, curl));
    EXPECT_EQ(curl.size(), 168u);

    for (const auto& c : curl) {
        EXPECT_TRUE(std::isfinite(c[0]));
        EXPECT_TRUE(std::isfinite(c[1]));
        EXPECT_TRUE(std::isfinite(c[2]));
    }
}

TEST(HigherOrderWedgePyramid, PyramidND3CurlFinite) {
    NedelecBasis basis(ElementType::Pyramid5, 3);

    math::Vector<Real, 3> xi{Real(0.1), Real(-0.15), Real(0.35)};

    std::vector<math::Vector<Real, 3>> curl;
    EXPECT_NO_THROW(basis.evaluate_curl(xi, curl));
    EXPECT_EQ(curl.size(), 116u);

    for (const auto& c : curl) {
        EXPECT_TRUE(std::isfinite(c[0]));
        EXPECT_TRUE(std::isfinite(c[1]));
        EXPECT_TRUE(std::isfinite(c[2]));
    }
}

// =============================================================================
// Moment-Based DOF Verification: RT Face-Normal Integrals
// =============================================================================
// For RT(k), the higher-order face DOFs in VectorBasis.cpp are assembled as
// reference-face moments against the local face Lagrange basis:
//   dof_f,m(N_i) = integral_f (N_i . n_ref) * phi_m dA
// The tests below verify those actual functionals rather than assuming that
// moment_index 0 always corresponds to an unweighted q = 1 face flux.
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeRT0FaceNormalMoment) {
    // Verify RT(0) on wedge: integral of N_i . n over face j = delta_ij
    RaviartThomasBasis basis(ElementType::Wedge6, 0);
    ASSERT_EQ(basis.size(), 5u);

    constexpr int n_faces = 5;
    for (int i = 0; i < static_cast<int>(basis.size()); ++i) {
        for (int f = 0; f < n_faces; ++f) {
            double flux = integrate_wedge_face_flux(basis, f, i);
            if (i == f) {
                EXPECT_NEAR(flux, 1.0, 1e-10)
                    << "RT(0) wedge: flux of N_" << i << " over face " << f
                    << " should be 1";
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-10)
                    << "RT(0) wedge: flux of N_" << i << " over face " << f
                    << " should be 0";
            }
        }
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT0FaceNormalMoment) {
    // Verify RT(0) on pyramid: integral of N_i . n over face j = delta_ij
    RaviartThomasBasis basis(ElementType::Pyramid5, 0);
    ASSERT_EQ(basis.size(), 5u);

    constexpr int n_faces = 5;
    for (int i = 0; i < static_cast<int>(basis.size()); ++i) {
        for (int f = 0; f < n_faces; ++f) {
            double flux = integrate_pyramid_face_flux(basis, f, i);
            if (i == f) {
                EXPECT_NEAR(flux, 1.0, 1e-10)
                    << "RT(0) pyramid: flux of N_" << i << " over face " << f
                    << " should be 1";
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-10)
                    << "RT(0) pyramid: flux of N_" << i << " over face " << f
                    << " should be 0";
            }
        }
    }
}

TEST(HigherOrderWedgePyramid, WedgeRT1FaceNormalMoment) {
    RaviartThomasBasis basis(ElementType::Wedge6, 1);
    ASSERT_EQ(basis.size(), 24u);

    auto assoc = basis.dof_associations();
    ASSERT_EQ(assoc.size(), 24u);
    const auto ref = elements::ReferenceElement::create(ElementType::Wedge6);
    for (std::size_t i = 0; i < assoc.size(); ++i) {
        for (std::size_t f = 0; f < ref.num_faces(); ++f) {
            const auto& face_nodes = ref.face_nodes(f);
            const std::size_t nface = (face_nodes.size() == 3u)
                ? static_cast<std::size_t>((basis.order() + 1) * (basis.order() + 2) / 2)
                : static_cast<std::size_t>((basis.order() + 1) * (basis.order() + 1));

            for (std::size_t m = 0; m < nface; ++m) {
                const double functional = integrate_rt_face_functional(
                    basis,
                    ElementType::Wedge6,
                    static_cast<int>(f),
                    static_cast<int>(m),
                    static_cast<int>(i));
                const bool is_own_face_dof =
                    assoc[i].entity_type == DofEntity::Face &&
                    assoc[i].entity_id == static_cast<int>(f) &&
                    assoc[i].moment_index == static_cast<int>(m);
                EXPECT_NEAR(functional, is_own_face_dof ? 1.0 : 0.0, 1e-8)
                    << "Wedge RT(1): face functional (" << f << ", " << m
                    << ") applied to basis function " << i
                    << " should be " << (is_own_face_dof ? 1.0 : 0.0);
            }
        }
    }
}

// =============================================================================
// Moment-Based DOF Verification: Nedelec Edge-Tangent Integrals
// =============================================================================
// For Nedelec(k), the edge DOF functional is:
//   dof_e(N_i) = integral_e (N_i . t) * q_m ds
// where t is the edge tangent and q_m is the m-th polynomial mode.
// For k=0 (zeroth mode q=1), this should give delta_{i,e}.
// =============================================================================

TEST(HigherOrderWedgePyramid, WedgeND0EdgeTangentMoment) {
    // Verify Nedelec(0) on wedge: integral of N_i . t along edge j = delta_ij
    NedelecBasis basis(ElementType::Wedge6, 0);
    ASSERT_EQ(basis.size(), 9u);

    constexpr int n_edges = 9;
    for (int i = 0; i < static_cast<int>(basis.size()); ++i) {
        for (int e = 0; e < n_edges; ++e) {
            double tangent_integral = integrate_wedge_edge_tangent(basis, e, i);
            if (i == e) {
                EXPECT_NEAR(tangent_integral, 1.0, 1e-10)
                    << "ND(0) wedge: tangent integral of N_" << i
                    << " along edge " << e << " should be 1";
            } else {
                EXPECT_NEAR(tangent_integral, 0.0, 1e-10)
                    << "ND(0) wedge: tangent integral of N_" << i
                    << " along edge " << e << " should be 0";
            }
        }
    }
}

TEST(HigherOrderWedgePyramid, PyramidND0EdgeTangentMoment) {
    // Verify Nedelec(0) on pyramid: integral of N_i . t along edge j = delta_ij
    NedelecBasis basis(ElementType::Pyramid5, 0);
    ASSERT_EQ(basis.size(), 8u);

    constexpr int n_edges = 8;
    for (int i = 0; i < static_cast<int>(basis.size()); ++i) {
        for (int e = 0; e < n_edges; ++e) {
            double tangent_integral = integrate_pyramid_edge_tangent(basis, e, i);
            if (i == e) {
                EXPECT_NEAR(tangent_integral, 1.0, 1e-10)
                    << "ND(0) pyramid: tangent integral of N_" << i
                    << " along edge " << e << " should be 1";
            } else {
                EXPECT_NEAR(tangent_integral, 0.0, 1e-10)
                    << "ND(0) pyramid: tangent integral of N_" << i
                    << " along edge " << e << " should be 0";
            }
        }
    }
}

TEST(HigherOrderWedgePyramid, WedgeND1EdgeTangentialMomentKronecker) {
    NedelecBasis basis(ElementType::Wedge6, 1);
    ASSERT_EQ(basis.size(), 34u);
    expect_nd_edge_kronecker(basis, ElementType::Wedge6, 5e-8);
}

TEST(HigherOrderWedgePyramid, PyramidND1EdgeTangentialMomentKronecker) {
    NedelecBasis basis(ElementType::Pyramid5, 1);
    ASSERT_EQ(basis.size(), 28u);
    expect_nd_edge_kronecker(basis, ElementType::Pyramid5, 5e-8);
}

TEST(HigherOrderWedgePyramid, WedgeND2EdgeTangentialMomentKronecker) {
    NedelecBasis basis(ElementType::Wedge6, 2);
    ASSERT_EQ(basis.size(), 84u);
    expect_nd_edge_kronecker(basis, ElementType::Wedge6, 2e-7);
}

TEST(HigherOrderWedgePyramid, PyramidND2EdgeTangentialMomentKronecker) {
    NedelecBasis basis(ElementType::Pyramid5, 2);
    ASSERT_EQ(basis.size(), 63u);
    expect_nd_edge_kronecker(basis, ElementType::Pyramid5, 2e-7);
}

TEST(HigherOrderWedgePyramid, WedgeND1FaceTangentialMomentKronecker) {
    NedelecBasis basis(ElementType::Wedge6, 1);
    ASSERT_EQ(basis.size(), 34u);
    expect_nd_face_kronecker(basis, ElementType::Wedge6, 5e-8);
}

TEST(HigherOrderWedgePyramid, PyramidND1FaceTangentialMomentKronecker) {
    NedelecBasis basis(ElementType::Pyramid5, 1);
    ASSERT_EQ(basis.size(), 28u);
    expect_nd_face_kronecker(basis, ElementType::Pyramid5, 5e-8);
}

TEST(HigherOrderWedgePyramid, WedgeND2FaceTangentialMomentKronecker) {
    NedelecBasis basis(ElementType::Wedge6, 2);
    ASSERT_EQ(basis.size(), 84u);
    expect_nd_face_kronecker(basis, ElementType::Wedge6, 2e-7);
}

TEST(HigherOrderWedgePyramid, PyramidND2FaceTangentialMomentKronecker) {
    NedelecBasis basis(ElementType::Pyramid5, 2);
    ASSERT_EQ(basis.size(), 63u);
    expect_nd_face_kronecker(basis, ElementType::Pyramid5, 2e-7);
}

TEST(HigherOrderWedgePyramid, WedgeND3EdgeTangentialMomentKronecker) {
    NedelecBasis basis(ElementType::Wedge6, 3);
    ASSERT_EQ(basis.size(), 168u);
    expect_nd_edge_kronecker(basis, ElementType::Wedge6, 5e-7);
}

TEST(HigherOrderWedgePyramid, PyramidND3EdgeTangentialMomentKronecker) {
    NedelecBasis basis(ElementType::Pyramid5, 3);
    ASSERT_EQ(basis.size(), 116u);
    expect_nd_edge_kronecker(basis, ElementType::Pyramid5, 5e-7);
}

TEST(HigherOrderWedgePyramid, WedgeND3FaceTangentialMomentKronecker) {
    NedelecBasis basis(ElementType::Wedge6, 3);
    ASSERT_EQ(basis.size(), 168u);
    expect_nd_face_kronecker(basis, ElementType::Wedge6, 8e-7);
}

TEST(HigherOrderWedgePyramid, PyramidND3FaceTangentialMomentKronecker) {
    NedelecBasis basis(ElementType::Pyramid5, 3);
    ASSERT_EQ(basis.size(), 116u);
    expect_nd_face_kronecker(basis, ElementType::Pyramid5, 8e-7);
}

// =============================================================================
// RT(1) Divergence Consistency With Outward Face Flux
// =============================================================================
// For any RT(k) basis function N_i, the divergence theorem gives:
//   integral_K div(N_i) dV = sum_f integral_f (N_i . n_out) dA
// =============================================================================

TEST(HigherOrderWedgePyramid, PyramidRT1FaceNormalMoment) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 1);
    ASSERT_EQ(basis.size(), 19u);

    auto assoc = basis.dof_associations();
    ASSERT_EQ(assoc.size(), 19u);

    const auto ref = elements::ReferenceElement::create(ElementType::Pyramid5);
    for (std::size_t i = 0; i < assoc.size(); ++i) {
        for (std::size_t f = 0; f < ref.num_faces(); ++f) {
            const auto& face_nodes = ref.face_nodes(f);
            const std::size_t nface = (face_nodes.size() == 3u)
                ? static_cast<std::size_t>((basis.order() + 1) * (basis.order() + 2) / 2)
                : static_cast<std::size_t>((basis.order() + 1) * (basis.order() + 1));

            for (std::size_t m = 0; m < nface; ++m) {
                const double functional = integrate_rt_face_functional(
                    basis,
                    ElementType::Pyramid5,
                    static_cast<int>(f),
                    static_cast<int>(m),
                    static_cast<int>(i));
                const bool is_own_face_dof =
                    assoc[i].entity_type == DofEntity::Face &&
                    assoc[i].entity_id == static_cast<int>(f) &&
                    assoc[i].moment_index == static_cast<int>(m);
                EXPECT_NEAR(functional, is_own_face_dof ? 1.0 : 0.0, 1e-8)
                    << "Pyramid RT(1): face functional (" << f << ", " << m
                    << ") applied to basis function " << i
                    << " should be " << (is_own_face_dof ? 1.0 : 0.0);
            }
        }
    }
}

TEST(HigherOrderWedgePyramid, WedgeRT2FaceNormalMoment) {
    RaviartThomasBasis basis(ElementType::Wedge6, 2);
    ASSERT_EQ(basis.size(), 66u);

    auto assoc = basis.dof_associations();
    ASSERT_EQ(assoc.size(), 66u);
    const auto ref = elements::ReferenceElement::create(ElementType::Wedge6);
    for (std::size_t i = 0; i < assoc.size(); ++i) {
        for (std::size_t f = 0; f < ref.num_faces(); ++f) {
            const auto& face_nodes = ref.face_nodes(f);
            const std::size_t nface = (face_nodes.size() == 3u)
                ? static_cast<std::size_t>((basis.order() + 1) * (basis.order() + 2) / 2)
                : static_cast<std::size_t>((basis.order() + 1) * (basis.order() + 1));

            for (std::size_t m = 0; m < nface; ++m) {
                const double functional = integrate_rt_face_functional(
                    basis,
                    ElementType::Wedge6,
                    static_cast<int>(f),
                    static_cast<int>(m),
                    static_cast<int>(i),
                    10);
                const bool is_own_face_dof =
                    assoc[i].entity_type == DofEntity::Face &&
                    assoc[i].entity_id == static_cast<int>(f) &&
                    assoc[i].moment_index == static_cast<int>(m);
                EXPECT_NEAR(functional, is_own_face_dof ? 1.0 : 0.0, 5e-8)
                    << "Wedge RT(2): face functional (" << f << ", " << m
                    << ") applied to basis function " << i
                    << " should be " << (is_own_face_dof ? 1.0 : 0.0);
            }
        }
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT2FaceNormalMoment) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 2);
    ASSERT_EQ(basis.size(), 57u);

    auto assoc = basis.dof_associations();
    ASSERT_EQ(assoc.size(), 57u);
    const auto ref = elements::ReferenceElement::create(ElementType::Pyramid5);
    for (std::size_t i = 0; i < assoc.size(); ++i) {
        for (std::size_t f = 0; f < ref.num_faces(); ++f) {
            const auto& face_nodes = ref.face_nodes(f);
            const std::size_t nface = (face_nodes.size() == 3u)
                ? static_cast<std::size_t>((basis.order() + 1) * (basis.order() + 2) / 2)
                : static_cast<std::size_t>((basis.order() + 1) * (basis.order() + 1));

            for (std::size_t m = 0; m < nface; ++m) {
                const double functional = integrate_rt_face_functional(
                    basis,
                    ElementType::Pyramid5,
                    static_cast<int>(f),
                    static_cast<int>(m),
                    static_cast<int>(i),
                    10);
                const bool is_own_face_dof =
                    assoc[i].entity_type == DofEntity::Face &&
                    assoc[i].entity_id == static_cast<int>(f) &&
                    assoc[i].moment_index == static_cast<int>(m);
                EXPECT_NEAR(functional, is_own_face_dof ? 1.0 : 0.0, 1e-7)
                    << "Pyramid RT(2): face functional (" << f << ", " << m
                    << ") applied to basis function " << i
                    << " should be " << (is_own_face_dof ? 1.0 : 0.0);
            }
        }
    }
}

TEST(HigherOrderWedgePyramid, WedgeRT3FaceNormalMoment) {
    RaviartThomasBasis basis(ElementType::Wedge6, 3);
    ASSERT_EQ(basis.size(), 140u);

    auto assoc = basis.dof_associations();
    ASSERT_EQ(assoc.size(), 140u);
    const auto ref = elements::ReferenceElement::create(ElementType::Wedge6);
    for (std::size_t i = 0; i < assoc.size(); ++i) {
        for (std::size_t f = 0; f < ref.num_faces(); ++f) {
            const auto& face_nodes = ref.face_nodes(f);
            const std::size_t nface = (face_nodes.size() == 3u)
                ? static_cast<std::size_t>((basis.order() + 1) * (basis.order() + 2) / 2)
                : static_cast<std::size_t>((basis.order() + 1) * (basis.order() + 1));

            for (std::size_t m = 0; m < nface; ++m) {
                const double functional = integrate_rt_face_functional(
                    basis,
                    ElementType::Wedge6,
                    static_cast<int>(f),
                    static_cast<int>(m),
                    static_cast<int>(i),
                    12);
                const bool is_own_face_dof =
                    assoc[i].entity_type == DofEntity::Face &&
                    assoc[i].entity_id == static_cast<int>(f) &&
                    assoc[i].moment_index == static_cast<int>(m);
                EXPECT_NEAR(functional, is_own_face_dof ? 1.0 : 0.0, 5e-7)
                    << "Wedge RT(3): face functional (" << f << ", " << m
                    << ") applied to basis function " << i
                    << " should be " << (is_own_face_dof ? 1.0 : 0.0);
            }
        }
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT3FaceNormalMoment) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 3);
    ASSERT_EQ(basis.size(), 137u);

    auto assoc = basis.dof_associations();
    ASSERT_EQ(assoc.size(), 137u);
    const auto ref = elements::ReferenceElement::create(ElementType::Pyramid5);
    for (std::size_t i = 0; i < assoc.size(); ++i) {
        for (std::size_t f = 0; f < ref.num_faces(); ++f) {
            const auto& face_nodes = ref.face_nodes(f);
            const std::size_t nface = (face_nodes.size() == 3u)
                ? static_cast<std::size_t>((basis.order() + 1) * (basis.order() + 2) / 2)
                : static_cast<std::size_t>((basis.order() + 1) * (basis.order() + 1));

            for (std::size_t m = 0; m < nface; ++m) {
                const double functional = integrate_rt_face_functional(
                    basis,
                    ElementType::Pyramid5,
                    static_cast<int>(f),
                    static_cast<int>(m),
                    static_cast<int>(i),
                    12);
                const bool is_own_face_dof =
                    assoc[i].entity_type == DofEntity::Face &&
                    assoc[i].entity_id == static_cast<int>(f) &&
                    assoc[i].moment_index == static_cast<int>(m);
                EXPECT_NEAR(functional, is_own_face_dof ? 1.0 : 0.0, 2e-6)
                    << "Pyramid RT(3): face functional (" << f << ", " << m
                    << ") applied to basis function " << i
                    << " should be " << (is_own_face_dof ? 1.0 : 0.0);
            }
        }
    }
}

TEST(HigherOrderWedgePyramid, WedgeRT1DivergenceFluxConsistency) {
    RaviartThomasBasis basis(ElementType::Wedge6, 1);
    ASSERT_EQ(basis.size(), 24u);

    constexpr int n_faces = 5;
    for (std::size_t i = 0; i < basis.size(); ++i) {
        const double vol_div = integrate_rt_volume_divergence(
            basis, ElementType::Wedge6, static_cast<int>(i), 8);

        double face_sum = 0.0;
        for (int f = 0; f < n_faces; ++f) {
            face_sum += integrate_rt_face_flux_outward(
                basis, ElementType::Wedge6, f, static_cast<int>(i), 8);
        }

        EXPECT_NEAR(vol_div, face_sum, 1e-8)
            << "Wedge RT(1): divergence theorem mismatch for DOF " << i;
    }
}

TEST(HigherOrderWedgePyramid, WedgeRT0LinearCombinationSatisfiesDivergenceTheorem) {
    RaviartThomasBasis basis(ElementType::Wedge6, 0);
    ASSERT_EQ(basis.size(), 5u);

    const std::vector<Real> coeffs = {
        Real(0.18), Real(-0.04), Real(0.09), Real(0.07), Real(-0.13)
    };

    double vol_div = 0.0;
    double face_sum = 0.0;
    for (std::size_t i = 0; i < basis.size(); ++i) {
        vol_div += static_cast<double>(coeffs[i]) *
                   integrate_rt_volume_divergence(
                       basis, ElementType::Wedge6, static_cast<int>(i), 8);
        for (int f = 0; f < 5; ++f) {
            face_sum += static_cast<double>(coeffs[i]) *
                        integrate_rt_face_flux_outward(
                            basis, ElementType::Wedge6, f, static_cast<int>(i), 8);
        }
    }

    EXPECT_NEAR(vol_div, face_sum, 1e-10);
}

TEST(HigherOrderWedgePyramid, PyramidRT1DivergenceFluxConsistency) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 1);
    ASSERT_EQ(basis.size(), 19u);

    constexpr int n_faces = 5;
    for (std::size_t i = 0; i < basis.size(); ++i) {
        const double vol_div = integrate_rt_volume_divergence(
            basis, ElementType::Pyramid5, static_cast<int>(i), 10);

        double face_sum = 0.0;
        for (int f = 0; f < n_faces; ++f) {
            face_sum += integrate_rt_face_flux_outward(
                basis, ElementType::Pyramid5, f, static_cast<int>(i), 10);
        }

        EXPECT_NEAR(vol_div, face_sum, 1e-6)
            << "Pyramid RT(1): divergence theorem mismatch for DOF " << i;
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT0LinearCombinationSatisfiesDivergenceTheorem) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 0);
    ASSERT_EQ(basis.size(), 5u);

    const std::vector<Real> coeffs = {
        Real(0.11), Real(-0.05), Real(0.14), Real(-0.02), Real(0.08)
    };

    double vol_div = 0.0;
    double face_sum = 0.0;
    for (std::size_t i = 0; i < basis.size(); ++i) {
        vol_div += static_cast<double>(coeffs[i]) *
                   integrate_rt_volume_divergence(
                       basis, ElementType::Pyramid5, static_cast<int>(i), 10);
        for (int f = 0; f < 5; ++f) {
            face_sum += static_cast<double>(coeffs[i]) *
                        integrate_rt_face_flux_outward(
                            basis, ElementType::Pyramid5, f, static_cast<int>(i), 10);
        }
    }

    EXPECT_NEAR(vol_div, face_sum, 1e-10);
}

TEST(HigherOrderWedgePyramid, WedgeRT2DivergenceFluxConsistency) {
    RaviartThomasBasis basis(ElementType::Wedge6, 2);
    ASSERT_EQ(basis.size(), 66u);

    constexpr int n_faces = 5;
    for (std::size_t i = 0; i < basis.size(); ++i) {
        const double vol_div = integrate_rt_volume_divergence(
            basis, ElementType::Wedge6, static_cast<int>(i), 10);

        double face_sum = 0.0;
        for (int f = 0; f < n_faces; ++f) {
            face_sum += integrate_rt_face_flux_outward(
                basis, ElementType::Wedge6, f, static_cast<int>(i), 10);
        }

        EXPECT_NEAR(vol_div, face_sum, 5e-7)
            << "Wedge RT(2): divergence theorem mismatch for DOF " << i;
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT2DivergenceFluxConsistency) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 2);
    ASSERT_EQ(basis.size(), 57u);

    constexpr int n_faces = 5;
    for (std::size_t i = 0; i < basis.size(); ++i) {
        const double vol_div = integrate_rt_volume_divergence(
            basis, ElementType::Pyramid5, static_cast<int>(i), 12);

        double face_sum = 0.0;
        for (int f = 0; f < n_faces; ++f) {
            face_sum += integrate_rt_face_flux_outward(
                basis, ElementType::Pyramid5, f, static_cast<int>(i), 12);
        }

        EXPECT_NEAR(vol_div, face_sum, 1e-6)
            << "Pyramid RT(2): divergence theorem mismatch for DOF " << i;
    }
}

TEST(HigherOrderWedgePyramid, WedgeRT3DivergenceFluxConsistency) {
    RaviartThomasBasis basis(ElementType::Wedge6, 3);
    ASSERT_EQ(basis.size(), 140u);

    constexpr int n_faces = 5;
    for (std::size_t i = 0; i < basis.size(); ++i) {
        const double vol_div = integrate_rt_volume_divergence(
            basis, ElementType::Wedge6, static_cast<int>(i), 12);

        double face_sum = 0.0;
        for (int f = 0; f < n_faces; ++f) {
            face_sum += integrate_rt_face_flux_outward(
                basis, ElementType::Wedge6, f, static_cast<int>(i), 12);
        }

        EXPECT_NEAR(vol_div, face_sum, 1e-6)
            << "Wedge RT(3): divergence theorem mismatch for DOF " << i;
    }
}

TEST(HigherOrderWedgePyramid, PyramidRT3DivergenceFluxConsistency) {
    RaviartThomasBasis basis(ElementType::Pyramid5, 3);
    ASSERT_EQ(basis.size(), 137u);

    constexpr int n_faces = 5;
    for (std::size_t i = 0; i < basis.size(); ++i) {
        const double vol_div = integrate_rt_volume_divergence(
            basis, ElementType::Pyramid5, static_cast<int>(i), 14);

        double face_sum = 0.0;
        for (int f = 0; f < n_faces; ++f) {
            face_sum += integrate_rt_face_flux_outward(
                basis, ElementType::Pyramid5, f, static_cast<int>(i), 14);
        }

        EXPECT_NEAR(vol_div, face_sum, 2e-6)
            << "Pyramid RT(3): divergence theorem mismatch for DOF " << i;
    }
}

TEST(HigherOrderWedgePyramid, WedgeND1StokesTheoremOnAllFaces) {
    NedelecBasis basis(ElementType::Wedge6, 1);
    const auto ref = elements::ReferenceElement::create(ElementType::Wedge6);

    for (std::size_t i = 0; i < basis.size(); ++i) {
        for (std::size_t f = 0; f < ref.num_faces(); ++f) {
            const double curl_flux = integrate_nd_face_curl_flux(
                basis, ElementType::Wedge6, static_cast<int>(f), static_cast<int>(i), 8);
            const double boundary = integrate_nd_face_boundary_circulation(
                basis, ElementType::Wedge6, static_cast<int>(f), static_cast<int>(i), 8);

            EXPECT_NEAR(curl_flux, boundary, 2e-4)
                << "Wedge ND(1): Stokes mismatch on face " << f
                << " for basis function " << i;
        }
    }
}

TEST(HigherOrderWedgePyramid, WedgeND0LinearCombinationSatisfiesStokesTheoremOnAllFaces) {
    NedelecBasis basis(ElementType::Wedge6, 0);
    const auto ref = elements::ReferenceElement::create(ElementType::Wedge6);
    const std::vector<Real> coeffs = {
        Real(0.10), Real(-0.15), Real(0.20), Real(0.05), Real(-0.08),
        Real(0.12), Real(-0.03), Real(0.09), Real(-0.11)
    };

    for (std::size_t f = 0; f < ref.num_faces(); ++f) {
        double curl_flux = 0.0;
        double boundary = 0.0;
        for (std::size_t i = 0; i < basis.size(); ++i) {
            curl_flux += static_cast<double>(coeffs[i]) *
                         integrate_nd_face_curl_flux(
                             basis, ElementType::Wedge6, static_cast<int>(f), static_cast<int>(i), 8);
            boundary += static_cast<double>(coeffs[i]) *
                        integrate_nd_face_boundary_circulation(
                            basis, ElementType::Wedge6, static_cast<int>(f), static_cast<int>(i), 8);
        }

        EXPECT_NEAR(curl_flux, boundary, 1e-10)
            << "Wedge ND(0): Stokes mismatch on face " << f;
    }
}

TEST(HigherOrderWedgePyramid, PyramidND1StokesTheoremOnAllFaces) {
    NedelecBasis basis(ElementType::Pyramid5, 1);
    const auto ref = elements::ReferenceElement::create(ElementType::Pyramid5);

    for (std::size_t i = 0; i < basis.size(); ++i) {
        for (std::size_t f = 0; f < ref.num_faces(); ++f) {
            const double curl_flux = integrate_nd_face_curl_flux(
                basis, ElementType::Pyramid5, static_cast<int>(f), static_cast<int>(i), 8);
            const double boundary = integrate_nd_face_boundary_circulation(
                basis, ElementType::Pyramid5, static_cast<int>(f), static_cast<int>(i), 8);

            EXPECT_NEAR(curl_flux, boundary, 5e-5)
                << "Pyramid ND(1): Stokes mismatch on face " << f
                << " for basis function " << i;
        }
    }
}

TEST(HigherOrderWedgePyramid, PyramidND0LinearCombinationSatisfiesStokesTheoremOnAllFaces) {
    NedelecBasis basis(ElementType::Pyramid5, 0);
    const auto ref = elements::ReferenceElement::create(ElementType::Pyramid5);
    const std::vector<Real> coeffs = {
        Real(0.14), Real(-0.07), Real(0.09), Real(0.05),
        Real(-0.11), Real(0.06), Real(-0.04), Real(0.12)
    };

    for (std::size_t f = 0; f < ref.num_faces(); ++f) {
        double curl_flux = 0.0;
        double boundary = 0.0;
        for (std::size_t i = 0; i < basis.size(); ++i) {
            curl_flux += static_cast<double>(coeffs[i]) *
                         integrate_nd_face_curl_flux(
                             basis, ElementType::Pyramid5, static_cast<int>(f), static_cast<int>(i), 8);
            boundary += static_cast<double>(coeffs[i]) *
                        integrate_nd_face_boundary_circulation(
                            basis, ElementType::Pyramid5, static_cast<int>(f), static_cast<int>(i), 8);
        }

        EXPECT_NEAR(curl_flux, boundary, 1e-10)
            << "Pyramid ND(0): Stokes mismatch on face " << f;
    }
}

TEST(HigherOrderWedgePyramid, WedgeND2StokesTheoremOnAllFaces) {
    NedelecBasis basis(ElementType::Wedge6, 2);
    const auto ref = elements::ReferenceElement::create(ElementType::Wedge6);

    for (std::size_t i = 0; i < basis.size(); ++i) {
        for (std::size_t f = 0; f < ref.num_faces(); ++f) {
            const double curl_flux = integrate_nd_face_curl_flux(
                basis, ElementType::Wedge6, static_cast<int>(f), static_cast<int>(i), 10);
            const double boundary = integrate_nd_face_boundary_circulation(
                basis, ElementType::Wedge6, static_cast<int>(f), static_cast<int>(i), 10);

            EXPECT_NEAR(curl_flux, boundary, 5e-4)
                << "Wedge ND(2): Stokes mismatch on face " << f
                << " for basis function " << i;
        }
    }
}

TEST(HigherOrderWedgePyramid, PyramidND2StokesTheoremOnAllFaces) {
    NedelecBasis basis(ElementType::Pyramid5, 2);
    const auto ref = elements::ReferenceElement::create(ElementType::Pyramid5);

    for (std::size_t i = 0; i < basis.size(); ++i) {
        for (std::size_t f = 0; f < ref.num_faces(); ++f) {
            const double curl_flux = integrate_nd_face_curl_flux(
                basis, ElementType::Pyramid5, static_cast<int>(f), static_cast<int>(i), 10);
            const double boundary = integrate_nd_face_boundary_circulation(
                basis, ElementType::Pyramid5, static_cast<int>(f), static_cast<int>(i), 10);

            EXPECT_NEAR(curl_flux, boundary, 5e-4)
                << "Pyramid ND(2): Stokes mismatch on face " << f
                << " for basis function " << i;
        }
    }
}

TEST(HigherOrderWedgePyramid, WedgeND3StokesTheoremOnAllFaces) {
    NedelecBasis basis(ElementType::Wedge6, 3);
    const auto ref = elements::ReferenceElement::create(ElementType::Wedge6);

    for (std::size_t i = 0; i < basis.size(); ++i) {
        for (std::size_t f = 0; f < ref.num_faces(); ++f) {
            const double curl_flux = integrate_nd_face_curl_flux(
                basis, ElementType::Wedge6, static_cast<int>(f), static_cast<int>(i), 12);
            const double boundary = integrate_nd_face_boundary_circulation(
                basis, ElementType::Wedge6, static_cast<int>(f), static_cast<int>(i), 12);

            EXPECT_NEAR(curl_flux, boundary, 2e-3)
                << "Wedge ND(3): Stokes mismatch on face " << f
                << " for basis function " << i;
        }
    }
}

TEST(HigherOrderWedgePyramid, PyramidND3StokesTheoremOnAllFaces) {
    NedelecBasis basis(ElementType::Pyramid5, 3);
    const auto ref = elements::ReferenceElement::create(ElementType::Pyramid5);

    for (std::size_t i = 0; i < basis.size(); ++i) {
        for (std::size_t f = 0; f < ref.num_faces(); ++f) {
            const double curl_flux = integrate_nd_face_curl_flux(
                basis, ElementType::Pyramid5, static_cast<int>(f), static_cast<int>(i), 12);
            const double boundary = integrate_nd_face_boundary_circulation(
                basis, ElementType::Pyramid5, static_cast<int>(f), static_cast<int>(i), 12);

            EXPECT_NEAR(curl_flux, boundary, 2e-3)
                << "Pyramid ND(3): Stokes mismatch on face " << f
                << " for basis function " << i;
        }
    }
}

TEST(HigherOrderWedgePyramid, WedgeND1CurlIsDivergenceFree) {
    NedelecBasis basis(ElementType::Wedge6, 1);
    expect_curl_is_divergence_free(
        basis,
        math::Vector<Real, 3>{Real(0.25), Real(0.25), Real(0.0)},
        Real(1e-3),
        Real(2e-5));
}

TEST(HigherOrderWedgePyramid, PyramidND1CurlIsDivergenceFree) {
    NedelecBasis basis(ElementType::Pyramid5, 1);
    expect_curl_is_divergence_free(
        basis,
        math::Vector<Real, 3>{Real(0.0), Real(0.0), Real(0.25)},
        Real(1e-3),
        Real(2e-5));
}

TEST(HigherOrderWedgePyramid, WedgeND2CurlIsDivergenceFree) {
    NedelecBasis basis(ElementType::Wedge6, 2);
    expect_curl_is_divergence_free(
        basis,
        math::Vector<Real, 3>{Real(0.25), Real(0.25), Real(0.0)},
        Real(1e-3),
        Real(5e-5));
}

TEST(HigherOrderWedgePyramid, PyramidND2CurlIsDivergenceFree) {
    NedelecBasis basis(ElementType::Pyramid5, 2);
    expect_curl_is_divergence_free(
        basis,
        math::Vector<Real, 3>{Real(0.0), Real(0.0), Real(0.25)},
        Real(1e-3),
        Real(5e-5));
}

TEST(HigherOrderWedgePyramid, WedgeND3CurlIsDivergenceFree) {
    NedelecBasis basis(ElementType::Wedge6, 3);
    expect_curl_is_divergence_free(
        basis,
        math::Vector<Real, 3>{Real(0.2), Real(0.3), Real(-0.1)},
        Real(1e-3),
        Real(1e-4));
}

TEST(HigherOrderWedgePyramid, PyramidND3CurlIsDivergenceFree) {
    NedelecBasis basis(ElementType::Pyramid5, 3);
    expect_curl_is_divergence_free(
        basis,
        math::Vector<Real, 3>{Real(0.1), Real(-0.15), Real(0.35)},
        Real(1e-3),
        Real(1e-4));
}

// =============================================================================
// Summary of supported orders
// =============================================================================
// RT on Wedge:   k=0 (5 DOFs), k=1 (24 DOFs), k=2 (66 DOFs), k=3 (140 DOFs)
// RT on Pyramid: k=0 (5 DOFs), k=1 (19 DOFs), k=2 (57 DOFs), k=3 (137 DOFs)
// ND on Wedge:   k=0 (9 DOFs), k=1 (34 DOFs), k=2 (84 DOFs), k=3 (168 DOFs)
// ND on Pyramid: k=0 (8 DOFs), k=1 (28 DOFs), k=2 (63 DOFs), k=3 (116 DOFs)
