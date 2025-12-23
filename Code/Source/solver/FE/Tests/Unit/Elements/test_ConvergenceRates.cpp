/**
 * @file test_ConvergenceRates.cpp
 * @brief Polynomial convergence rate tests for Lagrange interpolation
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "FE/Basis/LagrangeBasis.h"
#include "FE/Geometry/IsoparametricMapping.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include "FE/Quadrature/TriangleQuadrature.h"

using namespace svmp::FE;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

Real exact_f_2d(const math::Vector<Real, 3>& x) {
    return static_cast<Real>(std::sin(kPi * static_cast<double>(x[0])) *
                             std::sin(kPi * static_cast<double>(x[1])));
}

Real l2_error_on_quad_mesh(int p, int n) {
    basis::LagrangeBasis field_basis(ElementType::Quad4, p);
    auto geom_basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);

    // High-order quadrature to avoid quadrature-dominated error.
    quadrature::QuadrilateralQuadrature quad(12);

    const Real h = Real(1) / static_cast<Real>(n);
    Real err2 = Real(0);

    const auto geom_ref_nodes = geom_basis->nodes();
    const auto field_ref_nodes = field_basis.nodes();

    std::vector<math::Vector<Real, 3>> geom_nodes;
    geom_nodes.resize(geom_ref_nodes.size());

    std::vector<Real> coeffs;
    coeffs.resize(field_basis.size());

    std::vector<Real> phi;
    phi.resize(field_basis.size());

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const Real x0 = static_cast<Real>(i) * h;
            const Real y0 = static_cast<Real>(j) * h;

            // Geometry nodes in the order expected by geom_basis.
            for (std::size_t k = 0; k < geom_ref_nodes.size(); ++k) {
                const auto& xi = geom_ref_nodes[k];
                geom_nodes[k][0] = x0 + Real(0.5) * (xi[0] + Real(1)) * h;
                geom_nodes[k][1] = y0 + Real(0.5) * (xi[1] + Real(1)) * h;
                geom_nodes[k][2] = Real(0);
            }

            geometry::IsoparametricMapping mapping(geom_basis, geom_nodes);

            // Nodal interpolation coefficients: c_i = f(x(node_i))
            for (std::size_t a = 0; a < field_ref_nodes.size(); ++a) {
                const auto x_phys = mapping.map_to_physical(field_ref_nodes[a]);
                coeffs[a] = exact_f_2d(x_phys);
            }

            for (std::size_t q = 0; q < quad.num_points(); ++q) {
                const auto& xi_q = quad.point(q);
                const Real w = quad.weight(q);

                const auto x_phys = mapping.map_to_physical(xi_q);
                const Real f = exact_f_2d(x_phys);

                field_basis.evaluate_values(xi_q, phi);
                Real uh = Real(0);
                for (std::size_t a = 0; a < coeffs.size(); ++a) {
                    uh += coeffs[a] * phi[a];
                }

                const Real diff = f - uh;
                const Real detJ = std::abs(mapping.jacobian_determinant(xi_q));
                err2 += w * detJ * diff * diff;
            }
        }
    }

    return std::sqrt(err2);
}

Real l2_error_on_triangle_mesh(int p, int n) {
    basis::LagrangeBasis field_basis(ElementType::Triangle3, p);
    auto geom_basis = std::make_shared<basis::LagrangeBasis>(ElementType::Triangle3, 1);

    quadrature::TriangleQuadrature quad(12);

    const Real h = Real(1) / static_cast<Real>(n);
    Real err2 = Real(0);

    const auto geom_ref_nodes = geom_basis->nodes();
    const auto field_ref_nodes = field_basis.nodes();

    std::vector<math::Vector<Real, 3>> geom_nodes;
    geom_nodes.resize(geom_ref_nodes.size());

    std::vector<Real> coeffs;
    coeffs.resize(field_basis.size());

    std::vector<Real> phi;
    phi.resize(field_basis.size());

    auto map_affine_tri = [](const math::Vector<Real, 3>& xi,
                             const math::Vector<Real, 3>& v0,
                             const math::Vector<Real, 3>& v1,
                             const math::Vector<Real, 3>& v2) {
        math::Vector<Real, 3> x{};
        const Real r = xi[0];
        const Real s = xi[1];
        for (int d = 0; d < 3; ++d) {
            x[static_cast<std::size_t>(d)] =
                v0[static_cast<std::size_t>(d)] +
                r * (v1[static_cast<std::size_t>(d)] - v0[static_cast<std::size_t>(d)]) +
                s * (v2[static_cast<std::size_t>(d)] - v0[static_cast<std::size_t>(d)]);
        }
        return x;
    };

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const Real x0 = static_cast<Real>(i) * h;
            const Real y0 = static_cast<Real>(j) * h;
            const Real x1 = x0 + h;
            const Real y1 = y0 + h;

            // Triangle A: (x0,y0) -> (x1,y0) -> (x0,y1)
            math::Vector<Real, 3> v0{ x0, y0, Real(0) };
            math::Vector<Real, 3> v1{ x1, y0, Real(0) };
            math::Vector<Real, 3> v2{ x0, y1, Real(0) };

            // Triangle B: (x1,y1) -> (x0,y1) -> (x1,y0)  (positive orientation)
            math::Vector<Real, 3> w0{ x1, y1, Real(0) };
            math::Vector<Real, 3> w1{ x0, y1, Real(0) };
            math::Vector<Real, 3> w2{ x1, y0, Real(0) };

            const std::array<std::array<math::Vector<Real, 3>, 3>, 2> tris = {{
                {v0, v1, v2},
                {w0, w1, w2}
            }};

            for (const auto& tri : tris) {
                const auto& a0 = tri[0];
                const auto& a1 = tri[1];
                const auto& a2 = tri[2];

                for (std::size_t k = 0; k < geom_ref_nodes.size(); ++k) {
                    geom_nodes[k] = map_affine_tri(geom_ref_nodes[k], a0, a1, a2);
                }

                geometry::IsoparametricMapping mapping(geom_basis, geom_nodes);

                for (std::size_t a = 0; a < field_ref_nodes.size(); ++a) {
                    const auto x_phys = mapping.map_to_physical(field_ref_nodes[a]);
                    coeffs[a] = exact_f_2d(x_phys);
                }

                for (std::size_t q = 0; q < quad.num_points(); ++q) {
                    const auto& xi_q = quad.point(q);
                    const Real w = quad.weight(q);

                    const auto x_phys = mapping.map_to_physical(xi_q);
                    const Real f = exact_f_2d(x_phys);

                    field_basis.evaluate_values(xi_q, phi);
                    Real uh = Real(0);
                    for (std::size_t a = 0; a < coeffs.size(); ++a) {
                        uh += coeffs[a] * phi[a];
                    }

                    const Real diff = f - uh;
                    const Real detJ = std::abs(mapping.jacobian_determinant(xi_q));
                    err2 += w * detJ * diff * diff;
                }
            }
        }
    }

    return std::sqrt(err2);
}

void expect_rate(const std::vector<Real>& errors, int expected_order) {
    ASSERT_GE(errors.size(), 2u);
    for (std::size_t k = 1; k < errors.size(); ++k) {
        EXPECT_LT(errors[k], errors[k - 1]);
    }
    const std::size_t last = errors.size() - 1;
    const double rate_last =
        std::log(static_cast<double>(errors[last - 1] / errors[last])) / std::log(2.0);
    EXPECT_NEAR(rate_last, static_cast<double>(expected_order), 0.35);
}

} // namespace

TEST(ConvergenceRates, QuadLagrangeInterpolationL2) {
    const std::vector<int> refinements = {8, 16, 32};

    for (int p : {1, 2}) {
        std::vector<Real> errors;
        errors.reserve(refinements.size());
        for (int n : refinements) {
            errors.push_back(l2_error_on_quad_mesh(p, n));
        }
        expect_rate(errors, p + 1);
    }
}

TEST(ConvergenceRates, TriangleLagrangeInterpolationL2) {
    const std::vector<int> refinements = {8, 16, 32};

    for (int p : {1, 2}) {
        std::vector<Real> errors;
        errors.reserve(refinements.size());
        for (int n : refinements) {
            errors.push_back(l2_error_on_triangle_mesh(p, n));
        }
        expect_rate(errors, p + 1);
    }
}
