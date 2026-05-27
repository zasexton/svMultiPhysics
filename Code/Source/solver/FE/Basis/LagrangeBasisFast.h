/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_LAGRANGEBASISFAST_H
#define SVMP_FE_BASIS_LAGRANGEBASISFAST_H

/**
 * @file LagrangeBasisFast.h
 * @brief Header-only zero-overhead specializations of the Lagrange basis
 *
 * Provides templated static methods for the common nodal Lagrange families
 * with compile-time-known polynomial order. Callers that know their basis
 * type and order at compile time use these directly — there is no virtual
 * dispatch, no std::vector allocation, no scratch lookup, and no topology
 * switch. The output buffers are stack-allocated std::array, sized at
 * compile time. The compiler fully unrolls and constant-folds.
 *
 * These specializations are an alternative entry point to the runtime path
 * provided by `LagrangeBasis`. The runtime path remains the canonical API
 * for generic callers; these specializations serve hot loops that know the
 * element type.
 *
 * Node orderings match `ReferenceNodeLayout::get_lagrange_node_coords(...)` (VTK).
 */

#include "Core/Types.h"
#include "Math/Vector.h"
#include "Math/Matrix.h"
#include <array>
#include <cstddef>

namespace svmp {
namespace FE {
namespace basis {

using Gradient = math::Vector<Real, 3>;
using Hessian  = math::Matrix<Real, 3, 3>;

namespace detail {

constexpr Gradient scaled_gradient(const Gradient& gradient, Real scale) {
    return Gradient{scale * gradient[0], scale * gradient[1], scale * gradient[2]};
}

constexpr Gradient p2_edge_gradient(Real left,
                                    const Gradient& left_gradient,
                                    Real right,
                                    const Gradient& right_gradient) {
    return Gradient{
        Real(4) * (left_gradient[0] * right + right_gradient[0] * left),
        Real(4) * (left_gradient[1] * right + right_gradient[1] * left),
        Real(4) * (left_gradient[2] * right + right_gradient[2] * left),
    };
}

constexpr Hessian p2_vertex_hessian(const Gradient& gradient) {
    Hessian hessian{};
    for (std::size_t row = 0; row < 3u; ++row) {
        for (std::size_t col = 0; col < 3u; ++col) {
            hessian(row, col) = Real(4) * gradient[row] * gradient[col];
        }
    }
    return hessian;
}

constexpr Hessian p2_edge_hessian(const Gradient& left_gradient,
                                  const Gradient& right_gradient) {
    Hessian hessian{};
    for (std::size_t row = 0; row < 3u; ++row) {
        for (std::size_t col = 0; col < 3u; ++col) {
            hessian(row, col) = Real(4) * (
                left_gradient[row] * right_gradient[col] +
                right_gradient[row] * left_gradient[col]);
        }
    }
    return hessian;
}

} // namespace detail

// ---------------------------------------------------------------------------
// LagrangeLineFast<Order>
// ---------------------------------------------------------------------------
template<int Order>
struct LagrangeLineFast;

template<>
struct LagrangeLineFast<1> {
    static constexpr int n_dofs = 2;

    static constexpr void evaluate(const math::Vector<Real, 3>& xi, std::array<Real, n_dofs>& out) {
        out[0] = (Real(1) - xi[0]) * Real(0.5);
        out[1] = (Real(1) + xi[0]) * Real(0.5);
    }

    static constexpr void evaluate_gradients(const math::Vector<Real, 3>& /*xi*/,
                                             std::array<Gradient, n_dofs>& out) {
        out[0] = Gradient{Real(-0.5), Real(0), Real(0)};
        out[1] = Gradient{Real( 0.5), Real(0), Real(0)};
    }

    static constexpr void evaluate_hessians(const math::Vector<Real, 3>& /*xi*/,
                                            std::array<Hessian, n_dofs>& out) {
        out[0] = Hessian{};
        out[1] = Hessian{};
    }
};

template<>
struct LagrangeLineFast<2> {
    static constexpr int n_dofs = 3;

    static constexpr void evaluate(const math::Vector<Real, 3>& xi, std::array<Real, n_dofs>& out) {
        const Real x = xi[0];
        out[0] = x * (x - Real(1)) * Real(0.5);
        out[1] = x * (x + Real(1)) * Real(0.5);
        out[2] = (Real(1) - x) * (Real(1) + x);
    }

    static constexpr void evaluate_gradients(const math::Vector<Real, 3>& xi,
                                             std::array<Gradient, n_dofs>& out) {
        const Real x = xi[0];
        out[0] = Gradient{x - Real(0.5), Real(0), Real(0)};
        out[1] = Gradient{x + Real(0.5), Real(0), Real(0)};
        out[2] = Gradient{Real(-2) * x, Real(0), Real(0)};
    }

    static constexpr void evaluate_hessians(const math::Vector<Real, 3>& /*xi*/,
                                            std::array<Hessian, n_dofs>& out) {
        out[0] = Hessian{};
        out[1] = Hessian{};
        out[2] = Hessian{};
        out[0](0, 0) = Real(1);
        out[1](0, 0) = Real(1);
        out[2](0, 0) = Real(-2);
    }
};

// ---------------------------------------------------------------------------
// LagrangeQuadFast<Order>
// ---------------------------------------------------------------------------
template<int Order>
struct LagrangeQuadFast;

template<>
struct LagrangeQuadFast<1> {
    static constexpr int n_dofs = 4;

    // VTK Quad4 corner ordering: (-,-), (+,-), (+,+), (-,+).
    static constexpr void evaluate(const math::Vector<Real, 3>& xi, std::array<Real, n_dofs>& out) {
        const Real lx = (Real(1) - xi[0]) * Real(0.5);
        const Real ly = (Real(1) - xi[1]) * Real(0.5);
        const Real ux = (Real(1) + xi[0]) * Real(0.5);
        const Real uy = (Real(1) + xi[1]) * Real(0.5);
        out[0] = lx * ly;
        out[1] = ux * ly;
        out[2] = ux * uy;
        out[3] = lx * uy;
    }

    static constexpr void evaluate_gradients(const math::Vector<Real, 3>& xi,
                                             std::array<Gradient, n_dofs>& out) {
        const Real lx = (Real(1) - xi[0]) * Real(0.5);
        const Real ly = (Real(1) - xi[1]) * Real(0.5);
        const Real ux = (Real(1) + xi[0]) * Real(0.5);
        const Real uy = (Real(1) + xi[1]) * Real(0.5);
        out[0] = Gradient{Real(-0.5) * ly, Real(-0.5) * lx, Real(0)};
        out[1] = Gradient{Real( 0.5) * ly, Real(-0.5) * ux, Real(0)};
        out[2] = Gradient{Real( 0.5) * uy, Real( 0.5) * ux, Real(0)};
        out[3] = Gradient{Real(-0.5) * uy, Real( 0.5) * lx, Real(0)};
    }

    static constexpr void evaluate_hessians(const math::Vector<Real, 3>& /*xi*/,
                                            std::array<Hessian, n_dofs>& out) {
        out[0] = Hessian{};
        out[1] = Hessian{};
        out[2] = Hessian{};
        out[3] = Hessian{};
        constexpr Real qrt = Real(0.25);
        out[0](0, 1) = qrt;  out[0](1, 0) = qrt;
        out[1](0, 1) = -qrt; out[1](1, 0) = -qrt;
        out[2](0, 1) = qrt;  out[2](1, 0) = qrt;
        out[3](0, 1) = -qrt; out[3](1, 0) = -qrt;
    }
};

template<>
struct LagrangeQuadFast<2> {
    static constexpr int n_dofs = 9;

    static constexpr std::array<std::array<std::size_t, 2>, n_dofs> node_axes = {{
        {{0u, 0u}}, {{1u, 0u}}, {{1u, 1u}}, {{0u, 1u}},
        {{2u, 0u}}, {{1u, 2u}}, {{2u, 1u}}, {{0u, 2u}},
        {{2u, 2u}},
    }};

    static void evaluate(const math::Vector<Real, 3>& xi, std::array<Real, n_dofs>& out) {
        std::array<Real, LagrangeLineFast<2>::n_dofs> lx{};
        std::array<Real, LagrangeLineFast<2>::n_dofs> ly{};
        LagrangeLineFast<2>::evaluate({xi[0], Real(0), Real(0)}, lx);
        LagrangeLineFast<2>::evaluate({xi[1], Real(0), Real(0)}, ly);
        for (std::size_t n = 0; n < node_axes.size(); ++n) {
            out[n] = lx[node_axes[n][0]] * ly[node_axes[n][1]];
        }
    }

    static void evaluate_gradients(const math::Vector<Real, 3>& xi,
                                   std::array<Gradient, n_dofs>& out) {
        std::array<Real, LagrangeLineFast<2>::n_dofs> lx{};
        std::array<Real, LagrangeLineFast<2>::n_dofs> ly{};
        std::array<Gradient, LagrangeLineFast<2>::n_dofs> gx{};
        std::array<Gradient, LagrangeLineFast<2>::n_dofs> gy{};
        LagrangeLineFast<2>::evaluate({xi[0], Real(0), Real(0)}, lx);
        LagrangeLineFast<2>::evaluate({xi[1], Real(0), Real(0)}, ly);
        LagrangeLineFast<2>::evaluate_gradients({xi[0], Real(0), Real(0)}, gx);
        LagrangeLineFast<2>::evaluate_gradients({xi[1], Real(0), Real(0)}, gy);
        for (std::size_t n = 0; n < node_axes.size(); ++n) {
            const auto i = node_axes[n][0];
            const auto j = node_axes[n][1];
            out[n] = Gradient{gx[i][0] * ly[j], lx[i] * gy[j][0], Real(0)};
        }
    }

    static void evaluate_hessians(const math::Vector<Real, 3>& xi,
                                  std::array<Hessian, n_dofs>& out) {
        std::array<Real, LagrangeLineFast<2>::n_dofs> lx{};
        std::array<Real, LagrangeLineFast<2>::n_dofs> ly{};
        std::array<Gradient, LagrangeLineFast<2>::n_dofs> gx{};
        std::array<Gradient, LagrangeLineFast<2>::n_dofs> gy{};
        std::array<Hessian, LagrangeLineFast<2>::n_dofs> hx{};
        std::array<Hessian, LagrangeLineFast<2>::n_dofs> hy{};
        LagrangeLineFast<2>::evaluate({xi[0], Real(0), Real(0)}, lx);
        LagrangeLineFast<2>::evaluate({xi[1], Real(0), Real(0)}, ly);
        LagrangeLineFast<2>::evaluate_gradients({xi[0], Real(0), Real(0)}, gx);
        LagrangeLineFast<2>::evaluate_gradients({xi[1], Real(0), Real(0)}, gy);
        LagrangeLineFast<2>::evaluate_hessians({xi[0], Real(0), Real(0)}, hx);
        LagrangeLineFast<2>::evaluate_hessians({xi[1], Real(0), Real(0)}, hy);
        for (std::size_t n = 0; n < node_axes.size(); ++n) {
            const auto i = node_axes[n][0];
            const auto j = node_axes[n][1];
            Hessian H{};
            H(0, 0) = hx[i](0, 0) * ly[j];
            H(1, 1) = lx[i] * hy[j](0, 0);
            H(0, 1) = gx[i][0] * gy[j][0];
            H(1, 0) = H(0, 1);
            out[n] = H;
        }
    }
};

// ---------------------------------------------------------------------------
// LagrangeHexFast<Order>
// ---------------------------------------------------------------------------
template<int Order>
struct LagrangeHexFast;

template<>
struct LagrangeHexFast<1> {
    static constexpr int n_dofs = 8;

    // VTK Hex8 corner ordering: (-,-,-), (+,-,-), (+,+,-), (-,+,-),
    //                           (-,-,+), (+,-,+), (+,+,+), (-,+,+).
    static constexpr void evaluate(const math::Vector<Real, 3>& xi, std::array<Real, n_dofs>& out) {
        const Real lx = (Real(1) - xi[0]) * Real(0.5);
        const Real ly = (Real(1) - xi[1]) * Real(0.5);
        const Real lz = (Real(1) - xi[2]) * Real(0.5);
        const Real ux = (Real(1) + xi[0]) * Real(0.5);
        const Real uy = (Real(1) + xi[1]) * Real(0.5);
        const Real uz = (Real(1) + xi[2]) * Real(0.5);
        // Precompute z-plane partial products (sum factorization).
        const Real lxly = lx * ly;
        const Real uxly = ux * ly;
        const Real uxuy = ux * uy;
        const Real lxuy = lx * uy;
        out[0] = lxly * lz;
        out[1] = uxly * lz;
        out[2] = uxuy * lz;
        out[3] = lxuy * lz;
        out[4] = lxly * uz;
        out[5] = uxly * uz;
        out[6] = uxuy * uz;
        out[7] = lxuy * uz;
    }

    static constexpr void evaluate_gradients(const math::Vector<Real, 3>& xi,
                                             std::array<Gradient, n_dofs>& out) {
        const Real lx = (Real(1) - xi[0]) * Real(0.5);
        const Real ly = (Real(1) - xi[1]) * Real(0.5);
        const Real lz = (Real(1) - xi[2]) * Real(0.5);
        const Real ux = (Real(1) + xi[0]) * Real(0.5);
        const Real uy = (Real(1) + xi[1]) * Real(0.5);
        const Real uz = (Real(1) + xi[2]) * Real(0.5);
        // dL_0(x)/dx = -0.5, dL_1(x)/dx = +0.5 along each axis.
        out[0] = Gradient{Real(-0.5) * ly * lz, Real(-0.5) * lx * lz, Real(-0.5) * lx * ly};
        out[1] = Gradient{Real( 0.5) * ly * lz, Real(-0.5) * ux * lz, Real(-0.5) * ux * ly};
        out[2] = Gradient{Real( 0.5) * uy * lz, Real( 0.5) * ux * lz, Real(-0.5) * ux * uy};
        out[3] = Gradient{Real(-0.5) * uy * lz, Real( 0.5) * lx * lz, Real(-0.5) * lx * uy};
        out[4] = Gradient{Real(-0.5) * ly * uz, Real(-0.5) * lx * uz, Real( 0.5) * lx * ly};
        out[5] = Gradient{Real( 0.5) * ly * uz, Real(-0.5) * ux * uz, Real( 0.5) * ux * ly};
        out[6] = Gradient{Real( 0.5) * uy * uz, Real( 0.5) * ux * uz, Real( 0.5) * ux * uy};
        out[7] = Gradient{Real(-0.5) * uy * uz, Real( 0.5) * lx * uz, Real( 0.5) * lx * uy};
    }

    static constexpr void evaluate_hessians(const math::Vector<Real, 3>& xi,
                                            std::array<Hessian, n_dofs>& out) {
        const Real lx = (Real(1) - xi[0]) * Real(0.5);
        const Real ly = (Real(1) - xi[1]) * Real(0.5);
        const Real lz = (Real(1) - xi[2]) * Real(0.5);
        const Real ux = (Real(1) + xi[0]) * Real(0.5);
        const Real uy = (Real(1) + xi[1]) * Real(0.5);
        const Real uz = (Real(1) + xi[2]) * Real(0.5);
        const Real ax[8] = {lx, ux, ux, lx, lx, ux, ux, lx};
        const Real ay[8] = {ly, ly, uy, uy, ly, ly, uy, uy};
        const Real az[8] = {lz, lz, lz, lz, uz, uz, uz, uz};
        const int sx[8] = {-1, 1, 1, -1, -1, 1, 1, -1};
        const int sy[8] = {-1, -1, 1, 1, -1, -1, 1, 1};
        const int sz[8] = {-1, -1, -1, -1, 1, 1, 1, 1};
        constexpr Real qrt = Real(0.25);
        for (std::size_t n = 0; n < static_cast<std::size_t>(n_dofs); ++n) {
            out[n] = Hessian{};
            out[n](0, 1) = static_cast<Real>(sx[n] * sy[n]) * qrt * az[n];
            out[n](1, 0) = out[n](0, 1);
            out[n](0, 2) = static_cast<Real>(sx[n] * sz[n]) * qrt * ay[n];
            out[n](2, 0) = out[n](0, 2);
            out[n](1, 2) = static_cast<Real>(sy[n] * sz[n]) * qrt * ax[n];
            out[n](2, 1) = out[n](1, 2);
        }
    }
};

template<>
struct LagrangeHexFast<2> {
    static constexpr int n_dofs = 27;

    static constexpr std::array<std::array<std::size_t, 3>, n_dofs> node_axes = {{
        {{0u, 0u, 0u}}, {{1u, 0u, 0u}}, {{1u, 1u, 0u}}, {{0u, 1u, 0u}},
        {{0u, 0u, 1u}}, {{1u, 0u, 1u}}, {{1u, 1u, 1u}}, {{0u, 1u, 1u}},
        {{2u, 0u, 0u}}, {{1u, 2u, 0u}}, {{2u, 1u, 0u}}, {{0u, 2u, 0u}},
        {{2u, 0u, 1u}}, {{1u, 2u, 1u}}, {{2u, 1u, 1u}}, {{0u, 2u, 1u}},
        {{0u, 0u, 2u}}, {{1u, 0u, 2u}}, {{1u, 1u, 2u}}, {{0u, 1u, 2u}},
        {{2u, 2u, 0u}}, {{2u, 2u, 1u}}, {{2u, 0u, 2u}}, {{1u, 2u, 2u}},
        {{2u, 1u, 2u}}, {{0u, 2u, 2u}}, {{2u, 2u, 2u}},
    }};

    static void evaluate(const math::Vector<Real, 3>& xi, std::array<Real, n_dofs>& out) {
        std::array<Real, LagrangeLineFast<2>::n_dofs> lx{};
        std::array<Real, LagrangeLineFast<2>::n_dofs> ly{};
        std::array<Real, LagrangeLineFast<2>::n_dofs> lz{};
        LagrangeLineFast<2>::evaluate({xi[0], Real(0), Real(0)}, lx);
        LagrangeLineFast<2>::evaluate({xi[1], Real(0), Real(0)}, ly);
        LagrangeLineFast<2>::evaluate({xi[2], Real(0), Real(0)}, lz);
        for (std::size_t n = 0; n < node_axes.size(); ++n) {
            out[n] = lx[node_axes[n][0]] * ly[node_axes[n][1]] * lz[node_axes[n][2]];
        }
    }

    static void evaluate_gradients(const math::Vector<Real, 3>& xi,
                                   std::array<Gradient, n_dofs>& out) {
        std::array<Real, LagrangeLineFast<2>::n_dofs> lx{};
        std::array<Real, LagrangeLineFast<2>::n_dofs> ly{};
        std::array<Real, LagrangeLineFast<2>::n_dofs> lz{};
        std::array<Gradient, LagrangeLineFast<2>::n_dofs> gx{};
        std::array<Gradient, LagrangeLineFast<2>::n_dofs> gy{};
        std::array<Gradient, LagrangeLineFast<2>::n_dofs> gz{};
        LagrangeLineFast<2>::evaluate({xi[0], Real(0), Real(0)}, lx);
        LagrangeLineFast<2>::evaluate({xi[1], Real(0), Real(0)}, ly);
        LagrangeLineFast<2>::evaluate({xi[2], Real(0), Real(0)}, lz);
        LagrangeLineFast<2>::evaluate_gradients({xi[0], Real(0), Real(0)}, gx);
        LagrangeLineFast<2>::evaluate_gradients({xi[1], Real(0), Real(0)}, gy);
        LagrangeLineFast<2>::evaluate_gradients({xi[2], Real(0), Real(0)}, gz);
        for (std::size_t n = 0; n < node_axes.size(); ++n) {
            const auto i = node_axes[n][0];
            const auto j = node_axes[n][1];
            const auto k = node_axes[n][2];
            out[n] = Gradient{
                gx[i][0] * ly[j] * lz[k],
                lx[i] * gy[j][0] * lz[k],
                lx[i] * ly[j] * gz[k][0],
            };
        }
    }

    static void evaluate_hessians(const math::Vector<Real, 3>& xi,
                                  std::array<Hessian, n_dofs>& out) {
        std::array<Real, LagrangeLineFast<2>::n_dofs> lx{};
        std::array<Real, LagrangeLineFast<2>::n_dofs> ly{};
        std::array<Real, LagrangeLineFast<2>::n_dofs> lz{};
        std::array<Gradient, LagrangeLineFast<2>::n_dofs> gx{};
        std::array<Gradient, LagrangeLineFast<2>::n_dofs> gy{};
        std::array<Gradient, LagrangeLineFast<2>::n_dofs> gz{};
        std::array<Hessian, LagrangeLineFast<2>::n_dofs> hx{};
        std::array<Hessian, LagrangeLineFast<2>::n_dofs> hy{};
        std::array<Hessian, LagrangeLineFast<2>::n_dofs> hz{};
        LagrangeLineFast<2>::evaluate({xi[0], Real(0), Real(0)}, lx);
        LagrangeLineFast<2>::evaluate({xi[1], Real(0), Real(0)}, ly);
        LagrangeLineFast<2>::evaluate({xi[2], Real(0), Real(0)}, lz);
        LagrangeLineFast<2>::evaluate_gradients({xi[0], Real(0), Real(0)}, gx);
        LagrangeLineFast<2>::evaluate_gradients({xi[1], Real(0), Real(0)}, gy);
        LagrangeLineFast<2>::evaluate_gradients({xi[2], Real(0), Real(0)}, gz);
        LagrangeLineFast<2>::evaluate_hessians({xi[0], Real(0), Real(0)}, hx);
        LagrangeLineFast<2>::evaluate_hessians({xi[1], Real(0), Real(0)}, hy);
        LagrangeLineFast<2>::evaluate_hessians({xi[2], Real(0), Real(0)}, hz);
        for (std::size_t n = 0; n < node_axes.size(); ++n) {
            const auto i = node_axes[n][0];
            const auto j = node_axes[n][1];
            const auto k = node_axes[n][2];
            Hessian H{};
            H(0, 0) = hx[i](0, 0) * ly[j] * lz[k];
            H(1, 1) = lx[i] * hy[j](0, 0) * lz[k];
            H(2, 2) = lx[i] * ly[j] * hz[k](0, 0);
            H(0, 1) = gx[i][0] * gy[j][0] * lz[k];
            H(1, 0) = H(0, 1);
            H(0, 2) = gx[i][0] * ly[j] * gz[k][0];
            H(2, 0) = H(0, 2);
            H(1, 2) = lx[i] * gy[j][0] * gz[k][0];
            H(2, 1) = H(1, 2);
            out[n] = H;
        }
    }
};

// ---------------------------------------------------------------------------
// LagrangeTriFast<Order>
// ---------------------------------------------------------------------------
template<int Order>
struct LagrangeTriFast;

template<>
struct LagrangeTriFast<1> {
    static constexpr int n_dofs = 3;

    static constexpr void evaluate(const math::Vector<Real, 3>& xi, std::array<Real, n_dofs>& out) {
        out[0] = Real(1) - xi[0] - xi[1];
        out[1] = xi[0];
        out[2] = xi[1];
    }

    static constexpr void evaluate_gradients(const math::Vector<Real, 3>& /*xi*/,
                                             std::array<Gradient, n_dofs>& out) {
        out[0] = Gradient{Real(-1), Real(-1), Real(0)};
        out[1] = Gradient{Real( 1), Real( 0), Real(0)};
        out[2] = Gradient{Real( 0), Real( 1), Real(0)};
    }

    static constexpr void evaluate_hessians(const math::Vector<Real, 3>& /*xi*/,
                                            std::array<Hessian, n_dofs>& out) {
        out[0] = Hessian{};
        out[1] = Hessian{};
        out[2] = Hessian{};
    }
};

template<>
struct LagrangeTriFast<2> {
    static constexpr int n_dofs = 6;

    static constexpr void evaluate(const math::Vector<Real, 3>& xi, std::array<Real, n_dofs>& out) {
        const Real l1 = xi[0];
        const Real l2 = xi[1];
        const Real l0 = Real(1) - l1 - l2;

        out[0] = l0 * (Real(2) * l0 - Real(1));
        out[1] = l1 * (Real(2) * l1 - Real(1));
        out[2] = l2 * (Real(2) * l2 - Real(1));
        out[3] = Real(4) * l0 * l1;
        out[4] = Real(4) * l1 * l2;
        out[5] = Real(4) * l0 * l2;
    }

    static constexpr void evaluate_gradients(const math::Vector<Real, 3>& xi,
                                             std::array<Gradient, n_dofs>& out) {
        const Real l1 = xi[0];
        const Real l2 = xi[1];
        const Real l0 = Real(1) - l1 - l2;
        constexpr Gradient g0{Real(-1), Real(-1), Real(0)};
        constexpr Gradient g1{Real( 1), Real( 0), Real(0)};
        constexpr Gradient g2{Real( 0), Real( 1), Real(0)};

        out[0] = detail::scaled_gradient(g0, Real(4) * l0 - Real(1));
        out[1] = detail::scaled_gradient(g1, Real(4) * l1 - Real(1));
        out[2] = detail::scaled_gradient(g2, Real(4) * l2 - Real(1));
        out[3] = detail::p2_edge_gradient(l0, g0, l1, g1);
        out[4] = detail::p2_edge_gradient(l1, g1, l2, g2);
        out[5] = detail::p2_edge_gradient(l0, g0, l2, g2);
    }

    static constexpr void evaluate_hessians(const math::Vector<Real, 3>& /*xi*/,
                                            std::array<Hessian, n_dofs>& out) {
        constexpr Gradient g0{Real(-1), Real(-1), Real(0)};
        constexpr Gradient g1{Real( 1), Real( 0), Real(0)};
        constexpr Gradient g2{Real( 0), Real( 1), Real(0)};

        out[0] = detail::p2_vertex_hessian(g0);
        out[1] = detail::p2_vertex_hessian(g1);
        out[2] = detail::p2_vertex_hessian(g2);
        out[3] = detail::p2_edge_hessian(g0, g1);
        out[4] = detail::p2_edge_hessian(g1, g2);
        out[5] = detail::p2_edge_hessian(g0, g2);
    }
};

// ---------------------------------------------------------------------------
// LagrangeTetFast<Order>
// ---------------------------------------------------------------------------
template<int Order>
struct LagrangeTetFast;

template<>
struct LagrangeTetFast<1> {
    static constexpr int n_dofs = 4;

    static constexpr void evaluate(const math::Vector<Real, 3>& xi, std::array<Real, n_dofs>& out) {
        out[0] = Real(1) - xi[0] - xi[1] - xi[2];
        out[1] = xi[0];
        out[2] = xi[1];
        out[3] = xi[2];
    }

    static constexpr void evaluate_gradients(const math::Vector<Real, 3>& /*xi*/,
                                             std::array<Gradient, n_dofs>& out) {
        out[0] = Gradient{Real(-1), Real(-1), Real(-1)};
        out[1] = Gradient{Real( 1), Real( 0), Real( 0)};
        out[2] = Gradient{Real( 0), Real( 1), Real( 0)};
        out[3] = Gradient{Real( 0), Real( 0), Real( 1)};
    }

    static constexpr void evaluate_hessians(const math::Vector<Real, 3>& /*xi*/,
                                            std::array<Hessian, n_dofs>& out) {
        out[0] = Hessian{};
        out[1] = Hessian{};
        out[2] = Hessian{};
        out[3] = Hessian{};
    }
};

template<>
struct LagrangeTetFast<2> {
    static constexpr int n_dofs = 10;

    static constexpr void evaluate(const math::Vector<Real, 3>& xi, std::array<Real, n_dofs>& out) {
        const Real l1 = xi[0];
        const Real l2 = xi[1];
        const Real l3 = xi[2];
        const Real l0 = Real(1) - l1 - l2 - l3;

        out[0] = l0 * (Real(2) * l0 - Real(1));
        out[1] = l1 * (Real(2) * l1 - Real(1));
        out[2] = l2 * (Real(2) * l2 - Real(1));
        out[3] = l3 * (Real(2) * l3 - Real(1));
        out[4] = Real(4) * l0 * l1;
        out[5] = Real(4) * l1 * l2;
        out[6] = Real(4) * l0 * l2;
        out[7] = Real(4) * l0 * l3;
        out[8] = Real(4) * l1 * l3;
        out[9] = Real(4) * l2 * l3;
    }

    static constexpr void evaluate_gradients(const math::Vector<Real, 3>& xi,
                                             std::array<Gradient, n_dofs>& out) {
        const Real l1 = xi[0];
        const Real l2 = xi[1];
        const Real l3 = xi[2];
        const Real l0 = Real(1) - l1 - l2 - l3;
        constexpr Gradient g0{Real(-1), Real(-1), Real(-1)};
        constexpr Gradient g1{Real( 1), Real( 0), Real( 0)};
        constexpr Gradient g2{Real( 0), Real( 1), Real( 0)};
        constexpr Gradient g3{Real( 0), Real( 0), Real( 1)};

        out[0] = detail::scaled_gradient(g0, Real(4) * l0 - Real(1));
        out[1] = detail::scaled_gradient(g1, Real(4) * l1 - Real(1));
        out[2] = detail::scaled_gradient(g2, Real(4) * l2 - Real(1));
        out[3] = detail::scaled_gradient(g3, Real(4) * l3 - Real(1));
        out[4] = detail::p2_edge_gradient(l0, g0, l1, g1);
        out[5] = detail::p2_edge_gradient(l1, g1, l2, g2);
        out[6] = detail::p2_edge_gradient(l0, g0, l2, g2);
        out[7] = detail::p2_edge_gradient(l0, g0, l3, g3);
        out[8] = detail::p2_edge_gradient(l1, g1, l3, g3);
        out[9] = detail::p2_edge_gradient(l2, g2, l3, g3);
    }

    static constexpr void evaluate_hessians(const math::Vector<Real, 3>& /*xi*/,
                                            std::array<Hessian, n_dofs>& out) {
        constexpr Gradient g0{Real(-1), Real(-1), Real(-1)};
        constexpr Gradient g1{Real( 1), Real( 0), Real( 0)};
        constexpr Gradient g2{Real( 0), Real( 1), Real( 0)};
        constexpr Gradient g3{Real( 0), Real( 0), Real( 1)};

        out[0] = detail::p2_vertex_hessian(g0);
        out[1] = detail::p2_vertex_hessian(g1);
        out[2] = detail::p2_vertex_hessian(g2);
        out[3] = detail::p2_vertex_hessian(g3);
        out[4] = detail::p2_edge_hessian(g0, g1);
        out[5] = detail::p2_edge_hessian(g1, g2);
        out[6] = detail::p2_edge_hessian(g0, g2);
        out[7] = detail::p2_edge_hessian(g0, g3);
        out[8] = detail::p2_edge_hessian(g1, g3);
        out[9] = detail::p2_edge_hessian(g2, g3);
    }
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_LAGRANGEBASISFAST_H
