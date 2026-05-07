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
 * @brief Header-only zero-overhead specializations of the Lagrange basis (D4)
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
 * for generic callers; D4 is for hot loops that know the element type.
 *
 * Node orderings match `NodeOrdering::get_lagrange_node_coords(...)` (VTK).
 */

#include "Core/Types.h"
#include "Math/Vector.h"
#include "Math/Matrix.h"
#include <array>

namespace svmp {
namespace FE {
namespace basis {

using Gradient = math::Vector<Real, 3>;
using Hessian  = math::Matrix<Real, 3, 3>;

// ---------------------------------------------------------------------------
// LagrangeLineFast<Order>
// ---------------------------------------------------------------------------
template<int Order>
struct LagrangeLineFast;

template<>
struct LagrangeLineFast<1> {
    static constexpr int n_dofs = 2;

    static void evaluate(const math::Vector<Real, 3>& xi, std::array<Real, n_dofs>& out) {
        out[0] = (Real(1) - xi[0]) * Real(0.5);
        out[1] = (Real(1) + xi[0]) * Real(0.5);
    }

    static void evaluate_gradients(const math::Vector<Real, 3>& /*xi*/,
                                   std::array<Gradient, n_dofs>& out) {
        out[0] = Gradient{Real(-0.5), Real(0), Real(0)};
        out[1] = Gradient{Real( 0.5), Real(0), Real(0)};
    }
};

template<>
struct LagrangeLineFast<2> {
    static constexpr int n_dofs = 3;

    static void evaluate(const math::Vector<Real, 3>& xi, std::array<Real, n_dofs>& out) {
        const Real x = xi[0];
        out[0] = x * (x - Real(1)) * Real(0.5);   // L_0 at -1
        out[1] = (Real(1) - x) * (Real(1) + x);   // L_1 at  0
        out[2] = x * (x + Real(1)) * Real(0.5);   // L_2 at +1
    }

    static void evaluate_gradients(const math::Vector<Real, 3>& xi,
                                   std::array<Gradient, n_dofs>& out) {
        const Real x = xi[0];
        out[0] = Gradient{x - Real(0.5), Real(0), Real(0)};
        out[1] = Gradient{Real(-2) * x, Real(0), Real(0)};
        out[2] = Gradient{x + Real(0.5), Real(0), Real(0)};
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
    static void evaluate(const math::Vector<Real, 3>& xi, std::array<Real, n_dofs>& out) {
        const Real lx = (Real(1) - xi[0]) * Real(0.5);
        const Real ly = (Real(1) - xi[1]) * Real(0.5);
        const Real ux = (Real(1) + xi[0]) * Real(0.5);
        const Real uy = (Real(1) + xi[1]) * Real(0.5);
        out[0] = lx * ly;
        out[1] = ux * ly;
        out[2] = ux * uy;
        out[3] = lx * uy;
    }

    static void evaluate_gradients(const math::Vector<Real, 3>& xi,
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
    static void evaluate(const math::Vector<Real, 3>& xi, std::array<Real, n_dofs>& out) {
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

    static void evaluate_gradients(const math::Vector<Real, 3>& xi,
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
};

// ---------------------------------------------------------------------------
// LagrangeTriFast<Order>
// ---------------------------------------------------------------------------
template<int Order>
struct LagrangeTriFast;

template<>
struct LagrangeTriFast<1> {
    static constexpr int n_dofs = 3;

    static void evaluate(const math::Vector<Real, 3>& xi, std::array<Real, n_dofs>& out) {
        out[0] = Real(1) - xi[0] - xi[1];
        out[1] = xi[0];
        out[2] = xi[1];
    }

    static void evaluate_gradients(const math::Vector<Real, 3>& /*xi*/,
                                   std::array<Gradient, n_dofs>& out) {
        out[0] = Gradient{Real(-1), Real(-1), Real(0)};
        out[1] = Gradient{Real( 1), Real( 0), Real(0)};
        out[2] = Gradient{Real( 0), Real( 1), Real(0)};
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

    static void evaluate(const math::Vector<Real, 3>& xi, std::array<Real, n_dofs>& out) {
        out[0] = Real(1) - xi[0] - xi[1] - xi[2];
        out[1] = xi[0];
        out[2] = xi[1];
        out[3] = xi[2];
    }

    static void evaluate_gradients(const math::Vector<Real, 3>& /*xi*/,
                                   std::array<Gradient, n_dofs>& out) {
        out[0] = Gradient{Real(-1), Real(-1), Real(-1)};
        out[1] = Gradient{Real( 1), Real( 0), Real( 0)};
        out[2] = Gradient{Real( 0), Real( 1), Real( 0)};
        out[3] = Gradient{Real( 0), Real( 0), Real( 1)};
    }
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_LAGRANGEBASISFAST_H
