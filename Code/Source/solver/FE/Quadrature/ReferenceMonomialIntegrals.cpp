/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "ReferenceMonomialIntegrals.h"

#include "Core/FEException.h"

namespace svmp {
namespace FE {
namespace quadrature {
namespace reference_integrals {

namespace {

void check_nonnegative(bool condition, const char* message) {
    if (!condition) {
        throw FEException(message, __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }
}

Real factorial_int(int n) {
    check_nonnegative(n >= 0, "factorial_int: negative argument");
    Real result = Real(1);
    for (int k = 2; k <= n; ++k) {
        result *= static_cast<Real>(k);
    }
    return result;
}

} // namespace

Real integral_monomial_1d(int p) {
    check_nonnegative(p >= 0, "integral_monomial_1d: negative power");
    if (p % 2 != 0) {
        return Real(0);
    }
    return Real(2) / static_cast<Real>(p + 1);
}

Real integral_triangle_monomial(int px, int py) {
    check_nonnegative(px >= 0 && py >= 0,
                      "integral_triangle_monomial: negative power");
    return factorial_int(px) * factorial_int(py) / factorial_int(px + py + 2);
}

Real integral_tetra_monomial(int px, int py, int pz) {
    check_nonnegative(px >= 0 && py >= 0 && pz >= 0,
                      "integral_tetra_monomial: negative power");
    return factorial_int(px) * factorial_int(py) * factorial_int(pz) /
           factorial_int(px + py + pz + 3);
}

Real integral_pyramid_z(int power) {
    check_nonnegative(power >= 0, "integral_pyramid_z: negative power");
    return Real(1) / static_cast<Real>(power + 1);
}

Real integral_wedge_monomial(int px, int py, int pz) {
    return integral_triangle_monomial(px, py) * integral_monomial_1d(pz);
}

} // namespace reference_integrals
} // namespace quadrature
} // namespace FE
} // namespace svmp
