/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_COLLOCATION_METHODS_H
#define SVMP_FE_TIMESTEPPING_COLLOCATION_METHODS_H

#include "Core/FEException.h"

#include <cstdint>
#include <vector>

namespace svmp {
namespace FE {
namespace timestepping {
namespace collocation {

struct CollocationMethod {
    int stages{0};
    int order{0};
    std::vector<double> c{};
    std::vector<double> ainv{};     // row-major (stages x stages)
    std::vector<double> row_sums{}; // sum_j ainv[i,j]
    std::vector<double> final_w{};  // u_{n+1} = u_n + sum_j final_w[j] * (U_j - u_n)

    bool stiffly_accurate{false};
    int final_stage{0};
};

enum class CollocationFamily : std::uint8_t {
    RadauIIA,
    Gauss
};

struct SecondOrderCollocationData {
    int stages{0};
    int n_constraints{0}; // stages + 2 (u(0), u'(0), stage values)

    // p'(c_i) and p''(c_i) (derivatives in τ-space) as linear combinations of constraints:
    //   y = [u(0), dt*u'(0), U_0, ..., U_{s-1}]
    //   p'(c_i)  = d1_u0[i] * y0 + d1_dv0[i] * y1 + sum_j d1[i,j] * U_j
    //   p''(c_i) = d2_u0[i] * y0 + d2_dv0[i] * y1 + sum_j d2[i,j] * U_j
    std::vector<double> d1{}; // size stages*stages
    std::vector<double> d2{}; // size stages*stages
    std::vector<double> d1_u0{};  // size stages
    std::vector<double> d1_dv0{}; // size stages
    std::vector<double> d2_u0{};  // size stages
    std::vector<double> d2_dv0{}; // size stages

    // p(1), p'(1), p''(1) in τ-space as linear combinations of constraints y.
    std::vector<double> u1{};   // coefficients on U_j (size stages)
    std::vector<double> du1{};  // coefficients on U_j (size stages)
    std::vector<double> ddu1{}; // coefficients on U_j (size stages)
    double u1_u0{0.0};
    double u1_dv0{0.0};
    double du1_u0{0.0};
    double du1_dv0{0.0};
    double ddu1_u0{0.0};
    double ddu1_dv0{0.0};
};

[[nodiscard]] std::vector<double> gaussNodesUnit(int stages);
[[nodiscard]] std::vector<double> radauIIANodesUnit(int stages);

// Gaussian-elimination inverse for small dense matrices. Intended for collocation tableau sizes.
[[nodiscard]] std::vector<double> invertDenseMatrix(const std::vector<double>& A, int n);

[[nodiscard]] CollocationMethod buildCollocationMethod(CollocationFamily family, int stages);
[[nodiscard]] SecondOrderCollocationData buildSecondOrderCollocationData(const CollocationMethod& method);

} // namespace collocation
} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_COLLOCATION_METHODS_H

