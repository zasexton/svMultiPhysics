/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file test_FormSpatialDerivatives.cpp
 * @brief Unit tests for composite grad/div/H evaluation via spatial jets
 */

#include <gtest/gtest.h>

#include "Assembly/GlobalSystemView.h"
#include "Assembly/StandardAssembler.h"
#include "Forms/FormCompiler.h"
#include "Forms/FormKernels.h"
#include "Spaces/L2Space.h"
#include "Tests/Unit/Forms/FormsTestHelpers.h"

#include <array>
#include <cmath>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace test {

namespace {

dofs::DofMap createSingleTetraP0DofMap()
{
    dofs::DofMap dof_map(1, 1, 1);
    std::vector<GlobalIndex> cell_dofs = {0};
    dof_map.setCellDofs(0, cell_dofs);
    dof_map.setNumDofs(1);
    dof_map.setNumLocalDofs(1);
    dof_map.finalize();
    return dof_map;
}

Real assembleErrorIntegral(const FormExpr& error_expr)
{
    SingleTetraMeshAccess mesh;
    auto dof_map = createSingleTetraP0DofMap();
    spaces::L2Space space(ElementType::Tetra4, 0);

    FormCompiler compiler;
    const auto v = FormExpr::testFunction(space, "v");
    auto ir = compiler.compileLinear((error_expr * v).dx());
    FormKernel kernel(std::move(ir));

    assembly::StandardAssembler assembler;
    assembler.setDofMap(dof_map);

    assembly::DenseVectorView vec(1);
    vec.zero();
    (void)assembler.assembleVector(mesh, space, kernel, vec);
    return vec.getVectorEntry(0);
}

} // namespace

TEST(FormSpatialDerivativesTest, GradCompositeMatchesFiniteDifferences)
{
    constexpr Real h = 1e-7;

    ScalarCoefficient u_fn = [](Real x, Real y, Real z) {
        return 0.25 + x - 0.5 * y + 0.75 * z;
    };

    const auto u = FormExpr::coefficient("u", u_fn);
    const auto g = exp(u);

    VectorCoefficient grad_fd_fn = [=](Real x, Real y, Real z) {
        auto eval = [&](Real X, Real Y, Real Z) -> Real {
            return std::exp(u_fn(X, Y, Z));
        };
        std::array<Real, 3> grad{0.0, 0.0, 0.0};
        grad[0] = (eval(x + h, y, z) - eval(x - h, y, z)) / (2.0 * h);
        grad[1] = (eval(x, y + h, z) - eval(x, y - h, z)) / (2.0 * h);
        grad[2] = (eval(x, y, z + h) - eval(x, y, z - h)) / (2.0 * h);
        return grad;
    };
    const auto grad_fd = FormExpr::coefficient("grad_g_fd", grad_fd_fn);

    const auto diff = grad(g) - grad_fd;
    const auto error = inner(diff, diff);

    const Real err_int = assembleErrorIntegral(error);
    EXPECT_LT(err_int, 1e-12);
}

TEST(FormSpatialDerivativesTest, GradPowCompositeMatchesFiniteDifferences)
{
    constexpr Real h = 1e-7;

    ScalarCoefficient u_fn = [](Real x, Real y, Real z) {
        return 0.25 + x - 0.5 * y + 0.75 * z;
    };

    const auto u = FormExpr::coefficient("u", u_fn);
    const auto base = FormExpr::constant(1.0) + u * u;
    const auto exponent = FormExpr::constant(1.25);
    const auto g = pow(base, exponent);

    VectorCoefficient grad_fd_fn = [=](Real x, Real y, Real z) {
        auto eval = [&](Real X, Real Y, Real Z) -> Real {
            const Real uu = u_fn(X, Y, Z);
            return std::pow(1.0 + uu * uu, 1.25);
        };
        std::array<Real, 3> grad{0.0, 0.0, 0.0};
        grad[0] = (eval(x + h, y, z) - eval(x - h, y, z)) / (2.0 * h);
        grad[1] = (eval(x, y + h, z) - eval(x, y - h, z)) / (2.0 * h);
        grad[2] = (eval(x, y, z + h) - eval(x, y, z - h)) / (2.0 * h);
        return grad;
    };
    const auto grad_fd = FormExpr::coefficient("grad_g_fd", grad_fd_fn);

    const auto diff = grad(g) - grad_fd;
    const auto error = inner(diff, diff);

    const Real err_int = assembleErrorIntegral(error);
    EXPECT_LT(err_int, 1e-12);
}

TEST(FormSpatialDerivativesTest, URISExponentialMollifierMatchesReference)
{
    constexpr Real pi = 3.14159265358979323846;
    constexpr Real eps = 0.2;

    const std::array<Real, 4> sdf_abs = {0.02, 0.12, 0.31, 0.18};

    auto dist_eval = [=](Real x, Real y, Real z) -> Real {
        const Real N0 = 1.0 - x - y - z;
        return sdf_abs[0] * N0 + sdf_abs[1] * x + sdf_abs[2] * y + sdf_abs[3] * z;
    };

    ScalarCoefficient dist_fn = dist_eval;

    ScalarCoefficient ddir_ref_fn = [=](Real x, Real y, Real z) {
        const Real dist = dist_eval(x, y, z);
        if (dist <= eps) {
            return (1.0 + std::exp(-pi * dist / eps)) / (2.0 * eps * eps);
        }
        return 0.0;
    };

    const auto dist = FormExpr::coefficient("dist", dist_fn);
    const auto ddir_ref = FormExpr::coefficient("ddir_ref", ddir_ref_fn);

    const auto one = FormExpr::constant(1.0);
    const auto zero = FormExpr::constant(0.0);
    const auto two = FormExpr::constant(2.0);
    const auto eps_c = FormExpr::constant(eps);
    const auto pi_c = FormExpr::constant(pi);

    const auto arg = pi_c * dist / eps_c;
    const auto ddir_inside = (one + exp(-arg)) / (two * eps_c * eps_c);
    const auto ddir = conditional(le(dist, eps_c), ddir_inside, zero);

    const auto diff = ddir - ddir_ref;
    const auto error = diff * diff;

    const Real err_int = assembleErrorIntegral(error);
    EXPECT_LT(err_int, 1e-12);
}

TEST(FormSpatialDerivativesTest, HessianScalarCompositeMatchesFiniteDifferences)
{
    constexpr Real h = 1e-5;

    ScalarCoefficient u_fn = [](Real x, Real y, Real z) {
        return 1.0 + x * x + 2.0 * y * y + 3.0 * z * z;
    };

    const auto u = FormExpr::coefficient("u", u_fn);
    const auto g = sqrt(FormExpr::constant(1.0) + u * u);
    const auto Hg = g.hessian();

    MatrixCoefficient H_fd_fn = [=](Real x, Real y, Real z) {
        auto eval = [&](Real X, Real Y, Real Z) -> Real {
            const Real uu = u_fn(X, Y, Z);
            return std::sqrt(1.0 + uu * uu);
        };

        const Real f0 = eval(x, y, z);
        std::array<std::array<Real, 3>, 3> H{};

        const std::array<Real, 3> base{x, y, z};

        // Diagonal terms
        for (int i = 0; i < 3; ++i) {
            auto xp = base;
            auto xm = base;
            xp[static_cast<std::size_t>(i)] += h;
            xm[static_cast<std::size_t>(i)] -= h;
            const Real fp = eval(xp[0], xp[1], xp[2]);
            const Real fm = eval(xm[0], xm[1], xm[2]);
            H[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = (fp - 2.0 * f0 + fm) / (h * h);
        }

        // Mixed terms
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                auto xpp = base;
                auto xpm = base;
                auto xmp = base;
                auto xmm = base;
                xpp[static_cast<std::size_t>(i)] += h;
                xpp[static_cast<std::size_t>(j)] += h;
                xpm[static_cast<std::size_t>(i)] += h;
                xpm[static_cast<std::size_t>(j)] -= h;
                xmp[static_cast<std::size_t>(i)] -= h;
                xmp[static_cast<std::size_t>(j)] += h;
                xmm[static_cast<std::size_t>(i)] -= h;
                xmm[static_cast<std::size_t>(j)] -= h;

                const Real fpp = eval(xpp[0], xpp[1], xpp[2]);
                const Real fpm = eval(xpm[0], xpm[1], xpm[2]);
                const Real fmp = eval(xmp[0], xmp[1], xmp[2]);
                const Real fmm = eval(xmm[0], xmm[1], xmm[2]);

                const Real val = (fpp - fpm - fmp + fmm) / (4.0 * h * h);
                H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = val;
                H[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = val;
            }
        }

        return H;
    };
    const auto H_fd = FormExpr::coefficient("Hg_fd", H_fd_fn);

    const auto diff = Hg - H_fd;
    const auto error = inner(diff, diff);

    const Real err_int = assembleErrorIntegral(error);
    EXPECT_LT(err_int, 1e-8);
}

TEST(FormSpatialDerivativesTest, HessianVectorCompositeMatchesFiniteDifferences)
{
    constexpr Real h = 1e-5;

    VectorCoefficient u_fn = [](Real x, Real y, Real z) {
        return std::array<Real, 3>{1.0 + x * x, 2.0 + y * y, 3.0 + z * z};
    };

    auto w_fn = [=](Real x, Real y, Real z) -> std::array<Real, 3> {
        const auto u = u_fn(x, y, z);
        const Real n = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        const Real denom = 1.0 + n;
        return std::array<Real, 3>{u[0] / denom, u[1] / denom, u[2] / denom};
    };

    const auto u = FormExpr::coefficient("u", u_fn);
    const auto w = u / (FormExpr::constant(1.0) + norm(u));
    const auto Hw = w.hessian();

    Tensor3Coefficient H_fd_fn = [=](Real x, Real y, Real z) {
        std::array<Real, 27> H{};
        const std::array<Real, 3> base{x, y, z};
        const auto f0 = w_fn(x, y, z);

        auto eval = [&](const std::array<Real, 3>& p) -> std::array<Real, 3> {
            return w_fn(p[0], p[1], p[2]);
        };

        // Diagonal second derivatives
        for (int i = 0; i < 3; ++i) {
            auto xp = base;
            auto xm = base;
            xp[static_cast<std::size_t>(i)] += h;
            xm[static_cast<std::size_t>(i)] -= h;
            const auto fp = eval(xp);
            const auto fm = eval(xm);
            for (int comp = 0; comp < 3; ++comp) {
                const std::size_t idx = static_cast<std::size_t>((comp * 3 + i) * 3 + i);
                H[idx] = (fp[static_cast<std::size_t>(comp)] - 2.0 * f0[static_cast<std::size_t>(comp)] +
                          fm[static_cast<std::size_t>(comp)]) /
                         (h * h);
            }
        }

        // Mixed second derivatives
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                auto xpp = base;
                auto xpm = base;
                auto xmp = base;
                auto xmm = base;
                xpp[static_cast<std::size_t>(i)] += h;
                xpp[static_cast<std::size_t>(j)] += h;
                xpm[static_cast<std::size_t>(i)] += h;
                xpm[static_cast<std::size_t>(j)] -= h;
                xmp[static_cast<std::size_t>(i)] -= h;
                xmp[static_cast<std::size_t>(j)] += h;
                xmm[static_cast<std::size_t>(i)] -= h;
                xmm[static_cast<std::size_t>(j)] -= h;

                const auto fpp = eval(xpp);
                const auto fpm = eval(xpm);
                const auto fmp = eval(xmp);
                const auto fmm = eval(xmm);

                for (int comp = 0; comp < 3; ++comp) {
                    const Real val =
                        (fpp[static_cast<std::size_t>(comp)] - fpm[static_cast<std::size_t>(comp)] -
                         fmp[static_cast<std::size_t>(comp)] + fmm[static_cast<std::size_t>(comp)]) /
                        (4.0 * h * h);
                    const std::size_t idx_ij = static_cast<std::size_t>((comp * 3 + i) * 3 + j);
                    const std::size_t idx_ji = static_cast<std::size_t>((comp * 3 + j) * 3 + i);
                    H[idx_ij] = val;
                    H[idx_ji] = val;
                }
            }
        }

        return H;
    };
    const auto H_fd = FormExpr::coefficient("Hw_fd", H_fd_fn);

    const auto diff = Hw - H_fd;
    const auto error = inner(diff, diff);

    const Real err_int = assembleErrorIntegral(error);
    EXPECT_LT(err_int, 1e-6);
}

TEST(FormSpatialDerivativesTest, DivergenceMatrixCompositeMatchesFiniteDifferences)
{
    constexpr Real h = 1e-6;

    VectorCoefficient a_fn = [](Real x, Real y, Real z) {
        return std::array<Real, 3>{x + y, y + z, z + x};
    };
    VectorCoefficient b_fn = [](Real x, Real y, Real z) {
        return std::array<Real, 3>{1.0 + x * x, 2.0 + y, 3.0 + z * z};
    };

    auto A_fn = [=](Real x, Real y, Real z) -> std::array<std::array<Real, 3>, 3> {
        const auto a = a_fn(x, y, z);
        const auto b = b_fn(x, y, z);
        std::array<std::array<Real, 3>, 3> A{};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    a[static_cast<std::size_t>(i)] * b[static_cast<std::size_t>(j)];
            }
        }
        return A;
    };

    VectorCoefficient div_fd_fn = [=](Real x, Real y, Real z) {
        const std::array<Real, 3> base{x, y, z};
        std::array<Real, 3> div{0.0, 0.0, 0.0};
        for (int d = 0; d < 3; ++d) {
            auto xp = base;
            auto xm = base;
            xp[static_cast<std::size_t>(d)] += h;
            xm[static_cast<std::size_t>(d)] -= h;
            const auto Ap = A_fn(xp[0], xp[1], xp[2]);
            const auto Am = A_fn(xm[0], xm[1], xm[2]);
            for (int i = 0; i < 3; ++i) {
                div[static_cast<std::size_t>(i)] +=
                    (Ap[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)] -
                     Am[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)]) /
                    (2.0 * h);
            }
        }
        return div;
    };

    const auto a = FormExpr::coefficient("a", a_fn);
    const auto b = FormExpr::coefficient("b", b_fn);
    const auto divA = div(outer(a, b));
    const auto div_fd = FormExpr::coefficient("divA_fd", div_fd_fn);

    const auto diff = divA - div_fd;
    const auto error = inner(diff, diff);

    const Real err_int = assembleErrorIntegral(error);
    EXPECT_LT(err_int, 1e-8);
}

TEST(FormSpatialDerivativesTest, NestedDerivativesMatchFiniteDifferences)
{
    constexpr Real h = 1e-5;

    ScalarCoefficient u_fn = [](Real x, Real y, Real z) {
        return x * x + 2.0 * y * y + 3.0 * z * z;
    };
    const auto u = FormExpr::coefficient("u", u_fn);
    const auto lap_expr = div(grad(u));

    ScalarCoefficient lap_fd_fn = [=](Real x, Real y, Real z) {
        auto eval = [&](Real X, Real Y, Real Z) -> Real { return u_fn(X, Y, Z); };
        const Real f0 = eval(x, y, z);
        Real lap = 0.0;
        lap += (eval(x + h, y, z) - 2.0 * f0 + eval(x - h, y, z)) / (h * h);
        lap += (eval(x, y + h, z) - 2.0 * f0 + eval(x, y - h, z)) / (h * h);
        lap += (eval(x, y, z + h) - 2.0 * f0 + eval(x, y, z - h)) / (h * h);
        return lap;
    };
    const auto lap_fd = FormExpr::coefficient("lap_fd", lap_fd_fn);

    const auto lap_err = (lap_expr - lap_fd);
    const auto lap_error = lap_err * lap_err;

    const Real lap_err_int = assembleErrorIntegral(lap_error);
    EXPECT_LT(lap_err_int, 1e-10);

    VectorCoefficient v_fn = [](Real x, Real y, Real z) {
        return std::array<Real, 3>{x * x, y * y, z * z};
    };
    const auto v = FormExpr::coefficient("v", v_fn);
    const auto grad_div_expr = grad(div(v));

    VectorCoefficient grad_div_fd_fn = [=](Real x, Real y, Real z) {
        auto comp = [&](int k, Real X, Real Y, Real Z) -> Real {
            const auto vv = v_fn(X, Y, Z);
            return vv[static_cast<std::size_t>(k)];
        };

        auto hess_scalar = [&](int k) -> std::array<std::array<Real, 3>, 3> {
            std::array<std::array<Real, 3>, 3> H{};
            const std::array<Real, 3> base{x, y, z};
            const Real f0 = comp(k, x, y, z);

            for (int i = 0; i < 3; ++i) {
                auto xp = base;
                auto xm = base;
                xp[static_cast<std::size_t>(i)] += h;
                xm[static_cast<std::size_t>(i)] -= h;
                const Real fp = comp(k, xp[0], xp[1], xp[2]);
                const Real fm = comp(k, xm[0], xm[1], xm[2]);
                H[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = (fp - 2.0 * f0 + fm) / (h * h);
            }

            for (int i = 0; i < 3; ++i) {
                for (int j = i + 1; j < 3; ++j) {
                    auto xpp = base;
                    auto xpm = base;
                    auto xmp = base;
                    auto xmm = base;
                    xpp[static_cast<std::size_t>(i)] += h;
                    xpp[static_cast<std::size_t>(j)] += h;
                    xpm[static_cast<std::size_t>(i)] += h;
                    xpm[static_cast<std::size_t>(j)] -= h;
                    xmp[static_cast<std::size_t>(i)] -= h;
                    xmp[static_cast<std::size_t>(j)] += h;
                    xmm[static_cast<std::size_t>(i)] -= h;
                    xmm[static_cast<std::size_t>(j)] -= h;
                    const Real fpp = comp(k, xpp[0], xpp[1], xpp[2]);
                    const Real fpm = comp(k, xpm[0], xpm[1], xpm[2]);
                    const Real fmp = comp(k, xmp[0], xmp[1], xmp[2]);
                    const Real fmm = comp(k, xmm[0], xmm[1], xmm[2]);
                    const Real val = (fpp - fpm - fmp + fmm) / (4.0 * h * h);
                    H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = val;
                    H[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = val;
                }
            }
            return H;
        };

        const auto H0 = hess_scalar(0);
        const auto H1 = hess_scalar(1);
        const auto H2 = hess_scalar(2);

        std::array<Real, 3> g{0.0, 0.0, 0.0};
        for (int i = 0; i < 3; ++i) {
            g[static_cast<std::size_t>(i)] =
                H0[static_cast<std::size_t>(i)][0] +
                H1[static_cast<std::size_t>(i)][1] +
                H2[static_cast<std::size_t>(i)][2];
        }
        return g;
    };
    const auto grad_div_fd = FormExpr::coefficient("grad_div_fd", grad_div_fd_fn);

    const auto gdiff = grad_div_expr - grad_div_fd;
    const auto gerror = inner(gdiff, gdiff);

    const Real grad_div_err_int = assembleErrorIntegral(gerror);
    EXPECT_LT(grad_div_err_int, 1e-6);
}

TEST(FormSpatialDerivativesTest, CurlOfMatrixCompositeMatchesReference)
{
    // A(x) = x ⊗ x, with A_{ij} = x_i x_j.
    // Define curl row-wise: (curl A)_{i*} = curl(A_{i*}).
    // For i=0, (curl A)_{0,1} = -x_2 on a 3D domain.
    const auto x = FormExpr::coordinate();
    const auto A = outer(x, x);
    const auto c01 = component(curl(A), 0, 1);
    const auto z = component(x, 2);

    const auto diff = c01 + z;
    const auto error = diff * diff;

    const Real err_int = assembleErrorIntegral(error);
    EXPECT_LT(err_int, 1e-12);
}

} // namespace test
} // namespace forms
} // namespace FE
} // namespace svmp
