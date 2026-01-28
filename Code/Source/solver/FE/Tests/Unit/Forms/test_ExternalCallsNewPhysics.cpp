/**
 * @file test_ExternalCallsNewPhysics.cpp
 * @brief Unit tests for versioned ExternalCalls helpers used by new-physics FormExpr nodes
 */

#include <gtest/gtest.h>

#include "Forms/JIT/ExternalCalls.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace {

template <std::size_t N>
using Mat = std::array<double, N * N>;

template <std::size_t N>
using Vec = std::array<double, N>;

template <std::size_t N>
[[nodiscard]] Mat<N> addScaled(const Mat<N>& A, const Mat<N>& dA, double h)
{
    Mat<N> out{};
    for (std::size_t i = 0; i < N * N; ++i) out[i] = A[i] + h * dA[i];
    return out;
}

template <std::size_t N>
[[nodiscard]] Mat<N> finiteDifference(const Mat<N>& f_plus, const Mat<N>& f_minus, double h)
{
    Mat<N> out{};
    const double inv = 1.0 / (2.0 * h);
    for (std::size_t i = 0; i < N * N; ++i) out[i] = (f_plus[i] - f_minus[i]) * inv;
    return out;
}

[[nodiscard]] double tolRelAbs(double ref, double tol) noexcept
{
    return tol * (1.0 + std::abs(ref));
}

template <std::size_t N>
void expectNear(const Mat<N>& a, const Mat<N>& b, double tol)
{
    for (std::size_t i = 0; i < N * N; ++i) {
        EXPECT_NEAR(a[i], b[i], tolRelAbs(a[i], tol));
    }
}

template <std::size_t N>
void alignVectorSign(const Vec<N>& ref, Vec<N>& v)
{
    double dot = 0.0;
    for (std::size_t i = 0; i < N; ++i) dot += ref[i] * v[i];
    if (dot < 0.0) {
        for (std::size_t i = 0; i < N; ++i) v[i] = -v[i];
    }
}

template <std::size_t N>
void alignEigenvectorSigns(const Mat<N>& Qref, Mat<N>& Q)
{
    for (std::size_t col = 0; col < N; ++col) {
        double dot = 0.0;
        for (std::size_t r = 0; r < N; ++r) {
            dot += Qref[r * N + col] * Q[r * N + col];
        }
        if (dot < 0.0) {
            for (std::size_t r = 0; r < N; ++r) {
                Q[r * N + col] = -Q[r * N + col];
            }
        }
    }
}

template <std::size_t N>
[[nodiscard]] Mat<N> eigSymQ(const Mat<N>& A)
{
    Mat<N> Q{};
    if constexpr (N == 2) {
        double evals[2]{};
        double evecs[4]{};
        svmp::FE::forms::jit::svmp_fe_jit_eig_sym_2x2_v1(A.data(), evals, evecs);
        for (std::size_t i = 0; i < 4; ++i) Q[i] = evecs[i];
    } else if constexpr (N == 3) {
        double evals[3]{};
        double evecs[9]{};
        svmp::FE::forms::jit::svmp_fe_jit_eig_sym_3x3_v1(A.data(), evals, evecs);
        for (std::size_t i = 0; i < 9; ++i) Q[i] = evecs[i];
    }
    return Q;
}

template <std::size_t N>
[[nodiscard]] Vec<N> eigSymVec(const Mat<N>& A, std::int32_t which)
{
    const auto Q = eigSymQ<N>(A);
    which = std::max<std::int32_t>(0, std::min<std::int32_t>(which, static_cast<std::int32_t>(N - 1)));
    Vec<N> v{};
    for (std::size_t r = 0; r < N; ++r) v[r] = Q[r * N + static_cast<std::size_t>(which)];
    return v;
}

} // namespace

TEST(ExternalCallsNewPhysics, MatrixExpDD2x2MatchesFiniteDifference)
{
    const Mat<2> A{2.0, 0.3,
                   0.3, 1.7};
    const Mat<2> dA{0.1, -0.05,
                    -0.05, 0.2};

    const double h = 1e-6;
    const auto Ap = addScaled<2>(A, dA, h);
    const auto Am = addScaled<2>(A, dA, -h);

    Mat<2> fAp{};
    Mat<2> fAm{};
    svmp::FE::forms::jit::svmp_fe_jit_matrix_exp_2x2_v1(Ap.data(), fAp.data());
    svmp::FE::forms::jit::svmp_fe_jit_matrix_exp_2x2_v1(Am.data(), fAm.data());

    const auto fd = finiteDifference<2>(fAp, fAm, h);

    Mat<2> dd{};
    svmp::FE::forms::jit::svmp_fe_jit_matrix_exp_dd_2x2_v1(A.data(), dA.data(), dd.data());

    expectNear<2>(fd, dd, 5e-6);
}

TEST(ExternalCallsNewPhysics, MatrixLogDD3x3MatchesFiniteDifference)
{
    const Mat<3> A{
        2.5, 0.1, 0.0,
        0.1, 1.8, 0.2,
        0.0, 0.2, 1.2,
    };
    const Mat<3> dA{
        0.05, -0.02, 0.01,
        -0.02, 0.03, 0.0,
        0.01, 0.0, -0.04,
    };

    const double h = 1e-6;
    const auto Ap = addScaled<3>(A, dA, h);
    const auto Am = addScaled<3>(A, dA, -h);

    Mat<3> fAp{};
    Mat<3> fAm{};
    svmp::FE::forms::jit::svmp_fe_jit_matrix_log_3x3_v1(Ap.data(), fAp.data());
    svmp::FE::forms::jit::svmp_fe_jit_matrix_log_3x3_v1(Am.data(), fAm.data());

    const auto fd = finiteDifference<3>(fAp, fAm, h);

    Mat<3> dd{};
    svmp::FE::forms::jit::svmp_fe_jit_matrix_log_dd_3x3_v1(A.data(), dA.data(), dd.data());

    expectNear<3>(fd, dd, 5e-6);
}

TEST(ExternalCallsNewPhysics, MatrixPowDD2x2MatchesFiniteDifference)
{
    const Mat<2> A{2.0, 0.1,
                   0.1, 1.5};
    const Mat<2> dA{0.02, -0.01,
                    -0.01, 0.03};
    const double p = 1.3;

    const double h = 1e-6;
    const auto Ap = addScaled<2>(A, dA, h);
    const auto Am = addScaled<2>(A, dA, -h);

    Mat<2> fAp{};
    Mat<2> fAm{};
    svmp::FE::forms::jit::svmp_fe_jit_matrix_pow_2x2_v1(Ap.data(), p, fAp.data());
    svmp::FE::forms::jit::svmp_fe_jit_matrix_pow_2x2_v1(Am.data(), p, fAm.data());

    const auto fd = finiteDifference<2>(fAp, fAm, h);

    Mat<2> dd{};
    svmp::FE::forms::jit::svmp_fe_jit_matrix_pow_dd_2x2_v1(A.data(), dA.data(), p, dd.data());

    expectNear<2>(fd, dd, 1e-5);
}

TEST(ExternalCallsNewPhysics, MatrixPowDDReturnsZeroWhenPIsZero)
{
    const Mat<3> A{
        -1.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 3.0,
    };
    const Mat<3> dA{
        0.1, 0.0, 0.0,
        0.0, -0.2, 0.0,
        0.0, 0.0, 0.3,
    };

    Mat<3> dd{};
    svmp::FE::forms::jit::svmp_fe_jit_matrix_pow_dd_3x3_v1(A.data(), dA.data(), 0.0, dd.data());
    for (double v : dd) {
        EXPECT_EQ(v, 0.0);
    }
}

TEST(ExternalCallsNewPhysics, EigvecSymDD3x3MatchesFiniteDifferenceWithSignAlignment)
{
    const Mat<3> A{
        3.0, 0.2, 0.1,
        0.2, 2.0, 0.0,
        0.1, 0.0, 1.0,
    };
    const Mat<3> dA{
        0.03, -0.01, 0.02,
        -0.01, 0.01, 0.0,
        0.02, 0.0, -0.02,
    };
    const std::int32_t which = 0;
    const double h = 1e-6;

    const auto Ap = addScaled<3>(A, dA, h);
    const auto Am = addScaled<3>(A, dA, -h);

    auto v0 = eigSymVec<3>(A, which);
    auto vp = eigSymVec<3>(Ap, which);
    auto vm = eigSymVec<3>(Am, which);
    alignVectorSign(v0, vp);
    alignVectorSign(v0, vm);

    Vec<3> fd{};
    for (std::size_t i = 0; i < 3; ++i) {
        fd[i] = (vp[i] - vm[i]) / (2.0 * h);
    }

    Vec<3> dd{};
    svmp::FE::forms::jit::svmp_fe_jit_eigvec_sym_dd_3x3_v1(A.data(), dA.data(), which, dd.data());

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(fd[i], dd[i], tolRelAbs(fd[i], 5e-5));
    }
}

TEST(ExternalCallsNewPhysics, SpectralDecompDD3x3MatchesFiniteDifferenceWithSignAlignment)
{
    const Mat<3> A{
        3.0, 0.2, 0.1,
        0.2, 2.0, 0.0,
        0.1, 0.0, 1.0,
    };
    const Mat<3> dA{
        0.03, -0.01, 0.02,
        -0.01, 0.01, 0.0,
        0.02, 0.0, -0.02,
    };
    const double h = 1e-6;

    const auto Ap = addScaled<3>(A, dA, h);
    const auto Am = addScaled<3>(A, dA, -h);

    const auto Q0 = eigSymQ<3>(A);
    auto Qp = eigSymQ<3>(Ap);
    auto Qm = eigSymQ<3>(Am);
    alignEigenvectorSigns<3>(Q0, Qp);
    alignEigenvectorSigns<3>(Q0, Qm);

    const auto fd = finiteDifference<3>(Qp, Qm, h);

    Mat<3> dd{};
    svmp::FE::forms::jit::svmp_fe_jit_spectral_decomp_dd_3x3_v1(A.data(), dA.data(), dd.data());

    expectNear<3>(fd, dd, 5e-5);
}
