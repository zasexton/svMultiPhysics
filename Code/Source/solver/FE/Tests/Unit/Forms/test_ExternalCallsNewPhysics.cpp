/**
 * @file test_ExternalCallsNewPhysics.cpp
 * @brief Unit tests for versioned ExternalCalls helpers used by new-physics FormExpr nodes
 */

#include <gtest/gtest.h>

#include "Forms/JIT/ExternalCalls.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>

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
void expectAllNaN(const Mat<N>& a)
{
    for (double v : a) {
        EXPECT_TRUE(std::isnan(v));
    }
}

template <std::size_t N>
void expectIdentity(const Mat<N>& a, double tol)
{
    for (std::size_t r = 0; r < N; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            const double expected = (r == c) ? 1.0 : 0.0;
            EXPECT_NEAR(a[r * N + c], expected, tol);
        }
    }
}

template <std::size_t N>
Mat<N> symmetrize(const Mat<N>& A)
{
    Mat<N> S{};
    for (std::size_t r = 0; r < N; ++r) {
        S[r * N + r] = A[r * N + r];
        for (std::size_t c = r + 1; c < N; ++c) {
            const double v = 0.5 * (A[r * N + c] + A[c * N + r]);
            S[r * N + c] = v;
            S[c * N + r] = v;
        }
    }
    return S;
}

template <std::size_t N>
Mat<N> reconstructFromEigSym(const Vec<N>& evals, const Mat<N>& Q)
{
    Mat<N> out{};
    for (std::size_t r = 0; r < N; ++r) {
        for (std::size_t c = 0; c < N; ++c) {
            double sum = 0.0;
            for (std::size_t k = 0; k < N; ++k) {
                sum += evals[k] * Q[r * N + k] * Q[c * N + k];
            }
            out[r * N + c] = sum;
        }
    }
    return out;
}

template <std::size_t N>
Mat<N> qtq(const Mat<N>& Q)
{
    Mat<N> out{};
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (std::size_t r = 0; r < N; ++r) {
                sum += Q[r * N + i] * Q[r * N + j];
            }
            out[i * N + j] = sum;
        }
    }
    return out;
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

TEST(ExternalCallsNewPhysics, MatrixLogReturnsNaNsOnNonSPDInput)
{
    const Mat<2> A{-1.0, 0.0,
                   0.0, 2.0};
    Mat<2> logA{};
    svmp::FE::forms::jit::svmp_fe_jit_matrix_log_2x2_v1(A.data(), logA.data());
    expectAllNaN<2>(logA);
}

TEST(ExternalCallsNewPhysics, MatrixSqrtReturnsNaNsOnNonSPDInput)
{
    const Mat<3> A{
        1.0, 0.0, 0.0,
        0.0, -2.0, 0.0,
        0.0, 0.0, 3.0,
    };
    Mat<3> sqrtA{};
    svmp::FE::forms::jit::svmp_fe_jit_matrix_sqrt_3x3_v1(A.data(), sqrtA.data());
    expectAllNaN<3>(sqrtA);
}

TEST(ExternalCallsNewPhysics, MatrixPowReturnsIdentityWhenPIsZero)
{
    const Mat<3> A{
        2.0, 0.1, 0.0,
        0.1, 1.5, 0.0,
        0.0, 0.0, 3.0,
    };
    Mat<3> Ap{};
    svmp::FE::forms::jit::svmp_fe_jit_matrix_pow_3x3_v1(A.data(), 0.0, Ap.data());
    expectIdentity<3>(Ap, 1e-14);
}

TEST(ExternalCallsNewPhysics, MatrixPowReturnsNaNsWhenPNotFinite)
{
    const Mat<2> A{2.0, 0.0,
                   0.0, 1.0};
    Mat<2> Ap{};
    svmp::FE::forms::jit::svmp_fe_jit_matrix_pow_2x2_v1(A.data(), std::numeric_limits<double>::infinity(), Ap.data());
    expectAllNaN<2>(Ap);
}

TEST(ExternalCallsNewPhysics, MatrixExpMatchesDiagonal)
{
    const Mat<3> A{
        1.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 3.0,
    };
    Mat<3> expA{};
    svmp::FE::forms::jit::svmp_fe_jit_matrix_exp_3x3_v1(A.data(), expA.data());

    Mat<3> expected{};
    expected[0] = std::exp(1.0);
    expected[4] = std::exp(2.0);
    expected[8] = std::exp(3.0);
    expectNear<3>(expA, expected, 1e-12);
}

TEST(ExternalCallsNewPhysics, EigSymReconstructsRepeatedEigenvalueMatrix)
{
    const Mat<3> A{
        2.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 1.0,
    };

    Vec<3> evals{};
    Mat<3> Q{};
    svmp::FE::forms::jit::svmp_fe_jit_eig_sym_3x3_v1(A.data(), evals.data(), Q.data());

    const auto Arec = reconstructFromEigSym<3>(evals, Q);
    expectNear<3>(Arec, symmetrize<3>(A), 1e-12);

    const auto I = qtq<3>(Q);
    Mat<3> Iref{};
    Iref[0] = 1.0;
    Iref[4] = 1.0;
    Iref[8] = 1.0;
    expectNear<3>(I, Iref, 1e-12);
}

TEST(ExternalCallsNewPhysics, EigGeneralHandlesComplexConjugatePair)
{
    const Mat<3> A{
        0.0, -1.0, 0.0,
        1.0,  0.0, 0.0,
        0.0,  0.0, 2.0,
    };

    double eval_re[3]{};
    double eval_im[3]{};
    double evec_re[9]{};
    double evec_im[9]{};
    svmp::FE::forms::jit::svmp_fe_jit_eig_general_3x3_v1(A.data(), eval_re, eval_im, evec_re, evec_im);

    EXPECT_NEAR(eval_re[0], 2.0, 1e-12);
    EXPECT_NEAR(eval_im[0], 0.0, 1e-12);
    EXPECT_NEAR(eval_re[1], 0.0, 1e-12);
    EXPECT_NEAR(eval_im[1], 1.0, 1e-10);
    EXPECT_NEAR(eval_re[2], 0.0, 1e-12);
    EXPECT_NEAR(eval_im[2], -1.0, 1e-10);

    for (int i = 0; i < 3; ++i) {
        const std::complex<double> lambda{eval_re[i], eval_im[i]};
        std::array<std::complex<double>, 3> v{
            std::complex<double>(evec_re[0 * 3 + i], evec_im[0 * 3 + i]),
            std::complex<double>(evec_re[1 * 3 + i], evec_im[1 * 3 + i]),
            std::complex<double>(evec_re[2 * 3 + i], evec_im[2 * 3 + i]),
        };

        std::array<std::complex<double>, 3> Av{};
        for (int r = 0; r < 3; ++r) {
            std::complex<double> sum{0.0, 0.0};
            for (int c = 0; c < 3; ++c) {
                sum += A[static_cast<std::size_t>(r * 3 + c)] * v[c];
            }
            Av[r] = sum;
        }

        double res2 = 0.0;
        for (int r = 0; r < 3; ++r) {
            const auto res = Av[r] - lambda * v[r];
            res2 += std::norm(res);
        }
        EXPECT_LT(std::sqrt(res2), 1e-10);
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
