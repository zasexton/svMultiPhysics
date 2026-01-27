/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Forms/Tensor/SpectralEigen.h"

#include "Math/Eigensolvers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

[[nodiscard]] std::int32_t clampWhich(std::int32_t which, std::int32_t n) noexcept
{
    if (n <= 1) return 0;
    if (which < 0) return 0;
    if (which >= n) return n - 1;
    return which;
}

[[nodiscard]] std::array<double, 4> sym2(const double* A) noexcept
{
    // row-major: [a00 a01; a10 a11]
    const double a00 = A[0];
    const double a01 = 0.5 * (A[1] + A[2]);
    const double a11 = A[3];
    return {a00, a01, a01, a11};
}

[[nodiscard]] std::array<double, 9> sym3(const double* A) noexcept
{
    // row-major 3x3; symmetrize off-diagonals.
    std::array<double, 9> S{};
    for (int r = 0; r < 3; ++r) {
        S[static_cast<std::size_t>(r * 3 + r)] = A[static_cast<std::size_t>(r * 3 + r)];
    }
    for (int r = 0; r < 3; ++r) {
        for (int c = r + 1; c < 3; ++c) {
            const double v = 0.5 * (A[static_cast<std::size_t>(r * 3 + c)] + A[static_cast<std::size_t>(c * 3 + r)]);
            S[static_cast<std::size_t>(r * 3 + c)] = v;
            S[static_cast<std::size_t>(c * 3 + r)] = v;
        }
    }
    return S;
}

[[nodiscard]] double dot2(const std::array<double, 2>& a, const std::array<double, 2>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1];
}

[[nodiscard]] double dot3(const std::array<double, 3>& a, const std::array<double, 3>& b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] std::array<double, 2> mat2_vec(const std::array<double, 4>& M, const std::array<double, 2>& x) noexcept
{
    return {M[0] * x[0] + M[1] * x[1],
            M[2] * x[0] + M[3] * x[1]};
}

[[nodiscard]] std::array<double, 3> mat3_vec(const std::array<double, 9>& M, const std::array<double, 3>& x) noexcept
{
    std::array<double, 3> out{};
    for (int r = 0; r < 3; ++r) {
        out[static_cast<std::size_t>(r)] =
            M[static_cast<std::size_t>(r * 3 + 0)] * x[0] +
            M[static_cast<std::size_t>(r * 3 + 1)] * x[1] +
            M[static_cast<std::size_t>(r * 3 + 2)] * x[2];
    }
    return out;
}

[[nodiscard]] double quadForm2(const std::array<double, 4>& M, const std::array<double, 2>& v) noexcept
{
    const auto mv = mat2_vec(M, v);
    return dot2(v, mv);
}

[[nodiscard]] double quadForm3(const std::array<double, 9>& M, const std::array<double, 3>& v) noexcept
{
    const auto mv = mat3_vec(M, v);
    return dot3(v, mv);
}

[[nodiscard]] double safeDenom(double denom, double scale) noexcept
{
    // For repeated/near-repeated eigenvalues, eigenvector derivatives are not defined.
    // Avoid numerical blow-up by clamping the denominator away from 0.
    const double eps = 1e-12 * std::max(1.0, scale);
    if (std::abs(denom) < eps) {
        return (denom < 0.0) ? -eps : eps;
    }
    return denom;
}

} // namespace

extern "C" double svmp_fe_sym_eigenvalue_2x2_v1(const double* A, std::int32_t which)
{
    which = clampWhich(which, 2);
    svmp::FE::math::Matrix2x2<double> M;
    const auto S = sym2(A);
    M(0, 0) = S[0];
    M(0, 1) = S[1];
    M(1, 0) = S[2];
    M(1, 1) = S[3];

    const auto [evals, evecs] = svmp::FE::math::eigen_2x2_symmetric(M);
    (void)evecs;
    return evals[static_cast<std::size_t>(which)];
}

extern "C" double svmp_fe_sym_eigenvalue_3x3_v1(const double* A, std::int32_t which)
{
    which = clampWhich(which, 3);
    const std::int32_t asc_idx = 2 - which; // eigen_3x3_symmetric is ascending

    svmp::FE::math::Matrix3x3<double> M;
    const auto S = sym3(A);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            M(r, c) = S[static_cast<std::size_t>(r * 3 + c)];
        }
    }

    const auto [evals, evecs] = svmp::FE::math::eigen_3x3_symmetric(M);
    (void)evecs;
    return evals[static_cast<std::size_t>(asc_idx)];
}

extern "C" double svmp_fe_sym_eigenvalue_dd_2x2_v1(const double* A, const double* dA, std::int32_t which)
{
    which = clampWhich(which, 2);

    svmp::FE::math::Matrix2x2<double> M;
    const auto S = sym2(A);
    M(0, 0) = S[0];
    M(0, 1) = S[1];
    M(1, 0) = S[2];
    M(1, 1) = S[3];

    const auto [evals, evecs] = svmp::FE::math::eigen_2x2_symmetric(M);
    (void)evals;

    std::array<double, 2> v{
        evecs(0, static_cast<std::size_t>(which)),
        evecs(1, static_cast<std::size_t>(which)),
    };

    const auto Sd = sym2(dA);
    return quadForm2(Sd, v);
}

extern "C" double svmp_fe_sym_eigenvalue_dd_3x3_v1(const double* A, const double* dA, std::int32_t which)
{
    which = clampWhich(which, 3);
    const std::int32_t asc_idx = 2 - which; // eigen_3x3_symmetric is ascending

    svmp::FE::math::Matrix3x3<double> M;
    const auto S = sym3(A);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            M(r, c) = S[static_cast<std::size_t>(r * 3 + c)];
        }
    }

    const auto [evals, evecs] = svmp::FE::math::eigen_3x3_symmetric(M);
    (void)evals;

    std::array<double, 3> v{
        evecs(0, static_cast<std::size_t>(asc_idx)),
        evecs(1, static_cast<std::size_t>(asc_idx)),
        evecs(2, static_cast<std::size_t>(asc_idx)),
    };

    const auto Sd = sym3(dA);
    return quadForm3(Sd, v);
}

extern "C" double svmp_fe_sym_eigenvalue_ddA_2x2_v1(const double* A,
                                                    const double* B,
                                                    const double* dA,
                                                    std::int32_t which)
{
    which = clampWhich(which, 2);

    svmp::FE::math::Matrix2x2<double> M;
    const auto S = sym2(A);
    M(0, 0) = S[0];
    M(0, 1) = S[1];
    M(1, 0) = S[2];
    M(1, 1) = S[3];

    const auto [evals, evecs] = svmp::FE::math::eigen_2x2_symmetric(M);

    const int i = static_cast<int>(which);
    const int j = 1 - i;

    std::array<double, 2> vi{
        evecs(0, static_cast<std::size_t>(i)),
        evecs(1, static_cast<std::size_t>(i)),
    };
    std::array<double, 2> vj{
        evecs(0, static_cast<std::size_t>(j)),
        evecs(1, static_cast<std::size_t>(j)),
    };

    const auto SB = sym2(B);
    const auto Sd = sym2(dA);

    const double vjBvi = dot2(vj, mat2_vec(SB, vi));
    const double vjDvi = dot2(vj, mat2_vec(Sd, vi));

    const double denom_raw = evals[static_cast<std::size_t>(i)] - evals[static_cast<std::size_t>(j)];
    const double denom = safeDenom(denom_raw, std::max(std::abs(evals[0]), std::abs(evals[1])));

    return 2.0 * vjBvi * vjDvi / denom;
}

extern "C" double svmp_fe_sym_eigenvalue_ddA_3x3_v1(const double* A,
                                                    const double* B,
                                                    const double* dA,
                                                    std::int32_t which)
{
    which = clampWhich(which, 3);
    const std::int32_t asc_i = 2 - which; // descending -> ascending index

    svmp::FE::math::Matrix3x3<double> M;
    const auto S = sym3(A);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            M(r, c) = S[static_cast<std::size_t>(r * 3 + c)];
        }
    }

    const auto [evals, evecs] = svmp::FE::math::eigen_3x3_symmetric(M); // ascending

    std::array<double, 3> vi{
        evecs(0, static_cast<std::size_t>(asc_i)),
        evecs(1, static_cast<std::size_t>(asc_i)),
        evecs(2, static_cast<std::size_t>(asc_i)),
    };

    const auto SB = sym3(B);
    const auto Sd = sym3(dA);

    double sum = 0.0;
    const double scale = std::max({std::abs(evals[0]), std::abs(evals[1]), std::abs(evals[2])});
    for (int asc_j = 0; asc_j < 3; ++asc_j) {
        if (asc_j == asc_i) continue;

        std::array<double, 3> vj{
            evecs(0, static_cast<std::size_t>(asc_j)),
            evecs(1, static_cast<std::size_t>(asc_j)),
            evecs(2, static_cast<std::size_t>(asc_j)),
        };

        const double vjBvi = dot3(vj, mat3_vec(SB, vi));
        const double vjDvi = dot3(vj, mat3_vec(Sd, vi));

        const double denom_raw = evals[static_cast<std::size_t>(asc_i)] - evals[static_cast<std::size_t>(asc_j)];
        const double denom = safeDenom(denom_raw, scale);

        sum += 2.0 * vjBvi * vjDvi / denom;
    }

    return sum;
}

