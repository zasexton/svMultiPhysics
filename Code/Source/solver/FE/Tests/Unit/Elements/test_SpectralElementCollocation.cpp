/**
 * @file test_SpectralElementCollocation.cpp
 * @brief Tests for SpectralElement Gauss-Lobatto collocation properties
 */

#include <gtest/gtest.h>

#include "FE/Elements/SpectralElement.h"
#include "FE/Basis/SpectralBasis.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::elements;

namespace {

std::vector<Real> unique_sorted(std::vector<Real> values, Real tol) {
    std::sort(values.begin(), values.end());
    std::vector<Real> out;
    for (Real v : values) {
        if (out.empty() || std::abs(v - out.back()) > tol) {
            out.push_back(v);
        }
    }
    return out;
}

std::vector<Real> assemble_mass_matrix(const basis::BasisFunction& basis_fn,
                                       const quadrature::QuadratureRule& quad) {
    const std::size_t n = basis_fn.size();
    std::vector<Real> M(n * n, Real(0));

    std::vector<Real> values;
    for (std::size_t q = 0; q < quad.num_points(); ++q) {
        const auto qp = quad.point(q);
        const Real w = quad.weight(q);

        basis_fn.evaluate_values(qp, values);
        EXPECT_EQ(values.size(), n);

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                M[i * n + j] += w * values[i] * values[j];
            }
        }
    }

    return M;
}

void expect_mass_matrix_diagonal(const std::vector<Real>& M, std::size_t n, double rel_tol) {
    double max_diag = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        max_diag = std::max(max_diag, std::abs(static_cast<double>(M[i * n + i])));
    }
    const double tol = std::max(1e-14, rel_tol * max_diag);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_GT(static_cast<double>(M[i * n + i]), 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            EXPECT_NEAR(static_cast<double>(M[i * n + j]), 0.0, tol);
        }
    }
}

} // namespace

TEST(SpectralElement, CollocationAndDiagonalMassLine) {
    const int p = 4;
    SpectralElement elem(ElementType::Line2, p);

    auto basis_ptr = elem.basis_ptr();
    ASSERT_TRUE(basis_ptr);
    auto spectral = std::dynamic_pointer_cast<const basis::SpectralBasis>(basis_ptr);
    ASSERT_TRUE(spectral);

    auto quad = elem.quadrature();
    ASSERT_TRUE(quad);

    const auto& nodes_1d = spectral->nodes_1d();
    ASSERT_EQ(nodes_1d.size(), static_cast<std::size_t>(p + 1));
    EXPECT_EQ(quad->num_points(), nodes_1d.size());
    EXPECT_EQ(quad->order(), 2 * p - 1);

    for (std::size_t i = 0; i < nodes_1d.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(quad->point(i)[0]),
                    static_cast<double>(nodes_1d[i]),
                    1e-14);
    }

    const auto M = assemble_mass_matrix(*basis_ptr, *quad);
    expect_mass_matrix_diagonal(M, basis_ptr->size(), 1e-12);
}

TEST(SpectralElement, CollocationAndDiagonalMassQuad) {
    const int p = 3;
    SpectralElement elem(ElementType::Quad4, p);

    auto basis_ptr = elem.basis_ptr();
    ASSERT_TRUE(basis_ptr);
    auto spectral = std::dynamic_pointer_cast<const basis::SpectralBasis>(basis_ptr);
    ASSERT_TRUE(spectral);

    auto quad = elem.quadrature();
    ASSERT_TRUE(quad);

    const auto& nodes_1d = spectral->nodes_1d();
    ASSERT_EQ(nodes_1d.size(), static_cast<std::size_t>(p + 1));
    EXPECT_EQ(quad->num_points(), basis_ptr->size());
    EXPECT_EQ(quad->order(), 2 * p - 1);

    std::vector<Real> xs;
    std::vector<Real> ys;
    xs.reserve(quad->num_points());
    ys.reserve(quad->num_points());
    for (const auto& pt : quad->points()) {
        xs.push_back(pt[0]);
        ys.push_back(pt[1]);
    }

    const auto ux = unique_sorted(std::move(xs), Real(1e-14));
    const auto uy = unique_sorted(std::move(ys), Real(1e-14));
    ASSERT_EQ(ux.size(), nodes_1d.size());
    ASSERT_EQ(uy.size(), nodes_1d.size());
    for (std::size_t i = 0; i < nodes_1d.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(ux[i]), static_cast<double>(nodes_1d[i]), 1e-14);
        EXPECT_NEAR(static_cast<double>(uy[i]), static_cast<double>(nodes_1d[i]), 1e-14);
    }

    const auto M = assemble_mass_matrix(*basis_ptr, *quad);
    expect_mass_matrix_diagonal(M, basis_ptr->size(), 1e-12);
}

TEST(SpectralElement, CollocationAndDiagonalMassHex) {
    const int p = 3;
    SpectralElement elem(ElementType::Hex8, p);

    auto basis_ptr = elem.basis_ptr();
    ASSERT_TRUE(basis_ptr);
    auto spectral = std::dynamic_pointer_cast<const basis::SpectralBasis>(basis_ptr);
    ASSERT_TRUE(spectral);

    auto quad = elem.quadrature();
    ASSERT_TRUE(quad);

    const auto& nodes_1d = spectral->nodes_1d();
    ASSERT_EQ(nodes_1d.size(), static_cast<std::size_t>(p + 1));
    EXPECT_EQ(quad->num_points(), basis_ptr->size());
    EXPECT_EQ(quad->order(), 2 * p - 1);

    std::vector<Real> xs;
    std::vector<Real> ys;
    std::vector<Real> zs;
    xs.reserve(quad->num_points());
    ys.reserve(quad->num_points());
    zs.reserve(quad->num_points());
    for (const auto& pt : quad->points()) {
        xs.push_back(pt[0]);
        ys.push_back(pt[1]);
        zs.push_back(pt[2]);
    }

    const auto ux = unique_sorted(std::move(xs), Real(1e-14));
    const auto uy = unique_sorted(std::move(ys), Real(1e-14));
    const auto uz = unique_sorted(std::move(zs), Real(1e-14));
    ASSERT_EQ(ux.size(), nodes_1d.size());
    ASSERT_EQ(uy.size(), nodes_1d.size());
    ASSERT_EQ(uz.size(), nodes_1d.size());
    for (std::size_t i = 0; i < nodes_1d.size(); ++i) {
        EXPECT_NEAR(static_cast<double>(ux[i]), static_cast<double>(nodes_1d[i]), 1e-14);
        EXPECT_NEAR(static_cast<double>(uy[i]), static_cast<double>(nodes_1d[i]), 1e-14);
        EXPECT_NEAR(static_cast<double>(uz[i]), static_cast<double>(nodes_1d[i]), 1e-14);
    }

    const auto M = assemble_mass_matrix(*basis_ptr, *quad);
    expect_mass_matrix_diagonal(M, basis_ptr->size(), 1e-12);
}

