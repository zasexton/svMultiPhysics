/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

// Unit tests for the dense nodal-block inverse of the FSILS aggregation
// multigrid.

#include <gtest/gtest.h>

#include "Backends/FSILS/FsilsBlockPreconditioners.h"

#include <cmath>
#include <random>
#include <vector>

namespace svmp::FE::backends {
namespace {

TEST(FsilsBlockPreconditioners, DenseBlockInverseMatchesIdentity)
{
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (int d = 1; d <= 9; ++d) {
        std::vector<double> a(static_cast<std::size_t>(d * d));
        for (auto& v : a) {
            v = u(rng);
        }
        for (int i = 0; i < d; ++i) {
            a[static_cast<std::size_t>(i * d + i)] += 3.0;
        }
        std::vector<double> inv(a.size());
        EXPECT_EQ(invertDenseBlock(d, a.data(), inv.data()), 0);
        for (int r = 0; r < d; ++r) {
            for (int c = 0; c < d; ++c) {
                double s = 0.0;
                for (int k = 0; k < d; ++k) {
                    s += a[static_cast<std::size_t>(r * d + k)] * inv[static_cast<std::size_t>(k * d + c)];
                }
                EXPECT_NEAR(s, r == c ? 1.0 : 0.0, 1e-12) << "d=" << d;
            }
        }
    }
}

TEST(FsilsBlockPreconditioners, SingularAndZeroBlocksAreRegularized)
{
    const double singular[4] = {1.0, 2.0, 2.0, 4.0};
    double inv[4] = {};
    EXPECT_GT(invertDenseBlock(2, singular, inv), 0);
    for (const double v : inv) {
        EXPECT_TRUE(std::isfinite(v));
    }
    const double zero[4] = {0.0, 0.0, 0.0, 0.0};
    EXPECT_EQ(invertDenseBlock(2, zero, inv), 2);
    EXPECT_DOUBLE_EQ(inv[0], 1.0);
    EXPECT_DOUBLE_EQ(inv[1], 0.0);
    EXPECT_DOUBLE_EQ(inv[3], 1.0);
}

} // namespace
} // namespace svmp::FE::backends
