/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 * All Rights Reserved. See Copyright-SimVascular.txt for additional details.
 */
#pragma once

#include "Quadrature/QuadratureRule.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace svmp::FE::quadrature::test {

// Adjacent barycentric transpositions generate every vertex permutation.
inline void expectPermutationInvariantSamples(const QuadratureRule& rule)
{
    using Coordinates = std::array<Real, 4>;
    struct Sample { Coordinates lambda{}; Real weight{}; };
    std::vector<Sample> samples;
    for (std::size_t q = 0; q < rule.num_points(); ++q) {
        Sample s{{1.0, 0.0, 0.0, 0.0}, rule.weight(q)};
        for (int k = 0; k < rule.dimension(); ++k) {
            s.lambda[k + 1] = rule.point(q)[k];
            s.lambda[0] -= rule.point(q)[k];
        }
        samples.push_back(s);
    }
    std::sort(samples.begin(), samples.end(), [](const Sample& a, const Sample& b) {
        return a.lambda[0] < b.lambda[0];
    });
    constexpr Real tol = 64.0 * std::numeric_limits<Real>::epsilon();
    for (int k = 0; k < rule.dimension(); ++k) {
        std::vector<bool> matched(samples.size(), false);
        for (const auto& s : samples) {
            auto target = s.lambda;
            std::swap(target[k], target[k + 1]);
            auto candidate = std::lower_bound(samples.begin(), samples.end(), target[0] - tol,
                [](const Sample& p, Real first) { return p.lambda[0] < first; });
            bool found = false;
            for (; candidate != samples.end() && candidate->lambda[0] <= target[0] + tol; ++candidate) {
                const auto index = static_cast<std::size_t>(candidate - samples.begin());
                if (!matched[index] && std::abs(candidate->weight - s.weight) <= tol * std::abs(s.weight) &&
                    std::equal(target.begin(), target.end(), candidate->lambda.begin(),
                               [](Real a, Real b) { return std::abs(a - b) <= tol; })) {
                    matched[index] = true;
                    found = true;
                    break;
                }
            }
            ASSERT_TRUE(found) << "degree " << rule.order() << ", swapped vertices " << k << " and " << k+1
                               << ", barycentric point " << ::testing::PrintToString(target)
                               << ", weight " << s.weight;
        }
    }
}

} // namespace svmp::FE::quadrature::test
