/**
 * @file test_SpaceCompatibility.cpp
 * @brief Tests for SpaceCompatibility heuristics
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <map>

#include "FE/Basis/NodeOrderingConventions.h"
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Spaces/HDivSpace.h"

#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/L2Space.h"
#include "FE/Spaces/ProductSpace.h"
#include "FE/Spaces/SpaceCompatibility.h"

using namespace svmp::FE;
using namespace svmp::FE::spaces;

TEST(SpaceCompatibility, ConformityChecks) {
    H1Space a(ElementType::Triangle3, 1);
    H1Space b(ElementType::Triangle3, 1);
    L2Space c(ElementType::Triangle3, 1);

    auto r1 = SpaceCompatibility::check_conformity(a, b);
    EXPECT_TRUE(r1.ok);

    auto r2 = SpaceCompatibility::check_conformity(a, c);
    EXPECT_FALSE(r2.ok);
}

TEST(SpaceCompatibility, ContinuousQuadraticFieldsCanHaveAnUncontrolledLinearMultiplier)
{
    const auto type = ElementType::Triangle3;
    auto scalar = std::make_shared<H1Space>(type, 2);
    ProductSpace vector_field(scalar, 2);
    L2Space multiplier(type, 1);
    const auto rule = quadrature::QuadratureFactory::create(type, 4);
    const auto nodes = basis::ReferenceNodeLayout::get_lagrange_node_coords(type, 2);

    // Four right triangles meet at the origin. On each triangle, integration
    // against (12 lambda_origin - 3) / area evaluates a linear function at
    // that vertex. Alternate this multiplier's sign between adjacent cells.
    const std::array<std::array<int, 2>, 4> axis{{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}};
    std::map<std::array<int, 2>, std::array<Real, 2>> divergence_work;
    Real mean = 0, norm_squared = 0;
    for (int cell = 0; cell < 4; ++cell) {
        const auto& a = axis[cell];
        const auto& b = axis[(cell + 1) % 4];
        for (std::size_t k = 0; k < rule->num_points(); ++k) {
            const auto& point = rule->points()[k];
            const Real q = (cell % 2 ? -1 : 1) * (12 * (1 - point[0] - point[1]) - 3) / 0.5;
            const auto w = rule->weights()[k];
            mean += w * q;
            norm_squared += w * q * q;
            std::vector<basis::Gradient> gradient;
            scalar->element().basis().evaluate_gradients(point, gradient);
            ASSERT_EQ(gradient.size(), nodes.size());
            for (std::size_t j = 0; j < nodes.size(); ++j) {
                std::array<int, 2> key{};
                for (int d = 0; d < 2; ++d) {
                    key[d] = static_cast<int>(std::lround(2 * (a[d] * nodes[j][0] + b[d] * nodes[j][1])));
                }
                auto& work = divergence_work[key];
                for (int d = 0; d < 2; ++d) {
                    work[d] += w * q * (a[d] * gradient[j][0] + b[d] * gradient[j][1]);
                }
            }
        }
    }
    EXPECT_NEAR(mean, 0, 1e-13);
    EXPECT_NEAR(norm_squared, 72, 1e-11);
    ASSERT_EQ(divergence_work.size(), 13u);
    for (const auto& [node, work] : divergence_work) {
        EXPECT_NEAR(work[0], 0, 1e-12);
        EXPECT_NEAR(work[1], 0, 1e-12);
    }
    EXPECT_FALSE(SpaceCompatibility::check_inf_sup(vector_field, multiplier).ok);
    ProductSpace linear_field(std::make_shared<H1Space>(type, 1), 2);
    EXPECT_FALSE(SpaceCompatibility::check_inf_sup(linear_field, multiplier).ok);
}

TEST(SpaceCompatibility, MultiplierFitsTheReferenceDivergenceSpace)
{
    const auto type = ElementType::Triangle3;
    const HDivSpace rt0(type, 0, BasisType::RaviartThomas);
    const HDivSpace bdm1(type, 1, BasisType::BDM);
    const L2Space constant(type, 0), linear(type, 1);
    EXPECT_TRUE(SpaceCompatibility::check_inf_sup(rt0, constant).ok);
    EXPECT_TRUE(SpaceCompatibility::check_inf_sup(bdm1, constant).ok);
    EXPECT_FALSE(SpaceCompatibility::check_inf_sup(rt0, linear).ok);
    EXPECT_FALSE(SpaceCompatibility::check_inf_sup(bdm1, linear).ok);
    EXPECT_FALSE(SpaceCompatibility::check_inf_sup(rt0, L2Space(ElementType::Quad4, 0)).ok);
}
