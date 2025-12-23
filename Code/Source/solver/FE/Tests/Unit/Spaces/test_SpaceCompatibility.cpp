/**
 * @file test_SpaceCompatibility.cpp
 * @brief Tests for SpaceCompatibility heuristics
 */

#include <gtest/gtest.h>

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

TEST(SpaceCompatibility, InfSupForStokesPair) {
    auto vel_scalar = std::make_shared<H1Space>(ElementType::Triangle3, 2);
    ProductSpace velocity(vel_scalar, 2); // 2D velocity
    L2Space pressure(ElementType::Triangle3, 1);

    auto r = SpaceCompatibility::check_inf_sup(velocity, pressure);
    EXPECT_TRUE(r.ok);

    // Unstable pair: velocity order too low
    auto vel_low_scalar = std::make_shared<H1Space>(ElementType::Triangle3, 0);
    ProductSpace vel_low(vel_low_scalar, 2);
    auto r_low = SpaceCompatibility::check_inf_sup(vel_low, pressure);
    EXPECT_FALSE(r_low.ok);
}

