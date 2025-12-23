/**
 * @file test_SpaceCacheWorkspace.cpp
 * @brief Unit tests for SpaceCache and SpaceWorkspace
 */

#include <gtest/gtest.h>

#include "FE/Spaces/SpaceCache.h"
#include "FE/Spaces/SpaceWorkspace.h"
#include "FE/Spaces/H1Space.h"

using namespace svmp::FE;
using namespace svmp::FE::spaces;

// =============================================================================
// SpaceCache Tests
// =============================================================================

class SpaceCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        SpaceCache::instance().clear();
    }

    void TearDown() override {
        SpaceCache::instance().clear();
    }

    static constexpr Real tol = 1e-12;
};

TEST_F(SpaceCacheTest, SingletonInstance) {
    auto& c1 = SpaceCache::instance();
    auto& c2 = SpaceCache::instance();
    EXPECT_EQ(&c1, &c2);
}

TEST_F(SpaceCacheTest, ScalarBasisCaching) {
    H1Space space(ElementType::Quad4, 1);
    const auto& elem = space.element();

    auto& cache = SpaceCache::instance();
    const auto& data = cache.get(elem, space.polynomial_order());

    auto quad = elem.quadrature();
    ASSERT_TRUE(quad);

    EXPECT_EQ(data.num_dofs, elem.num_dofs());
    EXPECT_EQ(data.num_qpts, quad->num_points());
    ASSERT_EQ(data.basis_values.size(), elem.num_dofs());

    // Verify that cached basis values match direct evaluations
    const auto& basis = elem.basis();
    std::vector<Real> vals(elem.num_dofs());
    for (std::size_t q = 0; q < data.num_qpts; ++q) {
        basis.evaluate_values(quad->point(q), vals);
        for (std::size_t i = 0; i < elem.num_dofs(); ++i) {
            EXPECT_NEAR(data.basis_values[i][q], vals[i], tol);
        }
    }
}

TEST_F(SpaceCacheTest, CacheReusesEntries) {
    H1Space space(ElementType::Triangle3, 1);
    const auto& elem = space.element();

    auto& cache = SpaceCache::instance();
    const auto& d1 = cache.get(elem, space.polynomial_order());
    const auto& d2 = cache.get(elem, space.polynomial_order());

    EXPECT_EQ(&d1, &d2);
}

// =============================================================================
// SpaceWorkspace Tests
// =============================================================================

class SpaceWorkspaceTest : public ::testing::Test {
protected:
    static constexpr Real tol = 1e-12;
};

TEST_F(SpaceWorkspaceTest, ThreadLocalInstance) {
    auto& ws1 = SpaceWorkspace::local();
    auto& ws2 = SpaceWorkspace::local();
    EXPECT_EQ(&ws1, &ws2);
}

TEST_F(SpaceWorkspaceTest, GetVectorBasic) {
    auto& ws = SpaceWorkspace::local();
    auto& v = ws.get_vector(10);
    ASSERT_EQ(v.size(), 10u);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<Real>(i);
    }
    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_NEAR(v[i], static_cast<Real>(i), tol);
    }
}

TEST_F(SpaceWorkspaceTest, GetVectorResizesAndReserves) {
    auto& ws = SpaceWorkspace::local();
    auto& v1 = ws.get_vector(5, 0);
    auto* ptr1 = v1.data();

    // Request a larger vector in the same slot
    auto& v2 = ws.get_vector(20, 0);
    auto* ptr2 = v2.data();

    EXPECT_EQ(v2.size(), 20u);
    // Pointer may change due to reallocation, but vector must be valid
    (void)ptr1;
    (void)ptr2;
}

TEST_F(SpaceWorkspaceTest, DifferentSlotsIndependent) {
    auto& ws = SpaceWorkspace::local();
    auto& v0 = ws.get_vector(4, 0);
    auto& v1 = ws.get_vector(4, 1);

    EXPECT_NE(&v0, &v1);
    EXPECT_NE(v0.data(), v1.data());
}

