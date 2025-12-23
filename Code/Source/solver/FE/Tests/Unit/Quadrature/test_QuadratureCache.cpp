/**
 * @file test_QuadratureCache.cpp
 * @brief Unit tests for quadrature cache behavior
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Quadrature/QuadratureCache.h"
#include <thread>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::quadrature;

TEST(QuadratureCache, ReturnsSharedInstance) {
    QuadratureCache::instance().clear();
    auto q1 = QuadratureFactory::create(ElementType::Quad4, 3, QuadratureType::GaussLegendre, true);
    auto q2 = QuadratureFactory::create(ElementType::Quad4, 3, QuadratureType::GaussLegendre, true);
    ASSERT_TRUE(q1);
    ASSERT_TRUE(q2);
    EXPECT_EQ(q1.get(), q2.get());
}

TEST(QuadratureCache, DistinguishesByType) {
    QuadratureCache::instance().clear();
    auto gauss = QuadratureFactory::create(ElementType::Line2, 3, QuadratureType::GaussLegendre, true);
    auto lobatto = QuadratureFactory::create(ElementType::Line2, 3, QuadratureType::GaussLobatto, true);
    ASSERT_TRUE(gauss);
    ASSERT_TRUE(lobatto);
    EXPECT_NE(gauss.get(), lobatto.get());
}

TEST(QuadratureCache, ClearResetsSize) {
    QuadratureCache::instance().clear();
    (void)QuadratureFactory::create(ElementType::Quad4, 2, QuadratureType::GaussLegendre, true);
    EXPECT_GT(QuadratureCache::instance().size(), 0u);
    QuadratureCache::instance().clear();
    EXPECT_EQ(QuadratureCache::instance().size(), 0u);
}

TEST(QuadratureCache, MultithreadedAccessSharedInstance) {
    QuadratureCache::instance().clear();
    std::shared_ptr<const QuadratureRule> ref;
    std::vector<std::shared_ptr<const QuadratureRule>> seen(8);

    auto worker = [&](int idx) {
        seen[static_cast<std::size_t>(idx)] =
            QuadratureFactory::create(ElementType::Hex8, 3, QuadratureType::GaussLegendre, true);
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) threads.emplace_back(worker, i);
    for (auto& t : threads) t.join();

    for (const auto& s : seen) {
        ASSERT_TRUE(s);
        if (!ref) ref = s;
        EXPECT_EQ(ref.get(), s.get());
    }
}

TEST(QuadratureCache, ExpiredEntriesAreRegenerated) {
    QuadratureCache::instance().clear();
    auto q1 = QuadratureFactory::create(ElementType::Quad4, 2, QuadratureType::GaussLegendre, true);
    EXPECT_EQ(QuadratureCache::instance().size(), 1u);
    // Drop shared_ptr to leave only weak_ptr in cache
    q1.reset();
    auto q2 = QuadratureFactory::create(ElementType::Quad4, 2, QuadratureType::GaussLegendre, true);
    EXPECT_EQ(QuadratureCache::instance().size(), 1u);
    EXPECT_TRUE(q2);
}

TEST(QuadratureCache, PruneExpiredRemovesStaleEntries) {
    QuadratureCache::instance().clear();

    std::vector<std::shared_ptr<const QuadratureRule>> rules;
    for (int order = 1; order <= 20; ++order) {
        rules.push_back(
            QuadratureFactory::create(ElementType::Line2, order, QuadratureType::GaussLegendre, true));
    }
    EXPECT_EQ(QuadratureCache::instance().size(), 20u);

    // Release all strong references so cache entries are expired.
    rules.clear();
    QuadratureCache::instance().prune_expired();
    EXPECT_EQ(QuadratureCache::instance().size(), 0u);
}
