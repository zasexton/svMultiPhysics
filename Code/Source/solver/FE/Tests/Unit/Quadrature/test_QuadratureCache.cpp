/**
 * @file test_QuadratureCache.cpp
 * @brief Unit tests for quadrature cache behavior
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Quadrature/QuadratureCache.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>
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

namespace {

class PointSetRule final : public QuadratureRule {
public:
    PointSetRule(int dimension, std::vector<QuadPoint> points, std::vector<Real> weights)
        : QuadratureRule(svmp::CellFamily::Tetra, dimension, 1)
    {
        set_data(std::move(points), std::move(weights));
    }
};

void expectSameIdentityVerdict(const QuadratureRule& a,
                               const QuadratureRule& b,
                               bool expected,
                               const char* label)
{
    const bool text_equal = a.cache_identity() == b.cache_identity();
    EXPECT_EQ(text_equal, expected) << label;
    EXPECT_EQ(a.same_cache_identity(b), expected) << label;
    EXPECT_EQ(b.same_cache_identity(a), expected) << label;
}

} // namespace

TEST(QuadratureRuleIdentity, BinaryIdentityAgreesWithTextIdentity) {
    const std::vector<QuadPoint> points = {
        QuadPoint{Real(0.1), Real(0.2), Real(0.3)},
        QuadPoint{Real(1.0) / Real(3.0), Real(0.25), Real(0.0)}};
    const std::vector<Real> weights = {Real(0.05), Real(0.1)};
    const PointSetRule base(3, points, weights);

    EXPECT_TRUE(base.same_cache_identity(base));
    EXPECT_EQ(base.cache_identity().rfind("dim=3|npts=2|pt=", 0), 0u);

    // Weights are not part of the identity, as in the text form.
    expectSameIdentityVerdict(base, PointSetRule(3, points, {Real(1.0), Real(2.0)}),
                              true, "different weights");

    auto next_ulp = points;
    next_ulp[1][0] = std::nextafter(next_ulp[1][0], Real(1.0));
    expectSameIdentityVerdict(base, PointSetRule(3, next_ulp, weights), false,
                              "one coordinate one ulp apart");

    auto negative_zero = points;
    negative_zero[1][2] = -Real(0.0);
    expectSameIdentityVerdict(base, PointSetRule(3, negative_zero, weights), false,
                              "signed zero");

    expectSameIdentityVerdict(base, PointSetRule(2, points, weights), false,
                              "different dimension");

    auto extra = points;
    extra.push_back(QuadPoint{Real(0.4), Real(0.4), Real(0.1)});
    expectSameIdentityVerdict(base, PointSetRule(3, extra, {Real(0.05), Real(0.1), Real(0.0)}),
                              false, "extra point");

    auto reordered = points;
    std::swap(reordered[0], reordered[1]);
    expectSameIdentityVerdict(base, PointSetRule(3, reordered, weights), false,
                              "reordered points");
}

TEST(QuadratureRuleIdentity, BinaryIdentityDistinguishesFactoryRules) {
    const auto tet2 = QuadratureFactory::create(ElementType::Tetra4, 2);
    const auto tet3 = QuadratureFactory::create(ElementType::Tetra4, 3);
    const auto tri2 = QuadratureFactory::create(ElementType::Triangle3, 2);
    ASSERT_TRUE(tet2);
    ASSERT_TRUE(tet3);
    ASSERT_TRUE(tri2);

    // A copy of the points of a factory rule has its identity.
    const PointSetRule tet2_copy(tet2->dimension(), tet2->points(), tet2->weights());
    expectSameIdentityVerdict(*tet2, tet2_copy, true, "copied tetrahedron rule");
    expectSameIdentityVerdict(*tet2, *tet3, tet2->cache_identity() == tet3->cache_identity(),
                              "tetrahedron orders 2 and 3");
    expectSameIdentityVerdict(*tet2, *tri2, false, "tetrahedron and triangle");
}

namespace {

[[nodiscard]] std::uint64_t ruleRealBits(Real value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void expectSameRuleBits(const QuadratureRule& actual, const QuadratureRule& expected,
                        const char* label) {
    ASSERT_EQ(actual.num_points(), expected.num_points()) << label;
    for (std::size_t q = 0; q < expected.num_points(); ++q) {
        for (std::size_t d = 0; d < 3; ++d) {
            EXPECT_EQ(ruleRealBits(actual.point(q)[d]), ruleRealBits(expected.point(q)[d]))
                << label << " point " << q << " component " << d;
        }
        EXPECT_EQ(ruleRealBits(actual.weight(q)), ruleRealBits(expected.weight(q)))
            << label << " weight " << q;
    }
}

} // namespace

// Pinned rules are rules create() returned, kept alive per thread: they equal a
// freshly built rule bit for bit and are returned again as the same object.
// (Each test pins in its own thread, so the thread_local table starts empty and
// is released when the thread ends.)
TEST(QuadratureFactoryPinned, PinnedRulesEqualFreshRulesBitwise) {
    std::thread worker([] {
        struct Key {
            ElementType type;
            int order;
            QuadratureType rule_type;
        };
        const std::vector<Key> keys{
            {ElementType::Line2, 3, QuadratureType::GaussLegendre},
            {ElementType::Line2, 3, QuadratureType::GaussLobatto},
            {ElementType::Triangle3, 2, QuadratureType::GaussLegendre},
            {ElementType::Triangle3, 5, QuadratureType::GaussLegendre},
            {ElementType::Quad4, 3, QuadratureType::GaussLegendre},
            {ElementType::Tetra4, 4, QuadratureType::GaussLegendre},
            {ElementType::Hex8, 2, QuadratureType::GaussLegendre},
        };
        for (const auto& key : keys) {
            const auto pinned = QuadratureFactory::create_pinned(key.type, key.order, key.rule_type);
            const auto fresh = QuadratureFactory::create(key.type, key.order, key.rule_type,
                                                         /*use_cache=*/false);
            ASSERT_TRUE(pinned);
            ASSERT_TRUE(fresh);
            expectSameRuleBits(*pinned, *fresh, "pinned vs fresh");
            EXPECT_EQ(QuadratureFactory::create_pinned(key.type, key.order, key.rule_type).get(),
                      pinned.get());
        }
        // Gauss-Legendre and Gauss-Lobatto rules of one order are different keys.
        EXPECT_NE(QuadratureFactory::create_pinned(ElementType::Line2, 3,
                                                   QuadratureType::GaussLegendre).get(),
                  QuadratureFactory::create_pinned(ElementType::Line2, 3,
                                                   QuadratureType::GaussLobatto).get());

        const Real modifier = QuadratureFactory::default_legacy_modifier(ElementType::Tetra4);
        const auto legacy = QuadratureFactory::create_legacy_compatible_pinned(
            ElementType::Tetra4, modifier);
        const auto legacy_fresh = QuadratureFactory::create_legacy_compatible(
            ElementType::Tetra4, modifier, /*use_cache=*/false);
        expectSameRuleBits(*legacy, *legacy_fresh, "legacy pinned vs fresh");
        EXPECT_EQ(QuadratureFactory::create_legacy_compatible_pinned(ElementType::Tetra4, modifier).get(),
                  legacy.get());
        // A different modifier is a different key.
        const auto other = QuadratureFactory::create_legacy_compatible_pinned(
            ElementType::Tetra4, Real(0.25));
        EXPECT_NE(other.get(), legacy.get());
    });
    worker.join();
}

TEST(QuadratureFactoryPinned, KeysBeyondTheCapAreReturnedUnpinned) {
    std::thread worker([] {
        const int cap = static_cast<int>(QuadratureFactory::kMaxPinnedRulesPerThread);
        std::vector<std::weak_ptr<const QuadratureRule>> pinned;
        for (int order = 1; order <= cap; ++order) {
            pinned.push_back(QuadratureFactory::create_pinned(ElementType::Line2, order));
        }
        // The thread's table holds the first `cap` keys although no caller does.
        for (const auto& rule : pinned) {
            EXPECT_FALSE(rule.expired());
        }
        // The next key is created but not pinned: dropping it releases it.
        std::weak_ptr<const QuadratureRule> beyond =
            QuadratureFactory::create_pinned(ElementType::Line2, cap + 1);
        EXPECT_TRUE(beyond.expired());
        const auto again = QuadratureFactory::create_pinned(ElementType::Line2, cap + 1);
        ASSERT_TRUE(again);
        expectSameRuleBits(*again,
                           *QuadratureFactory::create(ElementType::Line2, cap + 1,
                                                      QuadratureType::GaussLegendre,
                                                      /*use_cache=*/false),
                           "unpinned beyond the cap");
        // Pinned keys keep returning the pinned object.
        EXPECT_EQ(QuadratureFactory::create_pinned(ElementType::Line2, 1).get(),
                  pinned.front().lock().get());
    });
    worker.join();
}
