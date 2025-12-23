/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_QUADRATURE_QUADRATURECACHE_H
#define SVMP_FE_QUADRATURE_QUADRATURECACHE_H

/**
 * @file QuadratureCache.h
 * @brief Thread-safe cache for reusable quadrature rules
 *
 * The cache supports both order-based and position-based quadrature rules.
 * For position-based rules, the cache key includes the position modifier
 * to distinguish rules with different point placements.
 */

#include "QuadratureRule.h"
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace quadrature {

/**
 * @brief Cache key for quadrature rules
 *
 * For position-based quadrature (QuadratureType::PositionBased), the
 * position_modifier field distinguishes rules with different point placements.
 * For other quadrature types, position_modifier is ignored in comparisons.
 */
struct QuadratureKey {
    ElementType element_type;
    int order;
    QuadratureType type;
    Real position_modifier = 0.0;  ///< Only used for PositionBased type

    /// Default constructor
    QuadratureKey() = default;

    /// Constructor for order-based rules
    QuadratureKey(ElementType elem, int ord, QuadratureType t)
        : element_type(elem), order(ord), type(t), position_modifier(0.0) {}

    /// Constructor for position-based rules
    QuadratureKey(ElementType elem, int ord, QuadratureType t, Real modifier)
        : element_type(elem), order(ord), type(t), position_modifier(modifier) {}

    bool operator==(const QuadratureKey& other) const noexcept {
        if (element_type != other.element_type ||
            order != other.order ||
            type != other.type) {
            return false;
        }
        // For position-based quadrature, also compare the modifier
        if (type == QuadratureType::PositionBased) {
            return std::abs(position_modifier - other.position_modifier) < 1e-14;
        }
        return true;
    }
};

/**
 * @brief Hash function for QuadratureKey
 *
 * For position-based rules, the hash includes a discretized version of
 * the position modifier to ensure consistent hashing while allowing
 * for small floating-point variations.
 */
struct QuadratureKeyHash {
    std::size_t operator()(const QuadratureKey& key) const noexcept {
        std::size_t h1 = std::hash<int>()(static_cast<int>(key.element_type));
        std::size_t h2 = std::hash<int>()(key.order);
        std::size_t h3 = std::hash<int>()(static_cast<int>(key.type));
        std::size_t base = h1 ^ (h2 << 1) ^ (h3 << 2);

        // For position-based quadrature, include the modifier in the hash
        // Discretize to avoid floating-point precision issues
        if (key.type == QuadratureType::PositionBased) {
            // Round to 10 decimal places for consistent hashing
            long long discretized = static_cast<long long>(key.position_modifier * 1e10);
            std::size_t h4 = std::hash<long long>()(discretized);
            base ^= (h4 << 3);
        }

        return base;
    }
};

class QuadratureCache {
public:
    static QuadratureCache& instance();

    /// Retrieve from cache or construct via @p factory if missing
    std::shared_ptr<const QuadratureRule> get_or_create(
        const QuadratureKey& key,
        const std::function<std::shared_ptr<QuadratureRule>()>& factory);

    /// Remove expired weak_ptr entries from the map.
    void prune_expired();

    void clear();
    std::size_t size() const;

private:
    QuadratureCache() = default;
    QuadratureCache(const QuadratureCache&) = delete;
    QuadratureCache& operator=(const QuadratureCache&) = delete;

    mutable std::mutex mutex_;
    std::unordered_map<QuadratureKey, std::weak_ptr<const QuadratureRule>, QuadratureKeyHash> cache_;
};

} // namespace quadrature
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_QUADRATURE_QUADRATURECACHE_H
