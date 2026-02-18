/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_GEOMETRY_JACOBIANCACHE_H
#define SVMP_FE_GEOMETRY_JACOBIANCACHE_H

/**
 * @file JacobianCache.h
 * @brief Thread-safe cache for Jacobian evaluations
 */

#include "GeometryMapping.h"
#include "Quadrature/QuadratureRule.h"
#include <memory>
#include <mutex>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace geometry {

/// Per-quadrature-point packed Jacobian data for cache-friendly access.
struct JacobianQPData {
    math::Matrix<Real, 3, 3> J;
    math::Matrix<Real, 3, 3> J_inv;
    math::Matrix<Real, 3, 3> J_invT;  // Pre-computed inverse transpose
    Real detJ;
};

struct JacobianCacheEntry {
    std::vector<JacobianQPData> data;  // packed per-qpt

    // Convenience accessors for backward compatibility
    const math::Matrix<Real, 3, 3>& J(std::size_t i) const { return data[i].J; }
    const math::Matrix<Real, 3, 3>& J_inv(std::size_t i) const { return data[i].J_inv; }
    const math::Matrix<Real, 3, 3>& J_invT(std::size_t i) const { return data[i].J_invT; }
    Real detJ_val(std::size_t i) const { return data[i].detJ; }
    std::size_t size() const { return data.size(); }
};

struct JacobianCacheKey {
    const GeometryMapping* mapping{nullptr};
    std::size_t quad_hash{0};
    bool operator==(const JacobianCacheKey& other) const noexcept {
        return mapping == other.mapping && quad_hash == other.quad_hash;
    }
};

struct JacobianCacheKeyHash {
    std::size_t operator()(const JacobianCacheKey& key) const noexcept {
        return std::hash<const GeometryMapping*>()(key.mapping) ^ (key.quad_hash << 1);
    }
};

class JacobianCache {
public:
    static JacobianCache& instance();

    const JacobianCacheEntry& get_or_compute(const GeometryMapping& mapping,
                                             const quadrature::QuadratureRule& quad);

    void clear();
    std::size_t size() const;

private:
    JacobianCache() = default;
    JacobianCache(const JacobianCache&) = delete;
    JacobianCache& operator=(const JacobianCache&) = delete;

    JacobianCacheEntry compute(const GeometryMapping& mapping,
                               const quadrature::QuadratureRule& quad) const;

    std::size_t quadrature_hash(const quadrature::QuadratureRule& quad) const;

    mutable std::mutex mutex_;
    std::unordered_map<JacobianCacheKey, std::shared_ptr<JacobianCacheEntry>, JacobianCacheKeyHash> cache_;
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_JACOBIANCACHE_H
