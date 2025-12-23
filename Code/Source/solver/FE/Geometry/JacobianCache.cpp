/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "JacobianCache.h"
#include <cstddef>

namespace svmp {
namespace FE {
namespace geometry {

JacobianCache& JacobianCache::instance() {
    static JacobianCache cache;
    return cache;
}

const JacobianCacheEntry& JacobianCache::get_or_compute(const GeometryMapping& mapping,
                                                        const quadrature::QuadratureRule& quad) {
    JacobianCacheKey key{&mapping, quadrature_hash(quad)};
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return *(it->second);
    }

    auto entry = std::make_shared<JacobianCacheEntry>(compute(mapping, quad));
    cache_.emplace(key, entry);
    return *entry;
}

void JacobianCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

std::size_t JacobianCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

JacobianCacheEntry JacobianCache::compute(const GeometryMapping& mapping,
                                          const quadrature::QuadratureRule& quad) const {
    JacobianCacheEntry entry;
    const auto& pts = quad.points();
    entry.J.resize(pts.size());
    entry.J_inv.resize(pts.size());
    entry.detJ.resize(pts.size());

    for (std::size_t i = 0; i < pts.size(); ++i) {
        entry.J[i] = mapping.jacobian(pts[i]);
        entry.J_inv[i] = entry.J[i].inverse();
        entry.detJ[i] = mapping.jacobian_determinant(pts[i]);
    }
    return entry;
}

std::size_t JacobianCache::quadrature_hash(const quadrature::QuadratureRule& quad) const {
    auto hash_combine = [](std::size_t& seed, std::size_t v) {
        seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    };

    std::size_t h = 0;
    hash_combine(h, quad.num_points());
    hash_combine(h, static_cast<std::size_t>(quad.dimension()));
    hash_combine(h, static_cast<std::size_t>(quad.order()));
    hash_combine(h, static_cast<std::size_t>(quad.cell_family()));

    const auto& pts = quad.points();
    const auto& wts = quad.weights();
    const std::hash<Real> rh{};
    for (std::size_t i = 0; i < pts.size(); ++i) {
        hash_combine(h, rh(pts[i][0]));
        hash_combine(h, rh(pts[i][1]));
        hash_combine(h, rh(pts[i][2]));
        hash_combine(h, rh(wts[i]));
    }
    return h;
}

} // namespace geometry
} // namespace FE
} // namespace svmp
