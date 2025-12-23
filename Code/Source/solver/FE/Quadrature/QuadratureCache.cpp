/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "QuadratureCache.h"

namespace svmp {
namespace FE {
namespace quadrature {

QuadratureCache& QuadratureCache::instance() {
    static QuadratureCache cache;
    return cache;
}

std::shared_ptr<const QuadratureRule> QuadratureCache::get_or_create(
    const QuadratureKey& key,
    const std::function<std::shared_ptr<QuadratureRule>()>& factory) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(key);
    if (it != cache_.end()) {
        if (auto existing = it->second.lock()) {
            return existing;
        }
        // The cached weak_ptr expired; erase the stale entry to avoid unbounded
        // growth of keys in long-running workflows.
        cache_.erase(it);
    }

    auto created = factory();
    cache_[key] = created;
    return created;
}

void QuadratureCache::prune_expired() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = cache_.begin(); it != cache_.end();) {
        if (it->second.expired()) {
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

void QuadratureCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

std::size_t QuadratureCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

} // namespace quadrature
} // namespace FE
} // namespace svmp
