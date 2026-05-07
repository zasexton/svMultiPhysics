/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "BasisCache.h"
namespace svmp {
namespace FE {
namespace basis {

BasisCache& BasisCache::instance() {
    static BasisCache cache;
    return cache;
}

const BasisCacheEntry& BasisCache::get_or_compute(
    const BasisFunction& basis,
    const quadrature::QuadratureRule& quad,
    bool gradients,
    bool hessians) {
    BasisCacheKey key{
        basis.cache_identity(),
        quad.cache_identity(),
        gradients,
        hessians
    };

    // Warm path: shared (reader) lock allows concurrent cache hits.
    {
        std::shared_lock<std::shared_mutex> read_lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return *(it->second);
        }
    }

    // Compute outside the lock so other readers can proceed.
    auto entry = std::make_shared<BasisCacheEntry>(compute(basis, quad, gradients, hessians));

    // Cold path: exclusive (writer) lock. Re-check in case another thread
    // populated the same key while we were computing — discard our entry
    // in favor of the one already in the cache for stable identity.
    std::unique_lock<std::shared_mutex> write_lock(mutex_);
    auto [it, inserted] = cache_.emplace(key, entry);
    return *(it->second);
}

void BasisCache::clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    cache_.clear();
}

std::size_t BasisCache::size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return cache_.size();
}

BasisCacheEntry BasisCache::compute(const BasisFunction& basis,
                                    const quadrature::QuadratureRule& quad,
                                    bool gradients,
                                    bool hessians) const {
    BasisCacheEntry entry;
    const auto& points = quad.points();
    entry.num_qpts = points.size();
    entry.num_dofs = basis.size();

    const bool vector_basis = basis.is_vector_valued();
    if (!vector_basis) {
        entry.scalar_values.assign(entry.num_dofs * entry.num_qpts, Real(0));
    }
    if (basis.is_vector_valued()) {
        entry.vector_values.resize(points.size());
    }
    if (gradients) {
        entry.gradients.resize(points.size());
    }
    if (hessians) {
        entry.hessians.resize(points.size());
    }

    std::vector<Real> scalar_values;
    if (!vector_basis) {
        scalar_values.resize(entry.num_dofs);
    }

    for (std::size_t qp = 0; qp < points.size(); ++qp) {
        if (vector_basis) {
            basis.evaluate_vector_values(points[qp], entry.vector_values[qp]);
        } else {
            basis.evaluate_values(points[qp], scalar_values);
            for (std::size_t dof = 0; dof < entry.num_dofs; ++dof) {
                entry.scalar_values[dof * entry.num_qpts + qp] = scalar_values[dof];
            }
        }

        if (gradients && !vector_basis) {
            basis.evaluate_gradients(points[qp], entry.gradients[qp]);
        }
        if (hessians && !vector_basis) {
            basis.evaluate_hessians(points[qp], entry.hessians[qp]);
        }
    }

    return entry;
}
} // namespace basis
} // namespace FE
} // namespace svmp
