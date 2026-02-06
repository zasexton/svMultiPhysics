/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "BasisCache.h"
#include <numeric>

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
        std::type_index(typeid(basis)),
        basis.basis_type(),
        basis.element_type(),
        basis.order(),
        quadrature_hash(quad),
        gradients,
        hessians,
        basis.is_vector_valued()
    };

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return *(it->second);
    }

    auto entry = std::make_shared<BasisCacheEntry>(compute(basis, quad, gradients, hessians));
    cache_.emplace(key, entry);
    return *entry;
}

void BasisCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

std::size_t BasisCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
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

std::size_t BasisCache::quadrature_hash(const quadrature::QuadratureRule& quad) const {
    std::size_t h = quad.num_points();
    h ^= static_cast<std::size_t>(quad.dimension()) << 8;
    h ^= static_cast<std::size_t>(quad.order()) << 16;
    const auto& pts = quad.points();
    for (std::size_t i = 0; i < std::min<std::size_t>(pts.size(), 4); ++i) {
        h ^= std::hash<Real>()(pts[i][0]) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<Real>()(pts[i][1]) + 0x9e3779b97f4a7c15ULL + (h << 5) + (h >> 3);
    }
    return h;
}

} // namespace basis
} // namespace FE
} // namespace svmp
