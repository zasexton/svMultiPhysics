/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Spaces/SpaceCache.h"

#include "Basis/BasisCache.h"
#include "Basis/BasisFunction.h"

namespace svmp {
namespace FE {
namespace spaces {

SpaceCache& SpaceCache::instance() {
    static SpaceCache cache;
    return cache;
}

const SpaceCache::CachedData&
SpaceCache::get(const elements::Element& elem, int poly_order) {
    const auto& basis = elem.basis();
    const auto quad = elem.quadrature();

    const int quad_order = quad ? quad->order() : 0;
    Key key{elem.element_type(), poly_order, quad_order};

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
    }

    // Compute scalar basis values; vector-valued bases are not cached
    CachedData data;
    if (!basis.is_vector_valued() && quad) {
        const auto& entry =
            basis::BasisCache::instance().get_or_compute(basis, *quad, /*gradients=*/false, /*hessians=*/false);

        const std::size_t ndofs = basis.size();
        const std::size_t nq = quad->num_points();

        data.num_dofs = ndofs;
        data.num_qpts = nq;
        FE_CHECK_ARG(entry.num_dofs == ndofs, "SpaceCache: BasisCache dof count mismatch");
        FE_CHECK_ARG(entry.num_qpts == nq, "SpaceCache: BasisCache quadrature size mismatch");
        FE_CHECK_ARG(entry.scalar_values.size() == ndofs * nq,
                     "SpaceCache: BasisCache scalar storage size mismatch");
        data.basis_values = std::span<const Real>(entry.scalar_values.data(), entry.scalar_values.size());
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.emplace(key, std::move(data)).first;
    return it->second;
}

void SpaceCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

} // namespace spaces
} // namespace FE
} // namespace svmp
