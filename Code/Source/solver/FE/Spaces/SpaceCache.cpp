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
SpaceCache::get(const elements::Element& elem, int /*poly_order*/) {
    const auto& basis = elem.basis();
    const auto quad = elem.quadrature();
    if (basis.is_vector_valued() || !quad) {
        static const CachedData empty;
        return empty;
    }
    auto handle = basis::BasisCache::instance().prewarm_handle(basis, *quad, false, false);
    const auto& entry = handle.entry();
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = cache_.try_emplace(&entry, CachedData{
        entry.num_dofs, entry.num_qpts, entry.scalar_values, std::move(handle)}).first;
    return it->second;
}

void SpaceCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

} // namespace spaces
} // namespace FE
} // namespace svmp
