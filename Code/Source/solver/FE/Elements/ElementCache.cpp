/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Elements/ElementCache.h"
#include "Core/FEException.h"

namespace svmp {
namespace FE {
namespace elements {

using svmp::FE::basis::BasisCache;
using svmp::FE::geometry::JacobianCache;

ElementCache& ElementCache::instance() {
    static ElementCache inst;
    return inst;
}

ElementCacheEntry ElementCache::get(const Element& element,
                                    const geometry::GeometryMapping& mapping) const {
    ElementCacheEntry out;

    const auto& quad = *element.quadrature();
    const auto& basis_entry =
        BasisCache::instance().get_or_compute(element.basis(), quad, /*gradients=*/true, /*hessians=*/false);
    const auto& jac_entry =
        JacobianCache::instance().get_or_compute(mapping, quad);

    out.basis    = &basis_entry;
    out.jacobian = &jac_entry;
    return out;
}

std::size_t ElementCache::size() const {
    return BasisCache::instance().size() + JacobianCache::instance().size();
}

void ElementCache::clear() {
    BasisCache::instance().clear();
    JacobianCache::instance().clear();
}

std::vector<ElementCacheEntry> ElementCache::get_batch(
    const std::vector<const Element*>& elements,
    const std::vector<const geometry::GeometryMapping*>& mappings,
    const BatchEvaluationHints& hints) const {

    std::vector<ElementCacheEntry> results;
    results.reserve(elements.size());

    if (elements.size() != mappings.size()) {
        throw FEException(
            "ElementCache::get_batch: elements and mappings vectors must have the same size",
            __FILE__, __LINE__, __func__, FEStatus::InvalidArgument);
    }

    // For now, delegate to single-element get() calls.
    // Future optimization: when hints.batch_size > 1 and elements share
    // the same type, we can use SIMD-optimized batch evaluation in the
    // underlying BasisCache and JacobianCache.
    //
    // The hints parameter is preserved for future SIMD optimization paths:
    // - hints.prefetch: could trigger cache line prefetching for next batch
    // - hints.align_memory: ensures returned data is SIMD-aligned
    // - hints.simd_width: selects between SSE/AVX/AVX-512 evaluation kernels
    (void)hints; // Mark as intentionally unused for now

    for (std::size_t i = 0; i < elements.size(); ++i) {
        if (elements[i] && mappings[i]) {
            results.push_back(get(*elements[i], *mappings[i]));
        } else {
            results.push_back(ElementCacheEntry{});
        }
    }

    return results;
}

int ElementCache::optimal_simd_width() noexcept {
    // Compile-time detection of SIMD support
    // Future: could use CPUID runtime detection for more accuracy
#if defined(__AVX512F__)
    return 16; // AVX-512: 512 bits = 16 floats or 8 doubles
#elif defined(__AVX__)
    return 8;  // AVX: 256 bits = 8 floats or 4 doubles
#elif defined(__SSE__)
    return 4;  // SSE: 128 bits = 4 floats or 2 doubles
#else
    return 1;  // No SIMD support detected
#endif
}

} // namespace elements
} // namespace FE
} // namespace svmp

