/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_ELEMENTS_ELEMENTCACHE_H
#define SVMP_FE_ELEMENTS_ELEMENTCACHE_H

/**
 * @file ElementCache.h
 * @brief Convenience access to basis and Jacobian caches for elements
 *
 * This wrapper does not introduce a new cache of its own; it simply ties
 * together the existing `BasisCache` and `JacobianCache` entries associated
 * with a given element, geometry mapping, and quadrature rule.
 */

#include "Elements/Element.h"
#include "Basis/BasisCache.h"
#include "Geometry/JacobianCache.h"

#include <vector>

namespace svmp {
namespace FE {
namespace elements {

struct ElementCacheEntry {
    const basis::BasisCacheEntry*    basis{nullptr};
    const geometry::JacobianCacheEntry* jacobian{nullptr};
};

/**
 * @brief Hints for SIMD batch evaluation optimization
 *
 * When evaluating multiple elements of the same type, callers can use these
 * hints to enable vectorized evaluation paths in the underlying caches.
 *
 * @note The current implementation of `ElementCache::get_batch()` preserves
 *       these hints for forward compatibility but does not yet use them to
 *       change evaluation behavior.
 */
struct BatchEvaluationHints {
    std::size_t batch_size{1};       ///< Number of elements to process in batch
    bool        prefetch{false};     ///< Enable cache prefetching for batch
    bool        align_memory{true};  ///< Ensure SIMD-aligned memory access
    int         simd_width{4};       ///< Preferred SIMD width (4 for SSE, 8 for AVX)
};

class ElementCache {
public:
    static ElementCache& instance();

    /**
     * @brief Retrieve cached basis and Jacobian data for an element
     *
     * The underlying BasisCache and JacobianCache are populated on-demand.
     * The returned pointers remain valid until the respective caches are
     * cleared.
     */
    ElementCacheEntry get(const Element& element,
                          const geometry::GeometryMapping& mapping) const;

    /**
     * @brief Retrieve cached data for multiple elements with SIMD hints
     *
     * This method is optimized for batch processing of elements of the same
     * type. When the hints specify batch_size > 1, the underlying caches may
     * use SIMD-optimized evaluation paths.
     *
     * @param elements Vector of elements to process (should be same type)
     * @param mappings Corresponding geometry mappings for each element
     * @param hints    SIMD optimization hints for batch processing
     * @return Vector of cache entries, one per element
     *
     * @note For best performance, ensure elements are of the same type and
     *       mappings share compatible quadrature rules.
     *
     * @note Currently, this method delegates to repeated `get()` calls and the
     *       `hints` parameter is ignored.
     */
    std::vector<ElementCacheEntry> get_batch(
        const std::vector<const Element*>& elements,
        const std::vector<const geometry::GeometryMapping*>& mappings,
        const BatchEvaluationHints& hints = {}) const;

    /**
     * @brief Query optimal SIMD width for the current platform
     *
     * Returns the recommended simd_width value for BatchEvaluationHints
     * based on CPU feature detection (SSE, AVX, AVX-512).
     */
    static int optimal_simd_width() noexcept;

    /// Total number of cached basis + Jacobian entries across underlying caches
    std::size_t size() const;

    /// Clear all underlying caches
    void clear();

private:
    ElementCache() = default;
};

} // namespace elements
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ELEMENTS_ELEMENTCACHE_H
