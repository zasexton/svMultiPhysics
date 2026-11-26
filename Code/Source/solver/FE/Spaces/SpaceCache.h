/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_SPACECACHE_H
#define SVMP_FE_SPACES_SPACECACHE_H

/**
 * @file SpaceCache.h
 * @brief Lightweight cache for basis evaluations on reference elements
 *
 * Provides scalar views of the owning basis tabulation for the element's
 * basis and actual quadrature points.
 */

#include "Core/Types.h"
#include "Basis/BasisCache.h"
#include "Elements/Element.h"
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace svmp {
namespace FE {
namespace spaces {

class SpaceCache {
public:
    /// Cached scalar basis evaluations
    struct CachedData {
        std::size_t num_dofs{0};
        std::size_t num_qpts{0};
        // basis_values[i * num_qpts + q] = phi_i(x_q)
        // The handle retains this storage across a BasisCache clear.
        std::span<const Real> basis_values;
        basis::BasisCacheHandle tabulation;

        [[nodiscard]] Real basisValue(std::size_t dof, std::size_t qpt) const noexcept {
            return basis_values[dof * num_qpts + qpt];
        }

        [[nodiscard]] std::span<const Real> basisValuesForDof(std::size_t dof) const noexcept {
            if (num_qpts == 0) return {};
            return std::span<const Real>(basis_values.data() + dof * num_qpts, num_qpts);
        }
    };

    /// Get or compute cached data for element and polynomial order
    const CachedData& get(const elements::Element& elem, int poly_order);

    /// Clear all cached data (mainly for testing)
    void clear();

    /// Singleton access
    static SpaceCache& instance();

private:
    SpaceCache() = default;

    std::unordered_map<const basis::BasisCacheEntry*, CachedData> cache_;
    std::mutex mutex_;
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_SPACECACHE_H
