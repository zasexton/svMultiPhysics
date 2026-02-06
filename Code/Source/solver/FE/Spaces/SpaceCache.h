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
 * This cache avoids repeated evaluation of scalar basis functions at
 * quadrature points for the same (element type, polynomial order,
 * quadrature order) configuration. It is intentionally independent of
 * Mesh and uses only FE Basis/Quadrature interfaces.
 */

#include "Core/Types.h"
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
    /// Cache key: (element type, polynomial order, quadrature order)
    struct Key {
        ElementType elem_type;
        int poly_order;
        int quad_order;

        bool operator==(const Key& other) const noexcept {
            return elem_type == other.elem_type &&
                   poly_order == other.poly_order &&
                   quad_order == other.quad_order;
        }
    };

    struct KeyHash {
        std::size_t operator()(const Key& key) const noexcept {
            std::size_t h1 = std::hash<int>{}(static_cast<int>(key.elem_type));
            std::size_t h2 = std::hash<int>{}(key.poly_order);
            std::size_t h3 = std::hash<int>{}(key.quad_order);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    /// Cached scalar basis evaluations
    struct CachedData {
        std::size_t num_dofs{0};
        std::size_t num_qpts{0};
        // basis_values[i * num_qpts + q] = phi_i(x_q)
        std::vector<Real> basis_values;

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

    std::unordered_map<Key, CachedData, KeyHash> cache_;
    std::mutex mutex_;
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_SPACECACHE_H
