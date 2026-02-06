/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_BASISCACHE_H
#define SVMP_FE_BASIS_BASISCACHE_H

/**
 * @file BasisCache.h
 * @brief Cache for basis evaluations at quadrature points
 */

#include "BasisFunction.h"
#include "Quadrature/QuadratureRule.h"
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <typeindex>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace basis {

struct BasisCacheKey {
    std::type_index dynamic_type;
    BasisType basis;
    ElementType elem;
    int order;
    std::size_t quad_hash;
    bool with_gradients;
    bool with_hessians;
    bool vector;

    bool operator==(const BasisCacheKey& other) const noexcept {
        return dynamic_type == other.dynamic_type &&
               basis == other.basis &&
               elem == other.elem &&
               order == other.order &&
               quad_hash == other.quad_hash &&
               with_gradients == other.with_gradients &&
               with_hessians == other.with_hessians &&
               vector == other.vector;
    }
};

struct BasisCacheKeyHash {
    std::size_t operator()(const BasisCacheKey& key) const noexcept {
        std::size_t h0 = std::hash<std::type_index>()(key.dynamic_type);
        std::size_t h1 = std::hash<int>()(static_cast<int>(key.basis));
        std::size_t h2 = std::hash<int>()(static_cast<int>(key.elem));
        std::size_t h3 = std::hash<int>()(key.order);
        std::size_t h4 = std::hash<std::size_t>()(key.quad_hash);
        std::size_t h5 = std::hash<bool>()(key.with_gradients);
        std::size_t h6 = std::hash<bool>()(key.with_hessians);
        std::size_t h7 = std::hash<bool>()(key.vector);
        return (((((h0 ^ (h1 << 1)) ^ (h2 << 2)) ^ (h3 << 3)) ^ (h4 << 4)) ^ (h5 << 5)) ^ (h6 << 6) ^ (h7 << 7);
    }
};

struct BasisCacheEntry {
    std::size_t num_qpts{0};
    std::size_t num_dofs{0};
    // Scalar basis values in dof-major SoA layout: [dof * num_qpts + qp].
    std::vector<Real> scalar_values;
    std::vector<std::vector<Gradient>> gradients;
    std::vector<std::vector<Hessian>> hessians;
    std::vector<std::vector<math::Vector<Real,3>>> vector_values;

    [[nodiscard]] Real scalarValue(std::size_t dof, std::size_t qp) const noexcept {
        return scalar_values[dof * num_qpts + qp];
    }

    [[nodiscard]] std::span<const Real> scalarValuesForDof(std::size_t dof) const noexcept {
        if (num_qpts == 0) return {};
        return std::span<const Real>(scalar_values.data() + dof * num_qpts, num_qpts);
    }
};

class BasisCache {
public:
    static BasisCache& instance();

    const BasisCacheEntry& get_or_compute(
        const BasisFunction& basis,
        const quadrature::QuadratureRule& quad,
        bool gradients = true,
        bool hessians = false);

    void clear();
    std::size_t size() const;

private:
    BasisCache() = default;
    BasisCache(const BasisCache&) = delete;
    BasisCache& operator=(const BasisCache&) = delete;

    BasisCacheEntry compute(const BasisFunction& basis,
                            const quadrature::QuadratureRule& quad,
                            bool gradients,
                            bool hessians) const;

    std::size_t quadrature_hash(const quadrature::QuadratureRule& quad) const;

    mutable std::mutex mutex_;
    std::unordered_map<BasisCacheKey, std::shared_ptr<BasisCacheEntry>, BasisCacheKeyHash> cache_;
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_BASISCACHE_H
