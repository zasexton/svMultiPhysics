/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_BATCHEVALUATOR_H
#define SVMP_FE_BASIS_BATCHEVALUATOR_H

/**
 * @file BatchEvaluator.h
 * @brief SIMD-optimized batch evaluation of basis functions
 *
 * Provides high-performance evaluation of basis functions at multiple
 * quadrature points simultaneously. Uses SIMD intrinsics when available
 * to vectorize across quadrature points, significantly improving throughput
 * for element-level computations in FE assembly.
 */

#include "BasisFunction.h"
#include "Quadrature/QuadratureRule.h"
#include "Math/AlignedAllocator.h"
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace basis {

namespace detail {

template<typename T, std::size_t Alignment = 64>
class AlignedNoInitAllocator : public math::simd::AlignedAllocator<T, Alignment> {
public:
    using value_type = T;

    AlignedNoInitAllocator() noexcept = default;

    template<class U>
    AlignedNoInitAllocator(const AlignedNoInitAllocator<U, Alignment>&) noexcept {}

    template<class U>
    struct rebind { using other = AlignedNoInitAllocator<U, Alignment>; };

    template<class U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        if constexpr (std::is_trivially_default_constructible_v<U>) {
            ::new (static_cast<void*>(p)) U;
        } else {
            ::new (static_cast<void*>(p)) U();
        }
    }

    template<class U, class... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
};

} // namespace detail

using AlignedRealVector = std::vector<Real, detail::AlignedNoInitAllocator<Real>>;

/**
 * @brief Storage layout for batched basis evaluation results
 *
 * Data is stored in Structure-of-Arrays (SoA) format for optimal
 * SIMD memory access patterns. Rows are padded to quad_stride for aligned
 * SIMD loads; accessors expose the logical num_quad_points range.
 *
 * values: [N_0(q_0), ..., N_0(q_{stride-1}), N_1(q_0), ...]
 *
 * This layout enables vectorization across quadrature points.
 */
struct BatchedBasisData {
    /// Basis function values at all quadrature points
    /// Layout: [num_basis][quad_stride]
    AlignedRealVector values;

    /// Gradients at all quadrature points
    /// Layout: [num_basis][3][quad_stride] (interleaved by dimension)
    AlignedRealVector gradients;

    /// Hessians at all quadrature points (optional)
    /// Layout: [num_basis][9][quad_stride]
    AlignedRealVector hessians;

    /// Number of basis functions
    std::size_t num_basis = 0;

    /// Number of quadrature points
    std::size_t num_quad_points = 0;

    /// Padded per-basis row length used by values/gradients/Hessians
    std::size_t quad_stride = 0;

    /// Whether gradients are populated
    bool has_gradients = false;

    /// Whether Hessians are populated
    bool has_hessians = false;

    /**
     * @brief Access value N_i(q_j)
     */
    [[nodiscard]] Real value(std::size_t i, std::size_t j) const noexcept {
        return values[i * quad_stride + j];
    }

    /**
     * @brief Access gradient dN_i/d(xi_d) at q_j
     * @param i Basis function index
     * @param d Dimension (0, 1, or 2)
     * @param j Quadrature point index
     */
    [[nodiscard]] Real gradient(std::size_t i, std::size_t d, std::size_t j) const noexcept {
        return gradients[(i * 3 + d) * quad_stride + j];
    }

    /**
     * @brief Access Hessian d^2N_i/(d xi_d1 d xi_d2) at q_j
     * @param i Basis function index
     * @param d1 First dimension
     * @param d2 Second dimension
     * @param j Quadrature point index
     */
    [[nodiscard]] Real hessian(std::size_t i,
                               std::size_t d1,
                               std::size_t d2,
                               std::size_t j) const noexcept {
        return hessians[(i * 9 + d1 * 3 + d2) * quad_stride + j];
    }

    /**
     * @brief Get pointer to contiguous values for basis function i
     */
    [[nodiscard]] const Real* values_for_basis(std::size_t i) const noexcept {
        return values.data() + i * quad_stride;
    }

    /**
     * @brief Get pointer to contiguous gradient component for basis function i
     */
    [[nodiscard]] const Real* gradients_for_basis(std::size_t i, std::size_t d) const noexcept {
        return gradients.data() + (i * 3 + d) * quad_stride;
    }
};

/**
 * @brief High-performance batch evaluator for basis functions
 *
 * Evaluates all basis functions at all quadrature points in a single
 * call, utilizing SIMD vectorization for improved throughput. The
 * results are stored in SoA format for optimal cache utilization
 * during subsequent computations (e.g., stiffness matrix assembly).
 *
 * This evaluator is intentionally scalar-basis only. Vector-valued bases use
 * BasisCacheEntry SoA accessors until an assembly caller needs a padded vector
 * batch view with a separate layout contract.
 */
class BatchEvaluator {
public:
    /**
     * @brief Create a batch evaluator for the given basis and quadrature
     * @param basis The basis function to evaluate
     * @param quad The quadrature rule providing evaluation points
     * @param compute_gradients Whether to compute gradients
     * @param compute_hessians Whether to compute Hessians
     */
    BatchEvaluator(const BasisFunction& basis,
                   const quadrature::QuadratureRule& quad,
                   bool compute_gradients = true,
                   bool compute_hessians = false);

    /**
     * @brief Get the precomputed batched data
     */
    [[nodiscard]] const BatchedBasisData& data() const noexcept { return data_; }

    /**
     * @brief Number of basis functions
     */
    [[nodiscard]] std::size_t num_basis() const noexcept { return data_.num_basis; }

    /**
     * @brief Number of quadrature points
     */
    [[nodiscard]] std::size_t num_quad_points() const noexcept { return data_.num_quad_points; }

    /**
     * @brief Reference dimension of the evaluated scalar basis
     */
    [[nodiscard]] int dimension() const noexcept { return dimension_; }

private:
    BatchedBasisData data_;
    int dimension_;
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_BATCHEVALUATOR_H
