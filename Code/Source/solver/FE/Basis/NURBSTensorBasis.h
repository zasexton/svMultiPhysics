/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_NURBSTENSORBASIS_H
#define SVMP_FE_BASIS_NURBSTENSORBASIS_H

/**
 * @file NURBSTensorBasis.h
 * @brief Rational tensor-product spline basis for quad/hex elements
 */

#include "Basis/BSplineBasis.h"

#include <array>
#include <string>

namespace svmp {
namespace FE {
namespace basis {

class NURBSTensorBasis : public BasisFunction {
public:
    struct ActiveTensorSupportRange {
        std::array<std::size_t, 3> first_indices{0u, 0u, 0u};
        std::array<std::size_t, 3> counts{0u, 0u, 0u};

        [[nodiscard]] std::size_t compact_size(int dimension) const noexcept {
            std::size_t result = 1u;
            for (int axis = 0; axis < dimension; ++axis) {
                result *= counts[static_cast<std::size_t>(axis)];
            }
            return result;
        }
    };

    NURBSTensorBasis(BSplineBasis bx,
                     BSplineBasis by,
                     std::vector<Real> weights,
                     std::vector<int> tensor_extents = {});

    NURBSTensorBasis(BSplineBasis bx,
                     BSplineBasis by,
                     BSplineBasis bz,
                     std::vector<Real> weights,
                     std::vector<int> tensor_extents = {});

    BasisType basis_type() const noexcept override { return BasisType::NURBS; }
    ElementType element_type() const noexcept override { return element_type_; }
    int dimension() const noexcept override { return dimension_; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return size_; }

    std::string cache_identity() const override;
    bool cache_identity_words(std::vector<std::uint64_t>& words) const override;
    bool cache_identity_fingerprint(std::uint64_t& hash_a,
                                    std::uint64_t& hash_b) const override;

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override;

    void evaluate_gradients(const math::Vector<Real, 3>& xi,
                            std::vector<Gradient>& gradients) const override;

    void evaluate_hessians(const math::Vector<Real, 3>& xi,
                           std::vector<Hessian>& hessians) const override;

    void evaluate_all(const math::Vector<Real, 3>& xi,
                      std::vector<Real>& values,
                      std::vector<Gradient>& gradients,
                      std::vector<Hessian>& hessians) const override;
    void evaluate_values_to(const math::Vector<Real, 3>& xi,
                            Real* SVMP_RESTRICT values_out) const override;
    void evaluate_gradients_to(const math::Vector<Real, 3>& xi,
                               Real* SVMP_RESTRICT gradients_out) const override;
    void evaluate_hessians_to(const math::Vector<Real, 3>& xi,
                              Real* SVMP_RESTRICT hessians_out) const override;
    void evaluate_at_quadrature_points(
        const std::vector<math::Vector<Real, 3>>& points,
        Real* SVMP_RESTRICT values_out,
        Real* SVMP_RESTRICT gradients_out,
        Real* SVMP_RESTRICT hessians_out) const override;
    void evaluate_at_quadrature_points_strided(
        const std::vector<math::Vector<Real, 3>>& points,
        std::size_t output_stride,
        Real* SVMP_RESTRICT values_out,
        Real* SVMP_RESTRICT gradients_out,
        Real* SVMP_RESTRICT hessians_out) const override;
    void fill_scalar_cache_entry(
        const std::vector<math::Vector<Real, 3>>& points,
        std::size_t output_stride,
        Real* SVMP_RESTRICT values_out,
        Real* SVMP_RESTRICT gradients_out,
        Real* SVMP_RESTRICT hessians_out) const override;
    ActiveTensorSupportRange active_tensor_support(
        const math::Vector<Real, 3>& xi) const;
    ActiveTensorSupportRange evaluate_active_support(
        const math::Vector<Real, 3>& xi,
        std::vector<std::size_t>& global_indices,
        std::vector<Real>* values = nullptr,
        std::vector<Gradient>* gradients = nullptr,
        std::vector<Hessian>* hessians = nullptr) const;

    const BSplineBasis& axis_basis(int axis) const noexcept {
        return axes_[static_cast<std::size_t>(axis)];
    }
    const std::vector<int>& tensor_extents() const noexcept { return tensor_extents_; }
    const std::vector<std::size_t>& axis_sizes() const noexcept { return axis_sizes_; }
    const std::vector<Real>& weights() const noexcept { return weights_; }

private:
    ElementType element_type_{ElementType::Unknown};
    int dimension_{0};
    int order_{0};
    std::size_t size_{0};
    std::vector<BSplineBasis> axes_;
    std::vector<std::size_t> axis_sizes_;
    std::vector<int> tensor_extents_;
    std::vector<Real> weights_;
    std::string cache_identity_;
    std::vector<std::uint64_t> cache_identity_words_;
    std::uint64_t cache_identity_hash_a_{0};
    std::uint64_t cache_identity_hash_b_{0};

    void initialize(std::vector<BSplineBasis> axes,
                    std::vector<Real> weights,
                    std::vector<int> tensor_extents);
    void rebuild_cache_identity();

    void evaluate_nonrational(const math::Vector<Real, 3>& xi,
                              std::vector<Real>& values,
                              std::vector<Gradient>* gradients,
                              std::vector<Hessian>* hessians = nullptr) const;

    void evaluate_rational_active_support(const math::Vector<Real, 3>& xi,
                                          std::vector<Real>* values,
                                          std::vector<Gradient>* gradients,
                                          std::vector<Hessian>* hessians) const;
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_NURBSTENSORBASIS_H
