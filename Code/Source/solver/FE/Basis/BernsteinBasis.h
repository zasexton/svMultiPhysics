/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_BERNSTEINBASIS_H
#define SVMP_FE_BASIS_BERNSTEINBASIS_H

/**
 * @file BernsteinBasis.h
 * @brief Bernstein polynomial basis for isogeometric-style formulations
 */

#include "BasisFunction.h"
#include <array>
#include <span>

namespace svmp {
namespace FE {
namespace basis {

class BernsteinBasis : public BasisFunction {
public:
    BernsteinBasis(ElementType type, int order);

    BasisType basis_type() const noexcept override { return BasisType::Bernstein; }
    ElementType element_type() const noexcept override { return element_type_; }
    int dimension() const noexcept override { return dimension_; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return size_; }
    bool cache_identity_is_structural() const noexcept override { return true; }

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

private:
    ElementType element_type_;
    int dimension_;
    int order_;
    std::size_t size_;

    std::vector<std::array<int, 4>> simplex_indices_;
    std::vector<Real> binomial_coefficients_;
    std::vector<Real> coefficients_;

    Real binomial(int n, int k) const;
    void evaluate_into_raw(const math::Vector<Real, 3>& xi,
                           std::size_t output_stride,
                           std::size_t output_offset,
                           Real* SVMP_RESTRICT values_out,
                           Real* SVMP_RESTRICT gradients_out,
                           Real* SVMP_RESTRICT hessians_out) const;
    void evaluate_into(const math::Vector<Real, 3>& xi,
                       std::span<Real> values,
                       std::span<Gradient> gradients,
                       std::span<Hessian> hessians) const;
    template <typename Writer>
    void evaluate_into_writer(const math::Vector<Real, 3>& xi,
                              Writer& writer) const;
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_BERNSTEINBASIS_H
