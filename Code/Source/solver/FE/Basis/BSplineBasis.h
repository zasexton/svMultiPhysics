/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_BSPLINEBASIS_H
#define SVMP_FE_BASIS_BSPLINEBASIS_H

/**
 * @file BSplineBasis.h
 * @brief Minimal B-spline basis for isogeometric analysis (IGA)
 *
 * This class implements a univariate B-spline basis on a line element using
 * the Cox-de Boor recursion and analytic first derivatives. It is intended
 * as a building block for tensor-product spline bases via TensorProductBasis.
 *
 * Notes:
 * - The basis is reported as BasisType::BSpline (non-rational). For rational
 *   NURBS evaluation, use the weights-based constructor (when available).
 * - Evaluation is performed in reference coordinates xi ∈ [-1,1] by mapping to
 *   the parametric coordinate u ∈ [u_min, u_max] implied by the knot vector.
 */

#include "Basis/BasisFunction.h"

#include <vector>

namespace svmp {
namespace FE {
namespace basis {

/**
 * @brief Univariate B-spline basis on a line element
 */
class BSplineBasis : public BasisFunction {
public:
    /**
     * @brief Construct a B-spline basis of degree @p degree with knot vector @p knots
     *
     * The knot vector must be non-decreasing and have length >= degree + 2.
     * The number of basis functions is knots.size() - degree - 1.
     */
    BSplineBasis(int degree, std::vector<Real> knots);

    /**
     * @brief Construct a rational NURBS basis with weights
     *
     * When weights are provided, evaluations apply the NURBS formula:
     *   R_i(u) = N_i(u) * w_i / sum_j(N_j(u) * w_j)
     * Weights must have size == knots.size() - degree - 1.
     */
    BSplineBasis(int degree, std::vector<Real> knots, std::vector<Real> weights);

    BasisType basis_type() const noexcept override { return BasisType::BSpline; }
    ElementType element_type() const noexcept override { return ElementType::Line2; }
    int dimension() const noexcept override { return 1; }
    int order() const noexcept override { return degree_; }
    std::size_t size() const noexcept override { return num_basis_; }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override;

    void evaluate_gradients(const math::Vector<Real, 3>& xi,
                            std::vector<Gradient>& gradients) const override;

    const std::vector<Real>& knots() const noexcept { return knots_; }

    /// Whether this basis uses rational (NURBS) weights
    bool is_rational() const noexcept { return !weights_.empty(); }

private:
    int degree_{0};
    std::vector<Real> knots_;
    std::size_t num_basis_{0};
    Real u_min_{Real(0)};
    Real u_max_{Real(1)};
    std::vector<Real> weights_;
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_BSPLINEBASIS_H

