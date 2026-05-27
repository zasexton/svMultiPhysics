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
 * - The weights-based constructor reports BasisType::NURBS because it changes
 *   the semantic basis family, even though the implementation shares the same
 *   storage/evaluation class.
 * - Evaluation is performed in reference coordinates xi ∈ [-1,1] by mapping to
 *   the parametric coordinate u ∈ [u_min, u_max] implied by the knot vector.
 */

#include "Basis/BasisFunction.h"

#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace basis {

/**
 * @brief Univariate B-spline basis on a line element
 */
class BSplineBasis : public BasisFunction {
public:
    struct ActiveSupportRange {
        std::size_t first_index{0};
        std::size_t count{0};
    };

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

    BasisType basis_type() const noexcept override { return semantic_type_; }
    ElementType element_type() const noexcept override { return ElementType::Line2; }
    int dimension() const noexcept override { return 1; }
    int order() const noexcept override { return degree_; }
    std::size_t size() const noexcept override { return num_basis_; }

    std::string cache_identity() const override;
    bool cache_identity_words(std::vector<std::uint64_t>& words) const override;
    bool cache_identity_fingerprint(std::uint64_t& hash_a,
                                    std::uint64_t& hash_b) const override;

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override;

    void evaluate_gradients(const math::Vector<Real, 3>& xi,
                            std::vector<Gradient>& gradients) const override;

    void evaluate_values_and_gradients(const math::Vector<Real, 3>& xi,
                                       std::vector<Real>& values,
                                       std::vector<Gradient>& gradients) const;
    ActiveSupportRange evaluate_active_support(
        const math::Vector<Real, 3>& xi,
        std::vector<Real>& values,
        std::vector<Real>* first_derivatives = nullptr,
        std::vector<Real>* second_derivatives = nullptr) const;

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

    const std::vector<Real>& knots() const noexcept { return knots_; }
    const std::vector<Real>& weights() const noexcept { return weights_; }

    /// Whether this basis uses rational (NURBS) weights
    bool is_rational() const noexcept { return !weights_.empty(); }

private:
    enum class OutputInitialization {
        ClearAllRequestedRows,
        CallerPrecleared
    };

    BasisType semantic_type_{BasisType::BSpline};
    int degree_{0};
    std::vector<Real> knots_;
    std::size_t num_basis_{0};
    Real u_min_{Real(0)};
    Real u_max_{Real(1)};
    std::vector<Real> weights_;
    std::string cache_identity_;
    std::vector<std::uint64_t> cache_identity_words_;
    std::uint64_t cache_identity_hash_a_{0};
    std::uint64_t cache_identity_hash_b_{0};

    void rebuild_cache_identity();
    void evaluate_point_strided(const math::Vector<Real, 3>& xi,
                                std::size_t output_stride,
                                std::size_t q,
                                Real* SVMP_RESTRICT values_out,
                                Real* SVMP_RESTRICT gradients_out,
                                Real* SVMP_RESTRICT hessians_out,
                                OutputInitialization initialization =
                                    OutputInitialization::ClearAllRequestedRows) const;
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_BSPLINEBASIS_H
