#ifndef SVMP_FE_FORMS_NITSCHE_BC_H
#define SVMP_FE_FORMS_NITSCHE_BC_H

/**
 * @file NitscheBC.h
 * @brief Weak Dirichlet boundary conditions (Nitsche) as BoundaryCondition objects
 */

#include "Forms/BoundaryCondition.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace bc {

/**
 * @brief Scalar Nitsche Dirichlet BC for diffusion-type operators
 *
 * Imposes u = g weakly on boundary marker Γ(m) via Nitsche's method:
 *  1) Consistency: -∫ k (∇u·n) v ds
 *  2) Adjoint:     ∓∫ k (∇v·n) (u-g) ds  (symmetric: -, unsymmetric: +)
 *  3) Penalty:     +∫ (γ k p^2 / h) (u-g) v ds
 *
 * where p is the trial polynomial order when available, and h is the facet-normal
 * element size h_n = 2|K|/|F| (cell volume divided by facet area).
 */
class ScalarNitscheBC final : public BoundaryCondition {
public:
    ScalarNitscheBC(int boundary_marker,
                    FormExpr value,
                    FormExpr diffusion_coeff,
                    Real penalty_gamma,
                    bool symmetric,
                    bool scale_with_p = true)
        : boundary_marker_(boundary_marker)
        , value_(std::move(value))
        , diffusion_coeff_(std::move(diffusion_coeff))
        , penalty_gamma_(penalty_gamma)
        , symmetric_(symmetric)
        , scale_with_p_(scale_with_p)
    {
        if (boundary_marker_ < 0) {
            throw std::invalid_argument("ScalarNitscheBC: boundary_marker must be >= 0");
        }
        if (!value_.isValid()) {
            throw std::invalid_argument("ScalarNitscheBC: invalid value expression");
        }
        if (!diffusion_coeff_.isValid()) {
            throw std::invalid_argument("ScalarNitscheBC: invalid diffusion coefficient expression");
        }
        if (!(penalty_gamma_ > Real(0.0))) {
            throw std::invalid_argument("ScalarNitscheBC: penalty_gamma must be > 0");
        }
        if (value_.hasTest() || value_.hasTrial()) {
            throw std::invalid_argument("ScalarNitscheBC: value must not contain test/trial functions");
        }
        if (diffusion_coeff_.hasTest() || diffusion_coeff_.hasTrial()) {
            throw std::invalid_argument("ScalarNitscheBC: diffusion_coeff must not contain test/trial functions");
        }
    }

    [[nodiscard]] int boundaryMarker() const override { return boundary_marker_; }

    void contributeToResidual(FormExpr& residual,
                              const FormExpr& u,
                              const FormExpr& v) const override
    {
        const auto n = FormExpr::normal();
        const auto h = (2.0 * FormExpr::cellVolume()) / FormExpr::facetArea();

        int p = 1;
        if (scale_with_p_) {
            p = detail::polynomialOrderOrDefault(u, /*default_order=*/1);
            if (p < 1) {
                p = 1;
            }
        }
        const auto p2 = FormExpr::constant(static_cast<Real>(p * p));
        const auto penalty = FormExpr::constant(penalty_gamma_) * diffusion_coeff_ * p2 / h;

        const auto diff = u - value_;

        residual = residual - (diffusion_coeff_ * inner(grad(u), n) * v).ds(boundary_marker_);
        if (symmetric_) {
            residual = residual - (diffusion_coeff_ * inner(grad(v), n) * diff).ds(boundary_marker_);
        } else {
            residual = residual + (diffusion_coeff_ * inner(grad(v), n) * diff).ds(boundary_marker_);
        }
        residual = residual + (penalty * diff * v).ds(boundary_marker_);
    }

    [[nodiscard]] std::vector<StrongDirichlet> getStrongConstraints(FieldId /*field_id*/) const override
    {
        return {};
    }

private:
    int boundary_marker_{-1};
    FormExpr value_{};
    FormExpr diffusion_coeff_{};
    Real penalty_gamma_{10.0};
    bool symmetric_{true};
    bool scale_with_p_{true};
};

} // namespace bc
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_NITSCHE_BC_H
