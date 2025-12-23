/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_HERMITEBASIS_H
#define SVMP_FE_BASIS_HERMITEBASIS_H

/**
 * @file HermiteBasis.h
 * @brief Hermite-type scalar basis functions for higher-continuity spaces
 *
 * This header provides a minimal Hermite basis family for scalar C¹ spaces.
 * The current implementation supports:
 *   - 1D cubic Hermite on Line2 (4 DOFs: values + first derivatives at endpoints)
 *   - 2D bicubic Hermite on Quad4 (16 DOFs: values, ∂/∂x, ∂/∂y, ∂²/(∂x∂y) at corners)
 *
 * Additional element types (e.g., tensor-product Hermite on hexes) can be
 * added in the future following the same family interface.
 */

#include "BasisFunction.h"

namespace svmp {
namespace FE {
namespace basis {

/**
 * @brief Cubic Hermite scalar basis on a 2-node line element
 *
 * Reference domain: ξ ∈ [-1, 1], mapped to t = (ξ + 1) / 2 ∈ [0, 1].
 * DOF ordering:
 *  - dof 0: value at left node (t=0)
 *  - dof 1: value at right node (t=1)
 *  - dof 2: derivative at left node (d/dt at t=0)
 *  - dof 3: derivative at right node (d/dt at t=1)
 *
 * Shape functions in parameter t:
 *  - H1(t) = 1 - 3t² + 2t³   (value at left node)
 *  - H2(t) = t - 2t² + t³    (slope at left node)
 *  - H3(t) = 3t² - 2t³       (value at right node)
 *  - H4(t) = -t² + t³        (slope at right node)
 *
 * Currently only ElementType::Line2 and ElementType::Quad4 with cubic order (3)
 * are supported. The constructor validates the requested element configuration
 * and throws for unsupported combinations, so the Hermite family can be
 * extended safely.
 */
class HermiteBasis : public BasisFunction {
public:
    HermiteBasis(ElementType element_type,
                 int order);

    BasisType basis_type() const noexcept override { return BasisType::Custom; }
    ElementType element_type() const noexcept override { return element_type_; }
    int dimension() const noexcept override { return dimension_; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return size_; }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override;

    void evaluate_gradients(const math::Vector<Real, 3>& xi,
                            std::vector<Gradient>& gradients) const override;

private:
    ElementType element_type_{ElementType::Unknown};
    int dimension_{0};
    int order_{0};
    std::size_t size_{0};
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_HERMITEBASIS_H
