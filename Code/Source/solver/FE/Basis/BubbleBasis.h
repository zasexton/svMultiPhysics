/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_BUBBLEBASIS_H
#define SVMP_FE_BASIS_BUBBLEBASIS_H

/**
 * @file BubbleBasis.h
 * @brief Scalar interior bubble functions for element enrichment
 *
 * Provides a single bubble function per element that vanishes on all
 * boundary faces/edges. Used in MINI elements (P1+bubble for Stokes),
 * stabilized methods, and enrichment via EnrichedSpace.
 *
 * Supported topologies (including the listed node-count aliases):
 *   - Line2 / Line3:
 *       b = 1 - xi^2                                           (quadratic, 1 DOF)
 *   - Triangle3 / Triangle6:
 *       b = 27 * L0 * L1 * L2                                  (cubic, 1 DOF)
 *   - Tetra4 / Tetra10:
 *       b = 256 * L0 * L1 * L2 * L3                            (quartic, 1 DOF)
 *   - Quad4 / Quad8 / Quad9:
 *       b = (1 - xi^2)(1 - eta^2)                              (biquadratic, 1 DOF)
 *   - Hex8 / Hex20 / Hex27:
 *       b = (1-xi^2)(1-eta^2)(1-zeta^2)                        (triquadratic, 1 DOF)
 *   - Wedge6 / Wedge15 / Wedge18:
 *       b = 27 * L0 * L1 * L2 * (1-z^2)                        (quintic, 1 DOF)
 *   - Pyramid5 / Pyramid13 / Pyramid14:
 *       b = (3125/256) z[(1-z)^2-x^2][(1-z)^2-y^2]             (quintic, 1 DOF)
 */

#include "BasisFunction.h"

namespace svmp {
namespace FE {
namespace basis {

class BubbleBasis : public BasisFunction {
public:
    explicit BubbleBasis(ElementType type);

    BasisType basis_type() const noexcept override { return BasisType::Bubble; }
    ElementType element_type() const noexcept override { return element_type_; }
    int dimension() const noexcept override { return dimension_; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return 1; }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override;

    void evaluate_gradients(const math::Vector<Real, 3>& xi,
                            std::vector<Gradient>& gradients) const override;

    void evaluate_hessians(const math::Vector<Real, 3>& xi,
                           std::vector<Hessian>& hessians) const override;

private:
    ElementType element_type_;
    int dimension_;
    int order_;
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_BUBBLEBASIS_H
