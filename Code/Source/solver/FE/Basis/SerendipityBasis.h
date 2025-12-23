/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_SERENDIPITYBASIS_H
#define SVMP_FE_BASIS_SERENDIPITYBASIS_H

/**
 * @file SerendipityBasis.h
 * @brief Reduced-degree-of-freedom serendipity bases
 */

#include "BasisFunction.h"

namespace svmp {
namespace FE {
namespace basis {

class SerendipityBasis : public BasisFunction {
public:
    SerendipityBasis(ElementType type, int order, bool geometry_mode = false);

    BasisType basis_type() const noexcept override { return BasisType::Serendipity; }
    ElementType element_type() const noexcept override { return element_type_; }
    int dimension() const noexcept override { return dimension_; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return size_; }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override;

    void evaluate_gradients(const math::Vector<Real, 3>& xi,
                            std::vector<Gradient>& gradients) const override;

private:
    ElementType element_type_;
    int dimension_;
    int order_;
    std::size_t size_;

    // When true, this basis is used purely for geometry mapping and may use
    // reduced polynomial order (e.g., Hex20 geometry as Hex8).
    bool geometry_mode_;
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_SERENDIPITYBASIS_H
