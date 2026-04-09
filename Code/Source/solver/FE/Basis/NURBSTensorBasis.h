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

namespace svmp {
namespace FE {
namespace basis {

class NURBSTensorBasis : public BasisFunction {
public:
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

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override;

    void evaluate_gradients(const math::Vector<Real, 3>& xi,
                            std::vector<Gradient>& gradients) const override;

    const std::vector<int>& tensor_extents() const noexcept { return tensor_extents_; }
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

    void initialize(std::vector<BSplineBasis> axes,
                    std::vector<Real> weights,
                    std::vector<int> tensor_extents);

    void evaluate_nonrational(const math::Vector<Real, 3>& xi,
                              std::vector<Real>& values,
                              std::vector<Gradient>* gradients) const;
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_NURBSTENSORBASIS_H
