/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_HIERARCHICALBASIS_H
#define SVMP_FE_BASIS_HIERARCHICALBASIS_H

/**
 * @file HierarchicalBasis.h
 * @brief Modal/hierarchical basis functions for p-adaptivity
 */

#include "BasisFunction.h"
#include "OrthogonalPolynomials.h"
#include <array>

namespace svmp {
namespace FE {
namespace basis {

/**
 * @brief Hierarchical/modal basis using orthogonal polynomials
 */
class HierarchicalBasis : public BasisFunction {
public:
    HierarchicalBasis(ElementType type, int order);

    BasisType basis_type() const noexcept override { return BasisType::Hierarchical; }
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

    std::vector<std::array<int, 3>> tensor_indices_;
    std::vector<std::array<int, 4>> simplex_indices_;
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_HIERARCHICALBASIS_H
