/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_SPECTRALBASIS_H
#define SVMP_FE_BASIS_SPECTRALBASIS_H

/**
 * @file SpectralBasis.h
 * @brief Gauss-Lobatto-Legendre spectral bases for high-order elements
 */

#include "BasisFunction.h"
#include <memory>
#include <vector>

namespace svmp {
namespace FE {
namespace basis {

class SpectralBasis : public BasisFunction {
public:
    SpectralBasis(ElementType type, int order);

    BasisType basis_type() const noexcept override { return BasisType::Spectral; }
    ElementType element_type() const noexcept override { return element_type_; }
    int dimension() const noexcept override { return dimension_; }
    int order() const noexcept override { return order_; }
    std::size_t size() const noexcept override { return size_; }

    const std::vector<Real>& nodes_1d() const noexcept { return nodes_1d_; }

    void evaluate_values(const math::Vector<Real, 3>& xi,
                         std::vector<Real>& values) const override;
    void evaluate_gradients(const math::Vector<Real, 3>& xi,
                            std::vector<Gradient>& gradients) const override;

private:
    ElementType element_type_;
    int dimension_;
    int order_;
    std::size_t size_;
    std::vector<Real> nodes_1d_;
    std::vector<Real> barycentric_weights_;

    void build_nodes();
    std::vector<Real> eval_1d(Real x) const;
    std::vector<Real> eval_1d_derivative(Real x) const;
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_SPECTRALBASIS_H
