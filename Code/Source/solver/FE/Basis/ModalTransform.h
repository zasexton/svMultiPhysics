/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_BASIS_MODALTRANSFORM_H
#define SVMP_FE_BASIS_MODALTRANSFORM_H

/**
 * @file ModalTransform.h
 * @brief Modal-to-nodal transformation utilities
 */

#include "BasisFunction.h"
#include "LagrangeBasis.h"
#include <vector>

namespace svmp {
namespace FE {
namespace basis {

class ModalTransform {
public:
    ModalTransform(const BasisFunction& modal_basis,
                   const LagrangeBasis& nodal_basis);

    /// Apply modal-to-nodal transform (nodal = V * modal)
    std::vector<Real> modal_to_nodal(const std::vector<Real>& modal_coeffs) const;

    /// Apply nodal-to-modal transform (modal = V^{-1} * nodal)
    std::vector<Real> nodal_to_modal(const std::vector<Real>& nodal_values) const;

    const std::vector<std::vector<Real>>& vandermonde() const noexcept { return vandermonde_; }
    const std::vector<std::vector<Real>>& vandermonde_inverse() const noexcept { return vandermonde_inv_; }

    /// Rough condition-number estimate using infinity norm
    Real condition_number() const;

private:
    const BasisFunction& modal_;
    const LagrangeBasis& nodal_;
    std::vector<std::vector<Real>> vandermonde_;
    std::vector<std::vector<Real>> vandermonde_inv_;

    void compute_vandermonde();
    void invert_vandermonde();
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_MODALTRANSFORM_H
