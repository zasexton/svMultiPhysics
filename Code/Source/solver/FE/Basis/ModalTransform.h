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
#include <memory>
#include <mutex>
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

    const std::vector<std::vector<Real>>& vandermonde() const noexcept;

    /// Diagnostic accessor. Materializes and caches the dense inverse on first
    /// use; repeated nodal-to-modal transforms should use nodal_to_modal().
    const std::vector<std::vector<Real>>& vandermonde_inverse() const;

    /// Rank-revealing condition-number estimate from the Vandermonde SVD.
    Real condition_number() const;

private:
    struct SolverStorage;
    struct TransformData;

    const BasisFunction& modal_;
    const LagrangeBasis& nodal_;
    std::shared_ptr<const TransformData> transform_data_;

    static std::shared_ptr<const TransformData> get_or_build_transform_data(
        const BasisFunction& modal_basis,
        const LagrangeBasis& nodal_basis);
    void materialize_vandermonde_inverse() const;
};

} // namespace basis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_BASIS_MODALTRANSFORM_H
