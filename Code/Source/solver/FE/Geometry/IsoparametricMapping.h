/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_GEOMETRY_ISOPARAMETRICMAPPING_H
#define SVMP_FE_GEOMETRY_ISOPARAMETRICMAPPING_H

/**
 * @file IsoparametricMapping.h
 * @brief Isoparametric geometric mapping driven by a basis function
 */

#include "GeometryMapping.h"
#include "FE/Basis/BasisFunction.h"
#include <memory>

namespace svmp {
namespace FE {
namespace geometry {

class IsoparametricMapping : public GeometryMapping {
public:
    IsoparametricMapping(std::shared_ptr<basis::BasisFunction> basis,
                         std::vector<math::Vector<Real, 3>> nodes);

    ElementType element_type() const noexcept override { return basis_->element_type(); }
    int dimension() const noexcept override { return basis_->dimension(); }
    std::size_t num_nodes() const noexcept override { return nodes_.size(); }
    const std::vector<math::Vector<Real, 3>>& nodes() const noexcept override { return nodes_; }

    math::Vector<Real, 3> map_to_physical(const math::Vector<Real, 3>& xi) const override;
    math::Vector<Real, 3> map_to_reference(const math::Vector<Real, 3>& x_phys,
                                           const math::Vector<Real, 3>& initial_guess = math::Vector<Real, 3>{}) const override;
    math::Matrix<Real, 3, 3> jacobian(const math::Vector<Real, 3>& xi) const override;
    MappingHessian mapping_hessian(const math::Vector<Real, 3>& xi) const override;

    void resetNodes(std::vector<math::Vector<Real, 3>> nodes) override { nodes_ = std::move(nodes); }
    const basis::BasisFunction& geometryBasis() const override { return *basis_; }

private:
    std::shared_ptr<basis::BasisFunction> basis_;
    std::vector<math::Vector<Real, 3>> nodes_;
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_ISOPARAMETRICMAPPING_H
