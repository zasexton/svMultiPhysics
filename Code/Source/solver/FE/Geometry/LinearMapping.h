/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_GEOMETRY_LINEARMAPPING_H
#define SVMP_FE_GEOMETRY_LINEARMAPPING_H

/**
 * @file LinearMapping.h
 * @brief Affine (linear) geometric mapping for simplex elements
 */

#include "GeometryMapping.h"
#include "FE/Basis/LagrangeBasis.h"
#include <memory>

namespace svmp {
namespace FE {
namespace geometry {

class LinearMapping : public GeometryMapping {
public:
    LinearMapping(ElementType type,
                  std::vector<math::Vector<Real, 3>> nodes);

    ElementType element_type() const noexcept override { return element_type_; }
    int dimension() const noexcept override { return dimension_; }
    std::size_t num_nodes() const noexcept override { return nodes_.size(); }
    const std::vector<math::Vector<Real, 3>>& nodes() const noexcept override { return nodes_; }

    math::Vector<Real, 3> map_to_physical(const math::Vector<Real, 3>& xi) const override;
    math::Vector<Real, 3> map_to_reference(const math::Vector<Real, 3>& x_phys,
                                           const math::Vector<Real, 3>& initial_guess = math::Vector<Real, 3>{}) const override;
    math::Matrix<Real, 3, 3> jacobian(const math::Vector<Real, 3>& xi) const override;

private:
    ElementType element_type_;
    int dimension_;
    std::vector<math::Vector<Real, 3>> nodes_;
    std::shared_ptr<basis::LagrangeBasis> basis_; // order 1
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_LINEARMAPPING_H
