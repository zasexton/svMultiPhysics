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
#include <span>
#include <vector>

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
    bool isAffine() const noexcept override { return true; }

    void resetNodes(const std::vector<math::Vector<Real, 3>>& nodes) override { nodes_ = nodes; }
    const basis::BasisFunction& geometryBasis() const override { return *basis_; }

    /**
     * @brief The Jacobian jacobian() returns for a LinearMapping of `type` with
     *        `nodes`, computed by the same code without constructing one.
     *
     * Throws as the constructor does for an unsupported type or a node count
     * that does not match the order-1 basis.
     */
    static math::Matrix<Real, 3, 3> affineJacobian(ElementType type,
                                                   std::span<const math::Vector<Real, 3>> nodes);

private:
    ElementType element_type_;
    int dimension_;
    std::vector<math::Vector<Real, 3>> nodes_;
    // Order 1, shared by every LinearMapping of the element type and never
    // modified (only const evaluations, which keep their scratch per thread).
    std::shared_ptr<const basis::LagrangeBasis> basis_;
    // Its gradients at the interior point jacobian() uses: one immutable table
    // per element type.
    const std::vector<basis::Gradient>* safe_gradients_{nullptr};
};

} // namespace geometry
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_GEOMETRY_LINEARMAPPING_H
