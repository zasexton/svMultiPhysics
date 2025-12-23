/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_PRODUCTSPACE_H
#define SVMP_FE_SPACES_PRODUCTSPACE_H

/**
 * @file ProductSpace.h
 * @brief Cartesian product of function spaces for vector-valued fields
 */

#include "Spaces/FunctionSpace.h"
#include <vector>

namespace svmp {
namespace FE {
namespace spaces {

/**
 * @brief Cartesian product of identical scalar spaces (e.g., vector H¹)
 *
 * ProductSpace represents vector-valued fields obtained by taking a Cartesian
 * product of a scalar base space with itself (typically 2 or 3 times).
 * Coefficients are stored in a single contiguous vector with blocks for each
 * component: [u_x, u_y, (u_z)].
 */
class ProductSpace : public FunctionSpace {
public:
    /// Construct a d-dimensional product of a scalar base space
    ProductSpace(std::shared_ptr<FunctionSpace> base_space,
                 int components);

    SpaceType space_type() const noexcept override { return SpaceType::Product; }
    FieldType field_type() const noexcept override { return FieldType::Vector; }
    Continuity continuity() const noexcept override { return base_->continuity(); }

    int value_dimension() const noexcept override { return components_; }
    int topological_dimension() const noexcept override { return base_->topological_dimension(); }
    int polynomial_order() const noexcept override { return base_->polynomial_order(); }
    ElementType element_type() const noexcept override { return base_->element_type(); }

    const elements::Element& element() const noexcept override { return base_->element(); }
    std::shared_ptr<const elements::Element> element_ptr() const noexcept override { return base_->element_ptr(); }

    /// Number of scalar DOFs per component
    std::size_t scalar_dofs_per_component() const noexcept { return base_->dofs_per_element(); }

    /// Total number of DOFs (components * scalar_dofs_per_component)
    std::size_t dofs_per_element() const noexcept override {
        return scalar_dofs_per_component() * static_cast<std::size_t>(components_);
    }

    /// Interpolate a vector-valued function component-wise into the base space.
    void interpolate(const ValueFunction& function,
                     std::vector<Real>& coefficients) const override;

    /// Evaluate vector field from concatenated coefficients
    Value evaluate(const Value& xi,
                   const std::vector<Real>& coefficients) const override;

private:
    std::shared_ptr<FunctionSpace> base_;
    int components_;
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_PRODUCTSPACE_H
