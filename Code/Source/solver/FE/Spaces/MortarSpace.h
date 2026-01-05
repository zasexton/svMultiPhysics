/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_MORTARSPACE_H
#define SVMP_FE_SPACES_MORTARSPACE_H

/**
 * @file MortarSpace.h
 * @brief Mortar/interface function spaces
 *
 * MortarSpace is a lightweight semantic wrapper around a (dim-1)-dimensional
 * FunctionSpace intended for interface coupling (e.g., mortar methods,
 * Lagrange multipliers on interfaces).
 *
 * This class is **mesh-agnostic** and does not own any interface topology or
 * master/slave pairing logic. Those responsibilities belong to Systems/Mesh.
 *
 * Note: FE/Assembly currently does not provide a dedicated "interface entity"
 * assembly loop for MortarSpace. It exists as a space vocabulary term for
 * future interface/mortar infrastructure.
 */

#include "Spaces/FunctionSpace.h"

#include <memory>

namespace svmp {
namespace FE {
namespace spaces {

class MortarSpace : public FunctionSpace {
public:
    explicit MortarSpace(std::shared_ptr<FunctionSpace> interface_space);

    SpaceType space_type() const noexcept override { return SpaceType::Mortar; }

    FieldType field_type() const noexcept override { return interface_space_->field_type(); }
    Continuity continuity() const noexcept override { return interface_space_->continuity(); }

    int value_dimension() const noexcept override { return interface_space_->value_dimension(); }
    int topological_dimension() const noexcept override { return interface_space_->topological_dimension(); }
    int polynomial_order() const noexcept override { return interface_space_->polynomial_order(); }
    ElementType element_type() const noexcept override { return interface_space_->element_type(); }

    const elements::Element& element() const noexcept override { return interface_space_->element(); }
    std::shared_ptr<const elements::Element> element_ptr() const noexcept override {
        return interface_space_->element_ptr();
    }

    const elements::Element& getElement(ElementType cell_type,
                                        GlobalIndex cell_id) const noexcept override {
        return interface_space_->getElement(cell_type, cell_id);
    }

    std::size_t dofs_per_element() const noexcept override { return interface_space_->dofs_per_element(); }

    Value evaluate(const Value& xi,
                   const std::vector<Real>& coefficients) const override {
        return interface_space_->evaluate(xi, coefficients);
    }

    void interpolate(const ValueFunction& function,
                     std::vector<Real>& coefficients) const override {
        interface_space_->interpolate(function, coefficients);
    }

    Gradient evaluate_gradient(const Value& xi,
                               const std::vector<Real>& coefficients) const override {
        return interface_space_->evaluate_gradient(xi, coefficients);
    }

    Real evaluate_divergence(const Value& xi,
                             const std::vector<Real>& coefficients) const override {
        return interface_space_->evaluate_divergence(xi, coefficients);
    }

    Value evaluate_curl(const Value& xi,
                        const std::vector<Real>& coefficients) const override {
        return interface_space_->evaluate_curl(xi, coefficients);
    }

    const FunctionSpace& interface_space() const noexcept { return *interface_space_; }
    std::shared_ptr<FunctionSpace> interface_space_ptr() const noexcept { return interface_space_; }

private:
    std::shared_ptr<FunctionSpace> interface_space_;
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_MORTARSPACE_H

