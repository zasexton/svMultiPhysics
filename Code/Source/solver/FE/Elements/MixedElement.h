/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_ELEMENTS_MIXEDELEMENT_H
#define SVMP_FE_ELEMENTS_MIXEDELEMENT_H

/**
 * @file MixedElement.h
 * @brief Mixed formulation elements combining multiple sub-elements
 */

#include "Elements/Element.h"

#include <vector>

namespace svmp {
namespace FE {
namespace elements {

/**
 * @brief Sub-element descriptor used by MixedElement
 *
 * Describes a component sub-element within a mixed formulation,
 * pairing the element with its associated field identifier for
 * block DOF assembly and multi-field coupling.
 */
struct MixedSubElement {
    /// The sub-element providing basis functions and quadrature
    std::shared_ptr<Element> element;

    /**
     * @brief Field identifier for this sub-element
     *
     * Used by DOF management and block assembly to determine which
     * physical field this sub-element approximates (e.g., velocity,
     * pressure). When set to INVALID_FIELD_ID, the field association
     * is unspecified and must be resolved by higher-level modules.
     *
     * Common use cases:
     * - Stokes/Navier-Stokes: field_id=0 for velocity, field_id=1 for pressure
     * - Elasticity: field_id=0 for displacement
     * - Multi-physics: distinct field_id for each coupled physics
     */
    FieldId                  field_id{INVALID_FIELD_ID};
};

/**
 * @brief Mixed finite element composed of multiple sub-elements
 *
 * This class does not attempt to implement any particular mixed formulation.
 * It simply aggregates a set of compatible sub-elements and exposes them as
 * a single element with `FieldType::Mixed`. Higher-level modules (Spaces,
 * Systems, Assembly) are responsible for interpreting the block structure.
 *
 * @note MixedElement is a container. The inherited Element interface methods
 *       `basis()`, `quadrature()`, and `num_nodes()` forward to the first
 *       ("primary") sub-element for convenience only and do not represent the
 *       full mixed space. Use `sub_elements()` for block-accurate assembly.
 */
class MixedElement : public Element {
public:
    explicit MixedElement(std::vector<MixedSubElement> sub_elements);

    ElementInfo info() const noexcept override { return info_; }
    int dimension() const noexcept override { return dimension_; }

    std::size_t num_dofs() const noexcept override { return num_dofs_; }
    std::size_t num_nodes() const noexcept override { return primary_->num_nodes(); }

    const basis::BasisFunction& basis() const noexcept override { return primary_->basis(); }
    std::shared_ptr<const basis::BasisFunction> basis_ptr() const noexcept override { return primary_->basis_ptr(); }

    std::shared_ptr<const quadrature::QuadratureRule> quadrature() const noexcept override { return primary_->quadrature(); }

    /// Access the underlying sub-elements (for block assembly and DOF layout)
    const std::vector<MixedSubElement>& sub_elements() const noexcept { return sub_elements_; }

private:
    ElementInfo info_;
    int dimension_;
    std::size_t num_dofs_;
    std::vector<MixedSubElement> sub_elements_;
    std::shared_ptr<Element> primary_;
};

} // namespace elements
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ELEMENTS_MIXEDELEMENT_H
