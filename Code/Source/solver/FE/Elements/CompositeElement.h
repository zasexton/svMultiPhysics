/* Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_ELEMENTS_COMPOSITEELEMENT_H
#define SVMP_FE_ELEMENTS_COMPOSITEELEMENT_H

/**
 * @file CompositeElement.h
 * @brief Composite/enriched elements built from multiple components
 */

#include "Elements/Element.h"

#include <vector>

namespace svmp {
namespace FE {
namespace elements {

/**
 * @brief Composite element that enriches a base element with additional
 *        approximation spaces (e.g., bubble functions, XFEM enrichment).
 *
 * For now this class acts as a simple container of compatible components.
 * Enrichment-specific logic (e.g., partition of unity, blending) is handled
 * at the physics/assembly level.
 *
 * @note CompositeElement is a container. The inherited Element interface
 *       methods `basis()`, `quadrature()`, and `num_nodes()` forward to the
 *       first ("primary") component for convenience only and do not represent
 *       the full enriched space. Use `components()` for full access.
 */
class CompositeElement : public Element {
public:
    explicit CompositeElement(std::vector<std::shared_ptr<Element>> components);

    ElementInfo info() const noexcept override { return info_; }
    int dimension() const noexcept override { return dimension_; }

    std::size_t num_dofs() const noexcept override { return num_dofs_; }
    std::size_t num_nodes() const noexcept override { return primary_->num_nodes(); }

    const basis::BasisFunction& basis() const noexcept override { return primary_->basis(); }
    std::shared_ptr<const basis::BasisFunction> basis_ptr() const noexcept override { return primary_->basis_ptr(); }

    std::shared_ptr<const quadrature::QuadratureRule> quadrature() const noexcept override { return primary_->quadrature(); }

    const std::vector<std::shared_ptr<Element>>& components() const noexcept { return components_; }

private:
    ElementInfo info_;
    int dimension_;
    std::size_t num_dofs_;
    std::vector<std::shared_ptr<Element>> components_;
    std::shared_ptr<Element> primary_;
};

} // namespace elements
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ELEMENTS_COMPOSITEELEMENT_H
