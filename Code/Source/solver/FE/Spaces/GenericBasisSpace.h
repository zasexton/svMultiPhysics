/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_GENERICBASISSPACE_H
#define SVMP_FE_SPACES_GENERICBASISSPACE_H

/**
 * @file GenericBasisSpace.h
 * @brief Function-space wrapper for basis-backed elements
 *
 * This space provides a thin FunctionSpace façade around either a prebuilt
 * Elements::Element instance or a direct basis/quadrature pair wrapped by
 * elements::GeneralBasisElement. It is intended for basis families that do not
 * need a dedicated named space class.
 */

#include "Spaces/FunctionSpace.h"
#include "Elements/GeneralBasisElement.h"

namespace svmp {
namespace FE {
namespace spaces {

class GenericBasisSpace : public FunctionSpace {
public:
    /// Construct from externally supplied basis, quadrature, and FE metadata.
    GenericBasisSpace(std::shared_ptr<basis::BasisFunction> basis,
                      std::shared_ptr<const quadrature::QuadratureRule> quadrature,
                      FieldType field_type = FieldType::Scalar,
                      Continuity continuity = Continuity::C0);

    /// Construct from an existing Element returned by ElementFactory or peers.
    explicit GenericBasisSpace(std::shared_ptr<elements::Element> element);

    SpaceType space_type() const noexcept override { return SpaceType::GenericBasis; }
    FieldType field_type() const noexcept override { return info_.field_type; }
    Continuity continuity() const noexcept override { return info_.continuity; }

    int value_dimension() const noexcept override {
        return field_type() == FieldType::Vector ? element_dimension_ : 1;
    }

    int topological_dimension() const noexcept override { return element_dimension_; }
    int polynomial_order() const noexcept override { return info_.order; }
    ElementType element_type() const noexcept override { return info_.element_type; }

    const elements::Element& element() const noexcept override { return *element_; }
    std::shared_ptr<const elements::Element> element_ptr() const noexcept override { return element_; }

private:
    elements::ElementInfo info_;
    int element_dimension_{0};
    std::shared_ptr<elements::Element> element_;
};

} // namespace spaces
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SPACES_GENERICBASISSPACE_H
