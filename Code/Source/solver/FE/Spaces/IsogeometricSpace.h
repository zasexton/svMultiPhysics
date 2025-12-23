/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_SPACES_ISOGEOMETRICSPACE_H
#define SVMP_FE_SPACES_ISOGEOMETRICSPACE_H

/**
 * @file IsogeometricSpace.h
 * @brief Function space wrapper for isogeometric (IGA) elements
 *
 * This space provides a thin FunctionSpace façade around
 * elements::IsogeometricElement. The actual B-spline/NURBS basis is
 * supplied externally via a BasisFunction implementation.
 */

#include "Spaces/FunctionSpace.h"
#include "Elements/IsogeometricElement.h"

namespace svmp {
namespace FE {
namespace spaces {

class IsogeometricSpace : public FunctionSpace {
public:
    /// Construct from basis, quadrature, and field/continuity information
    IsogeometricSpace(std::shared_ptr<basis::BasisFunction> basis,
                      std::shared_ptr<const quadrature::QuadratureRule> quadrature,
                      FieldType field_type = FieldType::Scalar,
                      Continuity continuity = Continuity::C0);

    SpaceType space_type() const noexcept override { return SpaceType::Isogeometric; }
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

#endif // SVMP_FE_SPACES_ISOGEOMETRICSPACE_H

