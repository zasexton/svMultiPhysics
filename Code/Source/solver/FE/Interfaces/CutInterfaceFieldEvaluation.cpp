/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Interfaces/CutInterfaceFieldEvaluation.h"

#include <stdexcept>

namespace svmp {
namespace FE {
namespace interfaces {

std::vector<Real> linearH1ShapeValues(ElementType element_type,
                                      const std::array<Real, 3>& parent_coordinate)
{
    const Real x = parent_coordinate[0];
    const Real y = parent_coordinate[1];
    const Real z = parent_coordinate[2];
    switch (element_type) {
    case ElementType::Triangle3:
        return {Real{1.0} - x - y, x, y};
    case ElementType::Quad4:
        return {(Real{1.0} - x) * (Real{1.0} - y),
                x * (Real{1.0} - y),
                x * y,
                (Real{1.0} - x) * y};
    case ElementType::Tetra4:
        return {Real{1.0} - x - y - z, x, y, z};
    default:
        throw std::invalid_argument("linear H1 cut-interface field evaluation requires Triangle3, Quad4, or Tetra4");
    }
}

CutInterfaceFieldValue evaluateH1FieldValueAtPoint(
    const H1NodalFieldData& field,
    const std::array<Real, 3>& parent_coordinate)
{
    if (field.components <= 0) {
        throw std::invalid_argument("H1 cut-interface field evaluation requires positive component count");
    }
    const auto shape = linearH1ShapeValues(field.element_type, parent_coordinate);
    const auto components = static_cast<std::size_t>(field.components);
    if (field.nodal_values.size() != shape.size() * components) {
        throw std::invalid_argument("H1 cut-interface field evaluation received inconsistent nodal data");
    }

    CutInterfaceFieldValue value;
    value.components.assign(components, Real{0.0});
    for (std::size_t node = 0; node < shape.size(); ++node) {
        for (std::size_t component = 0; component < components; ++component) {
            value.components[component] +=
                shape[node] * field.nodal_values[node * components + component];
        }
    }
    return value;
}

std::vector<CutInterfaceFieldValue> evaluateH1FieldValuesOnFragment(
    const H1NodalFieldData& field,
    const CutInterfaceFragment& fragment)
{
    std::vector<CutInterfaceFieldValue> values;
    values.reserve(fragment.quadrature_points.size());
    for (const auto& point : fragment.quadrature_points) {
        values.push_back(evaluateH1FieldValueAtPoint(field, point.parent_coordinate));
    }
    return values;
}

} // namespace interfaces
} // namespace FE
} // namespace svmp
