/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Interfaces/CutInterfaceFieldEvaluation.h"

#include <array>
#include <cmath>
#include <stdexcept>

namespace svmp {
namespace FE {
namespace interfaces {

namespace {

using Matrix3 = std::array<std::array<Real, 3>, 3>;

[[nodiscard]] Matrix3 fieldGeometryJacobian(
    const std::vector<std::array<Real, 3>>& shape_gradients,
    const std::vector<std::array<Real, 3>>& node_coordinates,
    int dimension)
{
    Matrix3 jacobian{};
    for (std::size_t node = 0; node < shape_gradients.size(); ++node) {
        for (int physical = 0; physical < dimension; ++physical) {
            for (int reference = 0; reference < dimension; ++reference) {
                jacobian[static_cast<std::size_t>(physical)]
                        [static_cast<std::size_t>(reference)] +=
                    node_coordinates[node][static_cast<std::size_t>(physical)] *
                    shape_gradients[node][static_cast<std::size_t>(reference)];
            }
        }
    }
    return jacobian;
}

[[nodiscard]] Matrix3 invertJacobian(const Matrix3& jacobian, int dimension)
{
    Matrix3 inverse{};
    if (dimension == 2) {
        const Real det = jacobian[0][0] * jacobian[1][1] -
                         jacobian[0][1] * jacobian[1][0];
        if (std::abs(det) <= Real{1.0e-14}) {
            throw std::invalid_argument("H1 cut-interface gradient evaluation received singular cell geometry");
        }
        inverse[0][0] = jacobian[1][1] / det;
        inverse[0][1] = -jacobian[0][1] / det;
        inverse[1][0] = -jacobian[1][0] / det;
        inverse[1][1] = jacobian[0][0] / det;
        return inverse;
    }

    if (dimension == 3) {
        const Real det =
            jacobian[0][0] * (jacobian[1][1] * jacobian[2][2] -
                              jacobian[1][2] * jacobian[2][1]) -
            jacobian[0][1] * (jacobian[1][0] * jacobian[2][2] -
                              jacobian[1][2] * jacobian[2][0]) +
            jacobian[0][2] * (jacobian[1][0] * jacobian[2][1] -
                              jacobian[1][1] * jacobian[2][0]);
        if (std::abs(det) <= Real{1.0e-14}) {
            throw std::invalid_argument("H1 cut-interface gradient evaluation received singular cell geometry");
        }

        inverse[0][0] = (jacobian[1][1] * jacobian[2][2] -
                         jacobian[1][2] * jacobian[2][1]) / det;
        inverse[0][1] = (jacobian[0][2] * jacobian[2][1] -
                         jacobian[0][1] * jacobian[2][2]) / det;
        inverse[0][2] = (jacobian[0][1] * jacobian[1][2] -
                         jacobian[0][2] * jacobian[1][1]) / det;
        inverse[1][0] = (jacobian[1][2] * jacobian[2][0] -
                         jacobian[1][0] * jacobian[2][2]) / det;
        inverse[1][1] = (jacobian[0][0] * jacobian[2][2] -
                         jacobian[0][2] * jacobian[2][0]) / det;
        inverse[1][2] = (jacobian[0][2] * jacobian[1][0] -
                         jacobian[0][0] * jacobian[1][2]) / det;
        inverse[2][0] = (jacobian[1][0] * jacobian[2][1] -
                         jacobian[1][1] * jacobian[2][0]) / det;
        inverse[2][1] = (jacobian[0][1] * jacobian[2][0] -
                         jacobian[0][0] * jacobian[2][1]) / det;
        inverse[2][2] = (jacobian[0][0] * jacobian[1][1] -
                         jacobian[0][1] * jacobian[1][0]) / det;
        return inverse;
    }

    throw std::invalid_argument("H1 cut-interface gradient evaluation requires a two- or three-dimensional element");
}

void validateTwoSidedBinding(const GeneratedInterfaceTwoSidedBinding& binding)
{
    if (!binding.complete()) {
        throw std::invalid_argument(
            "two-sided H1 cut-interface field evaluation requires a complete generated-interface binding");
    }
    if (binding.minus_side != CutInterfaceSideTag::Negative ||
        binding.plus_side != CutInterfaceSideTag::Positive) {
        throw std::invalid_argument(
            "two-sided H1 cut-interface field evaluation requires negative minus-side and positive plus-side tags");
    }
}

void validateFragmentMatchesBinding(
    const GeneratedInterfaceTwoSidedBinding& binding,
    const CutInterfaceFragment& fragment)
{
    validateTwoSidedBinding(binding);
    if (!fragment.active() ||
        fragment.interface_marker != binding.interface_marker ||
        fragment.parent_cell != binding.parent_cell ||
        fragment.stable_id != binding.interface_stable_id) {
        throw std::invalid_argument(
            "two-sided H1 cut-interface field evaluation received a fragment that does not match the generated-interface binding");
    }
}

[[nodiscard]] CutInterfaceFieldValue combineFieldValues(
    const CutInterfaceFieldValue& a,
    Real a_scale,
    const CutInterfaceFieldValue& b,
    Real b_scale,
    const char* context)
{
    if (a.components.size() != b.components.size()) {
        throw std::invalid_argument(context);
    }
    CutInterfaceFieldValue result;
    result.components.resize(a.components.size());
    for (std::size_t component = 0; component < result.components.size();
         ++component) {
        result.components[component] =
            a_scale * a.components[component] + b_scale * b.components[component];
    }
    return result;
}

[[nodiscard]] CutInterfaceFieldGradient combineFieldGradients(
    const CutInterfaceFieldGradient& a,
    Real a_scale,
    const CutInterfaceFieldGradient& b,
    Real b_scale,
    const char* context)
{
    if (a.components.size() != b.components.size()) {
        throw std::invalid_argument(context);
    }
    CutInterfaceFieldGradient result;
    result.components.resize(a.components.size());
    for (std::size_t component = 0; component < result.components.size();
         ++component) {
        for (std::size_t d = 0; d < result.components[component].size(); ++d) {
            result.components[component][d] =
                a_scale * a.components[component][d] +
                b_scale * b.components[component][d];
        }
    }
    return result;
}

void stampTwoSidedValueMetadata(
    CutInterfaceTwoSidedFieldValue& value,
    const GeneratedInterfaceTwoSidedBinding& binding)
{
    value.interface_marker = binding.interface_marker;
    value.parent_cell = binding.parent_cell;
    value.interface_stable_id = binding.interface_stable_id;
    value.minus_side = binding.minus_side;
    value.plus_side = binding.plus_side;
}

void stampTwoSidedGradientMetadata(
    CutInterfaceTwoSidedFieldGradient& gradient,
    const GeneratedInterfaceTwoSidedBinding& binding)
{
    gradient.interface_marker = binding.interface_marker;
    gradient.parent_cell = binding.parent_cell;
    gradient.interface_stable_id = binding.interface_stable_id;
    gradient.minus_side = binding.minus_side;
    gradient.plus_side = binding.plus_side;
}

} // namespace

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

std::vector<std::array<Real, 3>> linearH1ShapeGradients(
    ElementType element_type,
    const std::array<Real, 3>& parent_coordinate)
{
    const Real x = parent_coordinate[0];
    const Real y = parent_coordinate[1];
    switch (element_type) {
    case ElementType::Triangle3:
        return {{-Real{1.0}, -Real{1.0}, Real{0.0}},
                {Real{1.0}, Real{0.0}, Real{0.0}},
                {Real{0.0}, Real{1.0}, Real{0.0}}};
    case ElementType::Quad4:
        return {{-(Real{1.0} - y), -(Real{1.0} - x), Real{0.0}},
                {Real{1.0} - y, -x, Real{0.0}},
                {y, x, Real{0.0}},
                {-y, Real{1.0} - x, Real{0.0}}};
    case ElementType::Tetra4:
        return {{-Real{1.0}, -Real{1.0}, -Real{1.0}},
                {Real{1.0}, Real{0.0}, Real{0.0}},
                {Real{0.0}, Real{1.0}, Real{0.0}},
                {Real{0.0}, Real{0.0}, Real{1.0}}};
    default:
        throw std::invalid_argument("linear H1 cut-interface gradient evaluation requires Triangle3, Quad4, or Tetra4");
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

CutInterfaceFieldGradient evaluateH1FieldGradientAtPoint(
    const H1NodalFieldData& field,
    const std::array<Real, 3>& parent_coordinate)
{
    if (field.components <= 0) {
        throw std::invalid_argument("H1 cut-interface gradient evaluation requires positive component count");
    }
    const auto shape_gradients =
        linearH1ShapeGradients(field.element_type, parent_coordinate);
    const auto components = static_cast<std::size_t>(field.components);
    if (field.nodal_values.size() != shape_gradients.size() * components) {
        throw std::invalid_argument("H1 cut-interface gradient evaluation received inconsistent nodal data");
    }
    if (field.node_coordinates.size() != shape_gradients.size()) {
        throw std::invalid_argument("H1 cut-interface gradient evaluation requires matching node coordinates");
    }

    const int dimension = element_dimension(field.element_type);
    const auto jacobian =
        fieldGeometryJacobian(shape_gradients, field.node_coordinates, dimension);
    const auto inverse_jacobian = invertJacobian(jacobian, dimension);

    CutInterfaceFieldGradient gradient;
    gradient.components.assign(components, std::array<Real, 3>{{0.0, 0.0, 0.0}});
    for (std::size_t component = 0; component < components; ++component) {
        std::array<Real, 3> reference_gradient{{0.0, 0.0, 0.0}};
        for (std::size_t node = 0; node < shape_gradients.size(); ++node) {
            const Real value = field.nodal_values[node * components + component];
            for (int reference = 0; reference < dimension; ++reference) {
                reference_gradient[static_cast<std::size_t>(reference)] +=
                    value * shape_gradients[node][static_cast<std::size_t>(reference)];
            }
        }
        for (int physical = 0; physical < dimension; ++physical) {
            for (int reference = 0; reference < dimension; ++reference) {
                gradient.components[component][static_cast<std::size_t>(physical)] +=
                    inverse_jacobian[static_cast<std::size_t>(reference)]
                                    [static_cast<std::size_t>(physical)] *
                    reference_gradient[static_cast<std::size_t>(reference)];
            }
        }
    }
    return gradient;
}

std::vector<CutInterfaceFieldGradient> evaluateH1FieldGradientsOnFragment(
    const H1NodalFieldData& field,
    const CutInterfaceFragment& fragment)
{
    std::vector<CutInterfaceFieldGradient> gradients;
    gradients.reserve(fragment.quadrature_points.size());
    for (const auto& point : fragment.quadrature_points) {
        gradients.push_back(evaluateH1FieldGradientAtPoint(field, point.parent_coordinate));
    }
    return gradients;
}

CutInterfaceTwoSidedFieldValue evaluateH1TwoSidedFieldValueAtPoint(
    const H1NodalFieldData& negative_side_field,
    const H1NodalFieldData& positive_side_field,
    const GeneratedInterfaceTwoSidedBinding& binding,
    const std::array<Real, 3>& parent_coordinate)
{
    validateTwoSidedBinding(binding);
    CutInterfaceTwoSidedFieldValue value;
    stampTwoSidedValueMetadata(value, binding);
    value.minus =
        evaluateH1FieldValueAtPoint(negative_side_field, parent_coordinate);
    value.plus =
        evaluateH1FieldValueAtPoint(positive_side_field, parent_coordinate);
    value.jump = combineFieldValues(
        value.plus,
        Real{1.0},
        value.minus,
        Real{-1.0},
        "two-sided H1 cut-interface value evaluation requires matching component counts");
    value.average = combineFieldValues(
        value.plus,
        Real{0.5},
        value.minus,
        Real{0.5},
        "two-sided H1 cut-interface value evaluation requires matching component counts");
    return value;
}

std::vector<CutInterfaceTwoSidedFieldValue>
evaluateH1TwoSidedFieldValuesOnFragment(
    const H1NodalFieldData& negative_side_field,
    const H1NodalFieldData& positive_side_field,
    const GeneratedInterfaceTwoSidedBinding& binding,
    const CutInterfaceFragment& fragment)
{
    validateFragmentMatchesBinding(binding, fragment);
    std::vector<CutInterfaceTwoSidedFieldValue> values;
    values.reserve(fragment.quadrature_points.size());
    for (const auto& point : fragment.quadrature_points) {
        values.push_back(evaluateH1TwoSidedFieldValueAtPoint(
            negative_side_field, positive_side_field, binding, point.parent_coordinate));
    }
    return values;
}

CutInterfaceTwoSidedFieldGradient evaluateH1TwoSidedFieldGradientAtPoint(
    const H1NodalFieldData& negative_side_field,
    const H1NodalFieldData& positive_side_field,
    const GeneratedInterfaceTwoSidedBinding& binding,
    const std::array<Real, 3>& parent_coordinate)
{
    validateTwoSidedBinding(binding);
    CutInterfaceTwoSidedFieldGradient gradient;
    stampTwoSidedGradientMetadata(gradient, binding);
    gradient.minus =
        evaluateH1FieldGradientAtPoint(negative_side_field, parent_coordinate);
    gradient.plus =
        evaluateH1FieldGradientAtPoint(positive_side_field, parent_coordinate);
    gradient.jump = combineFieldGradients(
        gradient.plus,
        Real{1.0},
        gradient.minus,
        Real{-1.0},
        "two-sided H1 cut-interface gradient evaluation requires matching component counts");
    gradient.average = combineFieldGradients(
        gradient.plus,
        Real{0.5},
        gradient.minus,
        Real{0.5},
        "two-sided H1 cut-interface gradient evaluation requires matching component counts");
    return gradient;
}

std::vector<CutInterfaceTwoSidedFieldGradient>
evaluateH1TwoSidedFieldGradientsOnFragment(
    const H1NodalFieldData& negative_side_field,
    const H1NodalFieldData& positive_side_field,
    const GeneratedInterfaceTwoSidedBinding& binding,
    const CutInterfaceFragment& fragment)
{
    validateFragmentMatchesBinding(binding, fragment);
    std::vector<CutInterfaceTwoSidedFieldGradient> gradients;
    gradients.reserve(fragment.quadrature_points.size());
    for (const auto& point : fragment.quadrature_points) {
        gradients.push_back(evaluateH1TwoSidedFieldGradientAtPoint(
            negative_side_field, positive_side_field, binding, point.parent_coordinate));
    }
    return gradients;
}

} // namespace interfaces
} // namespace FE
} // namespace svmp
