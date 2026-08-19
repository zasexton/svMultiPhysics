#ifndef SVMP_FE_FORMS_BOUNDARYCONDITIONS_H
#define SVMP_FE_FORMS_BOUNDARYCONDITIONS_H

/**
 * @file BoundaryConditions.h
 * @brief Forms-level boundary-condition declarations (not assembled directly)
 *
 * This header defines small POD-like declarations that allow Physics modules
 * to express boundary conditions in a "weak-form adjacent" way for readability.
 *
 * These declarations are not assembled by FE/Forms. Instead, FE/Systems is
 * responsible for lowering them to the appropriate enforcement mechanism
 * (e.g., strong Dirichlet -> constraints).
 */

#include "Forms/FormExpr.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <stdexcept>
#include <utility>
#include <vector>
#include <variant>

namespace svmp {
namespace FE {
namespace forms {
namespace bc {

enum class ComponentValueNameStyle : std::uint8_t {
    Indexed,
    Component
};

namespace detail {

[[noreturn]] inline void throwInvalidMarker(std::string_view where)
{
    throw std::invalid_argument(std::string(where) + ": invalid boundary_marker (< 0)");
}

template <class BC>
[[nodiscard]] inline int boundaryMarkerOrThrow(const BC& bc, std::string_view where)
{
    const int marker = bc.boundary_marker;
    if (marker < 0) {
        throwInvalidMarker(where);
    }
    return marker;
}

[[nodiscard]] inline std::string makeValueName(std::string_view prefix,
                                              int boundary_marker,
                                              std::size_t i)
{
    const std::string marker = std::to_string(boundary_marker);
    const std::string idx = std::to_string(i);

    std::string out;
    if (prefix.empty()) {
        out.reserve(marker.size() + 1 + idx.size());
        out.append(marker);
    } else {
        out.reserve(prefix.size() + 1 + marker.size() + 1 + idx.size());
        out.append(prefix.data(), prefix.size());
        out.push_back('_');
        out.append(marker);
    }
    out.push_back('_');
    out.append(idx);
    return out;
}

[[nodiscard]] inline std::string makeMarkerValueName(std::string_view prefix,
                                                     int boundary_marker)
{
    const std::string marker = std::to_string(boundary_marker);

    std::string out;
    if (prefix.empty()) {
        out = marker;
    } else {
        out.reserve(prefix.size() + 1 + marker.size());
        out.append(prefix.data(), prefix.size());
        out.push_back('_');
        out.append(marker);
    }
    return out;
}

[[nodiscard]] inline std::string makeComponentValueName(std::string_view prefix,
                                                        int boundary_marker,
                                                        int component)
{
    const std::string marker = std::to_string(boundary_marker);
    const std::string comp = std::to_string(component);

    std::string out;
    if (prefix.empty()) {
        out.reserve(marker.size() + 2 + comp.size());
        out.append(marker);
    } else {
        out.reserve(prefix.size() + 1 + marker.size() + 2 + comp.size());
        out.append(prefix.data(), prefix.size());
        out.push_back('_');
        out.append(marker);
    }
    out.append("_c");
    out.append(comp);
    return out;
}

[[nodiscard]] inline std::string makeComponentValueName(std::string_view prefix,
                                                        int boundary_marker,
                                                        int component,
                                                        ComponentValueNameStyle style)
{
    switch (style) {
    case ComponentValueNameStyle::Indexed:
        return makeValueName(prefix, boundary_marker, static_cast<std::size_t>(component));
    case ComponentValueNameStyle::Component:
        return makeComponentValueName(prefix, boundary_marker, component);
    }

    return makeValueName(prefix, boundary_marker, static_cast<std::size_t>(component));
}

[[nodiscard]] inline int polynomialOrderOrDefault(const FormExpr& expr, int default_order = 1)
{
    if (!expr.isValid()) {
        return default_order;
    }
    const auto* node = expr.node();
    if (!node) {
        return default_order;
    }
    const auto* sig = node->spaceSignature();
    if (!sig) {
        return default_order;
    }
    return sig->polynomial_order;
}

[[nodiscard]] inline bool spaceSignaturesMatch(
    const SpaceSignature& left,
    const SpaceSignature& right) noexcept
{
    return left.space_type == right.space_type &&
           left.field_type == right.field_type &&
           left.continuity == right.continuity &&
           left.value_dimension == right.value_dimension &&
           left.topological_dimension == right.topological_dimension &&
           left.polynomial_order == right.polynomial_order &&
           left.element_type == right.element_type;
}

[[nodiscard]] inline bool hasForbiddenPrescribedValueDependency(
    const FormExprNode& node)
{
    if (node.hasTrial() || node.hasTest()) {
        return true;
    }
    switch (node.type()) {
    case FormExprType::DiscreteField:
    case FormExprType::StateField:
    case FormExprType::PreviousSolutionRef:
    case FormExprType::BoundaryFunctionalSymbol:
    case FormExprType::BoundaryIntegralSymbol:
    case FormExprType::BoundaryIntegralRef:
    case FormExprType::AuxiliaryStateSymbol:
    case FormExprType::AuxiliaryStateRef:
    case FormExprType::AuxiliaryInputSymbol:
    case FormExprType::AuxiliaryInputRef:
    case FormExprType::AuxiliaryOutputSymbol:
    case FormExprType::AuxiliaryOutputRef:
    case FormExprType::MaterialStateOldRef:
    case FormExprType::MaterialStateWorkRef:
    case FormExprType::MeshDisplacement:
    case FormExprType::MeshVelocity:
    case FormExprType::MeshAcceleration:
    case FormExprType::CurrentCoordinate:
    case FormExprType::CurrentJacobian:
    case FormExprType::CurrentJacobianDeterminant:
    case FormExprType::CurrentNormal:
    case FormExprType::CurrentMeanCurvature:
    case FormExprType::CurrentMeasure:
    case FormExprType::SurfaceJacobian:
    case FormExprType::GeometryTrialVectorVariation:
    case FormExprType::GeometryTrialJacobianVariation:
    case FormExprType::MeshVelocityVariation:
    case FormExprType::CurrentMeasureVariation:
    case FormExprType::CurrentNormalVariation:
    case FormExprType::SurfaceJacobianVariation:
    case FormExprType::Constitutive:
    case FormExprType::ConstitutiveOutput:
    case FormExprType::CellIntegral:
    case FormExprType::BoundaryIntegral:
    case FormExprType::InteriorFaceIntegral:
    case FormExprType::InterfaceIntegral:
    case FormExprType::CutVolumeIntegral:
        return true;
    default:
        break;
    }
    for (const auto& child : node.childrenShared()) {
        if (child != nullptr &&
            hasForbiddenPrescribedValueDependency(*child)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline bool isCompatiblePrescribedVector(
    const FormExprNode& node,
    const SpaceSignature& expected)
{
    if (expected.field_type != FieldType::Vector ||
        expected.value_dimension < 1) {
        return false;
    }

    if (node.type() == FormExprType::AsVector) {
        const auto components = node.childrenShared();
        if (components.size() !=
            static_cast<std::size_t>(expected.value_dimension)) {
            return false;
        }
        for (const auto& component : components) {
            if (component == nullptr) {
                return false;
            }
        }
        return true;
    }

    if (const auto* signature = node.spaceSignature();
        signature != nullptr) {
        return signature->field_type == FieldType::Vector &&
               signature->value_dimension ==
                   expected.value_dimension;
    }

    if (node.vectorCoefficient() != nullptr) {
        return expected.topological_dimension >= 1 &&
               expected.topological_dimension <= 3 &&
               expected.value_dimension ==
                   expected.topological_dimension;
    }

    return false;
}

} // namespace detail

enum class ScalarTraceOperator : std::uint8_t {
    Identity,
    NormalComponent
};

enum class InterfaceTraceReduction : std::uint8_t {
    Minus,
    Plus,
    Jump,
    Average
};

[[nodiscard]] inline FormExpr applyScalarTrace(const FormExpr& expr,
                                               ScalarTraceOperator op)
{
    switch (op) {
    case ScalarTraceOperator::Identity:
        return expr;
    case ScalarTraceOperator::NormalComponent:
        return dot(expr, FormExpr::normal());
    }

    throw std::invalid_argument("forms::bc::applyScalarTrace: unsupported scalar trace operator");
}

[[nodiscard]] inline FormExpr normalComponent(const FormExpr& expr)
{
    return applyScalarTrace(expr, ScalarTraceOperator::NormalComponent);
}

[[nodiscard]] inline FormExpr applyInterfaceScalarTrace(const FormExpr& expr,
                                                        ScalarTraceOperator op,
                                                        InterfaceTraceReduction reduction)
{
    const auto minus_value = [&]() -> FormExpr {
        switch (op) {
        case ScalarTraceOperator::Identity:
            return expr.minus();
        case ScalarTraceOperator::NormalComponent:
            return dot(expr.minus(), FormExpr::normal().minus());
        }

        throw std::invalid_argument(
            "forms::bc::applyInterfaceScalarTrace: unsupported scalar trace operator");
    };

    const auto plus_value = [&]() -> FormExpr {
        switch (op) {
        case ScalarTraceOperator::Identity:
            return expr.plus();
        case ScalarTraceOperator::NormalComponent:
            return dot(expr.plus(), FormExpr::normal().plus());
        }

        throw std::invalid_argument(
            "forms::bc::applyInterfaceScalarTrace: unsupported scalar trace operator");
    };

    const auto tau_minus = minus_value();
    const auto tau_plus = plus_value();

    switch (reduction) {
    case InterfaceTraceReduction::Minus:
        return tau_minus;
    case InterfaceTraceReduction::Plus:
        return tau_plus;
    case InterfaceTraceReduction::Jump:
        if (op == ScalarTraceOperator::Identity) {
            return tau_plus - tau_minus;
        }
        return tau_plus + tau_minus;
    case InterfaceTraceReduction::Average:
        if (op == ScalarTraceOperator::Identity) {
            return 0.5 * (tau_plus + tau_minus);
        }
        return 0.5 * (tau_plus + tau_minus);
    }

    throw std::invalid_argument(
        "forms::bc::applyInterfaceScalarTrace: unsupported interface trace reduction");
}

[[nodiscard]] inline FormExpr interfaceNormalComponent(const FormExpr& expr,
                                                       InterfaceTraceReduction reduction =
                                                           InterfaceTraceReduction::Minus)
{
    return applyInterfaceScalarTrace(expr, ScalarTraceOperator::NormalComponent, reduction);
}

/**
 * @brief Canonical scalar-valued boundary condition value type
 *
 * This is intended to be used by Physics modules when defining boundary
 * condition option structs to avoid re-defining common variants.
 */
using ScalarValue = std::variant<Real, ScalarCoefficient, TimeScalarCoefficient, FormExpr>;

[[nodiscard]] inline bool isConstantScalarValue(const ScalarValue& value)
{
    return std::holds_alternative<Real>(value);
}

[[nodiscard]] inline bool isZeroConstantScalarValue(const ScalarValue& value)
{
    const auto* real = std::get_if<Real>(&value);
    return real && *real == Real{0.0};
}

/**
 * @brief Convert common scalar value types into a scalar FormExpr
 *
 * Supported inputs:
 * - Real                         -> constant(...)
 * - ScalarCoefficient            -> coefficient(name, ...)
 * - TimeScalarCoefficient        -> coefficient(name, ...)
 * - FormExpr                     -> returned unchanged
 * - std::variant of the above    -> visited and converted
 */
[[nodiscard]] inline FormExpr toScalarExpr(Real value, std::string_view /*name*/)
{
    return FormExpr::constant(value);
}

[[nodiscard]] inline FormExpr toScalarExpr(const ScalarCoefficient& value, std::string_view name)
{
    return FormExpr::coefficient(std::string(name), value);
}

[[nodiscard]] inline FormExpr toScalarExpr(const TimeScalarCoefficient& value, std::string_view name)
{
    return FormExpr::coefficient(std::string(name), value);
}

[[nodiscard]] inline FormExpr toScalarExpr(const FormExpr& value, std::string_view /*name*/)
{
    return value;
}

template <class... Ts>
[[nodiscard]] inline FormExpr toScalarExpr(const std::variant<Ts...>& value, std::string_view name)
{
    return std::visit([&](const auto& v) { return toScalarExpr(v, name); }, value);
}

[[nodiscard]] inline std::string markerValueName(std::string_view prefix,
                                                 int boundary_marker)
{
    return detail::makeMarkerValueName(prefix, boundary_marker);
}

[[nodiscard]] inline std::string indexedComponentValueName(std::string_view prefix,
                                                          int boundary_marker,
                                                          int component)
{
    return detail::makeValueName(prefix, boundary_marker, static_cast<std::size_t>(component));
}

[[nodiscard]] inline std::string componentValueName(std::string_view prefix,
                                                    int boundary_marker,
                                                    int component)
{
    return detail::makeComponentValueName(prefix, boundary_marker, component);
}

template <class Values>
[[nodiscard]] inline std::vector<FormExpr> toVectorExpr(
    const Values& values,
    int dim,
    std::string_view name_prefix,
    int boundary_marker,
    ComponentValueNameStyle name_style = ComponentValueNameStyle::Indexed)
{
    std::vector<FormExpr> out;
    out.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        out.push_back(toScalarExpr(
            values[static_cast<std::size_t>(d)],
            detail::makeComponentValueName(name_prefix, boundary_marker, d, name_style)));
    }
    return out;
}

/**
 * @brief Strong (essential) Dirichlet boundary condition declaration
 *
 * Represents:
 *   u = g(x,t) on boundary marker Γ(marker)
 *
 * where `g` is a scalar FormExpr that must not depend on test/trial functions.
 *
 * Enforcement is handled by FE/Systems (lowered to constraints).
 */
struct StrongDirichlet {
    FieldId field{INVALID_FIELD_ID};
    int boundary_marker{-1};
    int component{-1};  // -1 means "all/unspecified" (scalar fields or apply uniformly to all components)
    FormExpr value{};

    // Optional symbol name for diagnostics / pretty-printing.
    std::string symbol{"u"};

    [[nodiscard]] bool isValid() const noexcept
    {
        return field != INVALID_FIELD_ID && component >= -1 && value.isValid();
    }

    [[nodiscard]] std::string toString() const
    {
        const std::string sym = (component >= 0) ? (symbol + "[" + std::to_string(component) + "]") : symbol;
        const std::string where = (boundary_marker >= 0)
                                      ? ("ds(" + std::to_string(boundary_marker) + ")")
                                      : "ds(*)";
        return sym + " = " + value.toString() + " on " + where;
    }
};

inline StrongDirichlet strongDirichlet(FieldId field,
                                       int boundary_marker,
                                       FormExpr value,
                                       std::string symbol = "u",
                                       int component = -1)
{
    return StrongDirichlet{field, boundary_marker, component, std::move(value), std::move(symbol)};
}

/**
 * @brief Apply scalar natural boundary data to a residual form without explicit loops
 *
 * For each BC on boundary marker Γ(m):
 *   k∇u·n = g  ⇒  adds  -∫ g v ds(m)
 *
 * Requirements on BC type:
 * - Must have an `int boundary_marker` member.
 *
 * The `fluxExpr(bc, i)` callback must return a scalar FormExpr `g` that is
 * independent of test/trial functions.
 */
template <class NeumannBC, class FluxExprFn>
[[nodiscard]] inline FormExpr applyNeumann(FormExpr residual,
                                           const FormExpr& v,
                                           std::span<const NeumannBC> bcs,
                                           FluxExprFn&& fluxExpr)
{
    for (std::size_t i = 0; i < bcs.size(); ++i) {
        const auto& bc = bcs[i];
        const int marker = detail::boundaryMarkerOrThrow(bc, "forms::bc::applyNeumann");
        residual = residual - (fluxExpr(bc, i) * v).ds(marker);
    }
    return residual;
}

/**
 * @brief Apply scalar natural boundary data on explicitly selected exterior
 * domains
 *
 * `measureExpr(bc, i)` must return an ExteriorBoundaryMeasure whose physical
 * marker matches `bc.boundary_marker`. This overload is the cut-boundary path:
 * callers can select a generated active subset without changing the flux
 * expression.
 */
template <class NeumannBC, class FluxExprFn, class MeasureExprFn>
[[nodiscard]] inline FormExpr applyNeumann(FormExpr residual,
                                           const FormExpr& v,
                                           std::span<const NeumannBC> bcs,
                                           FluxExprFn&& fluxExpr,
                                           MeasureExprFn&& measureExpr)
{
    for (std::size_t i = 0; i < bcs.size(); ++i) {
        const auto& bc = bcs[i];
        const int marker = detail::boundaryMarkerOrThrow(
            bc, "forms::bc::applyNeumann");
        const auto measure = measureExpr(bc, i);
        if (measure.physicalBoundaryMarker() != marker) {
            throw std::invalid_argument(
                "forms::bc::applyNeumann: exterior-boundary measure physical marker does not match the boundary condition");
        }
        residual =
            residual -
            (fluxExpr(bc, i) * v).dExteriorBoundary(measure);
    }
    return residual;
}

/**
 * @brief Apply Neumann BCs where the flux is stored directly in the BC struct
 *
 * This overload avoids per-formulation boilerplate for turning common scalar
 * values (constants / spatial / time-dependent coefficients) into FormExpr.
 *
 * Requirements on BC type:
 * - Must have an `int boundary_marker` member.
 * - Must have a value member (e.g., `flux`) of a type supported by `toScalarExpr`.
 */
template <class NeumannBC, class FluxValue>
[[nodiscard]] inline FormExpr applyNeumannValue(FormExpr residual,
                                                const FormExpr& v,
                                                std::span<const NeumannBC> bcs,
                                                FluxValue NeumannBC::*flux,
                                                std::string_view name_prefix)
{
    for (std::size_t i = 0; i < bcs.size(); ++i) {
        const auto& bc = bcs[i];
        const int marker = detail::boundaryMarkerOrThrow(bc, "forms::bc::applyNeumannValue");
        const auto g = toScalarExpr(bc.*flux, detail::makeValueName(name_prefix, marker, i));
        residual = residual - (g * v).ds(marker);
    }
    return residual;
}

template <class NeumannBC, class FluxValue, class MeasureExprFn>
[[nodiscard]] inline FormExpr applyNeumannValue(FormExpr residual,
                                                const FormExpr& v,
                                                std::span<const NeumannBC> bcs,
                                                FluxValue NeumannBC::*flux,
                                                std::string_view name_prefix,
                                                MeasureExprFn&& measureExpr)
{
    for (std::size_t i = 0; i < bcs.size(); ++i) {
        const auto& bc = bcs[i];
        const int marker = detail::boundaryMarkerOrThrow(
            bc, "forms::bc::applyNeumannValue");
        const auto measure = measureExpr(bc, i);
        if (measure.physicalBoundaryMarker() != marker) {
            throw std::invalid_argument(
                "forms::bc::applyNeumannValue: exterior-boundary measure physical marker does not match the boundary condition");
        }
        const auto g = toScalarExpr(
            bc.*flux,
            detail::makeValueName(name_prefix, marker, i));
        residual = residual - (g * v).dExteriorBoundary(measure);
    }
    return residual;
}

/**
 * @brief Apply scalar Robin-style boundary data to a residual form without explicit loops
 *
 * For each BC on boundary marker Γ(m):
 *   k∇u·n + α u = r  ⇒  adds  ∫ α u v ds(m) - ∫ r v ds(m)
 *
 * Requirements on BC type:
 * - Must have an `int boundary_marker` member.
 *
 * The callbacks must return scalar FormExprs `alpha` and `rhs` that are
 * independent of test/trial functions (except for the explicit `u`/`v` usage here).
 */
template <class RobinBC, class AlphaExprFn, class RhsExprFn>
[[nodiscard]] inline FormExpr applyRobin(FormExpr residual,
                                         const FormExpr& u,
                                         const FormExpr& v,
                                         std::span<const RobinBC> bcs,
                                         AlphaExprFn&& alphaExpr,
                                         RhsExprFn&& rhsExpr)
{
    for (std::size_t i = 0; i < bcs.size(); ++i) {
        const auto& bc = bcs[i];
        const int marker = detail::boundaryMarkerOrThrow(bc, "forms::bc::applyRobin");
        const auto a = alphaExpr(bc, i);
        const auto r = rhsExpr(bc, i);
        residual =
            residual + (a * u * v).ds(marker) -
            (r * v).ds(marker);
    }
    return residual;
}

template <class RobinBC,
          class AlphaExprFn,
          class RhsExprFn,
          class MeasureExprFn>
[[nodiscard]] inline FormExpr applyRobin(FormExpr residual,
                                         const FormExpr& u,
                                         const FormExpr& v,
                                         std::span<const RobinBC> bcs,
                                         AlphaExprFn&& alphaExpr,
                                         RhsExprFn&& rhsExpr,
                                         MeasureExprFn&& measureExpr)
{
    for (std::size_t i = 0; i < bcs.size(); ++i) {
        const auto& bc = bcs[i];
        const int marker = detail::boundaryMarkerOrThrow(
            bc, "forms::bc::applyRobin");
        const auto measure = measureExpr(bc, i);
        if (measure.physicalBoundaryMarker() != marker) {
            throw std::invalid_argument(
                "forms::bc::applyRobin: exterior-boundary measure physical marker does not match the boundary condition");
        }
        const auto a = alphaExpr(bc, i);
        const auto r = rhsExpr(bc, i);
        residual =
            residual + (a * u * v).dExteriorBoundary(measure) -
            (r * v).dExteriorBoundary(measure);
    }
    return residual;
}

/**
 * @brief Apply Robin BCs where alpha/rhs are stored directly in the BC struct
 *
 * Requirements on BC type:
 * - Must have an `int boundary_marker` member.
 * - Must have value members (e.g., `alpha`, `rhs`) of types supported by `toScalarExpr`.
 */
template <class RobinBC, class AlphaValue, class RhsValue>
[[nodiscard]] inline FormExpr applyRobinValue(FormExpr residual,
                                              const FormExpr& u,
                                              const FormExpr& v,
                                              std::span<const RobinBC> bcs,
                                              AlphaValue RobinBC::*alpha,
                                              std::string_view alpha_name_prefix,
                                              RhsValue RobinBC::*rhs,
                                              std::string_view rhs_name_prefix)
{
    for (std::size_t i = 0; i < bcs.size(); ++i) {
        const auto& bc = bcs[i];
        const int marker = detail::boundaryMarkerOrThrow(bc, "forms::bc::applyRobinValue");
        const auto a = toScalarExpr(bc.*alpha, detail::makeValueName(alpha_name_prefix, marker, i));
        const auto r = toScalarExpr(bc.*rhs, detail::makeValueName(rhs_name_prefix, marker, i));
        residual =
            residual + (a * u * v).ds(marker) -
            (r * v).ds(marker);
    }
    return residual;
}

template <class RobinBC,
          class AlphaValue,
          class RhsValue,
          class MeasureExprFn>
[[nodiscard]] inline FormExpr applyRobinValue(
    FormExpr residual,
    const FormExpr& u,
    const FormExpr& v,
    std::span<const RobinBC> bcs,
    AlphaValue RobinBC::*alpha,
    std::string_view alpha_name_prefix,
    RhsValue RobinBC::*rhs,
    std::string_view rhs_name_prefix,
    MeasureExprFn&& measureExpr)
{
    for (std::size_t i = 0; i < bcs.size(); ++i) {
        const auto& bc = bcs[i];
        const int marker = detail::boundaryMarkerOrThrow(
            bc, "forms::bc::applyRobinValue");
        const auto measure = measureExpr(bc, i);
        if (measure.physicalBoundaryMarker() != marker) {
            throw std::invalid_argument(
                "forms::bc::applyRobinValue: exterior-boundary measure physical marker does not match the boundary condition");
        }
        const auto a = toScalarExpr(
            bc.*alpha,
            detail::makeValueName(
                alpha_name_prefix, marker, i));
        const auto r = toScalarExpr(
            bc.*rhs,
            detail::makeValueName(
                rhs_name_prefix, marker, i));
        residual =
            residual + (a * u * v).dExteriorBoundary(measure) -
            (r * v).dExteriorBoundary(measure);
    }
    return residual;
}

/**
 * @brief Options for weak Dirichlet enforcement via Nitsche's method
 */
enum class NitscheVariant {
    Symmetric,
    Unsymmetric
};

struct NitscheDirichletOptions {
    Real gamma{10.0};  ///< Penalty parameter multiplier (scaled by k/h and optionally p^2)
    NitscheVariant variant{NitscheVariant::Symmetric};
    bool scale_with_p{true};  ///< Scale penalty by p^2 using TrialFunction polynomial order when available
};

using TraceNitscheOptions = NitscheDirichletOptions;

class GeneratedBoundaryNitscheTraceFormBinding;
struct GeneratedBoundaryNitscheTraceTerms;

[[nodiscard]] GeneratedBoundaryNitscheTraceTerms
buildGeneratedBoundarySymmetricGradientNitscheTraceTerms(
    const FormExpr& u,
    const FormExpr& v,
    const FormExpr& prescribed_value,
    const FormExpr& dynamic_viscosity,
    int physical_boundary_marker,
    int generated_active_boundary_marker,
    const TraceNitscheOptions& opts = {});

/**
 * Immutable proof that a generated-interface symmetric-gradient Nitsche
 * contribution was constructed from canonical state/test terminals with a
 * matching complete function-space signature by the canonical FE form
 * helper.
 *
 * Callers can copy a valid binding but cannot manufacture or edit one. The
 * formulation installer additionally requires the exact immutable route
 * anchor to occur once as an unscaled top-level additive summand in the
 * original residual being installed before it derives a generated-boundary
 * trace-certificate policy. This binding authenticates only that boundary
 * route; it does not authenticate a matching bulk viscous form. Any
 * coercivity interpretation of the resulting trace policy is therefore
 * conditional on the caller providing the corresponding bulk energy.
 */
class GeneratedBoundaryNitscheTraceFormBinding {
public:
    GeneratedBoundaryNitscheTraceFormBinding(
        const GeneratedBoundaryNitscheTraceFormBinding&) = default;
    GeneratedBoundaryNitscheTraceFormBinding&
    operator=(const GeneratedBoundaryNitscheTraceFormBinding&) = default;
    GeneratedBoundaryNitscheTraceFormBinding(
        GeneratedBoundaryNitscheTraceFormBinding&&) noexcept = default;
    GeneratedBoundaryNitscheTraceFormBinding&
    operator=(GeneratedBoundaryNitscheTraceFormBinding&&) noexcept = default;

    [[nodiscard]] FieldId velocityField() const noexcept
    {
        return velocity_field_;
    }
    [[nodiscard]] const SpaceSignature&
    velocitySpaceSignature() const noexcept
    {
        return velocity_space_signature_;
    }
    [[nodiscard]] int physicalBoundaryMarker() const noexcept
    {
        return physical_boundary_marker_;
    }
    [[nodiscard]] int generatedActiveBoundaryMarker() const noexcept
    {
        return generated_active_boundary_marker_;
    }
    [[nodiscard]] Real dynamicViscosity() const noexcept
    {
        return dynamic_viscosity_;
    }
    [[nodiscard]] Real penaltyGamma() const noexcept
    {
        return penalty_gamma_;
    }
    [[nodiscard]] bool scaleWithPolynomialOrder() const noexcept
    {
        return scale_with_polynomial_order_;
    }
    [[nodiscard]] int penaltyPolynomialOrder() const noexcept
    {
        return penalty_polynomial_order_;
    }
    [[nodiscard]] Real effectivePenaltyMultiplier() const noexcept
    {
        return effective_penalty_multiplier_;
    }
    [[nodiscard]] bool symmetric() const noexcept
    {
        return symmetric_;
    }
    [[nodiscard]] std::uint64_t metadataDigest() const noexcept
    {
        return metadata_digest_;
    }
    [[nodiscard]] const FormExpr& routeAnchor() const noexcept
    {
        return route_anchor_;
    }

private:
    GeneratedBoundaryNitscheTraceFormBinding(
        FieldId velocity_field,
        SpaceSignature velocity_space_signature,
        int physical_boundary_marker,
        int generated_active_boundary_marker,
        Real dynamic_viscosity,
        Real penalty_gamma,
        bool scale_with_polynomial_order,
        int penalty_polynomial_order,
        Real effective_penalty_multiplier,
        bool symmetric,
        std::uint64_t metadata_digest,
        FormExpr route_anchor)
        : velocity_field_(velocity_field)
        , velocity_space_signature_(
              std::move(velocity_space_signature))
        , physical_boundary_marker_(physical_boundary_marker)
        , generated_active_boundary_marker_(
              generated_active_boundary_marker)
        , dynamic_viscosity_(dynamic_viscosity)
        , penalty_gamma_(penalty_gamma)
        , scale_with_polynomial_order_(scale_with_polynomial_order)
        , penalty_polynomial_order_(penalty_polynomial_order)
        , effective_penalty_multiplier_(
              effective_penalty_multiplier)
        , symmetric_(symmetric)
        , metadata_digest_(metadata_digest)
        , route_anchor_(std::move(route_anchor))
    {
    }

    friend GeneratedBoundaryNitscheTraceTerms
    buildGeneratedBoundarySymmetricGradientNitscheTraceTerms(
        const FormExpr&,
        const FormExpr&,
        const FormExpr&,
        const FormExpr&,
        int,
        int,
        const TraceNitscheOptions&);

    FieldId velocity_field_{INVALID_FIELD_ID};
    SpaceSignature velocity_space_signature_{};
    int physical_boundary_marker_{-1};
    int generated_active_boundary_marker_{-1};
    Real dynamic_viscosity_{0.0};
    Real penalty_gamma_{0.0};
    bool scale_with_polynomial_order_{true};
    int penalty_polynomial_order_{0};
    Real effective_penalty_multiplier_{0.0};
    bool symmetric_{true};
    std::uint64_t metadata_digest_{0u};
    FormExpr route_anchor_{};
};

struct GeneratedBoundaryNitscheTraceTerms {
    FormExpr route_contribution{};
    FormExpr homogeneous_symmetric_consistency{};
    FormExpr homogeneous_penalty{};
    GeneratedBoundaryNitscheTraceFormBinding binding;
};

[[nodiscard]] inline GeneratedBoundaryNitscheTraceTerms
buildGeneratedBoundarySymmetricGradientNitscheTraceTerms(
    const FormExpr& u,
    const FormExpr& v,
    const FormExpr& prescribed_value,
    const FormExpr& dynamic_viscosity,
    int physical_boundary_marker,
    int generated_active_boundary_marker,
    const TraceNitscheOptions& opts)
{
    if (!u.isValid() || !v.isValid() ||
        !prescribed_value.isValid() ||
        !dynamic_viscosity.isValid()) {
        throw std::invalid_argument(
            "forms::bc::buildGeneratedBoundarySymmetricGradientNitscheTraceTerms: invalid form input");
    }
    if (physical_boundary_marker < 0 ||
        generated_active_boundary_marker < 0) {
        throw std::invalid_argument(
            "forms::bc::buildGeneratedBoundarySymmetricGradientNitscheTraceTerms: markers must be nonnegative");
    }
    if (!(opts.gamma > Real{0.0}) ||
        !std::isfinite(opts.gamma)) {
        throw std::invalid_argument(
            "forms::bc::buildGeneratedBoundarySymmetricGradientNitscheTraceTerms: gamma must be finite and > 0");
    }
    if (opts.variant != NitscheVariant::Symmetric &&
        opts.variant != NitscheVariant::Unsymmetric) {
        throw std::invalid_argument(
            "forms::bc::buildGeneratedBoundarySymmetricGradientNitscheTraceTerms: Nitsche variant is invalid");
    }
    const auto* u_node = u.node();
    const auto* v_node = v.node();
    const auto u_field =
        u_node == nullptr ? std::optional<FieldId>{}
                          : u_node->fieldId();
    const auto v_field =
        v_node == nullptr ? std::optional<FieldId>{}
                          : v_node->fieldId();
    if (!u_field.has_value() || !v_field.has_value() ||
        *u_field == INVALID_FIELD_ID || *u_field != *v_field) {
        throw std::invalid_argument(
            "forms::bc::buildGeneratedBoundarySymmetricGradientNitscheTraceTerms: u and v must be bound to the same valid field");
    }
    if (u_node->type() != FormExprType::StateField ||
        v_node->type() != FormExprType::TestFunction) {
        throw std::invalid_argument(
            "forms::bc::buildGeneratedBoundarySymmetricGradientNitscheTraceTerms: u must be a state-field terminal and v must be a test-function terminal");
    }
    const auto* u_space = u_node->spaceSignature();
    const auto* v_space = v_node->spaceSignature();
    if (u_space == nullptr || v_space == nullptr ||
        !detail::spaceSignaturesMatch(*u_space, *v_space)) {
        throw std::invalid_argument(
            "forms::bc::buildGeneratedBoundarySymmetricGradientNitscheTraceTerms: u and v must use the same complete function-space signature");
    }
    const auto* prescribed_value_node =
        prescribed_value.node();
    if (prescribed_value_node == nullptr ||
        detail::hasForbiddenPrescribedValueDependency(
            *prescribed_value_node)) {
        throw std::invalid_argument(
            "forms::bc::buildGeneratedBoundarySymmetricGradientNitscheTraceTerms: prescribed value must be independent of FE state, coupled data, and variational geometry");
    }
    if (!detail::isCompatiblePrescribedVector(
            *prescribed_value_node,
            *u_space)) {
        throw std::invalid_argument(
            "forms::bc::buildGeneratedBoundarySymmetricGradientNitscheTraceTerms: prescribed value must be a vector compatible with u");
    }
    const auto viscosity =
        dynamic_viscosity.node() == nullptr
            ? std::optional<Real>{}
            : dynamic_viscosity.node()->constantValue();
    if (!viscosity.has_value() ||
        !(*viscosity > Real{0.0}) ||
        !std::isfinite(*viscosity)) {
        throw std::invalid_argument(
            "forms::bc::buildGeneratedBoundarySymmetricGradientNitscheTraceTerms: dynamic viscosity must be a finite positive constant");
    }

    const int penalty_polynomial_order =
        u_space->polynomial_order;
    if (penalty_polynomial_order < 1) {
        throw std::invalid_argument(
            "forms::bc::buildGeneratedBoundarySymmetricGradientNitscheTraceTerms: polynomial order must be at least one");
    }
    const Real order_scale =
        opts.scale_with_p
            ? static_cast<Real>(penalty_polynomial_order) *
                  static_cast<Real>(penalty_polynomial_order)
            : Real{1.0};
    const Real effective_penalty_multiplier =
        opts.gamma * order_scale;
    if (!(effective_penalty_multiplier > Real{0.0}) ||
        !std::isfinite(effective_penalty_multiplier)) {
        throw std::invalid_argument(
            "forms::bc::buildGeneratedBoundarySymmetricGradientNitscheTraceTerms: effective penalty multiplier is invalid");
    }

    const auto n = FormExpr::normal();
    const auto diff = u - prescribed_value;
    const auto stress_u =
        FormExpr::constant(Real{2.0}) *
        dynamic_viscosity * sym(grad(u));
    const auto stress_v =
        FormExpr::constant(Real{2.0}) *
        dynamic_viscosity * sym(grad(v));
    const auto h_normal =
        (FormExpr::constant(Real{2.0}) *
         FormExpr::cellVolume()) /
        FormExpr::facetArea();
    const auto penalty =
        FormExpr::constant(effective_penalty_multiplier) *
        dynamic_viscosity / h_normal;
    const auto measure =
        ExteriorBoundaryMeasure::generatedActiveSubset(
            physical_boundary_marker,
            generated_active_boundary_marker);
    const auto primal_consistency =
        inner(stress_u * n, v)
            .dExteriorBoundary(measure);
    const auto adjoint_consistency =
        inner(stress_v * n, diff)
            .dExteriorBoundary(measure);
    const auto penalty_term =
        (penalty * inner(diff, v))
            .dExteriorBoundary(measure);
    const auto route_contribution =
        opts.variant == NitscheVariant::Symmetric
            ? (-primal_consistency -
               adjoint_consistency +
               penalty_term)
            : (-primal_consistency +
               adjoint_consistency +
               penalty_term);

    const auto homogeneous_symmetric_consistency =
        -primal_consistency -
        inner(stress_v * n, u)
            .dExteriorBoundary(measure);
    const auto homogeneous_penalty =
        (penalty * inner(u, v))
            .dExteriorBoundary(measure);

    constexpr std::uint64_t offset = 1469598103934665603ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    auto metadata_digest = offset;
    const auto mix = [&](std::uint64_t value) {
        metadata_digest ^= value;
        metadata_digest *= prime;
    };
    static_assert(sizeof(Real) == sizeof(std::uint64_t));
    mix(1u);
    mix(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(*u_field)));
    mix(static_cast<std::uint64_t>(
        u_space->space_type));
    mix(static_cast<std::uint64_t>(
        u_space->field_type));
    mix(static_cast<std::uint64_t>(
        u_space->continuity));
    mix(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(
            u_space->value_dimension)));
    mix(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(
            u_space->topological_dimension)));
    mix(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(
            u_space->polynomial_order)));
    mix(static_cast<std::uint64_t>(
        u_space->element_type));
    mix(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(
            physical_boundary_marker)));
    mix(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(
            generated_active_boundary_marker)));
    mix(std::bit_cast<std::uint64_t>(*viscosity));
    mix(std::bit_cast<std::uint64_t>(opts.gamma));
    mix(opts.scale_with_p ? 1u : 0u);
    mix(static_cast<std::uint64_t>(
        penalty_polynomial_order));
    mix(std::bit_cast<std::uint64_t>(
        effective_penalty_multiplier));
    mix(opts.variant == NitscheVariant::Symmetric
            ? 1u
            : 0u);
    if (metadata_digest == 0u) {
        metadata_digest = 1u;
    }

    auto binding =
        GeneratedBoundaryNitscheTraceFormBinding(
            *u_field,
            *u_space,
            physical_boundary_marker,
            generated_active_boundary_marker,
            *viscosity,
            opts.gamma,
            opts.scale_with_p,
            penalty_polynomial_order,
            effective_penalty_multiplier,
            opts.variant == NitscheVariant::Symmetric,
            metadata_digest,
            route_contribution);
    return GeneratedBoundaryNitscheTraceTerms{
        .route_contribution = route_contribution,
        .homogeneous_symmetric_consistency =
            homogeneous_symmetric_consistency,
        .homogeneous_penalty = homogeneous_penalty,
        .binding = std::move(binding),
    };
}

enum class TraceInequalitySense : std::uint8_t {
    LessEqual,
    GreaterEqual
};

enum class TraceInequalityLinearization : std::uint8_t {
    SemiSmooth,
    Smooth
};

struct TraceInequalityOptions {
    ScalarTraceOperator trace_operator{ScalarTraceOperator::NormalComponent};
    TraceInequalitySense sense{TraceInequalitySense::LessEqual};
    TraceInequalityLinearization linearization{TraceInequalityLinearization::SemiSmooth};
    FormExpr smoothing_epsilon{};
};

[[nodiscard]] inline FormExpr traceInequalityViolation(
    const FormExpr& trace_value,
    const FormExpr& bound,
    const TraceInequalityOptions& opts = {})
{
    const auto signed_gap =
        (opts.sense == TraceInequalitySense::LessEqual)
            ? (trace_value - bound)
            : (bound - trace_value);

    if (opts.linearization == TraceInequalityLinearization::Smooth) {
        if (!opts.smoothing_epsilon.isValid()) {
            throw std::invalid_argument(
                "forms::bc::traceInequalityViolation: smooth linearization requires a valid smoothing_epsilon");
        }
        return smoothMax(FormExpr::constant(0.0), signed_gap, opts.smoothing_epsilon);
    }

    return max(FormExpr::constant(0.0), signed_gap);
}

[[nodiscard]] inline FormExpr applyTraceInequality(
    FormExpr residual,
    const FormExpr& u,
    const FormExpr& v,
    const ExteriorBoundaryMeasure& measure,
    const FormExpr& bound,
    const FormExpr& penalty_weight,
    const TraceInequalityOptions& opts = {})
{
    if (!bound.isValid()) {
        throw std::invalid_argument("forms::bc::applyTraceInequality: invalid bound expression");
    }
    if (!penalty_weight.isValid()) {
        throw std::invalid_argument("forms::bc::applyTraceInequality: invalid penalty_weight expression");
    }

    const auto tau_u = applyScalarTrace(u, opts.trace_operator);
    const auto tau_v = applyScalarTrace(v, opts.trace_operator);
    const auto violation = traceInequalityViolation(tau_u, bound, opts);
    const Real direction = (opts.sense == TraceInequalitySense::LessEqual) ? Real(1.0) : Real(-1.0);

    residual = residual +
               (FormExpr::constant(direction) * penalty_weight * violation * tau_v)
                   .dExteriorBoundary(measure);
    return residual;
}

[[nodiscard]] inline FormExpr applyTraceInequality(
    FormExpr residual,
    const FormExpr& u,
    const FormExpr& v,
    int boundary_marker,
    const FormExpr& bound,
    const FormExpr& penalty_weight,
    const TraceInequalityOptions& opts = {})
{
    if (boundary_marker < 0) {
        throw std::invalid_argument(
            "forms::bc::applyTraceInequality: boundary_marker must be >= 0");
    }
    if (!bound.isValid()) {
        throw std::invalid_argument(
            "forms::bc::applyTraceInequality: invalid bound expression");
    }
    if (!penalty_weight.isValid()) {
        throw std::invalid_argument(
            "forms::bc::applyTraceInequality: invalid penalty_weight expression");
    }

    const auto tau_u = applyScalarTrace(u, opts.trace_operator);
    const auto tau_v = applyScalarTrace(v, opts.trace_operator);
    const auto violation =
        traceInequalityViolation(tau_u, bound, opts);
    const Real direction =
        opts.sense == TraceInequalitySense::LessEqual
            ? Real(1.0)
            : Real(-1.0);
    return residual +
           (FormExpr::constant(direction) * penalty_weight *
            violation * tau_v)
               .ds(boundary_marker);
}

[[nodiscard]] inline FormExpr buildTraceNitschePenalty(const FormExpr& penalty_weight,
                                                       const FormExpr& trial_trace_source,
                                                       const TraceNitscheOptions& opts = {})
{
    if (opts.gamma <= Real(0.0)) {
        throw std::invalid_argument("forms::bc::buildTraceNitschePenalty: gamma must be > 0");
    }

    int p = 1;
    if (opts.scale_with_p) {
        p = detail::polynomialOrderOrDefault(trial_trace_source, /*default_order=*/1);
        if (p < 1) {
            p = 1;
        }
    }

    const auto p2 = FormExpr::constant(static_cast<Real>(p * p));
    return FormExpr::constant(opts.gamma) * penalty_weight * p2;
}

[[nodiscard]] inline FormExpr applyTraceNitsche(FormExpr residual,
                                                const FormExpr& u,
                                                const FormExpr& v,
                                                const ExteriorBoundaryMeasure& measure,
                                                const FormExpr& value,
                                                const FormExpr& consistency_flux_u,
                                                const FormExpr& adjoint_flux_v,
                                                const FormExpr& penalty_weight,
                                                ScalarTraceOperator trace_operator =
                                                    ScalarTraceOperator::NormalComponent,
                                                const TraceNitscheOptions& opts = {})
{
    const auto tau_u = applyScalarTrace(u, trace_operator);
    const auto tau_v = applyScalarTrace(v, trace_operator);
    const auto diff = tau_u - value;
    const auto penalty = buildTraceNitschePenalty(penalty_weight, u, opts);

    residual =
        residual -
        (consistency_flux_u * tau_v).dExteriorBoundary(measure);
    if (opts.variant == NitscheVariant::Symmetric) {
        residual =
            residual -
            (adjoint_flux_v * diff).dExteriorBoundary(measure);
    } else {
        residual =
            residual +
            (adjoint_flux_v * diff).dExteriorBoundary(measure);
    }
    residual =
        residual +
        (penalty * diff * tau_v).dExteriorBoundary(measure);
    return residual;
}

[[nodiscard]] inline FormExpr applyTraceNitsche(FormExpr residual,
                                                const FormExpr& u,
                                                const FormExpr& v,
                                                int boundary_marker,
                                                const FormExpr& value,
                                                const FormExpr& consistency_flux_u,
                                                const FormExpr& adjoint_flux_v,
                                                const FormExpr& penalty_weight,
                                                ScalarTraceOperator trace_operator =
                                                    ScalarTraceOperator::NormalComponent,
                                                const TraceNitscheOptions& opts = {})
{
    if (boundary_marker < 0) {
        throw std::invalid_argument(
            "forms::bc::applyTraceNitsche: boundary_marker must be >= 0");
    }
    const auto tau_u = applyScalarTrace(u, trace_operator);
    const auto tau_v = applyScalarTrace(v, trace_operator);
    const auto diff = tau_u - value;
    const auto penalty =
        buildTraceNitschePenalty(penalty_weight, u, opts);

    residual =
        residual -
        (consistency_flux_u * tau_v).ds(boundary_marker);
    if (opts.variant == NitscheVariant::Symmetric) {
        residual =
            residual -
            (adjoint_flux_v * diff).ds(boundary_marker);
    } else {
        residual =
            residual +
            (adjoint_flux_v * diff).ds(boundary_marker);
    }
    return residual +
           (penalty * diff * tau_v).ds(boundary_marker);
}

[[nodiscard]] inline FormExpr applyInterfaceTraceNitsche(
    FormExpr residual,
    const FormExpr& u,
    const FormExpr& v,
    int interface_marker,
    const FormExpr& value,
    const FormExpr& consistency_flux_u,
    const FormExpr& adjoint_flux_v,
    const FormExpr& penalty_weight,
    ScalarTraceOperator trace_operator = ScalarTraceOperator::NormalComponent,
    InterfaceTraceReduction reduction = InterfaceTraceReduction::Jump,
    const TraceNitscheOptions& opts = {})
{
    if (interface_marker < 0) {
        throw std::invalid_argument("forms::bc::applyInterfaceTraceNitsche: interface_marker must be >= 0");
    }

    const auto tau_u = applyInterfaceScalarTrace(u, trace_operator, reduction);
    const auto tau_v = applyInterfaceScalarTrace(v, trace_operator, reduction);
    const auto diff = tau_u - value;
    const auto penalty = buildTraceNitschePenalty(penalty_weight, u, opts);

    residual = residual - (consistency_flux_u * tau_v).dI(interface_marker);
    if (opts.variant == NitscheVariant::Symmetric) {
        residual = residual - (adjoint_flux_v * diff).dI(interface_marker);
    } else {
        residual = residual + (adjoint_flux_v * diff).dI(interface_marker);
    }
    residual = residual + (penalty * diff * tau_v).dI(interface_marker);
    return residual;
}

/**
 * @brief Legacy scalar-diffusion Nitsche helper.
 *
 * Prefer `applyTraceNitsche(...)` in new formulation code so the consistency
 * flux, adjoint flux, and penalty weight remain explicit in the form file.
 * This compatibility helper is retained for existing scalar Poisson-style
 * modules.
 *
 * Imposes (on each boundary marker Γ(m)):
 *   u = uD
 *
 * Residual contributions (symmetric variant):
 *   -∫ k (∇u·n) v ds
 *   -∫ k (∇v·n) (u-uD) ds
 *   +∫ (γ k p^2 / h) (u-uD) v ds
 *
 * with the facet-normal element size h_n = 2|K|/|F| (cell volume divided by facet area).
 *
 * The unsymmetric variant flips the sign of the second term.
 *
 * Requirements on BC type:
 * - Must have an `int boundary_marker` member.
 *
 * The `valueExpr(bc, i)` callback must return a scalar FormExpr `uD` that is
 * independent of test/trial functions.
 */
template <class DirichletBC, class ValueExprFn>
[[nodiscard]] inline FormExpr applyNitscheDirichletPoisson(FormExpr residual,
                                                           const FormExpr& k,
                                                           const FormExpr& u,
                                                           const FormExpr& v,
                                                           std::span<const DirichletBC> bcs,
                                                           ValueExprFn&& valueExpr,
                                                           const NitscheDirichletOptions& opts = {})
{
    if (opts.gamma <= Real(0.0)) {
        throw std::invalid_argument("forms::bc::applyNitscheDirichletPoisson: gamma must be > 0");
    }

    const auto n = FormExpr::normal();
    const auto h = (2.0 * FormExpr::cellVolume()) / FormExpr::facetArea();

    for (std::size_t i = 0; i < bcs.size(); ++i) {
        const auto& bc = bcs[i];
        const int marker = detail::boundaryMarkerOrThrow(bc, "forms::bc::applyNitscheDirichletPoisson");
        const auto uD = valueExpr(bc, i);
        residual = applyTraceNitsche(std::move(residual),
                                     u,
                                     v,
                                     marker,
                                     uD,
                                     k * inner(grad(u), n),
                                     k * inner(grad(v), n),
                                     k / h,
                                     ScalarTraceOperator::Identity,
                                     opts);
    }
    return residual;
}

/**
 * @brief Apply weak Dirichlet BCs for scalar Poisson diffusion (value stored in BC struct)
 *
 * This overload avoids per-formulation boilerplate for turning common scalar
 * values into FormExpr via `toScalarExpr`.
 *
 * Requirements on BC type:
 * - Must have an `int boundary_marker` member.
 * - Must have a value member (e.g., `value`) of a type supported by `toScalarExpr`.
 */
template <class DirichletBC, class Value>
[[nodiscard]] inline FormExpr applyNitscheDirichletPoissonValue(FormExpr residual,
                                                                const FormExpr& k,
                                                                const FormExpr& u,
                                                                const FormExpr& v,
                                                                std::span<const DirichletBC> bcs,
                                                                Value DirichletBC::*value,
                                                                std::string_view name_prefix,
                                                                const NitscheDirichletOptions& opts = {})
{
    return applyNitscheDirichletPoisson(
        std::move(residual),
        k,
        u,
        v,
        bcs,
        [&](const DirichletBC& bc, std::size_t i) {
            const int marker = detail::boundaryMarkerOrThrow(bc, "forms::bc::applyNitscheDirichletPoissonValue");
            return toScalarExpr(bc.*value, detail::makeValueName(name_prefix, marker, i));
        },
        opts);
}

/**
 * @brief Build a `StrongDirichlet` declaration list without explicit loops in formulations
 *
 * Requirements on BC type:
 * - Must have an `int boundary_marker` member.
 *
 * The `valueExpr(bc, i)` callback must return a scalar FormExpr `uD` that is
 * independent of test/trial functions.
 */
template <class DirichletBC, class ValueExprFn>
[[nodiscard]] inline std::vector<StrongDirichlet> makeStrongDirichletList(
    FieldId field,
    std::span<const DirichletBC> bcs,
    ValueExprFn&& valueExpr,
    std::string symbol = "u")
{
    std::vector<StrongDirichlet> out;
    out.reserve(bcs.size());
    for (std::size_t i = 0; i < bcs.size(); ++i) {
        const auto& bc = bcs[i];
        const int marker = detail::boundaryMarkerOrThrow(bc, "forms::bc::makeStrongDirichletList");
        out.push_back(strongDirichlet(field, marker, valueExpr(bc, i), symbol));
    }
    return out;
}

/**
 * @brief Build StrongDirichlet declarations where the value is stored directly in the BC struct
 *
 * Requirements on BC type:
 * - Must have an `int boundary_marker` member.
 * - Must have a value member (e.g., `value`) of a type supported by `toScalarExpr`.
 */
template <class DirichletBC, class Value>
[[nodiscard]] inline std::vector<StrongDirichlet> makeStrongDirichletListValue(
    FieldId field,
    std::span<const DirichletBC> bcs,
    Value DirichletBC::*value,
    std::string_view name_prefix,
    std::string symbol = "u")
{
    std::vector<StrongDirichlet> out;
    out.reserve(bcs.size());
    for (std::size_t i = 0; i < bcs.size(); ++i) {
        const auto& bc = bcs[i];
        const int marker = detail::boundaryMarkerOrThrow(bc, "forms::bc::makeStrongDirichletListValue");
        out.push_back(strongDirichlet(
            field, marker, toScalarExpr(bc.*value, detail::makeValueName(name_prefix, marker, i)), symbol));
    }
    return out;
}

} // namespace bc
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_BOUNDARYCONDITIONS_H
