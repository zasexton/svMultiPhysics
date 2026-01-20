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

#include <cstddef>
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

} // namespace detail

/**
 * @brief Canonical scalar-valued boundary condition value type
 *
 * This is intended to be used by Physics modules when defining boundary
 * condition option structs to avoid re-defining common variants.
 */
using ScalarValue = std::variant<Real, ScalarCoefficient, TimeScalarCoefficient, FormExpr>;

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
 * @brief Apply a list of Poisson-style Neumann BCs to a residual form without explicit loops
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

/**
 * @brief Apply a list of Poisson-style Robin BCs to a residual form without explicit loops
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
        residual = residual + (a * u * v).ds(marker) - (r * v).ds(marker);
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
        residual = residual + (a * u * v).ds(marker) - (r * v).ds(marker);
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

/**
 * @brief Apply weak Dirichlet BCs for scalar Poisson diffusion using Nitsche's method
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

    int p = 1;
    if (opts.scale_with_p) {
        p = detail::polynomialOrderOrDefault(u, /*default_order=*/1);
        if (p < 1) {
            p = 1;
        }
    }
    const auto p2 = FormExpr::constant(static_cast<Real>(p * p));
    const auto penalty = FormExpr::constant(opts.gamma) * k * p2 / h;

    for (std::size_t i = 0; i < bcs.size(); ++i) {
        const auto& bc = bcs[i];
        const int marker = detail::boundaryMarkerOrThrow(bc, "forms::bc::applyNitscheDirichletPoisson");
        const auto uD = valueExpr(bc, i);
        const auto diff = u - uD;

        residual = residual - (k * inner(grad(u), n) * v).ds(marker);
        if (opts.variant == NitscheVariant::Symmetric) {
            residual = residual - (k * inner(grad(v), n) * diff).ds(marker);
        } else {
            residual = residual + (k * inner(grad(v), n) * diff).ds(marker);
        }
        residual = residual + (penalty * diff * v).ds(marker);
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
