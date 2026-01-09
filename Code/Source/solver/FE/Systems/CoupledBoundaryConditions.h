#ifndef SVMP_FE_SYSTEMS_COUPLED_BOUNDARY_CONDITIONS_H
#define SVMP_FE_SYSTEMS_COUPLED_BOUNDARY_CONDITIONS_H

/**
 * @file CoupledBoundaryConditions.h
 * @brief Loop-free helper APIs for coupled (non-local) boundary conditions
 *
 * These helpers mirror the style of `forms::bc::applyNeumann/applyRobin` but
 * additionally wire the coupled-BC orchestration managed by
 * `systems::CoupledBoundaryManager`:
 *  - register required boundary functionals (non-local integrals),
 *  - register auxiliary (0D) state variables and evolution callbacks,
 *  - add boundary-integral terms to a Forms residual expression.
 *
 * The non-local data is provided to coefficient evaluation via a stable
 * `constraints::CoupledBCContext` owned by the manager and updated just before
 * each PDE assembly call.
 */

#include "Core/Types.h"
#include "Core/FEException.h"

#include "Constraints/CoupledNeumannBC.h"
#include "Constraints/CoupledRobinBC.h"

#include "Forms/BoundaryFunctional.h"
#include "Forms/FormExpr.h"

#include "Systems/CoupledBoundaryManager.h"
#include "Systems/FESystem.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {
namespace bc {

namespace detail {

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

[[nodiscard]] inline bool isIntegralNodeType(forms::FormExprType t) noexcept
{
    return t == forms::FormExprType::CellIntegral ||
           t == forms::FormExprType::BoundaryIntegral ||
           t == forms::FormExprType::InteriorFaceIntegral;
}

inline void collectCoupledSymbolsImpl(const std::shared_ptr<forms::FormExprNode>& node,
                                      std::vector<forms::BoundaryFunctional>& functionals_out,
                                      std::unordered_map<std::string, std::pair<int, std::string>>& seen_functionals,
                                      std::vector<std::string>& aux_out,
                                      std::unordered_set<std::string>& seen_aux)
{
    if (!node) return;

    if (node->type() == forms::FormExprType::BoundaryFunctionalSymbol) {
        const int marker = node->boundaryMarker().value_or(-1);
        const auto name = node->symbolName();
        FE_THROW_IF(marker < 0 || !name || name->empty(), InvalidArgumentException,
                    "applyCoupled*: boundaryIntegral(...) symbol is missing boundary marker or name");

        const auto kids = node->childrenShared();
        FE_THROW_IF(kids.size() != 1u || !kids[0], InvalidArgumentException,
                    "applyCoupled*: boundaryIntegral(...) symbol is missing integrand");
        forms::FormExpr integrand(kids[0]);

        FE_THROW_IF(!integrand.isValid(), InvalidArgumentException,
                    "applyCoupled*: boundaryIntegral(...) integrand is invalid");
        FE_THROW_IF(integrand.hasTest() || integrand.hasTrial(), InvalidArgumentException,
                    "applyCoupled*: boundaryIntegral(...) integrand must not contain test/trial functions");

        // Disallow nested integrals and nested coupled placeholders inside the functional integrand.
        const auto contains_disallowed = integrand.transformNodes([](const forms::FormExprNode& n)
                                                                     -> std::optional<forms::FormExpr> {
            if (n.type() == forms::FormExprType::BoundaryFunctionalSymbol ||
                n.type() == forms::FormExprType::BoundaryIntegralSymbol ||
                n.type() == forms::FormExprType::AuxiliaryStateSymbol ||
                isIntegralNodeType(n.type())) {
                FE_THROW(InvalidArgumentException,
                         "applyCoupled*: boundaryIntegral(...) integrand must be a pure integrand (no measures or coupled symbols)");
            }
            return std::nullopt;
        });
        (void)contains_disallowed;

        const std::string key(*name);
        const std::string sig = integrand.toString();

        auto it = seen_functionals.find(key);
        if (it == seen_functionals.end()) {
            seen_functionals.emplace(key, std::make_pair(marker, sig));
            forms::BoundaryFunctional f;
            f.integrand = integrand;
            f.boundary_marker = marker;
            f.name = key;
            f.reduction = forms::BoundaryFunctional::Reduction::Sum;
            functionals_out.push_back(std::move(f));
        } else {
            FE_THROW_IF(it->second.first != marker || it->second.second != sig, InvalidArgumentException,
                        "applyCoupled*: boundaryIntegral(...) name '" + key + "' is used with inconsistent marker or integrand");
        }
    } else if (node->type() == forms::FormExprType::AuxiliaryStateSymbol) {
        const auto name = node->symbolName();
        FE_THROW_IF(!name || name->empty(), InvalidArgumentException,
                    "applyCoupled*: auxiliaryState(...) symbol is missing a name");
        const std::string key(*name);
        if (seen_aux.insert(key).second) {
            aux_out.push_back(key);
        }
    }

    for (const auto& child : node->childrenShared()) {
        collectCoupledSymbolsImpl(child, functionals_out, seen_functionals, aux_out, seen_aux);
    }
}

inline void collectCoupledSymbols(const forms::FormExpr& expr,
                                  std::vector<forms::BoundaryFunctional>& functionals_out,
                                  std::vector<std::string>& aux_out)
{
    functionals_out.clear();
    aux_out.clear();
    std::unordered_map<std::string, std::pair<int, std::string>> seen_functionals;
    std::unordered_set<std::string> seen_aux;
    collectCoupledSymbolsImpl(expr.nodeShared(), functionals_out, seen_functionals, aux_out, seen_aux);
}

[[nodiscard]] inline forms::FormExpr resolveCoupledSymbols(const forms::FormExpr& expr,
                                                           const std::unordered_map<std::string, std::size_t>& integral_index,
                                                           const std::unordered_map<std::string, std::size_t>& aux_index)
{
    return expr.transformNodes([&](const forms::FormExprNode& n) -> std::optional<forms::FormExpr> {
        if (n.type() == forms::FormExprType::BoundaryFunctionalSymbol ||
            n.type() == forms::FormExprType::BoundaryIntegralSymbol) {
            const auto name = n.symbolName();
            FE_THROW_IF(!name || name->empty(), InvalidArgumentException,
                        "resolveCoupledSymbols: boundaryIntegral(...) missing name");
            const std::string key(*name);

            auto it = integral_index.find(key);
            FE_THROW_IF(it == integral_index.end(), InvalidArgumentException,
                        "resolveCoupledSymbols: boundary integral '" + key + "' is not registered");
            const auto idx = it->second;
            FE_THROW_IF(idx > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()),
                        InvalidArgumentException,
                        "resolveCoupledSymbols: boundary integral slot overflow");
            return forms::FormExpr::boundaryIntegralRef(static_cast<std::uint32_t>(idx));
        }

        if (n.type() == forms::FormExprType::AuxiliaryStateSymbol) {
            const auto name = n.symbolName();
            FE_THROW_IF(!name || name->empty(), InvalidArgumentException,
                        "resolveCoupledSymbols: auxiliaryState(...) missing name");
            const std::string key(*name);

            auto it = aux_index.find(key);
            FE_THROW_IF(it == aux_index.end(), InvalidArgumentException,
                        "resolveCoupledSymbols: auxiliary state '" + key + "' is not registered");
            const auto idx = it->second;
            FE_THROW_IF(idx > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()),
                        InvalidArgumentException,
                        "resolveCoupledSymbols: auxiliary state slot overflow");
            return forms::FormExpr::auxiliaryStateRef(static_cast<std::uint32_t>(idx));
        }

        return std::nullopt;
    });
}

} // namespace detail

/**
 * @brief Apply coupled Neumann boundary conditions (loop-free helper)
 *
 * For each BC on boundary marker Γ(m):
 *   k∇u·n = g(ctx, x, t)  ⇒  adds  -∫ g v ds(m)
 *
 * Note: this overload wraps the user-provided evaluator callbacks into `FormExpr::coefficient(...)`
 * nodes, so it is compatible with the interpreter but is not "JIT-fast" (opaque call boundary).
 * Prefer the expression-aware overload below when targeting LLVM JIT kernels.
 *
 * The coupled context is updated automatically during assembly if the system
 * has a CoupledBoundaryManager enabled (this function enables it as needed).
 *
 * @param system FE system (used to register functionals/state and access context).
 * @param primary_field Field used as the discrete solution source for BoundaryFunctional evaluation.
 * @param residual Residual form (must contain TrialFunction and TestFunction).
 * @param v Test function.
 * @param bcs Coupled Neumann BCs (each may declare required boundary integrals).
 * @param aux Optional auxiliary-state registrations (ODE updates).
 * @param flux_name_prefix Prefix for auto-generated coefficient names.
 */
[[nodiscard]] inline forms::FormExpr applyCoupledNeumann(
    FESystem& system,
    FieldId primary_field,
    forms::FormExpr residual,
    const forms::FormExpr& v,
    std::span<const constraints::CoupledNeumannBC> bcs,
    std::span<const AuxiliaryStateRegistration> aux = {},
    std::string_view flux_name_prefix = "coupled_neumann")
{
    if (bcs.empty() && aux.empty()) {
        return residual;
    }

    auto& coupled = system.coupledBoundaryManager(primary_field);

    // Register auxiliary (0D) state and any functionals it depends on.
    for (const auto& reg : aux) {
        coupled.addAuxiliaryState(reg);
    }

    // Register required boundary functionals.
    for (const auto& bc : bcs) {
        coupled.addCoupledNeumannBC(bc);
    }

    const auto* ctx_ptr = coupled.contextPtr();
    FE_CHECK_NOT_NULL(ctx_ptr, "applyCoupledNeumann: coupled context");

    for (std::size_t i = 0; i < bcs.size(); ++i) {
        const auto& bc = bcs[i];
        const int marker = bc.boundaryMarker();
        FE_THROW_IF(marker < 0, InvalidArgumentException,
                    "applyCoupledNeumann: invalid boundary_marker (< 0)");

        // Capture BC by value to keep the coefficient independent of the caller's storage.
        constraints::CoupledNeumannBC bc_copy = bc;
        forms::TimeScalarCoefficient flux =
            [ctx_ptr, bc_copy](Real x, Real y, Real z, Real /*t*/) -> Real {
                return bc_copy.evaluate(*ctx_ptr, x, y, z);
            };

        const auto name = detail::makeValueName(flux_name_prefix, marker, i);
        const auto g = forms::FormExpr::coefficient(name, std::move(flux));
        residual = residual - (g * v).ds(marker);
    }

    return residual;
}

/**
 * @brief Apply a coupled Neumann boundary condition specified as a symbolic flux expression
 *
 * This overload is "expression-aware": the flux expression may contain the
 * coupled-BC placeholder terminals:
 * - `FormExpr::boundaryIntegral(integrand, marker, name)`
 * - `FormExpr::auxiliaryState(name)`
 *
 * The helper:
 * - auto-registers any referenced BoundaryFunctionals with the CoupledBoundaryManager,
 * - registers any provided auxiliary ODE state variables,
 * - resolves placeholders to standard coefficient nodes capturing a stable CoupledBCContext pointer,
 * - adds the Neumann contribution `-∫ flux * v ds(boundary_marker)` to the residual.
 */
[[nodiscard]] inline forms::FormExpr applyCoupledNeumann(
    FESystem& system,
    FieldId primary_field,
    forms::FormExpr residual,
    const forms::FormExpr& v,
    int boundary_marker,
    const forms::FormExpr& flux,
    std::span<const AuxiliaryStateRegistration> aux_states = {},
    std::string_view integral_symbol_prefix = "coupled_integral",
    std::string_view aux_symbol_prefix = "coupled_aux")
{
    (void)integral_symbol_prefix;
    (void)aux_symbol_prefix;

    FE_THROW_IF(boundary_marker < 0, InvalidArgumentException,
                "applyCoupledNeumann: invalid boundary_marker (< 0)");
    FE_THROW_IF(!flux.isValid(), InvalidArgumentException,
                "applyCoupledNeumann: invalid flux expression");
    FE_THROW_IF(flux.hasTest() || flux.hasTrial(), InvalidArgumentException,
                "applyCoupledNeumann: flux expression must not contain test/trial functions");

    auto& coupled = system.coupledBoundaryManager(primary_field);

    for (const auto& reg : aux_states) {
        coupled.addAuxiliaryState(reg);
    }

    std::vector<forms::BoundaryFunctional> functionals;
    std::vector<std::string> aux_refs;
    detail::collectCoupledSymbols(flux, functionals, aux_refs);

    std::unordered_map<std::string, std::size_t> integral_index;
    integral_index.reserve(functionals.size());
    for (auto& f : functionals) {
        integral_index.emplace(f.name, 0u);
        coupled.addBoundaryFunctional(std::move(f));
    }

    for (auto& kv : integral_index) {
        kv.second = coupled.integrals().indexOf(kv.first);
    }

    std::unordered_map<std::string, std::size_t> aux_index;
    aux_index.reserve(aux_refs.size());
    for (const auto& nm : aux_refs) {
        aux_index.emplace(nm, coupled.auxiliaryState().indexOf(nm));
    }

    const auto flux_resolved = detail::resolveCoupledSymbols(flux,
                                                             integral_index,
                                                             aux_index);

    return residual - (flux_resolved * v).ds(boundary_marker);
}

/**
 * @brief Apply coupled Robin boundary conditions (loop-free helper)
 *
 * This helper follows the same weak-form convention as `forms::bc::applyRobin`:
 * on boundary marker Γ(m), the flux-form Robin condition
 *   k∇u·n + α(ctx,x,t) u = r(ctx,x,t)
 * contributes
 *   +∫ α u v ds(m) - ∫ r v ds(m).
 *
 * Note: `constraints::CoupledRobinBC` also provides a `beta` evaluator for
 * more general mixed conditions. In this helper, `beta` is ignored; callers
 * should rewrite their BC into flux form (absorbing any scaling) before use.
 *
 * Note: this overload wraps user-provided evaluator callbacks into `FormExpr::coefficient(...)`
 * nodes, so it is compatible with the interpreter but is not "JIT-fast" (opaque call boundary).
 * Prefer the expression-aware overload below when targeting LLVM JIT kernels.
 */
[[nodiscard]] inline forms::FormExpr applyCoupledRobin(
    FESystem& system,
    FieldId primary_field,
    forms::FormExpr residual,
    const forms::FormExpr& u,
    const forms::FormExpr& v,
    std::span<const constraints::CoupledRobinBC> bcs,
    std::span<const AuxiliaryStateRegistration> aux = {},
    std::string_view alpha_name_prefix = "coupled_robin_alpha",
    std::string_view rhs_name_prefix = "coupled_robin_rhs")
{
    if (bcs.empty() && aux.empty()) {
        return residual;
    }

    auto& coupled = system.coupledBoundaryManager(primary_field);

    for (const auto& reg : aux) {
        coupled.addAuxiliaryState(reg);
    }

    for (const auto& bc : bcs) {
        coupled.addCoupledRobinBC(bc);
    }

    const auto* ctx_ptr = coupled.contextPtr();
    FE_CHECK_NOT_NULL(ctx_ptr, "applyCoupledRobin: coupled context");

    for (std::size_t i = 0; i < bcs.size(); ++i) {
        const auto& bc = bcs[i];
        const int marker = bc.boundaryMarker();
        FE_THROW_IF(marker < 0, InvalidArgumentException,
                    "applyCoupledRobin: invalid boundary_marker (< 0)");

        constraints::CoupledRobinBC bc_copy = bc;

        forms::TimeScalarCoefficient alpha =
            [ctx_ptr, bc_copy](Real x, Real y, Real z, Real /*t*/) -> Real {
                return bc_copy.alpha(*ctx_ptr, x, y, z);
            };

        forms::TimeScalarCoefficient rhs =
            [ctx_ptr, bc_copy](Real x, Real y, Real z, Real /*t*/) -> Real {
                return bc_copy.g(*ctx_ptr, x, y, z);
            };

        const auto alpha_name = detail::makeValueName(alpha_name_prefix, marker, i);
        const auto rhs_name = detail::makeValueName(rhs_name_prefix, marker, i);

        const auto a = forms::FormExpr::coefficient(alpha_name, std::move(alpha));
        const auto r = forms::FormExpr::coefficient(rhs_name, std::move(rhs));

        residual = residual + (a * u * v).ds(marker) - (r * v).ds(marker);
    }

    return residual;
}

/**
 * @brief Apply a coupled Robin boundary condition specified as symbolic alpha/rhs expressions
 *
 * This overload is expression-aware; `alpha` and `rhs` may contain
 * `boundaryIntegral(...)` and `auxiliaryState(...)` placeholders.
 *
 * Flux-form convention (same as `forms::bc::applyRobin`):
 *   k∇u·n + α u = r  ⇒  +∫ α u v ds - ∫ r v ds
 */
[[nodiscard]] inline forms::FormExpr applyCoupledRobin(
    FESystem& system,
    FieldId primary_field,
    forms::FormExpr residual,
    const forms::FormExpr& u,
    const forms::FormExpr& v,
    int boundary_marker,
    const forms::FormExpr& alpha,
    const forms::FormExpr& rhs,
    std::span<const AuxiliaryStateRegistration> aux_states = {},
    std::string_view integral_symbol_prefix = "coupled_integral",
    std::string_view aux_symbol_prefix = "coupled_aux")
{
    (void)integral_symbol_prefix;
    (void)aux_symbol_prefix;

    FE_THROW_IF(boundary_marker < 0, InvalidArgumentException,
                "applyCoupledRobin: invalid boundary_marker (< 0)");
    FE_THROW_IF(!alpha.isValid(), InvalidArgumentException,
                "applyCoupledRobin: invalid alpha expression");
    FE_THROW_IF(!rhs.isValid(), InvalidArgumentException,
                "applyCoupledRobin: invalid rhs expression");
    FE_THROW_IF(alpha.hasTest() || alpha.hasTrial(), InvalidArgumentException,
                "applyCoupledRobin: alpha expression must not contain test/trial functions");
    FE_THROW_IF(rhs.hasTest() || rhs.hasTrial(), InvalidArgumentException,
                "applyCoupledRobin: rhs expression must not contain test/trial functions");

    auto& coupled = system.coupledBoundaryManager(primary_field);

    for (const auto& reg : aux_states) {
        coupled.addAuxiliaryState(reg);
    }

    std::unordered_map<std::string, std::size_t> integral_index;
    std::unordered_map<std::string, std::size_t> aux_index;

    auto register_expr = [&](const forms::FormExpr& expr) {
        std::vector<forms::BoundaryFunctional> fn;
        std::vector<std::string> aux;
        detail::collectCoupledSymbols(expr, fn, aux);
        for (auto& f : fn) {
            integral_index.emplace(f.name, 0u);
            coupled.addBoundaryFunctional(std::move(f));
        }
        for (auto& nm : aux) {
            // Resolve index now to validate registration.
            aux_index.emplace(std::move(nm), 0u);
        }
    };

    register_expr(alpha);
    register_expr(rhs);

    for (auto& kv : integral_index) {
        kv.second = coupled.integrals().indexOf(kv.first);
    }
    for (auto& kv : aux_index) {
        kv.second = coupled.auxiliaryState().indexOf(kv.first);
    }

    const auto alpha_resolved = detail::resolveCoupledSymbols(alpha,
                                                              integral_index,
                                                              aux_index);
    const auto rhs_resolved = detail::resolveCoupledSymbols(rhs,
                                                            integral_index,
                                                            aux_index);

    return residual + (alpha_resolved * u * v).ds(boundary_marker) - (rhs_resolved * v).ds(boundary_marker);
}

} // namespace bc
} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_COUPLED_BOUNDARY_CONDITIONS_H
