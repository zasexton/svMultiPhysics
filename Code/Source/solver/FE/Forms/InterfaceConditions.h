#ifndef SVMP_FE_FORMS_INTERFACE_CONDITIONS_H
#define SVMP_FE_FORMS_INTERFACE_CONDITIONS_H

/**
 * @file InterfaceConditions.h
 * @brief Forms-level interface coupling templates (internal surfaces / InterfaceMesh)
 *
 * This header provides small helpers to assemble common interface-coupling terms
 * using the existing DG operators (`jump`, `avg`) and the interface-face measure
 * `.dI(marker)`.
 */

#include "Forms/BoundaryConditions.h"
#include "Forms/Vocabulary.h"

#include <stdexcept>

namespace svmp {
namespace FE {
namespace forms {
namespace interface {

/**
 * @brief Nitsche variant for interface conditions (symmetric vs unsymmetric)
 */
enum class NitscheVariant {
    Symmetric,
    Unsymmetric
};

struct NitscheInterfaceOptions {
    Real gamma{10.0};  ///< Penalty multiplier (scaled by k and 1/h_n and optionally p^2)
    NitscheVariant variant{NitscheVariant::Symmetric};
    bool scale_with_p{true};
};

struct ScalarContinuityPenaltyTerms {
    FormExpr first_side;
    FormExpr second_side;
};

/**
 * @brief Equal-and-opposite scalar continuity penalty terms for two traced sides
 */
[[nodiscard]] inline ScalarContinuityPenaltyTerms scalarContinuityPenaltyTerms(
    const FormExpr& first_value,
    const FormExpr& first_test,
    const FormExpr& second_value,
    const FormExpr& second_test,
    const FormExpr& penalty_weight,
    const bc::TraceNitscheOptions& opts = {})
{
    const auto jump = first_value - second_value;
    const auto penalty =
        bc::buildTraceNitschePenalty(penalty_weight, first_value, opts);
    return ScalarContinuityPenaltyTerms{
        .first_side = penalty * jump * first_test,
        .second_side = -penalty * jump * second_test,
    };
}

/**
 * @brief Legacy Nitsche/SIPG-style interface integrand for scalar diffusion operators
 *
 * Prefer `bc::applyInterfaceTraceNitsche(...)` in new formulation code so
 * interface fluxes and penalty weights remain explicit Forms expressions.
 *
 * For scalar diffusion with coefficient k, a common symmetric interface term is:
 *   -⟨ {k ∇u}·n ⟩ [[v]]  -  ⟨ {k ∇v}·n ⟩ [[u]]  +  (γ p^2 {k}/h_n) [[u]] [[v]]
 *
 * where `n` is the facet normal on the (-) side, and h_n = 2|K|/|F|.
 */
[[nodiscard]] inline FormExpr nitschePoissonIntegrand(const FormExpr& k,
                                                      const FormExpr& u,
                                                      const FormExpr& v,
                                                      const NitscheInterfaceOptions& opts = {})
{
    if (!(opts.gamma > Real(0.0))) {
        throw std::invalid_argument("forms::interface::nitschePoissonIntegrand: gamma must be > 0");
    }

    int p = 1;
    if (opts.scale_with_p) {
        p = bc::detail::polynomialOrderOrDefault(u, /*default_order=*/1);
        if (p < 1) {
            p = 1;
        }
    }

    const auto n_minus = FormExpr::normal().minus();
    const auto ju = jump(u);
    const auto jv = jump(v);
    const auto flux_u = inner(avg(k * grad(u)), n_minus);
    const auto flux_v = inner(avg(k * grad(v)), n_minus);

    // Penalty: γ p^2 {k} * avg(1/h_n).
    const auto penalty = harmonicAverage(k) * interiorPenaltyCoefficient(opts.gamma, static_cast<Real>(p));

    FormExpr out = FormExpr::constant(0.0);
    out = out - flux_u * jv;
    if (opts.variant == NitscheVariant::Symmetric) {
        out = out - flux_v * ju;
    } else {
        out = out + flux_v * ju;
    }
    out = out + penalty * ju * jv;
    return out;
}

/**
 * @brief Legacy scalar-diffusion Nitsche interface shortcut over `.dI(interface_marker)`
 */
[[nodiscard]] inline FormExpr applyNitscheInterfacePoisson(FormExpr residual,
                                                          const FormExpr& k,
                                                          const FormExpr& u,
                                                          const FormExpr& v,
                                                          int interface_marker,
                                                          const NitscheInterfaceOptions& opts = {})
{
    if (interface_marker < 0) {
        throw std::invalid_argument("forms::interface::applyNitscheInterfacePoisson: interface_marker must be >= 0");
    }

    bc::TraceNitscheOptions trace_opts;
    trace_opts.gamma = opts.gamma;
    trace_opts.variant = (opts.variant == NitscheVariant::Symmetric)
                             ? bc::NitscheVariant::Symmetric
                             : bc::NitscheVariant::Unsymmetric;
    trace_opts.scale_with_p = opts.scale_with_p;

    const auto n_minus = FormExpr::normal().minus();
    const auto consistency_flux = inner(avg(k * grad(u)), n_minus);
    const auto adjoint_flux = inner(avg(k * grad(v)), n_minus);
    const auto penalty_weight = harmonicAverage(k) * avg(FormExpr::constant(1.0) / hNormal());

    return bc::applyInterfaceTraceNitsche(std::move(residual),
                                          u,
                                          v,
                                          interface_marker,
                                          FormExpr::constant(0.0),
                                          consistency_flux,
                                          adjoint_flux,
                                          penalty_weight,
                                          bc::ScalarTraceOperator::Identity,
                                          bc::InterfaceTraceReduction::Jump,
                                          trace_opts);
}

} // namespace interface
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_INTERFACE_CONDITIONS_H
