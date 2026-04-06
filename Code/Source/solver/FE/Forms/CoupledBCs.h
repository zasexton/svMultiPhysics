#ifndef SVMP_FE_FORMS_COUPLED_BCS_H
#define SVMP_FE_FORMS_COUPLED_BCS_H

/**
 * @file CoupledBCs.h
 * @brief BoundaryCondition implementations for coupled (non-local) boundary conditions
 */

#include "Forms/StandardBCs.h"

#include "Auxiliary/AuxiliaryState.h"
#include "Systems/CoupledBoundaryConditions.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {
namespace forms {
namespace bc {

/**
 * @brief Coupled natural BC (Neumann/traction) with expression-aware coupled placeholders
 *
 * The flux expression may contain:
 * - FormExpr::boundaryIntegral(integrand, marker, name)
 * - FormExpr::auxiliaryState(name)
 *
 * setup() registers discovered boundary functionals and auxiliary state
 * registrations with the system's CoupledBoundaryManager and resolves
 * placeholders to slot-based terminals so the residual can be compiled.
 */
class CoupledNaturalBC final : public NaturalBC {
public:
    CoupledNaturalBC(int boundary_marker,
                     FormExpr flux,
                     std::vector<systems::AuxiliaryStateRegistration> aux_registrations = {})
        : NaturalBC(boundary_marker, std::move(flux))
        , aux_registrations_(std::move(aux_registrations))
    {
    }

    void setup(systems::FESystem& system, FieldId field_id) override
    {
        if (setup_done_) {
            return;
        }
        setup_done_ = true;

        auto& coupled = system.coupledBoundaryManager(field_id);

        for (const auto& reg : aux_registrations_) {
            coupled.addAuxiliaryState(reg);
        }

        std::vector<forms::BoundaryFunctional> functionals;
        std::vector<std::string> aux_refs;
        systems::bc::detail::collectCoupledSymbols(flux_, functionals, aux_refs);

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

        flux_ = systems::bc::detail::resolveCoupledSymbols(flux_, integral_index, aux_index);
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = boundary_marker_;
        d.domain = analysis::DomainKind::CoupledBoundary;
        d.trace_kind = analysis::TraceKind::Flux;
        d.enforcement_kind = analysis::EnforcementKind::WeakConsistent;
        d.anchors_constant_mode = false;
        d.anchors_rigid_body_translation = false;
        d.anchors_rigid_body_rotation = false;
        d.introduces_global_coupling = true;
        d.source = "CoupledNaturalBC on marker " + std::to_string(boundary_marker_);

        // Add related variables from aux registrations
        for (const auto& reg : aux_registrations_) {
            d.related_variables.push_back(
                analysis::VariableKey::named(analysis::VariableKind::AuxiliaryState, reg.spec.name));
        }

        return {d};
    }

private:
    std::vector<systems::AuxiliaryStateRegistration> aux_registrations_{};
    bool setup_done_{false};
};

/**
 * @brief Coupled Robin BC with expression-aware coupled placeholders
 *
 * Follows the same flux-form Robin convention as RobinBC:
 *   k∇u·n + alpha u = rhs  ⇒  +∫ alpha (u·v) ds - ∫ rhs·v ds.
 */
class CoupledRobinBC final : public RobinBC {
public:
    CoupledRobinBC(int boundary_marker,
                   FormExpr alpha,
                   FormExpr rhs,
                   std::vector<systems::AuxiliaryStateRegistration> aux_registrations = {})
        : RobinBC(boundary_marker, std::move(alpha), std::move(rhs))
        , aux_registrations_(std::move(aux_registrations))
    {
    }

    void setup(systems::FESystem& system, FieldId field_id) override
    {
        if (setup_done_) {
            return;
        }
        setup_done_ = true;

        auto& coupled = system.coupledBoundaryManager(field_id);

        for (const auto& reg : aux_registrations_) {
            coupled.addAuxiliaryState(reg);
        }

        std::unordered_map<std::string, std::size_t> integral_index;
        std::unordered_map<std::string, std::size_t> aux_index;

        auto register_expr = [&](const FormExpr& expr) {
            std::vector<forms::BoundaryFunctional> functionals;
            std::vector<std::string> aux_refs;
            systems::bc::detail::collectCoupledSymbols(expr, functionals, aux_refs);

            for (auto& f : functionals) {
                integral_index.emplace(f.name, 0u);
                coupled.addBoundaryFunctional(std::move(f));
            }

            for (auto& nm : aux_refs) {
                aux_index.emplace(std::move(nm), 0u);
            }
        };

        register_expr(alpha_);
        register_expr(rhs_);

        for (auto& kv : integral_index) {
            kv.second = coupled.integrals().indexOf(kv.first);
        }
        for (auto& kv : aux_index) {
            kv.second = coupled.auxiliaryState().indexOf(kv.first);
        }

        alpha_ = systems::bc::detail::resolveCoupledSymbols(alpha_, integral_index, aux_index);
        rhs_ = systems::bc::detail::resolveCoupledSymbols(rhs_, integral_index, aux_index);
    }

    [[nodiscard]] std::vector<analysis::BoundaryConditionDescriptor>
    analysisMetadata(FieldId field_id, const systems::FESystem* /*system*/) const override
    {
        analysis::BoundaryConditionDescriptor d;
        d.primary_variable = analysis::VariableKey::field(field_id);
        d.boundary_marker = boundary_marker_;
        d.domain = analysis::DomainKind::CoupledBoundary;
        d.trace_kind = analysis::TraceKind::Mixed;
        d.enforcement_kind = analysis::EnforcementKind::WeakPenalty;
        d.anchors_constant_mode = true;
        d.anchors_rigid_body_translation = true;
        d.anchors_rigid_body_rotation = false;
        d.introduces_global_coupling = true;
        d.source = "CoupledRobinBC on marker " + std::to_string(boundary_marker_);

        for (const auto& reg : aux_registrations_) {
            d.related_variables.push_back(
                analysis::VariableKey::named(analysis::VariableKind::AuxiliaryState, reg.spec.name));
        }

        return {d};
    }

private:
    std::vector<systems::AuxiliaryStateRegistration> aux_registrations_{};
    bool setup_done_{false};
};

} // namespace bc
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_COUPLED_BCS_H

