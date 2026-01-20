#ifndef SVMP_FE_FORMS_COUPLED_BCS_H
#define SVMP_FE_FORMS_COUPLED_BCS_H

/**
 * @file CoupledBCs.h
 * @brief BoundaryCondition implementations for coupled (non-local) boundary conditions
 */

#include "Forms/StandardBCs.h"

#include "Systems/AuxiliaryState.h"
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

private:
    std::vector<systems::AuxiliaryStateRegistration> aux_registrations_{};
    bool setup_done_{false};
};

} // namespace bc
} // namespace forms
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_FORMS_COUPLED_BCS_H

