/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/Poisson/PoissonModule.h"

#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/BoundaryFunctional.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Forms/WeakForm.h"
#include "FE/Systems/AuxiliaryStateBuilder.h"
#include "FE/Systems/CoupledBoundaryConditions.h"
#include "FE/Systems/CoupledBoundaryManager.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace svmp {
namespace Physics {
namespace formulations {
namespace poisson {

PoissonModule::PoissonModule(std::shared_ptr<const FE::spaces::FunctionSpace> space,
                             PoissonOptions options)
    : space_(std::move(space))
    , options_(std::move(options))
{
}

void PoissonModule::registerOn(FE::systems::FESystem& system) const
{
    if (!space_) {
        throw std::invalid_argument("PoissonModule::registerOn: null space");
    }

    FE::systems::FieldSpec spec;
    spec.name = options_.field_name;
    spec.space = space_;
    const FE::FieldId u_id = system.addField(std::move(spec));

    system.addOperator("residual");
    system.addOperator("jacobian");

    using namespace svmp::FE::forms;

    auto u = FormExpr::trialFunction(*space_, options_.field_name);
    auto v = FormExpr::testFunction(*space_, "v");

    const auto k = FormExpr::constant(options_.diffusion);
    const auto f = FormExpr::constant(options_.source);

    const auto integrand = k * inner(grad(u), grad(v)) - f * v;
    auto residual = integrand.dx();

    if (!options_.dirichlet.empty() && !options_.dirichlet_weak.empty()) {
        for (const auto& strong : options_.dirichlet) {
            for (const auto& weak : options_.dirichlet_weak) {
                if (strong.boundary_marker >= 0 && strong.boundary_marker == weak.boundary_marker) {
                    throw std::invalid_argument(
                        "PoissonModule::registerOn: boundary_marker " + std::to_string(strong.boundary_marker) +
                        " appears in both strong and weak Dirichlet lists");
                }
            }
        }
    }

    // Neumann: k∇u·n = g  =>  -∫ g v ds
    residual = FE::forms::bc::applyNeumannValue(
        residual,
        v,
        std::span<const PoissonOptions::NeumannBC>(options_.neumann),
        &PoissonOptions::NeumannBC::flux,
        "poisson_neumann");

    // Robin: k∇u·n + α u = r  =>  ∫ α u v ds - ∫ r v ds
    residual = FE::forms::bc::applyRobinValue(
        residual,
        u,
        v,
        std::span<const PoissonOptions::RobinBC>(options_.robin),
        &PoissonOptions::RobinBC::alpha,
        "poisson_robin_alpha",
        &PoissonOptions::RobinBC::rhs,
        "poisson_robin_rhs");

    // Coupled Neumann (RCR-style demo): uses symbolic coupled placeholders and a loop-free helper.
    if (!options_.coupled_neumann_rcr.empty()) {
        const auto u_state = FormExpr::discreteField(u_id, *space_, options_.field_name);

        for (const auto& bc : options_.coupled_neumann_rcr) {
            if (bc.boundary_marker < 0) {
                throw std::invalid_argument(
                    "PoissonModule::registerOn: CoupledRCRNeumannBC boundary_marker must be >= 0");
            }

            const std::string q_name = bc.functional_name.empty()
                                           ? ("poisson_Q_" + std::to_string(bc.boundary_marker))
                                           : bc.functional_name;
            const std::string x_name = bc.state_name.empty()
                                           ? ("poisson_X_" + std::to_string(bc.boundary_marker))
                                           : bc.state_name;

            FE::forms::BoundaryFunctional Q;
            Q.integrand = u_state;
            Q.boundary_marker = bc.boundary_marker;
            Q.name = q_name;
            Q.reduction = FE::forms::BoundaryFunctional::Reduction::Sum;

            const FE::Real Rp = bc.Rp;
            const FE::Real C = bc.C;
            const FE::Real Rd = bc.Rd;
            const FE::Real Pd = bc.Pd;

            if (Rd == 0.0) {
                throw std::invalid_argument("CoupledRCRNeumannBC: Rd must be nonzero");
            }

            const auto Qsym = FormExpr::boundaryIntegral(u_state, bc.boundary_marker, q_name);

            if (C == 0.0) {
                // Purely resistive limit: X = Pd + Rd*Q, so flux = X + Rp*Q = Pd + (Rd+Rp)*Q.
                const FE::Real Rsum = Rd + Rp;
                const auto flux = FormExpr::constant(Pd) + FormExpr::constant(Rsum) * Qsym;
                residual = FE::systems::bc::applyCoupledNeumann(
                    system,
                    u_id,
                    residual,
                    v,
                    bc.boundary_marker,
                    flux);
            } else {
                // ODE: dX/dt = (Q - (X - Pd)/Rd) / C
                const auto Xsym = FormExpr::auxiliaryState(x_name);
                const auto rhs = (Qsym - (Xsym - FormExpr::constant(Pd)) / FormExpr::constant(Rd)) / FormExpr::constant(C);
                const auto d_rhs_dX = FormExpr::constant(-1.0 / (Rd * C));

                auto reg = FE::systems::auxiliaryODE(x_name, bc.X0)
                               .requiresIntegral(Q)
                               .withRHS(rhs)
                               .withJacobian(d_rhs_dX)
                               .withIntegrator(FE::systems::ODEMethod::BackwardEuler)
                               .build();

                const auto flux = Xsym + FormExpr::constant(Rp) * Qsym;

                residual = FE::systems::bc::applyCoupledNeumann(
                    system,
                    u_id,
                    residual,
                    v,
                    bc.boundary_marker,
                    flux,
                    std::span<const FE::systems::AuxiliaryStateRegistration>(&reg, 1),
                    /*integral_symbol_prefix=*/"poisson_coupled_integral",
                    /*aux_symbol_prefix=*/"poisson_coupled_aux");
            }
        }
    }

    // Weak Dirichlet (Nitsche): u = uD  => boundary terms assembled through ds(marker).
    if (!options_.dirichlet_weak.empty()) {
        FE::forms::bc::NitscheDirichletOptions nitsche_opts;
        nitsche_opts.gamma = options_.nitsche_gamma;
        nitsche_opts.variant = options_.nitsche_symmetric ? FE::forms::bc::NitscheVariant::Symmetric
                                                          : FE::forms::bc::NitscheVariant::Unsymmetric;
        nitsche_opts.scale_with_p = options_.nitsche_scale_with_p;

        residual = FE::forms::bc::applyNitscheDirichletPoissonValue(
            residual,
            k,
            u,
            v,
            std::span<const PoissonOptions::DirichletBC>(options_.dirichlet_weak),
            &PoissonOptions::DirichletBC::value,
            "poisson_nitsche_dirichlet",
            nitsche_opts);
    }

    // Strong Dirichlet: u = uD on boundary marker.
    // Lowered by FE/Systems to system-aware constraints (no DOF boilerplate here).
    const auto dirichlet_bcs = FE::forms::bc::makeStrongDirichletListValue(
        u_id,
        std::span<const PoissonOptions::DirichletBC>(options_.dirichlet),
        &PoissonOptions::DirichletBC::value,
        "poisson_dirichlet",
        options_.field_name);

    FE::forms::WeakForm weak_form;
    weak_form.residual = residual;
    weak_form.strong_constraints = dirichlet_bcs;

    // Register the same weak form on both operator tags used by FESystem.
    // The underlying kernel can produce either vector or matrix outputs
    // depending on the AssemblyRequest (and may auto-select an optimized
    // LinearFormKernel when the residual is affine in the TrialFunction).
    FE::systems::installWeakForm(system, {"residual", "jacobian"}, u_id, u_id, weak_form);
}

} // namespace poisson
} // namespace formulations
} // namespace Physics
} // namespace svmp
