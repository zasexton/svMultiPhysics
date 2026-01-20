/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"

#include "FE/Forms/Vocabulary.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

IncompressibleNavierStokesVMSModule::IncompressibleNavierStokesVMSModule(
    std::shared_ptr<const FE::spaces::FunctionSpace> velocity_space,
    std::shared_ptr<const FE::spaces::FunctionSpace> pressure_space,
    IncompressibleNavierStokesVMSOptions options)
    : velocity_space_(std::move(velocity_space))
    , pressure_space_(std::move(pressure_space))
    , options_(std::move(options))
{
}

void IncompressibleNavierStokesVMSModule::registerOn(FE::systems::FESystem& system) const
{
    if (!velocity_space_) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: null velocity_space");
    }
    if (!pressure_space_) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: null pressure_space");
    }

    const int dim = velocity_space_->value_dimension();
    if (dim < 1 || dim > 3) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: velocity space must have 1..3 components");
    }
    if (pressure_space_->value_dimension() != 1) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: pressure space must be scalar");
    }
    if (!(options_.density > 0.0)) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: density must be > 0");
    }
    if (options_.viscosity_model == nullptr && !(options_.viscosity > 0.0)) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: viscosity must be > 0 when viscosity_model is not provided");
    }
    if (options_.enable_vms && !(options_.stabilization_epsilon > 0.0)) {
        throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: stabilization_epsilon must be > 0 when VMS is enabled");
    }

    FE::systems::FieldSpec u_spec;
    u_spec.name = options_.velocity_field_name;
    u_spec.space = velocity_space_;
    u_spec.components = dim;
    const FE::FieldId u_id = system.addField(std::move(u_spec));

    FE::systems::FieldSpec p_spec;
    p_spec.name = options_.pressure_field_name;
    p_spec.space = pressure_space_;
    p_spec.components = 1;
    const FE::FieldId p_id = system.addField(std::move(p_spec));

    system.addOperator("residual");
    system.addOperator("jacobian");

    using namespace svmp::FE::forms;

    const auto u = FormExpr::stateField(u_id, *velocity_space_, options_.velocity_field_name);
    const auto p = FormExpr::stateField(p_id, *pressure_space_, options_.pressure_field_name);

    const auto v = TestFunction(*velocity_space_, "v");
    const auto q = TestFunction(*pressure_space_, "q");

    const auto rho = FormExpr::constant(options_.density);

    // Body force (constant vector).
    std::vector<FormExpr> f_comp;
    f_comp.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        f_comp.push_back(FormExpr::constant(options_.body_force[static_cast<std::size_t>(d)]));
    }
    const auto f = FormExpr::asVector(std::move(f_comp));

    // Viscosity (constant or constitutive mu(gamma)).
    FormExpr mu;
    if (options_.viscosity_model) {
        const auto eps_u = sym(grad(u));
        const auto gamma = sqrt(FormExpr::constant(2.0) * inner(eps_u, eps_u));
        mu = constitutive(options_.viscosity_model, gamma).out(0);
    } else {
        mu = FormExpr::constant(options_.viscosity);
    }

    const auto a = u; // convection velocity (no ALE/mesh motion here)

    // Strong momentum residual (full, including dt(u)):
    //   R_m = rho*(dt(u) + (u·∇)u - f) + grad(p) - div(2 mu sym(grad(u))).
    const auto stress = FormExpr::constant(2.0) * mu * sym(grad(u));
    const auto r_m = rho * (dt(u) + grad(u) * a - f) + grad(p) - div(stress);

    // Galerkin terms.
    const auto inertia = rho * inner(dt(u), v);
    const auto convection = rho * inner(grad(u) * a, v);
    const auto viscous = FormExpr::constant(2.0) * mu * inner(sym(grad(u)), sym(grad(v)));
    const auto pressure = -p * div(v);
    const auto forcing = -rho * inner(f, v);

    FormExpr momentum_form = (inertia + convection + viscous + pressure + forcing).dx();
    FormExpr continuity_form = (q * div(u)).dx();

    if (options_.enable_vms) {
        // Residual-based VMS with static subscales:
        //   u' = -tau_M * R_m
        //   p' = -tau_C * (div u)
        // and coarse-scale stabilization terms assembled from (u', p').
        const auto eps = FormExpr::constant(options_.stabilization_epsilon);
        const auto dt_step = FormExpr::timeStep();
        const auto ct_m = FormExpr::constant(options_.ct_m);
        const auto ct_c = FormExpr::constant(options_.ct_c);

        // Element metric tensor Kxi = J^{-T} J^{-1}.
        const auto Jinv_expr = Jinv();
        const auto K = transpose(Jinv_expr) * Jinv_expr;
        const auto nu = mu / rho;

        // Legacy-inspired tau_M (stored here as tau_M/rho, matching legacy fluid.cpp naming).
        const auto kT = FormExpr::constant(4.0) * (ct_m * ct_m) / (dt_step * dt_step);
        const auto kU = inner(a, K * a);
        const auto kS = ct_c * doubleContraction(K, K) * (nu * nu);
        const auto tau_m = FormExpr::constant(1.0) / (rho * sqrt(kT + kU + kS + eps));

        const auto tau_c = FormExpr::constant(1.0) / (tau_m * trace(K) + eps);

        const auto u_sub = -tau_m * r_m;
        const auto p_sub = -tau_c * div(u);

        const auto u_adv = u + u_sub;
        const auto p_adv = p + p_sub;

        // Momentum: Galerkin + VMS (SUPG-like) + pressure-subscale (LSIC-like).
        const auto convection_adv = rho * inner(grad(u) * u_adv, v);
        const auto pressure_adv = -p_adv * div(v);
        // Legacy-style full VMS: use the subscale-augmented advection velocity in the
        // test-function stabilization term and include the tauB-based cross-stress closure.
        const auto supg = -rho * inner(grad(v) * u_adv, u_sub);

        // tauB cross-stress closure (legacy fluid.cpp):
        //   tauB = rho / sqrt( u'^T Kxi u' )
        // and adds + (u' · ∇v) · ( tauB * (u' · ∇)u ).
        const auto tau_b = rho / sqrt(inner(u_sub, K * u_sub) + eps);
        const auto rV_tau = tau_b * (grad(u) * u_sub); // (tauB * (u'·∇)u)
        const auto cross_stress = inner(grad(v) * u_sub, rV_tau);

        momentum_form = (inertia + convection_adv + viscous + pressure_adv + forcing + supg + cross_stress).dx();

        // Continuity: Galerkin + VMS (PSPG-like).
        continuity_form = (q * div(u) - inner(grad(q), u_sub)).dx();
    }

    // ---------------------------------------------------------------------
    // Boundary conditions (installer + factories)
    // ---------------------------------------------------------------------

    FE::systems::BoundaryConditionManager bc_manager;

    // Weak velocity Dirichlet is applied directly to the Forms residual (affects both momentum and continuity).
    // Reserve the marker here so validate() catches conflicts with other BC types.
    bc_manager.install(options_.velocity_dirichlet_weak, Factories::reserveMarker);

    bc_manager.install(options_.traction_neumann, [&](const auto& bc) { return Factories::toTractionBC(bc, dim); });
    bc_manager.install(options_.traction_robin, [&](const auto& bc) { return Factories::toTractionRobinBC(bc, dim); });
    bc_manager.install(options_.pressure_outflow, [&](const auto& bc) { return Factories::toOutflowBC(bc, u, rho); });
    bc_manager.install(options_.coupled_outflow_rcr, [&](const auto& bc) {
        return Factories::toCoupledOutflowBC(bc, u_id, *velocity_space_, options_.velocity_field_name, u, rho);
    });
    bc_manager.install(options_.velocity_dirichlet,
                       [&](const auto& bc) { return Factories::toVelocityEssentialBC(bc, dim, options_.velocity_field_name); });

    bc_manager.validate();

    auto velocity_constraints = bc_manager.getStrongConstraints(u_id);
    bc_manager.apply(system, momentum_form, u, v, u_id);

    FE::systems::BoundaryConditionManager p_bc_manager;
    p_bc_manager.install(options_.pressure_dirichlet,
                         [&](const auto& bc) { return Factories::toPressureEssentialBC(bc, options_.pressure_field_name); });
    p_bc_manager.validate();

    auto pressure_constraints = p_bc_manager.getStrongConstraints(p_id);
    {
        FormExpr dummy;
        p_bc_manager.apply(system, dummy, p, q, p_id);
    }

    Factories::applyVelocityNitscheBCs(momentum_form,
                                       continuity_form,
                                       options_,
                                       *velocity_space_,
                                       dim,
                                       u,
                                       p,
                                       v,
                                       q,
                                       mu);

    // Strong Dirichlet constraints (installed once; independent of operator tag).
    if (!velocity_constraints.empty()) {
        FE::systems::installStrongDirichlet(system, velocity_constraints);
    }
    if (!pressure_constraints.empty()) {
        FE::systems::installStrongDirichlet(system, pressure_constraints);
    }

    // Install coupled residual on both operator tags used by FESystem convenience calls.
    FE::forms::BlockLinearForm residual(/*tests=*/2);
    residual.setBlock(0, momentum_form);
    residual.setBlock(1, continuity_form);

    const std::array<FE::FieldId, 2> fields = {u_id, p_id};
    (void)FE::systems::installCoupledResidual(system, "residual", fields, fields, residual);
    (void)FE::systems::installCoupledResidual(system, "jacobian", fields, fields, residual);
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
