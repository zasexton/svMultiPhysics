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

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef SVMP_FE_ENABLE_LLVM_JIT
#define SVMP_FE_ENABLE_LLVM_JIT 0
#endif

namespace {
namespace forms = svmp::FE::forms;

svmp::FE::forms::SymbolicOptions makeBoundaryFunctionalCompilerOptions(bool enable_jit,
                                                                       bool enable_jit_specialization)
{
    svmp::FE::forms::SymbolicOptions options{};
#if SVMP_FE_ENABLE_LLVM_JIT
    options.jit.enable = enable_jit;
    if (options.jit.enable) {
        options.jit.optimization_level = 3;
        options.jit.specialization.enable = enable_jit_specialization;
        options.jit.specialization.specialize_n_qpts = enable_jit_specialization;
        options.jit.specialization.specialize_dofs = enable_jit_specialization;
    }
#else
    (void)enable_jit;
    (void)enable_jit_specialization;
#endif
    return options;
}

[[nodiscard]] forms::FormExpr zeroVector(int dim)
{
    std::vector<forms::FormExpr> components;
    components.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        components.push_back(forms::FormExpr::constant(0.0));
    }
    return forms::FormExpr::asVector(std::move(components));
}

[[nodiscard]] forms::FormExpr meshVelocityVector(int dim)
{
    std::vector<forms::FormExpr> components;
    components.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        components.push_back(forms::component(forms::meshVelocity(), d));
    }
    return forms::FormExpr::asVector(std::move(components));
}

[[nodiscard]] forms::FormExpr meshVelocityOrZero(int dim, bool ale_enabled)
{
    return ale_enabled ? meshVelocityVector(dim) : zeroVector(dim);
}

[[nodiscard]] forms::FormExpr relativeConvectiveVelocity(const forms::FormExpr& material_velocity,
                                                        int dim,
                                                        bool ale_enabled)
{
    return ale_enabled ? (material_velocity - meshVelocityVector(dim)) : material_velocity;
}

[[nodiscard]] forms::FormExpr movingControlVolumeVectorTransient(const forms::FormExpr& field,
                                                                const forms::FormExpr& test,
                                                                const forms::FormExpr& density,
                                                                int dim,
                                                                bool enabled)
{
    if (!enabled) {
        return forms::FormExpr::constant(0.0);
    }
    return density * forms::div(meshVelocityVector(dim)) * forms::inner(field, test);
}
} // namespace

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

    if (options_.enable_ale) {
        auto mesh_velocity_space = options_.mesh_velocity_space ? options_.mesh_velocity_space : velocity_space_;
        if (!mesh_velocity_space) {
            throw std::invalid_argument("IncompressibleNavierStokesVMSModule::registerOn: null mesh_velocity_space");
        }
        if (mesh_velocity_space->value_dimension() != dim) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule::registerOn: mesh velocity space dimension must match velocity space");
        }

        FE::FieldId mesh_velocity_id = system.findFieldByName(options_.mesh_velocity_field_name);
        if (mesh_velocity_id == FE::INVALID_FIELD_ID) {
            if (!options_.auto_register_mesh_velocity_field) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule::registerOn: ALE is enabled but mesh velocity field '" +
                    options_.mesh_velocity_field_name + "' is not registered");
            }

            FE::systems::FieldSpec w_spec;
            w_spec.name = options_.mesh_velocity_field_name;
            w_spec.space = std::move(mesh_velocity_space);
            w_spec.components = dim;
            mesh_velocity_id = system.addField(std::move(w_spec));
        }
        system.bindMeshMotionField(FE::systems::MeshMotionFieldRole::Velocity, mesh_velocity_id);
    }

    system.addOperator("equations");

    using namespace svmp::FE::forms;

    const auto u = StateField(u_id, *velocity_space_, options_.velocity_field_name);
    const auto p = StateField(p_id, *pressure_space_, options_.pressure_field_name);

    const auto v = TestField(u_id, *velocity_space_, "v");
    const auto q = TestField(p_id, *pressure_space_, "q");

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

    // ALE uses relative convection u - w_mesh. Static/default paths remain unchanged.
    const auto zero = zeroVector(dim);
    const auto mesh_velocity = meshVelocityOrZero(dim, options_.enable_ale);
    const auto a = options_.enable_convection
                       ? relativeConvectiveVelocity(u, dim, options_.enable_ale)
                       : zero;
    const bool include_mcv =
        options_.enable_ale && options_.include_moving_control_volume_transient;
    const auto moving_volume_strong =
        include_mcv ? (div(mesh_velocity) * u) : zero;

    // Strong momentum residual (full, including dt(u)):
    //   R_m = rho*(dt(u) + (u·∇)u - f) + grad(p) - div(2 mu sym(grad(u))).
    const auto stress = FormExpr::constant(2.0) * mu * sym(grad(u));
    const auto r_m = rho * (dt(u) + grad(u) * a + moving_volume_strong - f) + grad(p) - div(stress);

    // Galerkin terms.
    const auto inertia = rho * inner(dt(u), v);
    const auto moving_volume =
        movingControlVolumeVectorTransient(u, v, rho, dim, include_mcv);
    const auto convection = rho * inner(grad(u) * a, v);
    const auto viscous = FormExpr::constant(2.0) * mu * inner(sym(grad(u)), sym(grad(v)));
    const auto pressure = -p * div(v);
    const auto forcing = -rho * inner(f, v);

    FormExpr momentum_form = (inertia + moving_volume + convection + viscous + pressure + forcing).dx();
    FormExpr continuity_form = (q * div(u)).dx();

    if (options_.enable_vms) {
        // Residual-based VMS with static subscales:
        //   u' = -tau_M * R_m
        //   p' = -tau_C * (div u)
        // and coarse-scale stabilization terms assembled from (u', p').
        const auto eps = FormExpr::constant(options_.stabilization_epsilon);
        const auto dt_step = FormExpr::effectiveTimeStep();
        const auto ct_m = FormExpr::constant(options_.ct_m);
        const auto ct_c = FormExpr::constant(options_.ct_c);

        // Element metric tensor Kxi = J^{-T} J^{-1}.
        //
        // Geometry Jacobians are stored internally as 3x3 "frame" matrices so they are invertible
        // even for dim<3 mappings. Restrict all metric contractions to the physical spatial
        // dimension so the dummy thickness component does not contribute to trace(K) or (K:K).
        const auto Jinv_expr = Jinv();
        FormExpr Jinv_phys = Jinv_expr;
        if (dim == 1) {
            Jinv_phys = FormExpr::asTensor({{Jinv_expr.component(0, 0)}});
        } else if (dim == 2) {
            Jinv_phys = FormExpr::asTensor({{Jinv_expr.component(0, 0), Jinv_expr.component(0, 1)},
                                            {Jinv_expr.component(1, 0), Jinv_expr.component(1, 1)}});
        }
        const auto K = transpose(Jinv_phys) * Jinv_phys;
        const auto nu = mu / rho;

        // Legacy-inspired tau_M (stored here as tau_M/rho, matching legacy fluid.cpp naming).
        const auto kT = FormExpr::constant(4.0) * (ct_m * ct_m) / (dt_step * dt_step);
        const auto kU = inner(a, K * a);
        const auto kS = ct_c * doubleContraction(K, K) * (nu * nu);
        const auto tau_m = FormExpr::constant(1.0) / (rho * sqrt(kT + kU + kS + eps));

        const auto tau_c = FormExpr::constant(1.0) / (tau_m * trace(K) + eps);

        const auto u_sub = -tau_m * r_m;
        const auto p_sub = -tau_c * div(u);

        // Advection velocity for convection-related terms (disabled for Stokes).
        const auto u_adv = options_.enable_convection ? (u + u_sub - mesh_velocity) : a;
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
        FormExpr cross_stress = FormExpr::constant(0.0);
        if (options_.enable_convection) {
            const auto tau_b = rho / sqrt(inner(u_sub, K * u_sub) + eps);
            const auto rV_tau = tau_b * (grad(u) * u_sub); // (tauB * (u'·∇)u)
            cross_stress = inner(grad(v) * u_sub, rV_tau);
        }

        momentum_form = (inertia + moving_volume + convection_adv + viscous + pressure_adv + forcing + supg + cross_stress).dx();

        // Continuity: Galerkin + VMS (PSPG-like).
        continuity_form = (q * div(u) - inner(grad(q), u_sub)).dx();
    }

    // ---------------------------------------------------------------------
    // Boundary conditions (installer + factories)
    // ---------------------------------------------------------------------

    if (!options_.coupled_outflow_rcr.empty() || !options_.coupled_outflow_rcrcr.empty()) {
        const auto compiler_options =
            makeBoundaryFunctionalCompilerOptions(options_.enable_jit, options_.enable_jit_specialization);
        system.boundaryReductionService(u_id).setCompilerOptions(compiler_options);
    }

    FE::systems::BoundaryConditionManager bc_manager;

    // Weak velocity Dirichlet is applied directly to the Forms residual (affects both momentum and continuity).
    // Reserve the marker here so validate() catches conflicts with other BC types.
    bc_manager.install(options_.velocity_dirichlet_weak, Factories::reserveMarker);

    bc_manager.install(options_.traction_neumann, [&](const auto& bc) { return Factories::toTractionBC(bc, dim); });
    bc_manager.install(options_.traction_robin, [&](const auto& bc) { return Factories::toTractionRobinBC(bc, dim); });
    bc_manager.install(options_.pressure_outflow, [&](const auto& bc) { return Factories::toOutflowBC(bc, u, rho); });
    bc_manager.install(options_.coupled_outflow_rcr, [&](const auto& bc) {
        return Factories::toCoupledOutflowBC(bc, system, u, rho);
    });
    bc_manager.install(options_.coupled_outflow_rcrcr, [&](const auto& bc) {
        return Factories::toCoupledOutflowBC(bc, system, u, rho);
    });
    bc_manager.install(options_.velocity_dirichlet,
                       [&](const auto& bc) {
        return Factories::toVelocityEssentialBC(bc, dim, options_.velocity_field_name);
    });

    bc_manager.applyAll(system, momentum_form, u, v, u_id);

    FE::systems::BoundaryConditionManager p_bc_manager;
    p_bc_manager.install(options_.pressure_dirichlet,
                         [&](const auto& bc) { return Factories::toPressureEssentialBC(bc, options_.pressure_field_name); });
    p_bc_manager.applyAll(system, p_id);

    Factories::applyVelocityNitscheBCs(momentum_form, continuity_form, options_, *velocity_space_, dim, u, p, v, q, mu);

    // Install the complete residual (momentum + continuity) via the unified
    // installFormulation() entry point.  It auto-detects the two-field mixed
    // structure and sets up per-block Jacobian kernels with optimal assembly.
    const auto residual = momentum_form + continuity_form;

    FE::systems::FormInstallOptions install{};
    install.compiler_options.use_symbolic_tangent = true;
#if SVMP_FE_ENABLE_LLVM_JIT
    install.compiler_options.jit.enable = options_.enable_jit;
    if (install.compiler_options.jit.enable) {
        install.compiler_options.jit.optimization_level = 3;
        install.compiler_options.jit.specialization.enable = options_.enable_jit_specialization;
        install.compiler_options.jit.specialization.specialize_n_qpts = options_.enable_jit_specialization;
        install.compiler_options.jit.specialization.specialize_dofs = options_.enable_jit_specialization;
    }
#endif
    (void)FE::systems::installFormulation(system, "equations", {u_id, p_id}, residual, install);
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
