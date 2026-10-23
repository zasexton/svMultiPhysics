/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

/**
 * @file IncompressibleNavierStokesVMSModule.cpp
 * @brief Incompressible Navier-Stokes (residual-based VMS) with free surfaces
 *
 * Find (u, p) on the liquid Omega (the mesh, or the cut side Omega_h of a
 * level-set geometry) such that for all (v, q)
 *
 *   int_Omega rho (dt u + (grad u) a - f) . v + 2 mu eps(u) : eps(v) - p div v + q div u
 *     + SUPG/PSPG/LSIC and cross-stress terms of the static subscales u', p'
 *     + boundary and free-surface terms                                         = 0,
 *
 * with a = u - w the velocity relative to the mesh.  The boundary and
 * free-surface terms are stated in NavierStokesBCFactories.h.  Everything
 * that is not mathematics (option checks, field registration, certificates,
 * logs, diagnostic operators, the configuration artifact, the residual-work
 * ledger) is in the support headers included below.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"

// Not mathematics: option queries, preflight, fields and initial state,
// geometry adapters, logs, energy declarations, the residual-work ledger,
// the diagnostic operators and the configuration artifact.
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceConfigurationArtifact.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceDiagnosticOperators.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceDiagnostics.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergy.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceFittedALE.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Options/NavierStokesOptionRules.h"
#include "Physics/Options/NavierStokesOptionSchema.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceResidualWork.h"
#include "Physics/Formulations/NavierStokes/NavierStokesFields.h"
#include "Physics/Formulations/NavierStokes/NavierStokesGeneratedNitsche.h"
#include "Physics/Formulations/NavierStokes/NavierStokesInitialConditions.h"
#include "Physics/Formulations/NavierStokes/NavierStokesPreflight.h"

#include "FE/Analysis/ConstitutiveLawMetadata.h"
#include "FE/Constitutive/MetadataTaggedModel.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/FormulationBuilder.h"

#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

using namespace svmp::FE::forms;
namespace fs = svmp::FE::systems;
namespace cs = svmp::FE::constraints;
using FE::FieldId;
using FE::Real;
using Options = IncompressibleNavierStokesVMSOptions;

IncompressibleNavierStokesVMSModule::IncompressibleNavierStokesVMSModule(
    std::shared_ptr<const FE::spaces::FunctionSpace> velocity_space,
    std::shared_ptr<const FE::spaces::FunctionSpace> pressure_space,
    IncompressibleNavierStokesVMSOptions options)
    : velocity_space_(std::move(velocity_space))
    , pressure_space_(std::move(pressure_space))
    , options_(std::move(options))
{
}

// ---------------------------------------------------------------------------
// Bulk residual
// ---------------------------------------------------------------------------

NavierStokesVMSIntegrands navierStokesVMSIntegrands(const NavierStokesTerms& t,
                                                    const NavierStokesVMSSettings& s)
{
    const auto& [u, p, v, q, rho, mu, f, dim] = t;
    const auto zero = zeroVector(dim);
    const auto w_mesh = meshVelocity();
    const auto w = s.ale ? w_mesh : zero;                                 // mesh velocity
    const auto a = s.convection ? (s.ale ? (u - w_mesh) : u) : zero;      // velocity relative to the mesh
    const auto chi = s.moving_control_volume ? (div(w) * u) : zero;       // moving control volume

    // Strong momentum residual without and with the pressure gradient.
    const auto R_np = rho * (dt(u) + grad(u) * a + chi - f) - div(FormExpr::constant(2.0) * mu * sym(grad(u)));
    const auto R_m = R_np + grad(p);

    const auto inertia = rho * inner(dt(u), v);
    const auto mcv = s.moving_control_volume ? rho * div(w_mesh) * inner(u, v) : FormExpr::constant(0.0);
    const auto convection = rho * inner(grad(u) * a, v);
    const auto viscous = FormExpr::constant(2.0) * mu * inner(sym(grad(u)), sym(grad(v)));
    const auto pressure = -p * div(v);
    const auto forcing = -rho * inner(f, v);

    NavierStokesVMSIntegrands r;
    r.galerkin_momentum = inertia + mcv + convection + viscous + pressure + forcing;
    r.galerkin_continuity = q * div(u);
    r.momentum = r.galerkin_momentum;
    r.continuity = r.galerkin_continuity;
    r.convection = mcv + convection;
    r.pressure = pressure;
    r.forcing = forcing;
    if (!s.vms) {
        return r;
    }

    // Static subscales with the element metric K = J^{-T} J^{-1}:
    //   tau_M = 1 / (rho sqrt(4 ct_m^2/dt^2 + a.K a + ct_c K:K nu^2)),  tau_C = 1 / (tau_M tr K).
    const auto eps = FormExpr::constant(s.epsilon);
    const auto dts = FormExpr::effectiveTimeStep();
    const auto ct_m = FormExpr::constant(s.ct_m);
    const auto ct_c = FormExpr::constant(s.ct_c);
    const auto Jinv_expr = Jinv();
    const auto K = transpose(Jinv_expr) * Jinv_expr;
    const auto nu = mu / rho;
    const auto kT = FormExpr::constant(4.0) * (ct_m * ct_m) / (dts * dts);
    const auto tau_m = FormExpr::constant(1.0) /
                       (rho * sqrt(kT + inner(a, K * a) + ct_c * doubleContraction(K, K) * (nu * nu) + eps));
    const auto tau_c = FormExpr::constant(1.0) / (tau_m * trace(K) + eps);
    const auto u_s = -tau_m * R_m;                                         // u'
    const auto p_s = -tau_c * div(u);                                      // p'
    const auto u_adv = s.convection ? (u + u_s - w) : a;

    const auto convection_adv = rho * inner(grad(u) * u_adv, v);
    const auto pressure_adv = -(p + p_s) * div(v);
    const auto supg = -rho * inner(grad(v) * u_adv, u_s);
    // Cross stress (u'.grad v) . tau_B (u'.grad) u,  tau_B = rho / sqrt(u'.K u').
    const auto cross_stress = s.convection
        ? inner(grad(v) * u_s, (rho / sqrt(inner(u_s, K * u_s) + eps)) * (grad(u) * u_s))
        : FormExpr::constant(0.0);
    r.momentum = inertia + mcv + convection_adv + viscous + pressure_adv + forcing + supg + cross_stress;

    // PSPG, split: the pressure-gradient part is its own kernel group.  The
    // unit factors keep the installed trees of the previous module.
    r.pspg_pressure_gradient = FormExpr::constant(1.0) * FormExpr::constant(1.0) * inner(grad(q), tau_m * grad(p));
    r.pspg_without_pressure_gradient = FormExpr::constant(1.0) * inner(grad(q), tau_m * R_np);
    r.continuity = r.galerkin_continuity + r.pspg_without_pressure_gradient;
    return r;
}

namespace {

// ---------------------------------------------------------------------------
// Material data
// ---------------------------------------------------------------------------

/// mu, or mu(shear rate) with shear rate = sqrt(2 eps(u):eps(u)); the model is
/// tagged so that the installer can publish the viscosity law.
[[nodiscard]] FormExpr viscosity(const Options& o, const FormExpr& u)
{
    const auto eps = sym(grad(u));
    const auto shear_rate = sqrt(FormExpr::constant(2.0) * inner(eps, eps));
    if (!o.viscosity_model) {
        return FormExpr::constant(o.viscosity);
    }
    auto model = FE::constitutive::withConstitutiveLawMetadata(
        o.viscosity_model, 0u,
        FE::analysis::dynamicViscosityMetadata(FE::INVALID_FIELD_ID, o.viscosity, o.viscosity_model));
    return constitutive(std::move(model), shear_rate).out(0);
}

/// f = g + f(x, t) + f_h - 2 Omega x u   (constant, space-time, field and Coriolis parts).
[[nodiscard]] FormExpr bodyForce(const Options& o, const FormExpr& g, int dim, FieldId f_id,
                                 const FE::spaces::FunctionSpace& V, const FormExpr& u)
{
    FormExpr f = g;
    if (o.has_body_force_spacetime) {
        f = f + FormExpr::asVector(bc::toVectorExpr(o.body_force_spacetime, dim, "ns_body_force_spacetime_"));
    }
    if (f_id != FE::INVALID_FIELD_ID) {
        f = f + StateField(f_id, V, o.body_force_field_name);
    }
    if (o.rotating_frame_coriolis_enabled) {
        f = f + FormExpr::constant(-2.0) *
                    cross(FormExpr::asVector(bc::toVectorExpr(o.rotating_frame_angular_velocity, 3,
                                                              "ns_rotating_frame_angular_velocity_")),
                          u);
    }
    return f;
}

/// The free-surface term logs of the module (FreeSurfaceDiagnostics).
class FreeSurfaceTermLog final : public Factories::FreeSurfaceObserver {
public:
    void contactLineTerms(const FreeSurfaceBoundary& b, const FreeSurfaceContactLine& line, int contact_marker,
                          int wetted_marker) override
    {
        logContactLineTerms(b, line, contact_marker, wetted_marker);
    }
    void strongContactPointLaw(const FreeSurfaceContactLine& line) override { logStrongContactPointLaw(line); }
    void unfittedMode(const FreeSurfaceBoundary& b) override { logUnfittedFreeSurfaceMode(b); }
    void naturalUnfittedSurface(const FreeSurfaceBoundary& b) override { logNaturalUnfittedFreeSurface(b); }
    void traction(const FreeSurfaceBoundary& b) override { logFreeSurfaceTraction(b); }
    void cutCellStabilization(const FreeSurfaceBoundary& b, int velocity_order, int pressure_order) override
    {
        logCutCellStabilization(b, velocity_order, pressure_order);
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void IncompressibleNavierStokesVMSModule::registerOn(fs::FESystem& system) const
{
    const auto& o = options_;
    effective_configuration_artifact_.reset();
    diagnostics_ = std::make_shared<fs::DiagnosticRegistry>();   // published through diagnostics()
    declareNavierStokesDiagnosticSets(*diagnostics_);
    auto preflight = preflightNavierStokes(system, o, velocity_space_, pressure_space_);   // no mutation before
    const int dim = preflight.dim;
    const auto& free_surfaces = preflight.effective_free_surfaces;
    const auto ids = registerNavierStokesFields(system, o, preflight);              // u, p, f_h, u_ref
    const FieldId u_id = ids.velocity;
    const FieldId p_id = ids.pressure;
    declareFreeSurfaceDiscreteFunctionals(free_surfaces, system, u_id, o.density, o.body_force,
                                          !o.has_body_force_spacetime && ids.body_force == FE::INVALID_FIELD_ID,
                                          o.viscosity, o.viscosity_model == nullptr);
    // Exterior conditions act on the wetted part of a boundary when the liquid is a cut side.
    const auto wetted = wettedBoundaryResolver(system, free_surfaces);

    // Constraints, in the plan's phase order: vertex values < essential < inactive side < aggregation
    // < contact-point law.  With a cut liquid L the traction on Gamma_h fixes the pressure constant, the dry
    // side is pinned (u = 0, p = 0) and small cut cells are aggregated.
    cs::ConstraintPlan plan(system);
    const auto liquid = activePressureDomainFor(free_surfaces);
    const bool cut_liquid = liquid.has_value() &&
                            liquid->boundary->active_domain_method == FreeSurfaceActiveDomainMethod::CutVolume;
    if (cut_liquid && isExteriorOnePhaseBoundary(*liquid->boundary)) {
        plan.anchorGauge(o.operator_tag, Factories::freeSurfacePressureAnchor(p_id, *liquid->boundary));
    }
    const bool liquid_constraints = cut_liquid && hasNativeMeshOrWarn(system);
    const bool aggregated = liquid_constraints && liquid->boundary->small_cut_aggregation;
    if (liquid_constraints) {
        const auto L =
            freeSurfaceLevelSetGeometry(*liquid->boundary, system).side(activeDomainSide(liquid->active_domain));
        plan.inactiveSide({u_id}, L).inactiveSide({p_id}, L);
        if (aggregated) {
            plan.aggregate({u_id, p_id}, L, liquid->boundary->small_cut_aggregation_guards);
        }
    }
    if (!o.node_pressure_constraints.values.empty()) {
        plan.vertexValues(p_id, nodePressureValues(o), nodePressureIdMode(o));
    }
    plan.commitThrough(cs::ConstraintPhase::VertexValues);

    const auto u = StateField(u_id, *velocity_space_, o.velocity_field_name);
    const auto p = StateField(p_id, *pressure_space_, o.pressure_field_name);
    const auto v = TestField(u_id, *velocity_space_, "v");
    const auto q = TestField(p_id, *pressure_space_, "q");
    const auto ale = fs::resolveALEBinding(system, preflight.ale_options);
    auto fitted_normal_kinematics = fittedNormalKinematicDeclarations(ale, free_surfaces, u, u_id);
    if (!system.hasOperator(o.operator_tag)) {
        system.addOperator(o.operator_tag);
    }

    // The liquid Omega: the mesh, or the cut side {phi_h < c} or {phi_h > c}; the bulk residual on it.
    const auto active_volume = activeVolumeDomainFor(free_surfaces);
    const auto Omega = active_volume ? VolumeDomain::cut(active_volume->interface_marker, active_volume->side)
                                     : VolumeDomain::full();
    const auto rho = FormExpr::constant(o.density);
    const auto g = constantVector(o.body_force, dim);
    const NavierStokesTerms t{u, p, v, q, rho, viscosity(o, u),
                              bodyForce(o, g, dim, ids.body_force, *velocity_space_, u), dim};
    const bool mcv = o.enable_ale && o.include_moving_control_volume_transient;
    const auto r = navierStokesVMSIntegrands(
        t, {.convection = o.enable_convection, .ale = o.enable_ale, .moving_control_volume = mcv,
            .vms = preflight.enable_vms,
            .ct_m = o.ct_m, .ct_c = o.ct_c, .epsilon = o.stabilization_epsilon});
    FormExpr M = r.momentum.dx(Omega);
    FormExpr C = r.continuity.dx(Omega);
    const auto pspg_grad_p = r.pspg_pressure_gradient.isValid() ? r.pspg_pressure_gradient.dx(Omega) : FormExpr{};

    // Bookkeeping of the opt-in diagnostics and of the free-surface work ledger (exact additive groups of
    // the residual), collected while the terms are stated.
    const auto active_continuity = pspg_grad_p.isValid() ? C + pspg_grad_p : C;
    const bool nonconservative_force = o.has_body_force_spacetime || ids.body_force != FE::INVALID_FIELD_ID ||
                                       o.rotating_frame_coriolis_enabled;
    auto work = bulkResidualWork(r, M, C, pspg_grad_p, -rho * inner(g, v), nonconservative_force,
                                 preflight.enable_vms, Omega);
    std::optional<ConservativeBalanceForms> balance;
    if (conservativeBalanceDiagnosticSupported(*diagnostics_, free_surfaces, preflight.has_exterior_free_surface,
                                               o.conservative_balance_diagnostics)) {
        balance = ConservativeBalanceForms{.pressure = r.pressure.dx(Omega),
                                           .gravitational_potential = (-rho * inner(g, v)).dx(Omega)};
    }
    auto* const split = balance ? &*balance : nullptr;

    // Boundary conditions and free surfaces, in the order their terms enter M and C.
    if (!o.coupled_outflow_rcr.empty() || !o.coupled_outflow_rcrcr.empty()) {
        setBoundaryReductionCompilerOptions(system, u_id, o.jit_policy);
    }
    fs::BoundaryConditionManager bcs;
    bcs.install(o.velocity_dirichlet_weak, Factories::reserveMarker);   // Nitsche below; reserve the marker
    fs::FormulationBuilder F(system, o.operator_tag, physicsInstallOptions(o.jit_policy));
    F.symbolicTangent();
    FreeSurfaceTermLog term_log;
    for (const auto& b : free_surfaces) {
        if (isExteriorOnePhaseBoundary(b) && isUnfittedLevelSet(b)) {
            Factories::addUnfittedFreeSurface(M, system, plan, F, o, b, t, u_id, split, term_log);
        } else if (isExteriorOnePhaseBoundary(b)) {
            Factories::addFittedFreeSurface(M, system, plan, bcs, o, b, ale, t, u_id, F, split, term_log);
        }
        const auto J = Factories::cutFacetPressurePenalty(b, t, o, velocity_space_->polynomial_order(),
                                                          pressure_space_->polynomial_order(), term_log);
        if (J.isValid()) {
            C = C + J;
            work.ghost_penalty = sumOf(work.ghost_penalty, J);
        }
    }
    const auto on = [&](const auto& bc) { return wetted(bc.boundary_marker); };
    bcs.install(o.traction_neumann, [&](const auto& bc) { return Factories::toTractionBC(bc, dim, on(bc)); });
    bcs.install(o.traction_robin, [&](const auto& bc) { return Factories::toTractionRobinBC(bc, dim, on(bc)); });
    bcs.install(o.pressure_outflow, [&](const auto& bc) { return Factories::toOutflowBC(bc, u, rho, on(bc)); });
    const auto rcr = [&](const auto& bc) { return Factories::toCoupledOutflowBC(bc, system, u, rho, on(bc)); };
    bcs.install(o.coupled_outflow_rcr, rcr);
    bcs.install(o.coupled_outflow_rcrcr, rcr);
    bcs.install(o.velocity_dirichlet,
                [&](const auto& bc) { return Factories::toVelocityEssentialBC(bc, dim, o.velocity_field_name); });
    plan.essentials([&](fs::FESystem& target) { bcs.applyAll(target, M, u, v, u_id); });   // weak terms into M
    fs::BoundaryConditionManager pressure_bcs;
    pressure_bcs.install(o.pressure_dirichlet,
                         [&](const auto& bc) { return Factories::toPressureEssentialBC(bc, o.pressure_field_name); });
    plan.essentials([&](fs::FESystem& target) { pressure_bcs.applyAll(target, p_id); });
    logActiveDomainConstraints(liquid_constraints ? liquid->boundary : nullptr, aggregated);
    plan.commit();

    // Weak velocity Dirichlet (Nitsche), certified on generated wetted boundaries.
    std::vector<bc::GeneratedBoundaryNitscheTraceFormBinding> nitsche_traces;
    Factories::applyVelocityNitscheBCs(M, C, o, dim, u, p, v, q, t.mu, wetted,
                                       aggregated ? &nitsche_traces : nullptr,
                                       &work.weak_boundary_momentum, &work.weak_boundary_continuity);
    requireCertifiedNitscheTraces(o, wetted, aggregated, nitsche_traces);

    // Install: the residual, the residual-work channels, then the PSPG pressure-gradient group.
    if (!o.viscosity_model) {
        F.recordDynamicViscosity(u_id, o.viscosity);
    }
    if (aggregated) {
        F.certifyGeneratedBoundaryNitscheTraces(nitsche_traces, liquid->boundary->interface_marker,
                                                o.generated_boundary_nitsche_minimum_energy_ratio);
    }
    F.configure(ale);
    (void)F.install({u_id, p_id}, M + C);
    if (preflight.has_exterior_free_surface) {
        declareFreeSurfaceExternalBoundaryEnergy(system, o);
        installFreeSurfaceResidualWork(system, o, u_id, p_id, F.options(), work,
                                       {.include_mcv = mcv,
                                        .has_nonconservative_body_force = nonconservative_force,
                                        .enable_vms = preflight.enable_vms});
    }
    if (pspg_grad_p.isValid()) {
        (void)F.installGroup({p_id}, pspg_grad_p,
                             {.source_component_tag = "navier_stokes_vms_pspg_pressure_gradient",
                              .extra_trial_fields = {u_id},
                              .drop_generated_boundary_nitsche_trace_requests = true});
    }

    installNavierStokesDiagnosticOperators(*diagnostics_, system, u_id, p_id, F.options(), r, split,
                                           active_continuity, work.ghost_penalty, u, q, active_volume);
    declareFittedNormalKinematics(system, o.operator_tag, std::move(fitted_normal_kinematics), free_surfaces);
    effective_configuration_artifact_ = makeEffectiveConfigurationArtifact(
        o, free_surfaces, system, dim, preflight.enable_vms);
}

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

void IncompressibleNavierStokesVMSModule::applyInitialConditions(const fs::FESystem& system,
                                                                 FE::backends::GenericVector& u0) const
{
    // u0 and p0 from mesh vertex fields where given; hydrostatic initialization sets
    //   p0(x) = p_ref + rho g . (x - x_ref)   on the liquid support (p_ref on dry vertices).
    const auto& init = options_.hydrostatic_pressure_initialization;
    const auto hydrostatic = [this, &init](const std::array<Real, 3>& x) {
        Real p0 = init.reference_pressure;
        for (std::size_t d = 0; d < options_.body_force.size(); ++d) {
            p0 += options_.density * options_.body_force[d] * (x[d] - init.reference_point[d]);
        }
        return p0;
    };
    initializeNavierStokesState(system, u0, options_, hydrostatic);
}

std::optional<EffectiveConfigurationArtifact>
IncompressibleNavierStokesVMSModule::effectiveConfigurationArtifact() const
{
    return effective_configuration_artifact_;
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
