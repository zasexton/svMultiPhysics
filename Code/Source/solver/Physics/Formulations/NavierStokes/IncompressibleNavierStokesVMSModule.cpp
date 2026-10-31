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
 * free-surface terms are stated in NavierStokesBCFactories.h. Options
 * prepares the model requirements and domain identities; shared FE services
 * execute field, initialization, constraint and installation declarations.
 * The physical laws and the Forms consuming those declarations stay here.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"

// Model requirements and the authoritative energy/work declarations.
#include "Physics/Options/NavierStokesOptionRules.h"
#include "Physics/Options/NavierStokesOptionSchema.h"
#include "Physics/Formulations/NavierStokes/NavierStokesPostProcessing.h"

#include "FE/Systems/TwoFluidStageRecords.h"

#include "FE/Analysis/ConstitutiveLawMetadata.h"
#include "FE/Constitutive/MetadataTaggedModel.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Constraints/CoupledFieldGaugeConstraint.h"
#include "FE/Constraints/GaugeRegistry.h"
#include "FE/Constraints/VertexDirichletConstraint.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Interfaces/LevelSetInterfaceDomain.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/LevelSet/LevelSetSideSupport.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FieldBinding.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/FormulationBuilder.h"
#include "FE/Systems/MeshBoundaryKinematics.h"
#include "FE/Systems/VertexFieldInitialization.h"
#include "FE/Assembly/Assembler.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Core/Logger.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Geometry/FrameGeometry.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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

IncompressibleNavierStokesVMSModule::IncompressibleNavierStokesVMSModule(
    std::shared_ptr<const FE::spaces::FunctionSpace> negative_velocity_space,
    std::shared_ptr<const FE::spaces::FunctionSpace> negative_pressure_space,
    std::shared_ptr<const FE::spaces::FunctionSpace> positive_velocity_space,
    std::shared_ptr<const FE::spaces::FunctionSpace> positive_pressure_space,
    IncompressibleTwoFluidOptions options)
    : velocity_space_(std::move(negative_velocity_space))
    , pressure_space_(std::move(negative_pressure_space))
    , positive_velocity_space_(std::move(positive_velocity_space))
    , positive_pressure_space_(std::move(positive_pressure_space))
    , two_fluid_options_(std::move(options))
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
    // (a.grad) u, plus div(a) u for the conservative transport (div(a) taken
    // term by term, as divergences of the fields).
    const auto div_a = [&] { return s.ale ? div(u) - div(w_mesh) : div(u); };
    const auto transport = [&] { return s.conservative_transport ? advect(a, u) + div_a() * u : advect(a, u); };

    // Strong momentum residual without and with the pressure gradient.
    const auto R_np = rho * (dt(u) + transport() + chi - f) - div(FormExpr::constant(2.0) * mu * symgrad(u));
    const auto R_m = R_np + grad(p);

    const auto inertia = rho * inner(dt(u), v);
    const auto mcv = s.moving_control_volume ? rho * div(w_mesh) * inner(u, v) : FormExpr::constant(0.0);
    const auto convection = rho * inner(transport(), v);
    const auto viscous = FormExpr::constant(2.0) * mu * inner(symgrad(u), symgrad(v));
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

    const auto convection_adv = s.conservative_transport ? rho * inner(advect(u_adv, u) + div_a() * u, v)
                                                         : rho * inner(advect(u_adv, u), v);
    const auto pressure_adv = -(p + p_s) * div(v);
    const auto supg = -rho * inner(advect(u_adv, v), u_s);
    // Cross stress (u'.grad v) . tau_B (u'.grad) u,  tau_B = rho / sqrt(u'.K u').
    const auto cross_stress = s.convection
        ? inner(advect(u_s, v), (rho / sqrt(inner(u_s, K * u_s) + eps)) * (advect(u_s, u)))
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
    const auto eps = symgrad(u);
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
[[nodiscard]] FormExpr bodyForce(const Options& o, const FormExpr& g, int dim,
                                 const fs::BoundPrescribed& source, const FormExpr& u)
{
    FormExpr f = g;
    if (o.has_body_force_spacetime) {
        f = f + FormExpr::asVector(bc::toVectorExpr(o.body_force_spacetime, dim, "ns_body_force_spacetime_"));
    }
    if (source.id != FE::INVALID_FIELD_ID) {
        f = f + source.value;
    }
    if (o.rotating_frame_coriolis_enabled) {
        f = f + FormExpr::constant(-2.0) *
                    cross(FormExpr::asVector(bc::toVectorExpr(o.rotating_frame_angular_velocity, 3,
                                                              "ns_rotating_frame_angular_velocity_")),
                          u);
    }
    return f;
}

// ---------------------------------------------------------------------------
// Two-fluid composition
// ---------------------------------------------------------------------------

[[nodiscard]] int resolvedInterfaceMarker(
    const IncompressibleTwoFluidOptions& options,
    FE::FieldId level_set_field)
{
    FE::interfaces::GeneratedInterfaceMarkerKey key{};
    key.source =
        FE::interfaces::LevelSetInterfaceSource::fromField(level_set_field);
    key.domain_id = options.generated_interface_domain_id;
    key.isovalue = options.level_set_isovalue;
    key.requested_marker = options.interface_marker;
    return FE::interfaces::stableGeneratedInterfaceMarker(key);
}

void addSharedGaugeEvidence(FE::systems::FESystem& system,
                            FE::FieldId negative_pressure,
                            FE::FieldId positive_pressure)
{
    constexpr std::string_view source =
        "Coupled two-fluid shared pressure gauge constraint";
    for (const auto field : {negative_pressure, positive_pressure}) {
        system.gaugeRegistry().addAnchoring(FE::gauge::AnchoringEvidence{
            .field = field,
            .component = -1,
            .region = -1,
            .family = FE::gauge::NullspaceModeFamily::ScalarConstant,
            .verdict = FE::gauge::AnchoringVerdict::Anchored,
            .source = std::string(source),
        });
    }
}

[[nodiscard]] IncompressibleNavierStokesVMSOptions makePhaseOptions(
    const IncompressibleTwoFluidOptions& owner,
    const IncompressibleTwoFluidPhaseOptions& phase,
    FreeSurfaceActiveDomain active_domain,
    int interface_marker)
{
    IncompressibleNavierStokesVMSOptions options;
    options.velocity_field_name = phase.velocity_field_name;
    options.pressure_field_name = phase.pressure_field_name;
    options.operator_tag = owner.operator_tag;
    options.density = phase.density;
    options.viscosity = phase.viscosity;
    options.body_force = owner.body_force;
    options.enable_convection = owner.enable_convection;
    options.enable_vms = owner.enable_vms;
    options.ct_m = owner.ct_m;
    options.ct_c = owner.ct_c;
    options.stabilization_epsilon = owner.stabilization_epsilon;
    options.jit_policy = owner.jit_policy;
    options.velocity_dirichlet = phase.velocity_dirichlet;
    options.velocity_dirichlet.insert(
        options.velocity_dirichlet.end(),
        owner.shared_velocity_dirichlet.begin(),
        owner.shared_velocity_dirichlet.end());

    IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary interface;
    interface.role =
        FreeSurfaceBoundaryRole::InternalMaterialInterfaceVolume;
    interface.implementation =
        FreeSurfaceImplementation::UnfittedLevelSet;
    interface.boundary_marker = -1;
    interface.interface_marker = interface_marker;
    interface.level_set_field_name = owner.level_set_field_name;
    interface.generated_interface_domain_id =
        owner.generated_interface_domain_id;
    interface.generated_interface_geometry =
        owner.generated_interface_geometry;
    interface.geometry_tangent_policy = owner.geometry_tangent_policy;
    interface.level_set_isovalue = owner.level_set_isovalue;
    interface.active_domain = active_domain;
    interface.external_pressure = FE::Real{0.0};
    interface.surface_tension = FE::Real{0.0};
    interface.normal_kinematic_policy =
        FreeSurfaceNormalKinematicPolicy::None;
    interface.tangential_mesh_policy =
        FreeSurfaceTangentialMeshPolicy::Free;
    interface.kinematic_enforcement =
        FreeSurfaceKinematicEnforcement::None;
    interface.cut_cell_stabilization.enabled = true;
    interface.cut_cell_stabilization.pressure_gradient_penalty =
        owner.pressure_gradient_penalty;
    interface.contact_lines.clear();
    interface.small_cut_aggregation = true;
    interface.small_cut_aggregation_guards =
        owner.small_cut_aggregation_guards;
    options.free_surface.push_back(std::move(interface));
    return options;
}

} // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void IncompressibleNavierStokesVMSModule::registerOn(fs::FESystem& system) const
{
    if (two_fluid_options_) {
        registerTwoFluidOn(system);
        return;
    }
    const auto& o = options_;
    effective_configuration_artifact_.reset();
    auto preflight = preflightNavierStokes(system, o, velocity_space_, pressure_space_);   // no mutation before
    const int dim = preflight.dim;
    const auto& free_surfaces = preflight.effective_free_surfaces;
    fs::FieldBinding fields(system, "IncompressibleNavierStokesVMSModule::registerOn");
    const auto U = fields.unknownSymbols(std::move(preflight.u_spec), "velocity", o.velocity_field_name, "v");
    const auto P = fields.unknownSymbols(std::move(preflight.p_spec), "pressure", o.pressure_field_name, "q");
    const auto& [u_id, u, v] = U;
    const auto& [p_id, p, q] = P;
    const auto body_force = preflight.body_force_spec
        ? fields.prescribedSymbol(std::move(*preflight.body_force_spec),
                                  o.auto_register_body_force_field, "momentum source", o.body_force_field_name)
        : fs::BoundPrescribed{};
    const FieldId reference_velocity_id = preflight.semi_implicit_reference_spec
        ? fields.prescribed(std::move(*preflight.semi_implicit_reference_spec),
                            true, "semi-implicit reference velocity")
        : FE::INVALID_FIELD_ID;
    post::declareFreeSurfaceDiscreteFunctionals(free_surfaces, preflight.surface_identities, system, u_id, o.density, o.body_force,
                                          !o.has_body_force_spacetime && body_force.id == FE::INVALID_FIELD_ID,
                                          o.viscosity, o.viscosity_model == nullptr);
    // Exterior conditions act on the wetted part of a boundary when the liquid is a cut side.
    const auto wetted = FE::level_set::wettedBoundaryResolver(system, preflight.generated_active_boundary_for);

    // Constraints, in the plan's phase order: vertex values < essential < inactive side < aggregation
    // < contact-point law.  With a cut liquid L the traction on Gamma_h fixes the pressure constant, the dry
    // side is pinned (u = 0, p = 0) and small cut cells are aggregated.
    cs::ConstraintPlan plan(system);
    const auto* liquid = preflight.active_boundary_index
        ? &free_surfaces[*preflight.active_boundary_index] : nullptr;
    const bool cut_liquid = liquid != nullptr;
    if (cut_liquid && isExteriorOnePhaseBoundary(*liquid)) {
        plan.anchorGauge(o.operator_tag, Factories::freeSurfacePressureAnchor(p_id, *liquid));
    }
    const bool liquid_constraints = cut_liquid && hasNativeMeshOrWarn(system);
    const bool aggregated = liquid_constraints && liquid->small_cut_aggregation;
    if (liquid_constraints) {
        const auto L =
            preflight.surface_identities[*preflight.active_boundary_index].geometry.value()
                .side(activeDomainSide(liquid->active_domain));
        plan.inactiveSide({u_id}, L).inactiveSide({p_id}, L);
        if (aggregated) {
            plan.aggregate({u_id, p_id}, L, liquid->small_cut_aggregation_guards);
        }
    }
    if (!o.node_pressure_constraints.values.empty()) {
        plan.vertexValues(p_id, std::move(preflight.node_pressure_values), nodePressureIdMode(o));
    }
    plan.commitThrough(cs::ConstraintPhase::VertexValues);

    const auto ale = fs::resolveALEBinding(system, preflight.ale_options);
    std::vector<fs::MeshNormalKinematicDeclaration> fitted_normal_kinematics;
    if (ale.coupled()) {
        for (const auto& prepared : preflight.fitted_normal_boundaries) {
            const auto& b = free_surfaces[prepared.boundary_index];
            const bool mesh_nitsche = b.kinematic_enforcement == FreeSurfaceKinematicEnforcement::MeshNitsche;
            fitted_normal_kinematics.push_back({
                .constraint = {
                    .mesh_displacement_field = ale.mesh_displacement_field,
                    .boundary_marker = b.boundary_marker,
                    .quantity = fs::MeshNormalBoundaryQuantity::MeshVelocityTrace,
                    .target_kind = fs::MeshNormalBoundaryTargetKind::FluidNormalVelocity,
                    .target_expression = normalTrace(u, currentNormal()),
                    .enforcement_kind = FE::analysis::EnforcementKind::WeakPenalty,
                    .related_velocity_field = u_id,
                    .owner_component = "IncompressibleNavierStokesVMSModule.FreeSurfaceBoundary",
                    .requires_mesh_flux_consistency = mesh_nitsche,
                    .mesh_flux_consistency_nitsche_gamma = mesh_nitsche ? b.kinematic_nitsche_gamma : Real{0.0}},
                .binding = prepared.binding});
        }
    }
    fs::FormulationBuilder F(system, o.operator_tag, {});

    // The liquid Omega: the mesh, or the cut side {phi_h < c} or {phi_h > c}; the bulk residual on it.
    if (preflight.active_domain_log) {
        FE::logRecord(*preflight.active_domain_log);
    }
    const auto Omega = preflight.bulk_volume;
    const auto rho = FormExpr::constant(o.density);
    const auto g = constantVector(o.body_force, dim);
    const NavierStokesTerms t{u, p, v, q, rho, viscosity(o, u),
                              bodyForce(o, g, dim, body_force, u), dim};
    const bool mcv = o.enable_ale &&
                     (o.ale_momentum_form == ALEMomentumForm::Conservative ||
                      (o.ale_momentum_form == ALEMomentumForm::Legacy && o.include_moving_control_volume_transient));
    const bool conservative = o.enable_ale && o.enable_convection &&
                              o.ale_momentum_form == ALEMomentumForm::Conservative;
    const auto r = navierStokesVMSIntegrands(
        t, {.convection = o.enable_convection, .ale = o.enable_ale, .moving_control_volume = mcv,
            .conservative_transport = conservative, .vms = preflight.enable_vms,
            .ct_m = o.ct_m, .ct_c = o.ct_c, .epsilon = o.stabilization_epsilon});
    FormExpr M = r.momentum.dx(Omega);
    FormExpr C = r.continuity.dx(Omega);
    const auto pspg_grad_p = r.pspg_pressure_gradient.isValid() ? r.pspg_pressure_gradient.dx(Omega) : FormExpr{};

    // Bookkeeping of the opt-in diagnostics and of the free-surface work ledger (exact additive groups of
    // the residual), collected while the terms are stated.
    const auto active_continuity = pspg_grad_p.isValid() ? C + pspg_grad_p : C;
    const bool nonconservative_force = o.has_body_force_spacetime || body_force.id != FE::INVALID_FIELD_ID ||
                                       o.rotating_frame_coriolis_enabled;
    auto work = post::bulkResidualWork(r, M, C, pspg_grad_p, -rho * inner(g, v), nonconservative_force,
                                 preflight.enable_vms, Omega);
    auto& diagnostics = fs::diagnosticRegistry(system);   // shared with the other formulations of the system
    post::declareDiagnosticSets(diagnostics);
    std::optional<post::ConservativeBalanceForms> balance;
    if (post::conservativeBalanceDiagnosticSupported(diagnostics, free_surfaces, preflight.has_exterior_free_surface,
                                               o.conservative_balance_diagnostics)) {
        balance = post::ConservativeBalanceForms{.pressure = r.pressure.dx(Omega),
                                           .gravitational_potential = (-rho * inner(g, v)).dx(Omega)};
    }
    auto* const split = balance ? &*balance : nullptr;

    // Boundary conditions and free surfaces, in the order their terms enter M and C.
    if (!o.coupled_outflow_rcr.empty() || !o.coupled_outflow_rcrcr.empty()) {
        setBoundaryReductionCompilerOptions(system, u_id, o.jit_policy);
    }
    fs::BoundaryConditionManager bcs;
    bcs.install(o.velocity_dirichlet_weak, Factories::reserveMarker);   // Nitsche below; reserve the marker
    F.options() = physicsInstallOptions(o.jit_policy);
    F.symbolicTangent();
    // The energies, powers and measures of the free surfaces as forms (E3; NavierStokesPostProcessing.h).
    post::declareFreeSurfaceEnergyFunctionals(F, free_surfaces, *velocity_space_, dim);
    ResidualContributions cut_penalties{{p_id, C, {&work.ghost_penalty}}};
    for (std::size_t index = 0; index < free_surfaces.size(); ++index) {
        const auto& b = free_surfaces[index];
        auto& identity = preflight.surface_identities[index];
        if (isExteriorOnePhaseBoundary(b) && isUnfittedLevelSet(b)) {
            Factories::addUnfittedFreeSurface(M, system, plan, F, o, b, identity, t, u_id,
                                            reference_velocity_id, split);
        } else if (isExteriorOnePhaseBoundary(b)) {
            Factories::addFittedFreeSurface(M, system, plan, bcs, o, b, identity, ale, t, u_id, F, split);
        }
        const auto J = Factories::cutFacetPressurePenalty(b, t, o, velocity_space_->polynomial_order(),
                                                          pressure_space_->polynomial_order(), F);
        if (J.isValid()) {
            cut_penalties.append(p_id, J);
        }
    }
    const auto on = [&](const auto& bc) { return wetted(bc.boundary_marker); };
    bcs.install(o.traction_neumann, [&](const auto& bc) { return Factories::toTractionBC(bc, dim, on(bc)); });
    bcs.install(o.traction_robin, [&](const auto& bc) { return Factories::toTractionRobinBC(bc, dim, on(bc)); });
    bcs.install(o.pressure_outflow, [&](const auto& bc) { return Factories::toOutflowBC(bc, u, rho, on(bc)); });
    const auto rcr = [&](const auto& bc) { return Factories::toCoupledOutflowBC(bc, system, u, rho, on(bc)); };
    bcs.install(o.coupled_outflow_rcr, rcr);
    bcs.install(o.coupled_outflow_rcrcr, rcr);
    Factories::declareVelocityDirichletTimeFunctions(system, o.velocity_dirichlet);
    bcs.install(o.velocity_dirichlet,
                [&](const auto& bc) { return Factories::toVelocityEssentialBC(bc, dim, o.velocity_field_name); });
    plan.essentials(bcs, M, u, v, u_id);   // weak terms into M
    fs::BoundaryConditionManager pressure_bcs;
    pressure_bcs.install(o.pressure_dirichlet,
                         [&](const auto& bc) { return Factories::toPressureEssentialBC(bc, o.pressure_field_name); });
    plan.essentials(pressure_bcs, p_id);
    F.log(Factories::activeDomainConstraintLogs(liquid_constraints ? liquid : nullptr, aggregated));
    plan.commit();

    // Weak velocity Dirichlet (Nitsche), certified on generated wetted boundaries.
    ResidualContributions weak_boundaries{{u_id, M, {&work.weak_boundary_momentum}},
                                         {p_id, C, {&work.weak_boundary_continuity}}};
    auto nitsche_traces = Factories::appendVelocityNitscheBCs(
        weak_boundaries, o, preflight.prepared_weak_velocity_boundaries, U, P, t.mu,
        aggregated ? Factories::GeneratedTraceCertification::Required : Factories::GeneratedTraceCertification::None);
    fs::requireGeneratedBoundaryNitscheTraceBindings(
        preflight.required_nitsche_routes, nitsche_traces,
        "IncompressibleNavierStokesVMSModule::registerOn: generated-boundary Nitsche form binding count differs from the certified production routes");
    for (const auto& domain : preflight.required_nitsche_routes) {
        (void)F.registered(domain);
    }

    // Install: the residual, the residual-work channels, then the PSPG pressure-gradient group.
    if (!o.viscosity_model) {
        F.recordConstitutiveLaw(FE::analysis::dynamicViscosityMetadata(u_id, o.viscosity));
    }
    if (aggregated) {
        F.certifyGeneratedBoundaryNitscheTraces(nitsche_traces, liquid->interface_marker,
                                                o.generated_boundary_nitsche_minimum_energy_ratio);
    }
    F.configure(ale);
    (void)F.install({u_id, p_id}, M + C);
    if (preflight.has_exterior_free_surface) {
        post::declareFreeSurfaceExternalBoundaryEnergy(system, o);
        post::installResidualWork(F, o, u_id, p_id, work,
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

    post::installDiagnosticOperators(F, u_id, p_id, r, split, active_continuity, work.ghost_penalty, u, q, Omega);
    for (std::size_t index = 0; index < fitted_normal_kinematics.size(); ++index) {
        auto& declaration = fitted_normal_kinematics[index];
        const auto& boundary = free_surfaces[preflight.fitted_normal_boundaries[index].boundary_index];
        declaration.binding.prescribed_tangential_mesh_velocity = fittedTangentialMeasurementVelocity(boundary);
        fs::declareMeshNormalKinematics(system, o.operator_tag, {std::move(declaration)});
    }
    effective_configuration_artifact_ = makeEffectiveConfigurationArtifact(
        o, free_surfaces, system, dim, preflight.enable_vms);
}

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

void IncompressibleNavierStokesVMSModule::applyInitialConditions(const fs::FESystem& system,
                                                                 FE::backends::GenericVector& u0) const
{
    if (two_fluid_options_) {
        applyTwoFluidInitialConditions(system, u0);
        return;
    }
    const auto& o = options_;
    const auto& init = o.hydrostatic_pressure_initialization;
    auto initialized_dofs = fs::initializeFieldFromMeshVertexField(
        system, u0, system.findFieldByName(o.velocity_field_name),
        "IncompressibleNavierStokesVMSModule: mesh-field velocity initialization");
    if (!init.enabled) {
        initialized_dofs += fs::initializeFieldFromMeshVertexField(
            system, u0, system.findFieldByName(o.pressure_field_name),
            "IncompressibleNavierStokesVMSModule: mesh-field pressure initialization");
        if (initialized_dofs > 0u) {
            FE::logRecord(post::navierStokesLog("mesh-field initialization")
                              .diagnostic("mesh_field_initialization")
                              .integer("initialized_dofs", initialized_dofs)
                              .quoted("velocity_field", o.velocity_field_name)
                              .quoted("pressure_field", o.pressure_field_name));
        }
        return;
    }
    if (initialized_dofs > 0u) {
        FE::logRecord(post::navierStokesLog("mesh-field initialization")
                          .diagnostic("mesh_field_initialization")
                          .integer("initialized_dofs", initialized_dofs)
                          .quoted("velocity_field", o.velocity_field_name));
    }

    const auto pressure = fs::requireVertexField(
        system, o.pressure_field_name,
        "IncompressibleNavierStokesVMSModule: hydrostatic pressure initialization could not find pressure field '",
        "IncompressibleNavierStokesVMSModule: hydrostatic pressure initialization requires vertex DOF metadata");
    // p0(x) = p_ref + rho g . (x - x_ref) on liquid support, p_ref elsewhere.
    // A selected mesh vertex field supplies the supported values when given.
    const auto hydrostatic = [&o, &init](const std::array<Real, 3>& x) {
        Real p0 = init.reference_pressure;
        for (std::size_t d = 0; d < o.body_force.size(); ++d) {
            p0 += o.density * o.body_force[d] * (x[d] - init.reference_point[d]);
        }
        return p0;
    };
    const auto gauges = nodePressureValues(o);
    const auto prepared = hydrostaticPressureInitialization(o, hydrostatic, gauges);
    const auto initialized = FE::level_set::initializeVertexFieldOnLevelSetSide(
        system, u0, pressure, prepared.field);
    if (prepared.active_domain) {
        const auto finite = [](Real value) {
            return std::isfinite(value) ? value : Real{0.0};
        };
        const auto& statistics = initialized.statistics;
        const auto& comparison = initialized.comparisons;
        auto record =
            post::navierStokesLog("hydrostatic pressure initialization")
                .diagnostic("hydrostatic_initialization")
                .text("active_domain", activeDomainName(prepared.active_domain->active_domain))
                .integer("wet_pressure_vertices", statistics.on_side_vertices)
                .integer("dry_pressure_vertices", statistics.off_side_vertices)
                .integer("active_support_pressure_vertices", statistics.supported_vertices)
                .integer("dry_sign_active_support_pressure_vertices", statistics.off_side_supported_vertices)
                .real("reference_pressure", init.reference_pressure)
                .real("initialized_pressure_min", finite(statistics.value_min))
                .real("initialized_pressure_max", finite(statistics.value_max))
                .real("wet_pressure_min", finite(statistics.supported_value_min))
                .real("wet_pressure_max", finite(statistics.supported_value_max))
                .integer("gauge_constraints", gauges.size())
                .integer("checked_gauge_constraints", comparison.checked)
                .integer("skipped_gauge_constraints", comparison.skipped)
                .real("gauge_pressure_min", finite(comparison.prescribed_min))
                .real("gauge_pressure_max", finite(comparison.prescribed_max))
                .real("gauge_initialized_pressure_min", finite(comparison.sampled_min))
                .real("gauge_initialized_pressure_max", finite(comparison.sampled_max))
                .real("gauge_pressure_max_abs_error", comparison.maximum_absolute_error);
        if (!init.field_name.empty()) {
            record.quoted("pressure_field", init.field_name);
        }
        FE::logRecord(record);
    }
}

// ---------------------------------------------------------------------------
// The two-fluid mode: the statement above on each side of a generated
// material interface (the negative phase, then the positive phase), coupled
// on the interface (NavierStokesBCFactories.h) and through one shared
// pressure gauge.  velocity_space_ and pressure_space_ are the negative
// phase's.  The messages keep the two-fluid module's texts.
// ---------------------------------------------------------------------------

void IncompressibleNavierStokesVMSModule::registerTwoFluidOn(fs::FESystem& system) const
{
    const auto& o = *two_fluid_options_;
    effective_configuration_artifact_.reset();
    const auto preflight = preflightIncompressibleTwoFluid(system, o, velocity_space_, pressure_space_,
                                                           positive_velocity_space_, positive_pressure_space_);
    const auto dimension = preflight.dimension;
    const auto level_set = preflight.level_set;

    const int interface_marker =
        resolvedInterfaceMarker(o, level_set);
    const Factories::IncompressibleTwoFluidInterfaceParameters interface_parameters{
        .dimension = dimension,
        .interface_marker = interface_marker,
        .negative_density = o.negative_phase.density,
        .positive_density = o.positive_phase.density,
        .negative_viscosity = o.negative_phase.viscosity,
        .positive_viscosity = o.positive_phase.viscosity,
        .nitsche_gamma = o.interface_nitsche_gamma,
        .surface_tension = o.surface_tension,
        .include_transient_penalty =
            o.include_transient_interface_penalty,
        .prescribed_pressure_jump = o.prescribed_pressure_jump,
        .prescribed_viscous_traction_jump =
            o.prescribed_viscous_traction_jump,
    };
    const auto interface_weights =
        Factories::incompressibleTwoFluidInterfaceWeights(interface_parameters);

    auto negative_options = makePhaseOptions(
        o,
        o.negative_phase,
        FreeSurfaceActiveDomain::LevelSetNegative,
        interface_marker);
    auto positive_options = makePhaseOptions(
        o,
        o.positive_phase,
        FreeSurfaceActiveDomain::LevelSetPositive,
        interface_marker);

    IncompressibleNavierStokesVMSModule negative_module(
        velocity_space_,
        pressure_space_,
        std::move(negative_options));
    IncompressibleNavierStokesVMSModule positive_module(
        positive_velocity_space_,
        positive_pressure_space_,
        std::move(positive_options));
    negative_module.registerOn(system);
    positive_module.registerOn(system);

    const auto u_negative = system.findFieldByName(
        o.negative_phase.velocity_field_name);
    const auto p_negative = system.findFieldByName(
        o.negative_phase.pressure_field_name);
    const auto u_positive = system.findFieldByName(
        o.positive_phase.velocity_field_name);
    const auto p_positive = system.findFieldByName(
        o.positive_phase.pressure_field_name);
    if (u_negative == FE::INVALID_FIELD_ID ||
        p_negative == FE::INVALID_FIELD_ID ||
        u_positive == FE::INVALID_FIELD_ID ||
        p_positive == FE::INVALID_FIELD_ID) {
        throw std::logic_error(
            "IncompressibleTwoFluidModule: phase registration did not create all four unknown fields");
    }

    using namespace FE::forms;
    const Factories::IncompressiblePhaseSymbols negative{
        .velocity = StateField(u_negative, *velocity_space_, o.negative_phase.velocity_field_name),
        .pressure = StateField(p_negative, *pressure_space_, o.negative_phase.pressure_field_name),
        .velocity_test = TestField(u_negative, *velocity_space_, "v_negative"),
        .pressure_test = TestField(p_negative, *pressure_space_, "q_negative")};
    const Factories::IncompressiblePhaseSymbols positive{
        .velocity = StateField(u_positive, *positive_velocity_space_, o.positive_phase.velocity_field_name),
        .pressure = StateField(p_positive, *positive_pressure_space_, o.positive_phase.pressure_field_name),
        .velocity_test = TestField(u_positive, *positive_velocity_space_, "v_positive"),
        .pressure_test = TestField(p_positive, *positive_pressure_space_, "q_positive")};
    const auto forms = Factories::buildIncompressibleTwoFluidInterfaceForms(
        negative, positive, interface_parameters);

    auto install = physicsInstallOptions(o.jit_policy);
    install.compiler_options.use_symbolic_tangent = true;
    install.source_component_tag =
        o.operator_tag + "_two_fluid_material_interface";
    install.recordConstitutiveLaw(FE::analysis::dynamicViscosityMetadata(
        u_negative, o.negative_phase.viscosity, {}, "negative_dynamic_viscosity"));
    install.recordConstitutiveLaw(FE::analysis::dynamicViscosityMetadata(
        u_positive, o.positive_phase.viscosity, {}, "positive_dynamic_viscosity"));
    const std::array<FE::FieldId, 4> fields{
        u_negative, p_negative, u_positive, p_positive};
    (void)FE::systems::installFormulation(
        system,
        o.operator_tag,
        std::span<const FE::FieldId>(fields),
        forms.residual,
        install);

    if (o.shared_pressure_gauge.has_value()) {
        const auto& gauge = *o.shared_pressure_gauge;
        std::vector<FE::constraints::VertexDirichletValue> values{
            FE::constraints::VertexDirichletValue{
                .vertex_id = gauge.vertex_gid,
                .value = gauge.pressure,
            }};
        system.addSystemConstraint(
            std::make_unique<FE::constraints::VertexDirichletConstraint>(
                p_negative,
                std::move(values),
                FE::constraints::VertexIdMode::GlobalVertexGid));
    } else {
        system.addSystemConstraint(
            std::make_unique<FE::constraints::CoupledFieldGaugeConstraint>(
                p_negative, p_positive));
    }
    addSharedGaugeEvidence(system, p_negative, p_positive);

    system.declareMaterialInterfaceTransportVelocity(
        FE::interfaces::MaterialInterfaceTransportVelocityDeclaration{
            .dimension = dimension,
            .interface_marker = interface_marker,
            .level_set_field = level_set,
            .negative_velocity_field = u_negative,
            .positive_velocity_field = u_positive,
            .level_set_isovalue = o.level_set_isovalue,
            .negative_trace_weight =
                interface_weights.negative_complement,
            .positive_trace_weight =
                interface_weights.positive_complement,
            .geometry_domain_id = o.generated_interface_domain_id,
            .owner_component = "incompressible_two_fluid",
        });

    svmp::FE::systems::twoFluidStageRecords(system).declareTwoFluidAcceptedStageDiagnostics(
        FE::systems::TwoFluidAcceptedStageDiagnosticDeclaration{
            .interface_marker = interface_marker,
            .level_set_field = level_set,
            .negative_velocity_field = u_negative,
            .negative_pressure_field = p_negative,
            .positive_velocity_field = u_positive,
            .positive_pressure_field = p_positive,
            .operator_tag = o.operator_tag,
            .geometry_domain_id = o.generated_interface_domain_id,
            .level_set_isovalue = o.level_set_isovalue,
            .require_conservative_phase_momentum_reconciliation =
                o
                    .require_conservative_phase_momentum_reconciliation,
            .require_accepted_stage_numerics = true,
            .pressure_stabilization_coefficient =
                o.pressure_gradient_penalty,
            .parameters =
                FE::interfaces::IncompressibleTwoFluidDiagnosticParameters{
                    .dimension = dimension,
                    .interface_marker = interface_marker,
                    .negative_density = o.negative_phase.density,
                    .positive_density = o.positive_phase.density,
                    .negative_viscosity = o.negative_phase.viscosity,
                    .positive_viscosity = o.positive_phase.viscosity,
                    .nitsche_gamma = o.interface_nitsche_gamma,
                    .surface_tension = o.surface_tension,
                    .include_transient_penalty =
                        o.include_transient_interface_penalty,
                    .prescribed_pressure_jump =
                        o.prescribed_pressure_jump,
                    .prescribed_viscous_traction_jump =
                        o.prescribed_viscous_traction_jump,
                    .body_force = o.body_force,
                },
            .owner_component = "incompressible_two_fluid",
        });

    effective_configuration_artifact_ = makeIncompressibleTwoFluidConfigurationArtifact(
        o, dimension, interface_marker, interface_weights);
}

void IncompressibleNavierStokesVMSModule::applyTwoFluidInitialConditions(const fs::FESystem& system,
                                                                         FE::backends::GenericVector& u0) const
{
    const auto& o = *two_fluid_options_;
    const auto level_set =
        system.findFieldByName(o.level_set_field_name);
    if (level_set == FE::INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "IncompressibleTwoFluidModule: initial conditions require the registered level-set field '" +
            o.level_set_field_name + "'");
    }
    const auto interface_marker =
        resolvedInterfaceMarker(o, level_set);
    auto negative_options = makePhaseOptions(
        o,
        o.negative_phase,
        FreeSurfaceActiveDomain::LevelSetNegative,
        interface_marker);
    auto positive_options = makePhaseOptions(
        o,
        o.positive_phase,
        FreeSurfaceActiveDomain::LevelSetPositive,
        interface_marker);

    IncompressibleNavierStokesVMSModule negative_module(
        velocity_space_,
        pressure_space_,
        std::move(negative_options));
    IncompressibleNavierStokesVMSModule positive_module(
        positive_velocity_space_,
        positive_pressure_space_,
        std::move(positive_options));
    negative_module.applyInitialConditions(system, u0);
    positive_module.applyInitialConditions(system, u0);
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
