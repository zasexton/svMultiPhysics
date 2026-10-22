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
 * with a = u - w the velocity relative to the mesh.  Everything that is not
 * mathematics (option checks, field registration, certificates, logs,
 * diagnostic operators, the configuration artifact, the residual-work
 * ledger) is behind NavierStokesSupport.h.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Formulations/NavierStokes/NavierStokesSupport.h"
#include "Physics/Formulations/NavierStokes/NavierStokesVMSFormulation.h"

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

    // PSPG, split: the pressure-gradient part is its own kernel group, and with
    // the kinematic area-gradient traction it acts on dt_eff dt p (the traction
    // publishes a stationary pressure history).  The unit factors keep the
    // installed trees of the previous module.
    const auto grad_pp = grad(s.incremental_pspg_pressure ? FormExpr::effectiveTimeStep() * dt(p) : p);
    r.pspg_pressure_gradient = FormExpr::constant(1.0) * FormExpr::constant(1.0) * inner(grad(q), tau_m * grad_pp);
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

// ---------------------------------------------------------------------------
// Unfitted free surface Gamma_h = {phi_h = c}, bounding the liquid side L
// ---------------------------------------------------------------------------
//   - gamma cos(theta_e) int_CL v.m                    Young wall energy on the contact line CL = Gamma_h ^ wall,
//   + int_CL (xi u.m - gamma cos(theta_e)) v.m         or the weak Ren-E law, friction xi = 1/M,
//     or u.t_w = M gamma (cos(theta_e) - cos(theta_d))  the strong Ren-E law at the contact point (constraint);
//   + int_W (mu/l_s) u_t . v_t                         Navier slip on the wetted wall W_h;
//   + int_Gamma p_ext n.v + gamma (I - n x n) : grad v  the traction (SurfaceStress), or
//   + int_Gamma (p_ext + gamma kappa_h) n.v            (KinematicAreaGradientTraction);
//   + gamma dt int_Gamma grad_G((u - u_ref).n) . grad_G(v.n)   semi-implicit capillary term (D13).
// n points out of the liquid, n_w out of the liquid into the wall, m = conormal(n, n_w) is wall-tangential
// and points out of the wetted footprint.  u.n_w = 0 holds strongly on the walls, the dry side is pinned and
// small cuts are aggregated (registerOn), and phi is transported by the level-set equation (the kinematic
// condition).  gamma is a literal (validated); SurfaceStress already contains gamma cos(theta_d) in its
// conormal, so its wall term keeps only cos(theta_e).
void addUnfittedFreeSurface(FormExpr& M, fs::FESystem& system, cs::ConstraintPlan& plan, fs::FormulationBuilder& F,
                            const Options& o, const FreeSurfaceBoundary& b, const NavierStokesTerms& t, FieldId u_id,
                            ConservativeBalanceForms* balance)
{
    const auto& [u, p, v, q, rho, mu, f, dim] = t;
    const auto L = freeSurfaceLevelSetGeometry(b, system).side(activeDomainSide(b.active_domain));
    const auto gamma = freeSurfaceScalar(b.surface_tension, "ns_free_surface_surface_tension", b);
    const bool ss = usesSurfaceStress(b);
    const bool kag = usesKinematicAreaGradientTraction(b);   // its curvature carries the wall energy

    registerFreeSurfacePrescribedAngleGeometry(system, b);   // contact-geometry certificate at each refresh
    for (const auto& c : contactLaws(b, dim)) {
        const auto CL = F.registered(L.contactLine(c.wall, c.marker));
        if (c.dynamic) {
            registerDynamicContactWallNormalCheck(system, *c.line, CL.marker());
            if (c.strong && !ss) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: Contact_line_law_enforcement=Strong requires Surface_tension_form=SurfaceStress");
            }
        }
        // The generated-rule normal, or (CurvatureTraction) grad(phi)/|grad phi|.
        const auto n = usesGeneratedInterfaceNormal(b) ? L.outwardNormal()
                                                       : unfittedInterfaceNormal(b, levelSetStateField(b, system));
        const auto m = conormal(n, c.n_w);
        const auto wall_cos = ss ? c.cos_e : c.cos_e + dot(n, c.n_w);
        if (c.young && !kag && !bc::isZeroConstantScalarValue(b.surface_tension)) {
            const auto young = (-gamma * wall_cos * dot(v, m)).dL(CL);
            M = M + young;
            addSurfaceEnergy(ss ? balance : nullptr, young);
        }
        if (c.dynamic && !c.strong) {
            const auto friction = FormExpr::constant(1.0) / c.mobility * dot(u, m);
            M = M + ((kag ? friction : friction - gamma * wall_cos) * dot(v, m)).dL(CL);
            addSurfaceEnergy(ss ? balance : nullptr, (-gamma * c.cos_e * dot(v, m)).dL(CL));
        }
        int wetted_marker = -1;
        if (c.slip_length.isValid()) {
            const auto W = F.registered(L.wetted(c.wall));
            M = M + (mu / c.slip_length * dot(tangentialProjection(u, c.n_w), tangentialProjection(v, c.n_w))).ds(W);
            wetted_marker = W.generatedActiveBoundaryMarker();
        }
        logContactLineTerms(b, *c.line, CL.marker(), wetted_marker);
        if (c.strong) {   // PointLaw phase: registered after the aggregation, so it can expand aggregated DOFs
            plan.contactPointVelocity(u_id, L.geometry(), c.wall, c.wall_normal, c.contact_speed,
                                      "ren_e_contact_point:" + std::to_string(c.wall));
            logStrongContactPointLaw(*c.line);
        }
    }

    validateFreeSurfaceBoundary(b, o, o.enable_ale, system, dim);
    logUnfittedFreeSurfaceMode(b, system);
    if (bc::isZeroConstantScalarValue(b.external_pressure) && bc::isZeroConstantScalarValue(b.surface_tension)) {
        logNaturalUnfittedFreeSurface(b);   // sigma n = 0: no interface term
        return;
    }
    const auto Gamma = L.interface();
    const auto phi = freeSurfaceLevelSet(b, system);
    const auto n = usesGeneratedInterfaceNormal(b) ? L.outwardNormal() : unfittedInterfaceNormal(b, phi);
    const auto p_ext = freeSurfaceScalar(b.external_pressure, "ns_free_surface_external_pressure", b);
    if (ss) {
        const auto pressure = p_ext * inner(n, v);
        addExteriorPressure(balance, pressure.dI(Gamma));
        if (!surfaceStressActive(b)) {   // gamma = 0
            M = M + pressure.dI(Gamma);
        } else {
            const auto surface_energy = gamma * inner(tangentialProjector(n, dim), grad(v));
            addSurfaceEnergy(balance, surface_energy.dI(Gamma));
            M = M + (pressure + surface_energy).dI(Gamma);
        }
    } else if (kag) {
        const auto pressure = (p_ext * inner(n, v)).dI(Gamma);
        M = M + pressure;
        addExteriorPressure(balance, pressure);
        if (!bc::isZeroConstantScalarValue(b.surface_tension)) {
            const auto capillary = (gamma * unfittedTractionCurvature(b, system, phi) * inner(n, v)).dI(Gamma);
            M = M + capillary;
            addSurfaceEnergy(balance, capillary);
        }
    } else {
        M = M + unfittedCurvatureTraction(b, system, phi, p_ext, gamma, n, v).dI(Gamma);   // legacy
    }
    if (usesSemiImplicitNormalIncrement(b)) {
        const auto u_ref = freeSurfaceSemiImplicitReferenceVelocity(system);   // u at the last geometry refresh
        M = M + semiImplicitNormalIncrementIntegrand(gamma, u, u_ref, v, n).dI(Gamma);
    }
    logFreeSurfaceTraction(b);
}

// ---------------------------------------------------------------------------
// Fitted ALE free surface Gamma(t) = a boundary marker of the moving mesh
// ---------------------------------------------------------------------------
//   fluid rows: int_Gamma p_ext n.v + gamma (I - n x n) : grad v      (SurfaceStress, opt-in), or
//               - int_Gamma ((-p_ext - gamma kappa) n) . v             (CurvatureTraction, supplied kappa);
//   mesh rows:  int_Gamma (gamma_N/h_n) ((dt d - u).n)(psi.n)            (MeshNitsche, D35; or a penalty),
//               + the tangential mesh policy (smoothing, or a prescribed tangential rate);
//   a pinned contact line: d = 0 (strong).
void addFittedFreeSurface(FormExpr& M, FormExpr& C, fs::FESystem& system, cs::ConstraintPlan& plan,
                          fs::BoundaryConditionManager& bcs, const Options& o, const FreeSurfaceBoundary& b,
                          const fs::ALEBinding& ale, const NavierStokesTerms& t, const FormExpr& w, FieldId u_id,
                          fs::FormulationBuilder& F, FreeSurfaceResidualWorkForms& work,
                          ConservativeBalanceForms* balance)
{
    const auto& [u, p, v, q, rho, mu, f, dim] = t;
    applyFreeSurfaceContactLineConstraints(plan, b, ale, dim);   // pinned contact lines: d = 0
    bcs.add(std::make_unique<bc::ReservedBC>(b.boundary_marker));

    validateFreeSurfaceBoundary(b, o, o.enable_ale, system, dim);
    const auto Gamma = BoundaryDomain::fullPhysical(b.boundary_marker);
    const auto n = o.enable_ale ? currentNormal() : FormExpr::normal();   // normal of the integration frame
    const auto gamma = freeSurfaceScalar(b.surface_tension, "ns_free_surface_surface_tension", b);
    const auto p_ext = freeSurfaceScalar(b.external_pressure, "ns_free_surface_external_pressure", b);
    const bool dynamic_stress =
        !bc::isZeroConstantScalarValue(b.external_pressure) || !bc::isZeroConstantScalarValue(b.surface_tension);
    if (dynamic_stress && usesSurfaceStress(b)) {
        const auto pressure = p_ext * inner(n, v);
        addExteriorPressure(balance, pressure.ds(Gamma));
        if (!surfaceStressActive(b)) {   // gamma = 0
            M = M + pressure.ds(Gamma);
        } else {
            const auto surface_energy = gamma * inner(tangentialProjector(n, dim), grad(v));
            addSurfaceEnergy(balance, surface_energy.ds(Gamma));
            M = M + (pressure + surface_energy).ds(Gamma);
        }
    } else if (dynamic_stress) {
        const auto kappa = bc::isZeroConstantScalarValue(b.surface_tension)
                               ? FormExpr::constant(0.0)
                               : freeSurfaceScalar(b.curvature, "ns_free_surface_curvature", b);
        M = M + (FormExpr::constant(-1.0) * inner((-p_ext - gamma * kappa) * n, v)).ds(Gamma);
    }
    if (dynamic_stress) {
        logFreeSurfaceTraction(b);
    }
    addFittedFluidKinematicRow(M, C, b, u, p, v, q, w, mu, n, o.enable_ale,   // schema-1 legacy fluid rows
                               &work.weak_boundary_momentum, &work.weak_boundary_continuity);

    if (!ale.coupled()) {
        return;
    }
    const auto& rec = fittedMeshDisplacementRecord(system, ale);   // throws without a displacement unknown
    declareFittedTangentialMeshPolicy(system, ale, b);
    const bool normal_relation =
        b.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None &&
        b.normal_kinematic_policy == FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity;
    const auto psi = TestField(ale.mesh_displacement_field, *rec.space, "psi_free_surface_mesh");
    const auto d = StateField(ale.mesh_displacement_field, *rec.space, "d_mesh_free_surface");
    const auto nc = currentNormal();
    FormExpr rows;
    if (normal_relation) {   // (dt d - u).n = 0 on the mesh rows
        const auto penalty = b.kinematic_enforcement == FreeSurfaceKinematicEnforcement::Penalty
            ? freeSurfaceScalar(b.kinematic_penalty, "ns_free_surface_mesh_kinematic_penalty", b)
            : FormExpr::constant(b.kinematic_nitsche_gamma) / hNormal();
        rows = (penalty * normalTrace(dt(d) - u, nc) * normalTrace(psi, nc)).ds(Gamma);
    }
    if (const auto tangential = tangentialMeshPolicyResidual(b, d, psi, nc, rec.components, Gamma);
        tangential.isValid()) {
        rows = sumOf(rows, tangential);
    }
    if (rows.isValid()) {   // installed before the fluid residual
        fs::FormulationGroup group{.ale = &ale};
        if (normal_relation) {
            group.extra_trial_fields.push_back(u_id);
        }
        (void)F.installGroup({ale.mesh_displacement_field}, rows, group);
    }
    declareFittedMeshRowDescriptors(system, o.operator_tag, ale, b, u_id, rows.isValid());
}

// Pressure gradient-jump penalty on the facets next to the cut (Burman; Schott & Wall):
//   sum_F s gamma_p 0.01 h^3/(mu + rho h^2/dt) [grad p].[grad q]   (+ h^5 [d_nn p][d_nn q] for P2 pressure)
[[nodiscard]] FormExpr cutFacetPressurePenalty(const FreeSurfaceBoundary& b, const NavierStokesTerms& t,
                                               const Options& o, int velocity_order, int pressure_order)
{
    const auto& cut = b.cut_cell_stabilization;
    if (!isUnfittedLevelSet(b) || !cut.enabled) {
        return {};
    }
    logCutCellStabilization(b, velocity_order, pressure_order);
    if (!pressureStabilizationActive(b)) {
        return {};
    }
    const bool incremental = cut.pressure_policy == FreeSurfacePressureStabilizationPolicy::Incremental;
    const auto pp = incremental ? FormExpr::effectiveTimeStep() * dt(t.p) : t.p;
    auto s = cut.use_cut_metadata_scale ? cutStabilizationScale() : FormExpr::constant(1.0);
    if (cut.use_cut_metadata_scale && cut.cut_metadata_scale_cap.has_value()) {
        s = min(s, FormExpr::constant(*cut.cut_metadata_scale_cap));
    }
    const auto h = avg(hNormal());
    const auto h3 = h * h * h;
    const auto mu_gp = t.mu + FormExpr::constant(o.density) * h * h / FormExpr::effectiveTimeStep();
    const auto gamma_p = freeSurfaceScalar(cut.pressure_gradient_penalty, "ns_free_surface_cut_pressure_penalty", b);
    const auto c = FormExpr::constant(0.01);
    const auto F = b.active_domain == FreeSurfaceActiveDomain::None
                       ? FacetDomain::cutAdjacent(b.interface_marker)
                       : FacetDomain::cutAdjacent(b.interface_marker, activeDomainSide(b.active_domain));
    auto J = (s * gamma_p * c * h3 / (mu_gp + FormExpr::constant(o.stabilization_epsilon)) *
              inner(cutAdjacentFacetGradientJump(pp), cutAdjacentFacetGradientJump(t.q))).dS(F);
    if (!incremental && pressure_order > 1) {
        J = J + (s * gamma_p * c * (h3 * h * h) / (mu_gp + FormExpr::constant(o.stabilization_epsilon)) *
                 cutAdjacentFacetSecondNormalDerivativeJump(pp) *
                 cutAdjacentFacetSecondNormalDerivativeJump(t.q)).dS(F);
    }
    return J;
}

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
        plan.anchorGauge(o.operator_tag, freeSurfacePressureAnchor(p_id, *liquid->boundary));
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
            .incremental_pspg_pressure =
                preflight.pspg_pressure_gradient_form == PspgPressureGradientForm::Incremental,
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
    const auto w = o.enable_ale ? meshVelocity() : zeroVector(dim);
    for (const auto& b : free_surfaces) {
        if (isExteriorOnePhaseBoundary(b) && isUnfittedLevelSet(b)) {
            addUnfittedFreeSurface(M, system, plan, F, o, b, t, u_id, split);
        } else if (isExteriorOnePhaseBoundary(b)) {
            addFittedFreeSurface(M, C, system, plan, bcs, o, b, ale, t, w, u_id, F, work, split);
        }
        const auto J = cutFacetPressurePenalty(b, t, o, velocity_space_->polynomial_order(),
                                               pressure_space_->polynomial_order());
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
    appendUnfittedFreeSurfaceLevelSetTrialFields(free_surfaces, system, F.options());
    appendUnfittedFreeSurfaceCurvatureTrialFields(free_surfaces, system, F.options());
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
        o, free_surfaces, system, dim, preflight.enable_vms, preflight.pspg_pressure_gradient_form);
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
