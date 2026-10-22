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
 * ledger) is behind the calls into the support files included below.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Formulations/NavierStokes/NavierStokesVMSFormulation.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceConfigurationArtifact.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceDiagnosticOperators.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceDiagnostics.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergy.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceFittedALE.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceLegacyTerms.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfacePressureGauge.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceResidualWork.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceValidation.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Formulations/NavierStokes/NavierStokesFields.h"
#include "Physics/Formulations/NavierStokes/NavierStokesFormHelpers.h"
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
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <variant>
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

/// Registers the generated marker of a contact line or wetted wall and returns its domain.
ContactLineDomain registered(fs::FESystem& system, ContactLineDomain d)
{
    system.registerGeneratedEmbeddedInterfaceMarker(d.marker());
    return d;
}
BoundaryDomain registered(fs::FESystem& system, BoundaryDomain d)
{
    system.registerGeneratedEmbeddedInterfaceMarker(d.generatedActiveBoundaryMarker());
    return d;
}

/// A free-surface value as a form (a constant, or a coefficient named for the surface).
[[nodiscard]] FormExpr freeSurfaceScalar(const bc::ScalarValue& value, std::string_view name,
                                         const FreeSurfaceBoundary& b)
{
    return bc::toScalarExpr(value, freeSurfaceValueName(name, b));
}

// ---------------------------------------------------------------------------
// Unfitted free surface Gamma_h = {phi_h = c} bounding the liquid side L
// ---------------------------------------------------------------------------
//   int_CL (xi (u.m) - gamma cos(theta_e)) (v.m)       Young wall energy, Ren-E friction xi = 1/M (weak law)
//   int_W (mu/l_s) u_t . v_t                           Navier slip on the wetted wall W_h
//   int_Gamma p_ext n.v + gamma (I - n x n) : grad v    SurfaceStress (default), or
//   int_Gamma p_ext n.v + gamma kappa_h n.v             KinematicAreaGradientTraction
//   gamma dt int_Gamma grad_G((u - u_ref).n) . grad_G(v.n)   semi-implicit capillary term (D13)
// with u.n_w = 0 strong on the walls, the dry side pinned and small cuts
// aggregated (constraint plan), and phi transported by the level-set equation
// (the kinematic condition).  gamma is a literal (validated).
void addUnfittedFreeSurface(FormExpr& M, fs::FESystem& system, const Options& o, const FreeSurfaceBoundary& b,
                            const NavierStokesTerms& t, ConservativeBalanceForms* balance)
{
    const auto& [u, p, v, q, rho, mu, f, dim] = t;
    const auto L = freeSurfaceLevelSetGeometry(b, system).side(activeDomainSide(b.active_domain));
    const auto gamma = freeSurfaceScalar(b.surface_tension, "ns_free_surface_surface_tension", b);
    const bool kag = usesKinematicAreaGradientTraction(b);   // its curvature carries the wall energy

    registerFreeSurfacePrescribedAngleGeometry(system, b);   // contact-geometry certificate at each refresh
    for (const auto& c : b.contact_lines) {
        const auto kind = contactLineKind(c);
        if (kind != ContactLineKind::PrescribedAngle && kind != ContactLineKind::DynamicRenE) {
            continue;
        }
        const int wall = contactLineWallBoundaryMarker(c);
        const auto CL = registered(system, L.contactLine(wall, contactLineMarker(c)));   // Gamma_h on the wall
        if (kind == ContactLineKind::DynamicRenE) {
            registerDynamicContactWallNormalCheck(system, c, CL.marker());
            if (std::get<FreeSurfaceContactLine::DynamicRenE>(c.configuration).strong_law && !usesSurfaceStress(b)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: Contact_line_law_enforcement=Strong requires Surface_tension_form=SurfaceStress");
            }
        }
        // n outward from the liquid: the generated-rule normal, or (CurvatureTraction) grad(phi)/|grad phi|.
        const auto n = usesGeneratedInterfaceNormal(b) ? L.outwardNormal()
                                                       : unfittedInterfaceNormal(b, levelSetStateField(b, system));
        const auto n_w = wallNormalExpression(c, dim);
        const auto m = conormal(n, n_w);                       // wall-tangential, out of the wetted footprint
        const auto cos_e = FormExpr::constant(
            std::cos(constantScalarValueOrThrow(contactLineAngleRadians(c), "contact_angle_radians")));
        // SurfaceStress already contains gamma cos(theta_d) in gamma P:grad v.
        const auto wall_cos = usesSurfaceStress(b) ? cos_e : cos_e + dot(n, n_w);
        if (kind == ContactLineKind::PrescribedAngle && !kag && !bc::isZeroConstantScalarValue(b.surface_tension)) {
            const auto young = (-gamma * wall_cos * dot(v, m)).dL(CL);
            M = M + young;
            addSurfaceEnergy(usesSurfaceStress(b) ? balance : nullptr, young);
        }
        if (kind == ContactLineKind::DynamicRenE &&
            !std::get<FreeSurfaceContactLine::DynamicRenE>(c.configuration).strong_law) {
            const auto mobility = freeSurfaceScalar(contactLineMobility(c), "ns_dynamic_contact_angle_mobility", b);
            const auto friction = FormExpr::constant(1.0) / mobility * dot(u, m);   // xi (u.m), xi = 1/M
            M = M + ((kag ? friction : friction - gamma * wall_cos) * dot(v, m)).dL(CL);
            addSurfaceEnergy(usesSurfaceStress(b) ? balance : nullptr, (-gamma * cos_e * dot(v, m)).dL(CL));
        }
        int wetted_marker = -1;
        if (contactLineHasNavierSlip(c)) {
            const auto W = registered(system, L.wetted(wall));
            const auto l_s = freeSurfaceScalar(contactLineSlipLength(c), "ns_contact_wall_slip_length", b);
            M = M + (mu / l_s * dot(tangentialProjection(u, n_w), tangentialProjection(v, n_w))).ds(W);
            wetted_marker = W.generatedActiveBoundaryMarker();
        }
        logContactLineTerms(b, c, CL.marker(), wetted_marker);
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
    if (usesSurfaceStress(b)) {
        const auto pressure = p_ext * inner(n, v);
        addExteriorPressure(balance, pressure.dI(Gamma));
        if (!surfaceStressActive(b)) {
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
        M = M + semiImplicitNormalIncrementIntegrand(gamma, u, freeSurfaceSemiImplicitReferenceVelocity(system), v, n)
                    .dI(Gamma);
    }
    logFreeSurfaceTraction(b);
}

// ---------------------------------------------------------------------------
// Fitted ALE free surface Gamma(t) = boundary marker of the moving mesh
// ---------------------------------------------------------------------------
//   fluid rows: int_Gamma p_ext n.v + gamma (I - n x n) : grad v  (SurfaceStress, opt-in)
//               or  -int_Gamma ((-p_ext - gamma kappa) n) . v      (CurvatureTraction, supplied kappa),
//   mesh rows:  int_Gamma (gamma_N/h_n) ((dt d - u).n)(psi.n)        (MeshNitsche, D35)
//               + the tangential mesh policy (smoothing, or a prescribed tangential rate),
//   a pinned contact line: d = 0 (strong).
void addFittedFreeSurface(FormExpr& M, FormExpr& C, fs::FESystem& system, cs::ConstraintPlan& plan,
                          fs::BoundaryConditionManager& bcs, const Options& o, const FreeSurfaceBoundary& b,
                          const fs::ALEBinding& ale, const NavierStokesTerms& t, const FormExpr& w, FieldId u_id,
                          const fs::FormInstallOptions& base_install, FreeSurfaceResidualWorkForms& work,
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
        if (!surfaceStressActive(b)) {
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
    const auto d_mesh = StateField(ale.mesh_displacement_field, *rec.space, "d_mesh_free_surface");
    const auto nc = currentNormal();
    FormExpr residual;
    if (normal_relation) {   // (dt d - u).n = 0, weakly on the mesh rows
        const auto penalty = b.kinematic_enforcement == FreeSurfaceKinematicEnforcement::Penalty
            ? freeSurfaceScalar(b.kinematic_penalty, "ns_free_surface_mesh_kinematic_penalty", b)
            : FormExpr::constant(b.kinematic_nitsche_gamma) / hNormal();
        residual = (penalty * normalTrace(dt(d_mesh) - u, nc) * normalTrace(psi, nc)).ds(Gamma);
    }
    if (b.tangential_mesh_policy == FreeSurfaceTangentialMeshPolicy::SmoothingOnly) {   // int |P grad d P|^2
        const auto P = tangentialProjector(nc, rec.components);
        const auto weight =
            freeSurfaceScalar(b.tangential_mesh_penalty, "ns_free_surface_tangential_mesh_smoothing_weight", b);
        residual = sumOf(residual, (weight * inner(P * grad(d_mesh) * P, P * grad(psi) * P)).ds(Gamma));
    }
    if (b.tangential_mesh_policy == FreeSurfaceTangentialMeshPolicy::Prescribed) {   // (I - n x n)(dt d - w_t) = 0
        const auto target = FormExpr::asVector(bc::toVectorExpr(
            b.prescribed_tangential_mesh_velocity, rec.components, "ns_free_surface_tangential_mesh_velocity",
            b.boundary_marker, bc::ComponentValueNameStyle::Component));
        const auto gap = dt(d_mesh) - target;
        const auto penalty = freeSurfaceScalar(b.tangential_mesh_penalty, "ns_free_surface_tangential_mesh_penalty", b);
        residual = sumOf(residual,
                         (penalty * (inner(gap, psi) - normalTrace(gap, nc) * normalTrace(psi, nc))).ds(Gamma));
    }
    if (residual.isValid()) {   // installed before the fluid residual
        auto install = base_install;
        install.compiler_options.use_symbolic_tangent = true;
        ale.configureInstallOptions(install);
        if (normal_relation) {
            install.extra_trial_fields.push_back(u_id);
        }
        (void)fs::installFormulation(system, o.operator_tag, {ale.mesh_displacement_field}, residual, install);
    }
    declareFittedMeshRowDescriptors(system, o.operator_tag, ale, b, u_id, residual.isValid());
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

/// Ren-E contact-point law u.t_w = M gamma (cos(theta_e) - cos(theta_d)), imposed strongly.
[[nodiscard]] cs::ContactPointVelocityLaw renEContactPointLaw(const FreeSurfaceBoundary& b,
                                                              const FreeSurfaceContactLine& c)
{
    const auto wall_normal = normalizedWallNormal(c);
    const Real mobility = constantScalarValueOrThrow(contactLineMobility(c), "Strong DynamicContactAngle mobility");
    const Real gamma = constantScalarValueOrThrow(b.surface_tension, "Strong DynamicContactAngle surface tension");
    const Real cos_e = std::cos(constantScalarValueOrThrow(contactLineAngleRadians(c),
                                                           "Strong DynamicContactAngle equilibrium angle"));
    cs::ContactPointVelocityLaw law;
    law.wall_boundary_markers = {contactLineWallBoundaryMarker(c)};
    law.wall_normal = {wall_normal[0], wall_normal[1], wall_normal[2]};
    law.level_set_field_name = b.level_set_field_name;
    law.isovalue = b.level_set_isovalue;
    law.contact_speed = [mobility, gamma, cos_e](Real cos_d) { return mobility * gamma * (cos_e - cos_d); };
    law.name = "ren_e_contact_point:" + std::to_string(contactLineWallBoundaryMarker(c));
    return law;
}

} // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void IncompressibleNavierStokesVMSModule::registerOn(fs::FESystem& system) const
{
    const auto& o = options_;
    effective_configuration_artifact_.reset();
    auto preflight = preflightNavierStokes(system, o, velocity_space_, pressure_space_);   // no mutation before
    const int dim = preflight.dim;
    const auto& free_surfaces = preflight.effective_free_surfaces;
    const auto ids = registerNavierStokesFields(system, o, preflight);              // u, p, f_h, u_ref
    const FieldId u_id = ids.velocity;
    const FieldId p_id = ids.pressure;
    declareFreeSurfaceDiscreteFunctionals(free_surfaces, system, u_id, o.density, o.body_force,
                                          !o.has_body_force_spacetime && ids.body_force == FE::INVALID_FIELD_ID,
                                          o.viscosity, o.viscosity_model == nullptr);

    // Exterior conditions act on the wetted part of the boundary when the liquid is a cut side.
    const auto wetted = [&system, &free_surfaces](int marker) -> std::optional<int> {
        const auto generated = sharpActiveBoundaryMarkerFor(marker, free_surfaces, system);
        if (generated.has_value()) {
            system.registerGeneratedEmbeddedInterfaceMarker(*generated);
        }
        return generated;
    };

    // Constraints register in the plan's phase order:
    //   vertex values < essential < inactive side < aggregation < contact-point law.
    cs::ConstraintPlan plan(system);
    const auto liquid = activePressureDomainFor(free_surfaces);   // the active-domain free surface
    const bool cut_liquid = liquid.has_value() &&
                            liquid->boundary->active_domain_method == FreeSurfaceActiveDomainMethod::CutVolume;
    if (cut_liquid && isExteriorOnePhaseBoundary(*liquid->boundary)) {
        anchorPressureOnFreeSurface(system, o.operator_tag, p_id, *liquid->boundary);   // traction on Gamma_h fixes p
    }
    const bool liquid_constraints = cut_liquid && hasNativeMeshOrWarn(system);
    const bool aggregated = liquid_constraints && liquid->boundary->small_cut_aggregation;
    if (liquid_constraints) {
        const auto L =
            freeSurfaceLevelSetGeometry(*liquid->boundary, system).side(activeDomainSide(liquid->active_domain));
        plan.inactiveSide({u_id}, L);                                       // dry side: u = 0
        plan.inactiveSide({p_id}, L);                                       //           p = 0
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

    // The liquid: the mesh, or the cut side Omega_h = {phi_h < c} or {phi_h > c}.
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
    const auto active_continuity = pspg_grad_p.isValid() ? C + pspg_grad_p : C;   // pressure-row diagnostic
    const bool nonconservative_force = o.has_body_force_spacetime || ids.body_force != FE::INVALID_FIELD_ID ||
                                       o.rotating_frame_coriolis_enabled;
    // The exact additive groups of the residual that the free-surface work ledger pairs with
    // the state, and the conservative split of the opt-in balance diagnostic.
    auto work = bulkResidualWork(r, M, C, pspg_grad_p, -rho * inner(g, v), nonconservative_force,
                                 preflight.enable_vms, Omega);
    std::optional<ConservativeBalanceForms> balance;
    if (conservativeBalanceDiagnosticSupported(free_surfaces, preflight.has_exterior_free_surface)) {
        balance = ConservativeBalanceForms{.pressure = r.pressure.dx(Omega),
                                           .gravitational_potential = (-rho * inner(g, v)).dx(Omega)};
    }

    // Boundary conditions and free surfaces, in the order the terms enter M and C.
    if (!o.coupled_outflow_rcr.empty() || !o.coupled_outflow_rcrcr.empty()) {
        setBoundaryReductionCompilerOptions(system, u_id, o.jit_policy);
    }
    fs::BoundaryConditionManager bcs;
    bcs.install(o.velocity_dirichlet_weak, Factories::reserveMarker);   // Nitsche below; reserve the marker
    const auto mesh_rows_install = physicsInstallOptions(o.jit_policy);
    const auto w = o.enable_ale ? meshVelocity() : zeroVector(dim);
    for (const auto& b : free_surfaces) {
        if (isExteriorOnePhaseBoundary(b) && isUnfittedLevelSet(b)) {
            addUnfittedFreeSurface(M, system, o, b, t, balance ? &*balance : nullptr);
        } else if (isExteriorOnePhaseBoundary(b)) {
            addFittedFreeSurface(M, C, system, plan, bcs, o, b, ale, t, w, u_id, mesh_rows_install, work,
                                 balance ? &*balance : nullptr);
        }
        const auto J = cutFacetPressurePenalty(b, t, o, velocity_space_->polynomial_order(),
                                               pressure_space_->polynomial_order());
        if (J.isValid()) {
            C = C + J;
            work.ghost_penalty = sumOf(work.ghost_penalty, J);
        }
    }
    namespace F = Factories;
    const auto on = [&](const auto& bc) { return wetted(bc.boundary_marker); };
    bcs.install(o.traction_neumann, [&](const auto& bc) { return F::toTractionBC(bc, dim, on(bc)); });
    bcs.install(o.traction_robin, [&](const auto& bc) { return F::toTractionRobinBC(bc, dim, on(bc)); });
    bcs.install(o.pressure_outflow, [&](const auto& bc) { return F::toOutflowBC(bc, u, rho, on(bc)); });
    const auto rcr = [&](const auto& bc) { return F::toCoupledOutflowBC(bc, system, u, rho, on(bc)); };
    bcs.install(o.coupled_outflow_rcr, rcr);
    bcs.install(o.coupled_outflow_rcrcr, rcr);
    bcs.install(o.velocity_dirichlet,
                [&](const auto& bc) { return F::toVelocityEssentialBC(bc, dim, o.velocity_field_name); });
    plan.essentials([&](fs::FESystem& target) { bcs.applyAll(target, M, u, v, u_id); });   // weak terms into M
    fs::BoundaryConditionManager pressure_bcs;
    pressure_bcs.install(o.pressure_dirichlet,
                         [&](const auto& bc) { return F::toPressureEssentialBC(bc, o.pressure_field_name); });
    plan.essentials([&](fs::FESystem& target) { pressure_bcs.applyAll(target, p_id); });
    logActiveDomainConstraints(liquid_constraints ? liquid->boundary : nullptr, aggregated);
    for (const auto& b : o.free_surface) {
        for (const auto& c : b.contact_lines) {
            const auto* dynamic = std::get_if<FreeSurfaceContactLine::DynamicRenE>(&c.configuration);
            if (dynamic != nullptr && dynamic->strong_law) {
                plan.contactPointVelocity(u_id, renEContactPointLaw(b, c));   // expands aggregated DOFs: last
                logStrongContactPointLaw(c);
            }
        }
    }
    plan.commit();

    // Weak velocity Dirichlet (Nitsche), certified on generated wetted boundaries.
    std::vector<bc::GeneratedBoundaryNitscheTraceFormBinding> nitsche_traces;
    F::applyVelocityNitscheBCs(M, C, o, dim, u, p, v, q, t.mu, wetted, aggregated ? &nitsche_traces : nullptr,
                               &work.weak_boundary_momentum, &work.weak_boundary_continuity);
    requireCertifiedNitscheTraces(o, wetted, aggregated, nitsche_traces);

    // Install: the residual, the residual-work channels, then the PSPG pressure-gradient group.
    auto install = physicsInstallOptions(o.jit_policy);
    install.compiler_options.use_symbolic_tangent = true;
    if (!o.viscosity_model) {
        install.recordDynamicViscosity(u_id, o.viscosity);
    }
    if (aggregated) {
        addNitscheTraceRequests(install, nitsche_traces, liquid->boundary->interface_marker,
                                o.generated_boundary_nitsche_minimum_energy_ratio);
    }
    appendUnfittedFreeSurfaceLevelSetTrialFields(free_surfaces, system, install);
    appendUnfittedFreeSurfaceCurvatureTrialFields(free_surfaces, system, install);
    ale.configureInstallOptions(install);
    (void)fs::installFormulation(system, o.operator_tag, {u_id, p_id}, M + C, install);
    if (preflight.has_exterior_free_surface) {
        declareFreeSurfaceExternalBoundaryEnergy(system, o);
        installFreeSurfaceResidualWork(system, o, u_id, p_id, install, work,
                                       {.include_mcv = mcv,
                                        .has_nonconservative_body_force = nonconservative_force,
                                        .enable_vms = preflight.enable_vms});
    }
    if (pspg_grad_p.isValid()) {
        auto pspg = install;
        pspg.generated_boundary_nitsche_trace_requests.clear();
        pspg.source_component_tag = "navier_stokes_vms_pspg_pressure_gradient";
        appendUnique(pspg.extra_trial_fields, u_id);
        (void)fs::installFormulation(system, o.operator_tag, {p_id}, pspg_grad_p, pspg);
    }

    // Opt-in diagnostic operators (environment switches), after the production ones.
    if (balance) {
        installConservativeBalanceDiagnosticOperators(system, u_id, p_id, install, balance->pressure,
                                                      balance->external_pressure, balance->surface_energy,
                                                      balance->gravitational_potential, r.pressure, u, q,
                                                      active_volume);
    }
    if (pressureRowContributionDiagnosticEnabled()) {
        const auto vms_pspg = r.pspg_pressure_gradient.isValid()
                                  ? r.pspg_pressure_gradient + r.pspg_without_pressure_gradient
                                  : FormExpr{};
        installPressureRowContributionDiagnosticOperators(system, u_id, p_id, install, r.galerkin_continuity,
                                                          active_continuity, vms_pspg, r.pspg_pressure_gradient,
                                                          r.pspg_without_pressure_gradient, work.ghost_penalty,
                                                          active_volume);
    }
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
