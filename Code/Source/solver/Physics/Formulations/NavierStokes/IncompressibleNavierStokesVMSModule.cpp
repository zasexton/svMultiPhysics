/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceCollectives.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceConfigurationArtifact.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactCertificate.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceDiagnosticOperators.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergy.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceFittedALE.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfacePressureGauge.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceResidualWork.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceSemiImplicitSurfaceTension.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceValidation.h"
#include "Physics/Formulations/NavierStokes/NavierStokesBCFactories.h"
#include "Physics/Formulations/NavierStokes/NavierStokesFields.h"
#include "Physics/Formulations/NavierStokes/NavierStokesGeneratedNitsche.h"
#include "Physics/Formulations/NavierStokes/NavierStokesInitialConditions.h"
#include "Physics/Formulations/NavierStokes/NavierStokesPreflight.h"

#include "FE/Assembly/Assembler.h"
#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Constraints/ConstraintPlan.h"
#include "FE/Constraints/LevelSetActiveSideVertexDirichletConstraint.h"
#include "FE/Constraints/SmallCutAggregationConstraint.h"
#include "FE/Constraints/ContactPointVelocityConstraint.h"
#include "FE/Constraints/VertexDirichletConstraint.h"
#include "FE/Constitutive/MetadataTaggedModel.h"
#include "FE/Core/Logger.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Elements/ReferenceElement.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/Forms/SymbolicDifferentiation.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Geometry/FrameGeometry.h"
#include "FE/Geometry/MappingFactory.h"
#include "FE/LevelSet/LevelSetCellEvaluator.h"
#include "FE/LevelSet/LevelSetGeometry.h"
#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Systems/ALEBinding.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "Interfaces/GeneratedActiveBoundaryDomain.h"
#include "Interfaces/GeneratedInterfaceBoundaryIntersectionDomain.h"
#include "Interfaces/LevelSetInterfaceDomain.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <array>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
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

namespace {

using FreeSurfaceBoundary = IncompressibleNavierStokesVMSOptions::FreeSurfaceBoundary;
using FreeSurfaceContactLine =
    IncompressibleNavierStokesVMSOptions::FreeSurfaceContactLine;

[[nodiscard]] FE::Real hydrostaticPressureAt(
    const std::array<FE::Real, 3>& x,
    const IncompressibleNavierStokesVMSOptions& options,
    const IncompressibleNavierStokesVMSOptions::
        HydrostaticPressureInitialization& init) noexcept
{
    FE::Real pressure = init.reference_pressure;
    for (std::size_t d = 0; d < options.body_force.size(); ++d) {
        pressure += options.density * options.body_force[d] *
                    (x[d] - init.reference_point[d]);
    }
    return pressure;
}

} // namespace

void IncompressibleNavierStokesVMSModule::applyInitialConditions(
    const FE::systems::FESystem& system,
    FE::backends::GenericVector& u0) const
{
    const auto& init = options_.hydrostatic_pressure_initialization;
    initializeNavierStokesState(
        system,
        u0,
        options_,
        [this, &init](const std::array<FE::Real, 3>& x) {
            return hydrostaticPressureAt(x, options_, init);
        });
}

namespace {

void applyFreeSurfaceCutCellStabilization(
    FE::forms::FormExpr& momentum_form,
    FE::forms::FormExpr& continuity_form,
    const FreeSurfaceBoundary& bc,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& p,
    const FE::forms::FormExpr& v,
    const FE::forms::FormExpr& q,
    const FE::forms::FormExpr& mu,
    FE::Real density,
    FE::Real stabilization_epsilon,
    int velocity_components,
    int velocity_polynomial_order,
    int pressure_polynomial_order,
    FE::forms::FormExpr* pressure_stabilization_form = nullptr)
{
    if (!isUnfittedLevelSet(bc) || !bc.cut_cell_stabilization.enabled) {
        return;
    }

    namespace bc_forms = FE::forms::bc;
    const auto& cut = bc.cut_cell_stabilization;
    constexpr int supported_derivative_order = 2;
    const auto derivative_order_label = [](int max_order) -> const char* {
        if (max_order <= 0) {
            return "disabled";
        }
        return max_order > 1 ? "1,2" : "1";
    };
    const bool pressure_stabilization_enabled =
        pressureStabilizationActive(bc);
    const bool pressure_stabilization_incremental =
        pressure_stabilization_enabled &&
        cut.pressure_policy ==
            FreeSurfacePressureStabilizationPolicy::Incremental;
    const int pressure_derivative_order =
        pressure_stabilization_enabled
            ? (pressure_stabilization_incremental
                   ? 1
                   : (pressure_polynomial_order > 1
                          ? supported_derivative_order
                          : 1))
            : 0;
    const int max_derivative_order = pressure_derivative_order;
    const bool has_unsupported_derivative_order =
        pressure_stabilization_enabled &&
        !pressure_stabilization_incremental &&
        pressure_polynomial_order > supported_derivative_order;
    auto cut_scale = cut.use_cut_metadata_scale
        ? FE::forms::cutStabilizationScale()
        : FE::forms::FormExpr::constant(1.0);
    if (cut.use_cut_metadata_scale && cut.cut_metadata_scale_cap.has_value()) {
        cut_scale = FE::forms::min(
            cut_scale,
            FE::forms::FormExpr::constant(*cut.cut_metadata_scale_cap));
    }
    const auto h_f = FE::forms::avg(FE::forms::hNormal());
    const auto h3 = h_f * h_f * h_f;
    const auto h5 = h3 * h_f * h_f;
    // Transient ghost-penalty coefficient. The viscous scale mu alone
    // under-stabilizes inertia-dominated cut supports (rho*h^2/dt >> mu for
    // water-like parameters), so the penalty carries the generalized
    // diffusivity mu_gp = mu + rho*h^2/dt used by transient CutFEM analyses
    // (Schott & Wall; Burman, Fernandez & Massing). Velocity jump penalties
    // scale with mu_gp, pressure jump penalties with 1/mu_gp.
    const auto rho_gp = FE::forms::FormExpr::constant(density);
    const auto mu_gp =
        mu + rho_gp * h_f * h_f / FE::forms::FormExpr::effectiveTimeStep();
    // Dimensionless ghost-penalty calibration for the analytic transient
    // scaling: 0.01*(mu+rho*h^2/dt)*h (gradient jump) and the matching
    // h^3/h^5 variants. The eigenproblem-calibrated alternative was
    // retired: its coercivity-sufficient coefficients measured 2-6 orders
    // above the Newton-conditioning ceiling on the SPHERIC wet-bed cases
    // (the workable small-gamma regime is sub-coercive), and small-cut
    // aggregation now replaces the velocity penalty altogether for
    // vertex-covered (P1/iso-Q2) spaces. See
    // Documentation/free_surface_program_tracker.md, section 8.2.
    constexpr FE::Real kCutPenaltyTransientCalibration{0.01};
    const auto gp_calibration =
        FE::forms::FormExpr::constant(kCutPenaltyTransientCalibration);
    const auto interface_side =
        bc.active_domain == FreeSurfaceActiveDomain::LevelSetPositive
            ? "Plus"
            : (bc.active_domain == FreeSurfaceActiveDomain::LevelSetNegative
                   ? "Minus"
                   : "All");
    const auto active_domain_side =
        bc.active_domain == FreeSurfaceActiveDomain::None
            ? "FullDomain"
            : cutVolumeSideName(activeDomainSide(bc.active_domain));
    const auto facet_domain_side =
        bc.active_domain == FreeSurfaceActiveDomain::None
            ? std::optional<FE::forms::CutVolumeSide>{}
            : std::optional<FE::forms::CutVolumeSide>{
                  activeDomainSide(bc.active_domain)};
    const auto integrate_cut_adjacent =
        [&](const FE::forms::FormExpr& integrand) {
            return facet_domain_side.has_value()
                       ? integrand.dS(FE::forms::FacetDomain::cutAdjacent(
                             bc.interface_marker, *facet_domain_side))
                       : integrand.dS(FE::forms::FacetDomain::cutAdjacent(
                             bc.interface_marker));
        };

    std::ostringstream oss;
    oss << "IncompressibleNavierStokesVMSModule: cut-cell stabilization "
        << "marker=" << bc.interface_marker
        << " level_set_field='" << bc.level_set_field_name << "'"
        << " interface_side=" << interface_side
        << " active_domain=" << activeDomainName(bc.active_domain)
        << " active_domain_side=" << active_domain_side
        << " Active_domain_method="
        << activeDomainMethodName(bc.active_domain_method)
        << " generated_interface_geometry="
        << bc.generated_interface_geometry
        << " geometry_tangent_policy="
        << bc.geometry_tangent_policy
        << " use_cut_metadata_scale="
        << (cut.use_cut_metadata_scale ? "true" : "false")
        << " cut_metadata_scale_cap=";
    if (cut.cut_metadata_scale_cap.has_value()) {
        oss << *cut.cut_metadata_scale_cap;
    } else {
        oss << "unbounded";
    }
    oss
        << " pressure_stabilization_policy="
        << pressureStabilizationPolicyName(cut.pressure_policy)
        << " pressure_stabilization="
        << (pressure_stabilization_enabled ? "enabled" : "disabled")
        << " pressure_stabilization_form="
        << (pressure_stabilization_enabled
                ? (pressure_stabilization_incremental ? "incremental"
                                                      : "absolute")
                : "disabled")
        << " pressure_disabled_reason="
        << pressureStabilizationDisabledReason(bc)
        << " facet_scope=cut-adjacent"
        << " velocity_polynomial_order=" << velocity_polynomial_order
        << " pressure_polynomial_order=" << pressure_polynomial_order
        << " derivative_orders=" << derivative_order_label(max_derivative_order)
        << " pressure_derivative_orders="
        << derivative_order_label(pressure_derivative_order)
        << " cut_penalty_transient_calibration="
        << kCutPenaltyTransientCalibration
        << " small_cut_aggregation="
        << (bc.small_cut_aggregation ? "true" : "false")
        << " velocity_ghost_penalty_mode=retired_replaced_by_aggregation"
        << " pressure_scaling="
        << (pressure_derivative_order <= 0
                ? "disabled"
                : (pressure_derivative_order > 1
                       ? "0.01*h^3/(mu+rho*h^2/dt),0.01*h^5/(mu+rho*h^2/dt)"
                       : "0.01*h^3/(mu+rho*h^2/dt)"));
    FE_LOG_INFO(oss.str());

    if (has_unsupported_derivative_order) {
        FE_LOG_WARNING(
            "IncompressibleNavierStokesVMSModule: high-order cut-cell "
            "stabilization currently supports derivative_orders=1,2; "
            "higher-normal-derivative penalties above derivative_order=" +
            std::to_string(supported_derivative_order) +
            " are not yet available");
    }

    if (pressure_stabilization_enabled) {
        const auto pressure_penalty = bc_forms::toScalarExpr(
            cut.pressure_gradient_penalty,
            freeSurfaceValueName("ns_free_surface_cut_pressure_penalty", bc));
        const auto stabilized_pressure =
            pressure_stabilization_incremental
                ? FE::forms::FormExpr::effectiveTimeStep() * FE::forms::dt(p)
                : p;
        const auto pressure_jump_p =
            FE::forms::cutAdjacentFacetGradientJump(stabilized_pressure);
        const auto pressure_jump_q =
            FE::forms::cutAdjacentFacetGradientJump(q);
        auto pressure_form = integrate_cut_adjacent(
            cut_scale * pressure_penalty * gp_calibration * h3 /
                (mu_gp + FE::forms::FormExpr::constant(stabilization_epsilon)) *
                FE::forms::inner(pressure_jump_p, pressure_jump_q));

        if (pressure_derivative_order > 1) {
            const auto pressure_second_jump_p =
                FE::forms::cutAdjacentFacetSecondNormalDerivativeJump(
                    stabilized_pressure);
            const auto pressure_second_jump_q =
                FE::forms::cutAdjacentFacetSecondNormalDerivativeJump(q);
            pressure_form =
                pressure_form +
                integrate_cut_adjacent(
                    cut_scale * pressure_penalty * gp_calibration * h5 /
                        (mu_gp + FE::forms::FormExpr::constant(stabilization_epsilon)) *
                        pressure_second_jump_p * pressure_second_jump_q);
        }
        continuity_form = continuity_form + pressure_form;
        if (pressure_stabilization_form != nullptr) {
            *pressure_stabilization_form =
                pressure_stabilization_form->isValid()
                    ? (*pressure_stabilization_form + pressure_form)
                    : pressure_form;
        }
    }
}

void applyDynamicContactAngleResidual(
    FE::forms::FormExpr& momentum_form,
    FE::systems::FESystem& system,
    const FreeSurfaceBoundary& bc,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& v,
    const FE::forms::FormExpr& mu,
    int dim,
    FE::forms::FormExpr* conservative_surface_energy_form = nullptr)
{
    using namespace FE::forms;

    const auto install_navier_wall_slip =
        [&](const FreeSurfaceContactLine& contact_line) {
            const auto wall_n = wallNormalExpression(contact_line, dim);
            const auto slip_length = FE::forms::bc::toScalarExpr(
                contactLineSlipLength(contact_line),
                freeSurfaceValueName(
                    "ns_contact_wall_slip_length", bc));
            const auto u_tangent = tangentialProjection(u, wall_n);
            const auto v_tangent = tangentialProjection(v, wall_n);
            const int active_wall_marker = generatedActiveBoundaryMarker(
                bc, contactLineWallBoundaryMarker(contact_line), system);
            system.registerGeneratedEmbeddedInterfaceMarker(
                active_wall_marker);
            // The wetted part of the wall on the liquid side.
            momentum_form = momentum_form +
                (mu / slip_length * dot(u_tangent, v_tangent))
                    .ds(FE::forms::BoundaryDomain::generatedActiveSubset(
                        contactLineWallBoundaryMarker(contact_line),
                        active_wall_marker));
            return active_wall_marker;
        };

    for (const auto& contact_line : bc.contact_lines) {
        if (contactLineKind(contact_line) ==
                ContactLineKind::PrescribedAngle &&
            !FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension)) {
            if (usesKinematicAreaGradientTraction(bc)) {
                const int contact_marker =
                    generatedUnfittedContactLineMarker(
                        contact_line, bc, system);
                system.registerGeneratedEmbeddedInterfaceMarker(
                    contact_marker);
                FE_LOG_INFO(
                    std::string("IncompressibleNavierStokesVMSModule: equilibrium Young wall force is owned by the kinematic-area-gradient traction") +
                    " interface_marker=" +
                    std::to_string(bc.interface_marker) +
                    " contact_marker=" +
                    std::to_string(contact_marker) +
                    " surface_tension_form=KinematicAreaGradientTraction" +
                    " separate_equilibrium_line_force=omitted" +
                    " diagnostic=navier_stokes_total_energy_traction_wall_ownership");
                if (contactLineHasNavierSlip(contact_line)) {
                    (void)install_navier_wall_slip(contact_line);
                }
                continue;
            }
            // Decision D4: this solid--liquid/solid--gas wall force is the
            // only contact-angle mechanism; accepted-state level-set wall
            // maintenance only rescales contact cells.  SurfaceStress already
            // supplies the dynamic conormal and therefore adds only
            // -gamma*cos(theta_e); CurvatureTraction needs the full
            // cos(theta_d)-cos(theta_e) line force.  If all wall-tangential
            // test traces are essential (a no-slip wall), either form performs
            // no virtual work and contributes only to the reaction, so nothing
            // imposes the angle there.
            const auto phi_id = resolveLevelSetFieldId(bc, system);
            const auto& phi_record = system.fieldRecord(phi_id);
            const auto phi = StateField(
                phi_id,
                *phi_record.space,
                bc.level_set_field_name);
            const auto n = usesGeneratedInterfaceNormal(bc)
                               ? generatedInterfaceOutwardNormal(bc)
                               : unfittedInterfaceNormal(bc, phi);
            const auto wall_n = wallNormalExpression(contact_line, dim);
            const auto footprint_direction = conormal(n, wall_n);
            const auto contact_test = dot(v, footprint_direction);
            const auto gamma = FE::forms::bc::toScalarExpr(
                bc.surface_tension,
                freeSurfaceValueName(
                    "ns_prescribed_contact_angle_surface_tension", bc));
            const auto equilibrium_cosine = FormExpr::constant(std::cos(
                constantScalarValueOrThrow(
                    contactLineAngleRadians(contact_line),
                    "PrescribedContactAngle contact_angle_radians")));
            const int contact_marker = generatedUnfittedContactLineMarker(
                contact_line,
                bc,
                system);
            const auto wall_energy_force = usesSurfaceStress(bc)
                ? -gamma * equilibrium_cosine
                : -gamma * (equilibrium_cosine + dot(n, wall_n));
            const auto wall_energy_form = dInterfaceBoundary(
                wall_energy_force * contact_test,
                ContactLineDomain(
                    contact_marker,
                    contactLineWallBoundaryMarker(contact_line)));
            momentum_form = momentum_form + wall_energy_form;
            if (conservative_surface_energy_form != nullptr &&
                usesSurfaceStress(bc)) {
                *conservative_surface_energy_form =
                    conservative_surface_energy_form->isValid()
                        ? (*conservative_surface_energy_form + wall_energy_form)
                        : wall_energy_form;
            }
            FE_LOG_INFO(
                std::string("IncompressibleNavierStokesVMSModule: installed prescribed-angle wall-energy momentum term") +
                " interface_marker=" + std::to_string(bc.interface_marker) +
                " contact_marker=" + std::to_string(contact_marker) +
                " surface_tension_form=" + surfaceTensionFormName(bc) +
                " contact_force_discretization=" +
                (usesSurfaceStress(bc)
                     ? "surface_conormal_plus_equilibrium_wall_energy"
                     : "explicit_dynamic_angle_gap") +
                " diagnostic=navier_stokes_prescribed_contact_wall_energy");
            if (contactLineHasNavierSlip(contact_line)) {
                const int active_wall_marker =
                    install_navier_wall_slip(contact_line);
                FE_LOG_INFO(
                    std::string("IncompressibleNavierStokesVMSModule: installed prescribed-angle sharp Navier wall slip") +
                    " interface_marker=" +
                    std::to_string(bc.interface_marker) +
                    " wall_marker=" +
                    std::to_string(
                        contactLineWallBoundaryMarker(contact_line)) +
                    " active_wall_marker=" +
                    std::to_string(active_wall_marker) +
                    " wetted_wall_domain=sharp_generated_active_boundary" +
                    " line_friction=absent" +
                    " diagnostic=navier_stokes_prescribed_contact_navier_slip");
            }
            continue;
        }
        if (contactLineKind(contact_line) ==
            ContactLineKind::PrescribedAngle) {
            if (contactLineHasNavierSlip(contact_line)) {
                (void)install_navier_wall_slip(contact_line);
            }
            continue;
        }
        if (contactLineKind(contact_line) !=
            ContactLineKind::DynamicRenE) {
            continue;
        }

        const auto phi_id = resolveLevelSetFieldId(bc, system);
        const auto& phi_record = system.fieldRecord(phi_id);
        const auto phi = StateField(
            phi_id,
            *phi_record.space,
            bc.level_set_field_name);
        // Generated-geometry forms take contact geometry from the same mapped
        // rule as the surface measure.  The legacy curvature form retains its
        // Q1-gradient normal and full Ren--E angle gap.
        const auto n = usesGeneratedInterfaceNormal(bc)
                           ? generatedInterfaceOutwardNormal(bc)
                           : unfittedInterfaceNormal(bc, phi);
        const auto wall_n = wallNormalExpression(contact_line, dim);
        // n is the outward liquid normal.  Its wall-tangential projection
        // points out of the wetted footprint for either contact orientation.
        const auto footprint_direction = conormal(n, wall_n);
        const auto contact_velocity = dot(u, footprint_direction);
        const auto contact_test = dot(v, footprint_direction);

        const auto gamma = FE::forms::bc::toScalarExpr(
            bc.surface_tension,
            freeSurfaceValueName(
                "ns_dynamic_contact_angle_surface_tension", bc));
        const auto mobility = FE::forms::bc::toScalarExpr(
            contactLineMobility(contact_line),
            freeSurfaceValueName(
                "ns_dynamic_contact_angle_mobility", bc));
        const auto line_friction = FormExpr::constant(1.0) / mobility;
        const auto equilibrium_cosine = FormExpr::constant(std::cos(
            constantScalarValueOrThrow(
                contactLineAngleRadians(contact_line),
                "DynamicContactAngle equilibrium contact_angle_radians")));
        // With explicit kappa*n traction, the line term contains the complete
        // Ren--E gap cos(theta_e)-cos(theta_d).  With SurfaceStress, its
        // +gamma*cos(theta_d) conormal force is already contained in
        // gamma*P:grad(v), so the separate wall term is only
        // -gamma*cos(theta_e).  Keeping young_gap there double-counts the
        // dynamic-angle force.
        const auto contact_force =
            usesKinematicAreaGradientTraction(bc)
                ? line_friction * contact_velocity
                : usesSurfaceStress(bc)
                      ? (line_friction * contact_velocity -
                         gamma * equilibrium_cosine)
                      : (line_friction * contact_velocity -
                         gamma * (equilibrium_cosine + dot(n, wall_n)));
        const int contact_marker = generatedUnfittedContactLineMarker(
            contact_line,
            bc,
            system);
        system.registerGeneratedEmbeddedInterfaceMarker(contact_marker);
        registerDynamicContactWallNormalCheck(
            system, contact_line, contact_marker);
        const bool strong_contact_law =
            std::get<FreeSurfaceContactLine::DynamicRenE>(
                contact_line.configuration).strong_law;
        if (!strong_contact_law) {
        momentum_form = momentum_form + dInterfaceBoundary(
            contact_force * contact_test,
            ContactLineDomain(
                contact_marker,
                contactLineWallBoundaryMarker(contact_line)));
        }
        if (strong_contact_law) {
            if (!usesSurfaceStress(bc)) {
                throw std::invalid_argument(
                    "IncompressibleNavierStokesVMSModule: Contact_line_law_enforcement=Strong requires Surface_tension_form=SurfaceStress");
            }
            // The Ren-E law is the essential contact-point condition of
            // ContactPointVelocityConstraint (registered after aggregation);
            // its reaction replaces the line friction and the Young term.
            FE_LOG_INFO(
                std::string("IncompressibleNavierStokesVMSModule: DynamicContactAngle uses the strong contact-point law") +
                " contact_marker=" + std::to_string(contact_marker) +
                " Contact_line_law_enforcement=Strong line_friction_and_young_term=replaced_by_constraint_reaction" +
                " diagnostic=navier_stokes_dynamic_contact_strong_law");
        }
        if (!strong_contact_law && conservative_surface_energy_form != nullptr &&
            usesSurfaceStress(bc)) {
            // Only the Young wall-energy variation belongs to the
            // conservative split.  The line-friction and wetted-wall Navier
            // terms are dissipative and must remain absent from this
            // diagnostic even away from equilibrium.
            const auto wall_energy_form = dInterfaceBoundary(
                (-gamma * equilibrium_cosine) * contact_test,
                ContactLineDomain(
                    contact_marker,
                    contactLineWallBoundaryMarker(contact_line)));
            *conservative_surface_energy_form =
                conservative_surface_energy_form->isValid()
                    ? (*conservative_surface_energy_form + wall_energy_form)
                    : wall_energy_form;
        }

        const int active_wall_marker =
            install_navier_wall_slip(contact_line);

        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: installed coupled dissipative Ren--E DynamicContactAngle") +
            " interface_marker=" + std::to_string(bc.interface_marker) +
            " contact_marker=" + std::to_string(contact_marker) +
            " wall_marker=" +
            std::to_string(contactLineWallBoundaryMarker(contact_line)) +
            " convention=theta_through_liquid_n_outward_liquid_nw_fluid_to_solid" +
            " law=xi_Vcl_equals_gamma_cos_thetae_minus_cos_thetad" +
            " surface_tension_form=" + surfaceTensionFormName(bc) +
            " contact_force_discretization=" +
            (usesKinematicAreaGradientTraction(bc)
                 ? "total_energy_traction_plus_line_friction"
                 : usesSurfaceStress(bc)
                       ? "surface_conormal_plus_equilibrium_wall_energy"
                       : "explicit_dynamic_angle_gap") +
            " active_wall_marker=" +
            std::to_string(active_wall_marker) +
            " wetted_wall_domain=sharp_generated_active_boundary" +
            " diagnostic=navier_stokes_dynamic_contact_angle");
    }
}

void applyFreeSurfaceBoundary(FE::forms::FormExpr& momentum_form,
                              FE::forms::FormExpr& continuity_form,
                              const FreeSurfaceBoundary& bc,
                              const FE::systems::FESystem& system,
                              const FE::forms::FormExpr& u,
                              const FE::forms::FormExpr& p,
                              const FE::forms::FormExpr& v,
                              const FE::forms::FormExpr& q,
                              const FE::forms::FormExpr& mesh_velocity,
                              const FE::forms::FormExpr& mu,
                              const IncompressibleNavierStokesVMSOptions& options,
                              bool ale_enabled,
                              int dim,
                              FE::forms::FormExpr* conservative_pressure_form = nullptr,
                              FE::forms::FormExpr* conservative_external_pressure_form = nullptr,
                              FE::forms::FormExpr* conservative_surface_energy_form = nullptr,
                              FE::forms::FormExpr* weak_boundary_momentum_work_form = nullptr,
                              FE::forms::FormExpr* weak_boundary_continuity_work_form = nullptr)
{
    using namespace FE::forms;

    const auto append_weak_work = [](
        FormExpr* destination,
        const FormExpr& contribution) {
        if (destination == nullptr) {
            return;
        }
        *destination = destination->isValid()
                           ? (*destination + contribution)
                           : contribution;
    };

    validateFreeSurfaceBoundary(bc, options, ale_enabled, system, dim);

    const bool has_dynamic_stress =
        !bc::isZeroConstantScalarValue(bc.external_pressure) ||
        !bc::isZeroConstantScalarValue(bc.surface_tension);
    const bool needs_surface_normal =
        has_dynamic_stress ||
        bc.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None;
    if (isUnfittedLevelSet(bc)) {
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: unfitted free-surface boundary mode marker=") +
            std::to_string(bc.interface_marker) +
            " level_set_field='" + bc.level_set_field_name + "'" +
            " generated_interface_domain_id='" + bc.generated_interface_domain_id + "'" +
            " generated_interface_geometry=" + bc.generated_interface_geometry +
            " geometry_tangent_policy=" + bc.geometry_tangent_policy +
            " active_domain=" + activeDomainName(bc.active_domain) +
            " active_domain_method=" + activeDomainMethodName(bc.active_domain_method) +
            " dynamic_stress=" + (has_dynamic_stress ? "enabled" : "natural_zero") +
            " surface_tension_form=" + surfaceTensionFormName(bc) +
            " curvature_policy=" + freeSurfaceCurvaturePolicyName(bc) +
            " curvature_tangent_policy=" +
            freeSurfaceCurvatureTangentPolicyName(bc, system) +
            " kinematic_enforcement=" + kinematicEnforcementName(bc.kinematic_enforcement) +
            " cut_cell_stabilization=" + (bc.cut_cell_stabilization.enabled ? "enabled" : "disabled") +
            " pressure_stabilization_policy=" +
            pressureStabilizationPolicyName(
                bc.cut_cell_stabilization.pressure_policy) +
            " pressure_stabilization=" +
            (pressureStabilizationActive(bc) ? "active" : "inactive") +
            " pressure_stabilization_disabled_reason=" +
            pressureStabilizationDisabledReason(bc));
    }
    const auto p_ext = bc::toScalarExpr(
        bc.external_pressure,
        freeSurfaceValueName("ns_free_surface_external_pressure", bc));
    if (!needs_surface_normal) {
        if (isUnfittedLevelSet(bc)) {
            FE_LOG_WARNING(
                std::string("IncompressibleNavierStokesVMSModule: unfitted free surface installs no explicit dI boundary residual marker=") +
                std::to_string(bc.interface_marker) +
                " level_set_field='" + bc.level_set_field_name + "'" +
                " dynamic_stress=natural_zero"
                " kinematic_enforcement=None"
                " diagnostic=unfitted_free_surface_natural_mode");
        }
        return;
    }

    const auto phi = freeSurfaceLevelSet(bc, system);
    // Fitted boundaries use the boundary normal of the integration frame:
    // the current normal with ALE (the active frame is the current
    // configuration) and the static facet normal otherwise.
    const auto n = isUnfittedLevelSet(bc)
        ? (usesGeneratedInterfaceNormal(bc)
               ? generatedInterfaceOutwardNormal(bc)
               : unfittedInterfaceNormal(bc, phi))
        : (useFittedCurrentGeometry(bc, ale_enabled)
               ? currentNormal()
               : FormExpr::normal());
    const auto gamma = bc::toScalarExpr(
        bc.surface_tension,
        freeSurfaceValueName("ns_free_surface_surface_tension", bc));
    const auto curvature = [&]() {
        if (usesSurfaceStress(bc)) {
            return FormExpr::constant(0.0);
        }
        if (bc::isZeroConstantScalarValue(bc.surface_tension)) {
            return FormExpr::constant(0.0);
        }
        if (isUnfittedLevelSet(bc)) {
            return unfittedTractionCurvature(bc, system, phi);
        }
        return bc::toScalarExpr(
            bc.curvature,
            freeSurfaceValueName("ns_free_surface_curvature", bc));
    }();

    if (has_dynamic_stress) {
        if (usesSurfaceStress(bc)) {
            const auto pressure_integrand = p_ext * inner(n, v);
            const auto interface_pressure_form = integrateOnFreeSurface(
                pressure_integrand, bc, ale_enabled);
            if (conservative_pressure_form != nullptr) {
                *conservative_pressure_form =
                    conservative_pressure_form->isValid()
                        ? (*conservative_pressure_form +
                           interface_pressure_form)
                        : interface_pressure_form;
            }
            if (conservative_external_pressure_form != nullptr) {
                *conservative_external_pressure_form =
                    conservative_external_pressure_form->isValid()
                        ? (*conservative_external_pressure_form +
                           interface_pressure_form)
                        : interface_pressure_form;
            }
            if (!surfaceStressActive(bc)) {
                // A literal gamma=0 leaves only the external-pressure work.
                // Keep the generated-interface normal required by the
                // SurfaceStress geometry contract, but do not construct the
                // surface projector.  Besides being the exact algebraic
                // specialization, this prevents the zero-coefficient
                // projector/grad(v) tree from reaching the JIT compiler.
                momentum_form = momentum_form + interface_pressure_form;
                FE_LOG_INFO(
                    std::string("IncompressibleNavierStokesVMSModule: installed pressure-only free-surface stress") +
                    " marker=" + std::to_string(freeSurfaceMarker(bc)) +
                    " surface_tension_form=SurfaceStress" +
                    " normal_source=integration_rule_geometry" +
                    " form=p_external_n_dot_v" +
                    " surface_energy_form=omitted_literal_zero_gamma" +
                    " diagnostic=free_surface_pressure_only_surface_stress");
            } else {
                // Variation of the discrete surface energy gamma*|Gamma_h|.
                // Crucially, n is the geometric normal carried by the same
                // generated-interface quadrature rule as dI, not an
                // independently evaluated Q1-gradient normal.  The positive
                // sign follows div_Gamma(I-n*n)=-kappa*n for the
                // outward-liquid convention.
                const auto projector = tangentialProjector(n, dim);
                const auto surface_energy_integrand =
                    gamma * inner(projector, grad(v));
                const auto interface_surface_energy_form =
                    integrateOnFreeSurface(
                        surface_energy_integrand, bc, ale_enabled);
                if (conservative_surface_energy_form != nullptr) {
                    *conservative_surface_energy_form =
                        conservative_surface_energy_form->isValid()
                            ? (*conservative_surface_energy_form +
                               interface_surface_energy_form)
                            : interface_surface_energy_form;
                }
                const auto residual_integrand =
                    pressure_integrand + surface_energy_integrand;
                momentum_form = momentum_form + integrateOnFreeSurface(
                    residual_integrand, bc, ale_enabled);
                FE_LOG_INFO(
                    std::string("IncompressibleNavierStokesVMSModule: installed variational free-surface stress") +
                    " marker=" + std::to_string(freeSurfaceMarker(bc)) +
                    " surface_tension_form=SurfaceStress" +
                    " normal_source=integration_rule_geometry" +
                    " form=gamma_I_minus_n_outer_n_colon_grad_v" +
                    " capillary_balance_method=discrete_energy_volume_stationarity" +
                    " capillary_balance_qualification=prerequisite_only" +
                    " force_projection_applied=0" +
                    " static_geometry_stationarity_required=1" +
                    " diagnostic=free_surface_variational_surface_stress");
            }
        } else if (usesKinematicAreaGradientTraction(bc)) {
            const auto pressure_integrand = p_ext * inner(n, v);
            const auto interface_pressure_form = integrateOnFreeSurface(
                pressure_integrand, bc, ale_enabled);
            momentum_form = momentum_form + interface_pressure_form;
            if (conservative_pressure_form != nullptr) {
                *conservative_pressure_form =
                    conservative_pressure_form->isValid()
                        ? (*conservative_pressure_form +
                           interface_pressure_form)
                        : interface_pressure_form;
            }
            if (conservative_external_pressure_form != nullptr) {
                *conservative_external_pressure_form =
                    conservative_external_pressure_form->isValid()
                        ? (*conservative_external_pressure_form +
                           interface_pressure_form)
                        : interface_pressure_form;
            }
            if (!bc::isZeroConstantScalarValue(bc.surface_tension)) {
                const auto capillary_integrand =
                    gamma * curvature * inner(n, v);
                const auto total_energy_gradient_form =
                    integrateOnFreeSurface(
                        capillary_integrand, bc, ale_enabled);
                momentum_form =
                    momentum_form + total_energy_gradient_form;
                if (conservative_surface_energy_form != nullptr) {
                    *conservative_surface_energy_form =
                        conservative_surface_energy_form->isValid()
                            ? (*conservative_surface_energy_form +
                               total_energy_gradient_form)
                            : total_energy_gradient_form;
                }
            }
            FE_LOG_INFO(
                std::string("IncompressibleNavierStokesVMSModule: installed total discrete-energy-gradient traction") +
                " marker=" +
                std::to_string(freeSurfaceMarker(bc)) +
                " surface_tension_form=KinematicAreaGradientTraction" +
                " normal_source=integration_rule_geometry" +
                " curvature_source=" +
                freeSurfaceCurvaturePolicyName(bc) +
                " capillary_balance_method=kinematic_area_gradient_energy_traction" +
                " capillary_balance_qualification=prerequisite_only" +
                " equilibrium_line_force_owner=traction" +
                " force_projection_applied=0" +
                " diagnostic=free_surface_total_energy_gradient_traction");
        } else {
            const auto traction_scalar = -p_ext - gamma * curvature;
            const auto traction = traction_scalar * n;
            const auto residual_integrand =
                FE::forms::FormExpr::constant(FE::Real{-1.0}) *
                inner(traction, v);
            momentum_form = momentum_form + integrateOnFreeSurface(
                residual_integrand, bc, ale_enabled);
        }
    }

    if (usesSemiImplicitNormalIncrement(bc)) {
        // Lagged normal increment: zero at every refreshed residual, so it
        // is not part of any conservative or residual-work ledger channel;
        // it adds only a constant velocity block to the inner Jacobian.
        const auto semi_implicit_integrand =
            semiImplicitNormalIncrementIntegrand(
                gamma,
                u,
                freeSurfaceSemiImplicitReferenceVelocity(system),
                v,
                n);
        momentum_form = momentum_form + integrateOnFreeSurface(
            semi_implicit_integrand, bc, ale_enabled);
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: installed semi-implicit capillary term") +
            " marker=" + std::to_string(freeSurfaceMarker(bc)) +
            " surface_tension_semi_implicit=" +
            freeSurfaceSurfaceTensionSemiImplicitName(
                bc.surface_tension_semi_implicit) +
            " surface_tension_form=" + surfaceTensionFormName(bc) +
            " form=gamma_dt_eff_grad_gamma_du_dot_n_dot_grad_gamma_v_dot_n" +
            " reference_velocity_field=" +
            std::string(kFreeSurfaceSemiImplicitReferenceVelocityFieldName) +
            " reference_refresh=every_generated_state_refresh" +
            " fresh_residual_contribution=0" +
            " coefficient=surface_tension_times_effective_time_step" +
            " qualification=" +
            (usesKinematicAreaGradientTraction(bc) ? "Experimental"
                                                   : "Candidate") +
            " diagnostic=free_surface_semi_implicit_normal_increment");
    }

    switch (bc.kinematic_enforcement) {
    case FreeSurfaceKinematicEnforcement::None:
        return;
    case FreeSurfaceKinematicEnforcement::MeshNitsche:
        // The fluid keeps the natural dynamic condition assembled above;
        // the normal kinematics are owned by the mesh-displacement row
        // (installFittedFreeSurfaceMeshKinematics and the mesh-motion
        // consistency term).  A fluid kinematic row here would replace the
        // normal dynamic condition by u.n = w.n.
        if (isUnfittedLevelSet(bc)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: MeshNitsche free-surface kinematics are only supported on fitted ALE boundaries");
        }
        return;
    case FreeSurfaceKinematicEnforcement::Penalty: {
        if (bc.normal_kinematic_policy !=
            FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unsupported fitted free-surface normal kinematic policy");
        }
        const auto penalty = bc::toScalarExpr(
            bc.kinematic_penalty,
            freeSurfaceValueName("ns_free_surface_kinematic_penalty", bc));
        const auto normal_mismatch = normalTrace(u - mesh_velocity, n);
        const auto residual_integrand =
            penalty * normal_mismatch * normalTrace(v, n);
        const auto residual_form = integrateOnFreeSurface(
            residual_integrand, bc, ale_enabled);
        momentum_form = momentum_form + residual_form;
        append_weak_work(
            weak_boundary_momentum_work_form, residual_form);
        return;
    }
    case FreeSurfaceKinematicEnforcement::Nitsche: {
        if (isUnfittedLevelSet(bc)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: Nitsche free-surface kinematics are only supported on fitted ALE boundaries");
        }
        if (bc.normal_kinematic_policy !=
            FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: unsupported fitted free-surface normal kinematic policy");
        }
        if (!(bc.kinematic_nitsche_gamma > 0.0)) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: Nitsche free-surface kinematics require kinematic_nitsche_gamma > 0");
        }

        const auto normal_mismatch = normalTrace(u - mesh_velocity, n);
        const auto v_normal = normalTrace(v, n);
        const auto stress_u = FormExpr::constant(2.0) * mu * sym(grad(u));
        const auto stress_v = FormExpr::constant(2.0) * mu * sym(grad(v));
        const auto normal_stress_u = normalTrace(stress_u * n, n);
        const auto normal_stress_v = normalTrace(stress_v * n, n);
        const auto penalty = bc::buildTraceNitschePenalty(
            mu / hNormal(),
            u,
            bc::TraceNitscheOptions{
                .gamma = bc.kinematic_nitsche_gamma,
                .variant = bc.kinematic_nitsche_symmetric
                    ? bc::NitscheVariant::Symmetric
                    : bc::NitscheVariant::Unsymmetric,
                .scale_with_p = bc.kinematic_nitsche_scale_with_p});

        const auto consistency_penalty_form = integrateOnFreeSurface(
            (p - normal_stress_u) * v_normal +
            penalty * normal_mismatch * v_normal,
            bc,
            ale_enabled);
        momentum_form = momentum_form + consistency_penalty_form;
        append_weak_work(
            weak_boundary_momentum_work_form,
            consistency_penalty_form);
        if (bc.kinematic_nitsche_symmetric) {
            const auto adjoint_form = -integrateOnFreeSurface(
                normal_stress_v * normal_mismatch,
                bc,
                ale_enabled);
            const auto kinematic_continuity_form =
                integrateOnFreeSurface(
                    q * normal_mismatch, bc, ale_enabled);
            momentum_form = momentum_form + adjoint_form;
            continuity_form =
                continuity_form + kinematic_continuity_form;
            append_weak_work(
                weak_boundary_momentum_work_form, adjoint_form);
            append_weak_work(
                weak_boundary_continuity_work_form,
                kinematic_continuity_form);
        } else {
            const auto adjoint_form = integrateOnFreeSurface(
                normal_stress_v * normal_mismatch,
                bc,
                ale_enabled);
            const auto kinematic_continuity_form =
                -integrateOnFreeSurface(
                    q * normal_mismatch, bc, ale_enabled);
            momentum_form = momentum_form + adjoint_form;
            continuity_form =
                continuity_form + kinematic_continuity_form;
            append_weak_work(
                weak_boundary_momentum_work_form, adjoint_form);
            append_weak_work(
                weak_boundary_continuity_work_form,
                kinematic_continuity_form);
        }
        return;
    }
    }

    throw std::invalid_argument(
        "IncompressibleNavierStokesVMSModule: unsupported free-surface kinematic enforcement");
}

void installFittedFreeSurfaceMeshKinematics(
    FE::systems::FESystem& system,
    const FreeSurfaceBoundary& bc,
    const FE::systems::ALEBinding& ale_binding,
    const FE::forms::FormExpr& u,
    const IncompressibleNavierStokesVMSOptions& options,
    const FE::systems::FormInstallOptions& base_install_options,
    FE::FieldId velocity_field)
{
    using namespace FE::forms;

    if (bc.implementation != FreeSurfaceImplementation::FittedALE) {
        return;
    }
    const bool install_normal_relation =
        bc.kinematic_enforcement != FreeSurfaceKinematicEnforcement::None &&
        bc.normal_kinematic_policy ==
            FreeSurfaceNormalKinematicPolicy::MatchFluidNormalVelocity;
    const bool install_tangential_smoothing =
        bc.tangential_mesh_policy ==
        FreeSurfaceTangentialMeshPolicy::SmoothingOnly;
    const bool install_tangential_prescription =
        bc.tangential_mesh_policy ==
        FreeSurfaceTangentialMeshPolicy::Prescribed;
    if (!ale_binding.coupled()) {
        return;
    }
    if (ale_binding.mesh_displacement_field == FE::INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: fitted free-surface mesh kinematics require a coupled mesh displacement unknown");
    }

    const auto& rec = system.fieldRecord(ale_binding.mesh_displacement_field);
    if (!rec.space) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: fitted free-surface mesh displacement field has no function space");
    }

    system.declareMeshTangentialBoundaryPolicy(
        FE::systems::MeshTangentialBoundaryPolicyDeclaration{
            .mesh_displacement_field = ale_binding.mesh_displacement_field,
            .boundary_marker = bc.boundary_marker,
            .policy =
                systemTangentialMeshPolicy(bc.tangential_mesh_policy),
            .owner_component =
                "IncompressibleNavierStokesVMSModule.FreeSurfaceBoundary",
        });

    FE_LOG_INFO(
        std::string("IncompressibleNavierStokesVMSModule: fitted "
                    "free-surface mesh policy") +
        " marker=" + std::to_string(bc.boundary_marker) +
        " tangential_policy=" +
        tangentialMeshPolicyName(bc.tangential_mesh_policy) +
        " tangential_owner=free_surface_boundary" +
        " tangential_enforcement=" +
        (install_tangential_prescription
             ? "weak_velocity_penalty"
             : install_tangential_smoothing
                   ? "surface_smoothing_functional"
                   : "natural_no_tangential_constraint") +
        " diagnostic=fitted_free_surface_mesh_policy");

    if (!install_normal_relation && !install_tangential_smoothing &&
        !install_tangential_prescription) {
        FE::analysis::BoundaryConditionDescriptor descriptor;
        descriptor.primary_variable =
            FE::analysis::VariableKey::field(
                ale_binding.mesh_displacement_field);
        descriptor.boundary_marker = bc.boundary_marker;
        descriptor.trace_kind =
            FE::analysis::TraceKind::TangentialComponent;
        descriptor.enforcement_kind =
            FE::analysis::EnforcementKind::WeakConsistent;
        const auto consumer_source = fittedTangentialOperatorSource(
            bc.boundary_marker, bc.tangential_mesh_policy);
        descriptor.source = consumer_source;
        system.addBoundaryConditionDescriptor(
            std::move(descriptor), options.operator_tag);
        system.bindMeshTangentialBoundaryPolicyConsumer(
            ale_binding.mesh_displacement_field,
            bc.boundary_marker,
            systemTangentialMeshPolicy(bc.tangential_mesh_policy),
            options.operator_tag,
            consumer_source);
        return;
    }

    const auto psi = TestField(
        ale_binding.mesh_displacement_field,
        *rec.space,
        "psi_free_surface_mesh");
    const auto d_mesh = StateField(
        ale_binding.mesh_displacement_field,
        *rec.space,
        "d_mesh_free_surface");
    const auto n = currentNormal();
    FormExpr residual;
    if (install_normal_relation) {
        const auto normal_mismatch = normalTrace(dt(d_mesh) - u, n);
        const auto penalty = [&]() {
            switch (bc.kinematic_enforcement) {
            case FreeSurfaceKinematicEnforcement::Penalty:
                return bc::toScalarExpr(
                    bc.kinematic_penalty,
                    freeSurfaceValueName(
                        "ns_free_surface_mesh_kinematic_penalty", bc));
            case FreeSurfaceKinematicEnforcement::Nitsche:
                return FormExpr::constant(bc.kinematic_nitsche_gamma) /
                       hNormal();
            case FreeSurfaceKinematicEnforcement::MeshNitsche:
                // Nitsche penalty of the mesh-side kinematic row on the
                // normal mesh-velocity mismatch.  It balances the harmonic
                // mesh operator acting on the mesh velocity,
                // kappa grad(dt(d)):grad(psi) (dimensionless kappa, harmonic
                // quantity `velocity`), without a time scale; the mesh-motion
                // module adds the consistency term and checks coercivity,
                // gamma_N > 2 kappa (P1 trace inverse inequality with
                // h_n = 2|T|/|F|).  A penalty on the velocity mismatch of a
                // displacement operator would need deltat and would let the
                // P1 flux defect accumulate from step to step.
                return FormExpr::constant(bc.kinematic_nitsche_gamma) /
                       hNormal();
            case FreeSurfaceKinematicEnforcement::None:
                break;
            }
            return FormExpr::constant(0.0);
        }();
        residual = integrateOnFreeSurface(
            penalty * normal_mismatch * normalTrace(psi, n),
            bc,
            /*ale_enabled=*/true);
    }

    if (install_tangential_smoothing) {
        const auto projector = tangentialProjector(n, rec.components);
        const auto tangential_gradient =
            projector * grad(d_mesh) * projector;
        const auto tangential_test_gradient =
            projector * grad(psi) * projector;
        const auto smoothing_weight = bc::toScalarExpr(
            bc.tangential_mesh_penalty,
            freeSurfaceValueName(
                "ns_free_surface_tangential_mesh_smoothing_weight", bc));
        const auto tangential_smoothing_residual =
            integrateOnFreeSurface(
                smoothing_weight *
                    inner(tangential_gradient,
                          tangential_test_gradient),
                bc,
                /*ale_enabled=*/true);
        residual = residual.isValid()
                       ? residual + tangential_smoothing_residual
                       : tangential_smoothing_residual;
    }

    if (install_tangential_prescription) {
        auto target_components = bc::toVectorExpr(
            bc.prescribed_tangential_mesh_velocity,
            rec.components,
            "ns_free_surface_tangential_mesh_velocity",
            bc.boundary_marker,
            bc::ComponentValueNameStyle::Component);
        const auto target =
            FormExpr::asVector(std::move(target_components));
        const auto rate_gap = dt(d_mesh) - target;
        const auto tangential_work =
            inner(rate_gap, psi) -
            normalTrace(rate_gap, n) * normalTrace(psi, n);
        const auto tangential_penalty = bc::toScalarExpr(
            bc.tangential_mesh_penalty,
            freeSurfaceValueName(
                "ns_free_surface_tangential_mesh_penalty", bc));
        const auto tangential_residual = integrateOnFreeSurface(
            tangential_penalty * tangential_work,
            bc,
            /*ale_enabled=*/true);
        residual = residual.isValid()
                       ? residual + tangential_residual
                       : tangential_residual;
    }

    auto install = base_install_options;
    install.compiler_options.use_symbolic_tangent = true;
    ale_binding.configureInstallOptions(install);
    if (install_normal_relation &&
        velocity_field != FE::INVALID_FIELD_ID) {
        install.extra_trial_fields.push_back(velocity_field);
    }

    (void)FE::systems::installFormulation(
        system,
        options.operator_tag,
        {ale_binding.mesh_displacement_field},
        residual,
        install);

    if (install_normal_relation) {
        if (velocity_field == FE::INVALID_FIELD_ID) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: fitted normal "
                "kinematics require a related fluid velocity field");
        }
        const auto mesh_source =
            "Fitted free-surface mesh normal kinematic row on marker " +
            std::to_string(bc.boundary_marker);
        FE::analysis::BoundaryConditionDescriptor mesh_descriptor;
        mesh_descriptor.primary_variable =
            FE::analysis::VariableKey::field(
                ale_binding.mesh_displacement_field);
        mesh_descriptor.related_variables.push_back(
            FE::analysis::VariableKey::field(velocity_field));
        mesh_descriptor.boundary_marker = bc.boundary_marker;
        mesh_descriptor.trace_kind =
            FE::analysis::TraceKind::NormalComponent;
        mesh_descriptor.enforcement_kind =
            FE::analysis::EnforcementKind::WeakPenalty;
        mesh_descriptor.source = mesh_source;
        system.addBoundaryConditionDescriptor(
            std::move(mesh_descriptor), options.operator_tag);

        const auto fluid_source =
            fittedFluidNormalDescriptorSource(bc);
        FE::analysis::BoundaryConditionDescriptor fluid_descriptor;
        fluid_descriptor.primary_variable =
            FE::analysis::VariableKey::field(velocity_field);
        fluid_descriptor.related_variables.push_back(
            FE::analysis::VariableKey::field(
                ale_binding.mesh_displacement_field));
        fluid_descriptor.boundary_marker = bc.boundary_marker;
        fluid_descriptor.trace_kind =
            FE::analysis::TraceKind::NormalComponent;
        fluid_descriptor.enforcement_kind =
            bc.kinematic_enforcement ==
                    FreeSurfaceKinematicEnforcement::Nitsche
                ? FE::analysis::EnforcementKind::WeakNitsche
                : bc.kinematic_enforcement ==
                          FreeSurfaceKinematicEnforcement::MeshNitsche
                      ? FE::analysis::EnforcementKind::WeakConsistent
                      : FE::analysis::EnforcementKind::WeakPenalty;
        fluid_descriptor.source = fluid_source;
        system.addBoundaryConditionDescriptor(
            std::move(fluid_descriptor), options.operator_tag);
    }

    FE::analysis::BoundaryConditionDescriptor descriptor;
    descriptor.primary_variable =
        FE::analysis::VariableKey::field(
            ale_binding.mesh_displacement_field);
    descriptor.boundary_marker = bc.boundary_marker;
    descriptor.trace_kind =
        FE::analysis::TraceKind::TangentialComponent;
    descriptor.enforcement_kind =
        install_tangential_prescription
            ? FE::analysis::EnforcementKind::WeakPenalty
            : FE::analysis::EnforcementKind::WeakConsistent;
    const auto consumer_source = fittedTangentialOperatorSource(
        bc.boundary_marker, bc.tangential_mesh_policy);
    descriptor.source = consumer_source;
    system.addBoundaryConditionDescriptor(
        std::move(descriptor), options.operator_tag);
    system.bindMeshTangentialBoundaryPolicyConsumer(
        ale_binding.mesh_displacement_field,
        bc.boundary_marker,
        systemTangentialMeshPolicy(bc.tangential_mesh_policy),
        options.operator_tag,
        consumer_source);
}

} // namespace

void IncompressibleNavierStokesVMSModule::registerOn(FE::systems::FESystem& system) const
{
    effective_configuration_artifact_.reset();
    auto preflight = preflightNavierStokes(
        system, options_, velocity_space_, pressure_space_);
    const int dim = preflight.dim;
    const bool enable_vms = preflight.enable_vms;
    const PspgPressureGradientForm pspg_pressure_gradient_form =
        preflight.pspg_pressure_gradient_form;
    const auto& ale_options = preflight.ale_options;
    const auto& effective_free_surfaces = preflight.effective_free_surfaces;
    const bool has_exterior_free_surface = preflight.has_exterior_free_surface;
    const auto active_pressure_domain =
        activePressureDomainFor(effective_free_surfaces);

    const auto field_ids =
        registerNavierStokesFields(system, options_, preflight);
    const FE::FieldId u_id = field_ids.velocity;
    const FE::FieldId p_id = field_ids.pressure;
    const FE::FieldId body_force_field_id = field_ids.body_force;

    declareFreeSurfaceDiscreteFunctionals(
        effective_free_surfaces,
        system,
        u_id,
        options_.density,
        options_.body_force,
        !options_.has_body_force_spacetime &&
            body_force_field_id == FE::INVALID_FIELD_ID,
        options_.viscosity,
        options_.viscosity_model == nullptr);

    const auto generated_active_boundary_for =
        [&system, &effective_free_surfaces](int physical_boundary_marker)
            -> std::optional<int> {
            const auto marker = sharpActiveBoundaryMarkerFor(
                physical_boundary_marker,
                effective_free_surfaces,
                system);
            if (marker.has_value()) {
                system.registerGeneratedEmbeddedInterfaceMarker(*marker);
            }
            return marker;
        };
    // Every system constraint of this module is registered through the plan,
    // in the order VertexValues < Essential < InactiveSupport < Aggregation <
    // PointLaw (see FE/Constraints/ConstraintPlan.h).
    FE::constraints::ConstraintPlan constraint_plan(system);
    struct PendingSmallCutAggregation {
        FE::geometry::CutIntegrationSide side{
            FE::geometry::CutIntegrationSide::Negative};
        int interface_marker{-1};
        FE::constraints::SmallCutAggregationGuardOptions guards{};
    };
    struct PendingActiveDomainConstraints {
        FE::constraints::LevelSetConstraintSide side{
            FE::constraints::LevelSetConstraintSide::Negative};
        std::string level_set_field_name{};
        FE::Real level_set_isovalue{0.0};
        int interface_marker{-1};
    };
    std::optional<PendingSmallCutAggregation> pending_small_cut_aggregation;
    std::optional<PendingActiveDomainConstraints>
        pending_active_domain_constraints;
    if (active_pressure_domain.has_value() &&
        isExteriorOnePhaseBoundary(
            *active_pressure_domain->boundary) &&
        active_pressure_domain->boundary->active_domain_method ==
            FreeSurfaceActiveDomainMethod::CutVolume) {
        anchorPressureOnFreeSurface(
            system,
            options_.operator_tag,
            p_id,
            *active_pressure_domain->boundary);
    }

    if (active_pressure_domain.has_value() &&
        active_pressure_domain->boundary->active_domain_method ==
            FreeSurfaceActiveDomainMethod::CutVolume) {
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
        const bool native_mesh_available = system.mesh() != nullptr;
#else
        const bool native_mesh_available = false;
#endif
        if (!native_mesh_available) {
            FE_LOG_WARNING(
                "IncompressibleNavierStokesVMSModule: skipping active-domain "
                "inactive velocity/pressure constraints because native mesh support "
                "is unavailable in this FESystem");
        } else {
            const auto side =
                active_pressure_domain->active_domain ==
                        FreeSurfaceActiveDomain::LevelSetPositive
                    ? FE::constraints::LevelSetConstraintSide::Positive
                    : FE::constraints::LevelSetConstraintSide::Negative;
            pending_active_domain_constraints =
                PendingActiveDomainConstraints{
                    .side = side,
                    .level_set_field_name =
                        active_pressure_domain->boundary
                            ->level_set_field_name,
                    .level_set_isovalue =
                        active_pressure_domain->boundary
                            ->level_set_isovalue,
                    .interface_marker =
                        active_pressure_domain->boundary
                            ->interface_marker,
                };
            // Physical essential data owns the exterior trace: the plan
            // registers the artificial inactive-side extension after every
            // strong velocity and pressure condition, so its per-DOF
            // precedence rule fills only otherwise unconstrained support.
            const auto active_side =
                freeSurfaceLevelSetGeometry(
                    *active_pressure_domain->boundary, system)
                    .side(activeDomainSide(
                        active_pressure_domain->active_domain));
            constraint_plan.inactiveSide({u_id}, active_side);
            constraint_plan.inactiveSide({p_id}, active_side);
            if (active_pressure_domain->boundary->small_cut_aggregation) {
                const auto aggregation_side =
                    active_pressure_domain->active_domain ==
                            FreeSurfaceActiveDomain::LevelSetPositive
                        ? FE::geometry::CutIntegrationSide::Positive
                        : FE::geometry::CutIntegrationSide::Negative;
                // Defer aggregation until every strong velocity/pressure
                // constraint has been registered below.  Small-cut
                // aggregation then observes essential data component by
                // component: a normal-only impermeability condition replaces
                // only the normal line, while tangential slip DOFs retain
                // their aggregation support.  The former marker-wide
                // exclusion incorrectly removed those tangential lines.
                pending_small_cut_aggregation = PendingSmallCutAggregation{
                    .side = aggregation_side,
                    .interface_marker =
                        active_pressure_domain->boundary->interface_marker,
                    .guards = active_pressure_domain->boundary
                                  ->small_cut_aggregation_guards,
                };
                // Mixed problems aggregate both spaces (see the
                // registration log below).
                constraint_plan.aggregate(
                    {u_id, p_id},
                    active_side,
                    active_pressure_domain->boundary
                        ->small_cut_aggregation_guards);
            }
        }
    }

    if (!options_.node_pressure_constraints.values.empty()) {
        constraint_plan.vertexValues(
            p_id, nodePressureValues(options_), nodePressureIdMode(options_));
    }
    constraint_plan.commitThrough(
        FE::constraints::ConstraintPhase::VertexValues);

    using namespace svmp::FE::forms;

    const auto u = StateField(u_id, *velocity_space_, options_.velocity_field_name);
    const auto p = StateField(p_id, *pressure_space_, options_.pressure_field_name);

    const auto v = TestField(u_id, *velocity_space_, "v");
    const auto q = TestField(p_id, *pressure_space_, "q");

    const auto ale_binding =
        FE::systems::resolveALEBinding(system, ale_options);

    auto normal_declarations = fittedNormalKinematicDeclarations(
        ale_binding, effective_free_surfaces, u, u_id);

    if (!system.hasOperator(options_.operator_tag)) {
        system.addOperator(options_.operator_tag);
    }

    const auto active_volume_domain =
        activeVolumeDomainFor(effective_free_surfaces);

    const auto rho = FormExpr::constant(options_.density);

    // Body force/source acceleration. The optional field is evaluated as
    // prescribed data, so it contributes to both Galerkin forcing and VMS
    // strong residual without adding unknowns to the system.
    std::vector<FormExpr> constant_body_force_components;
    constant_body_force_components.reserve(static_cast<std::size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        constant_body_force_components.push_back(
            FormExpr::constant(
                options_.body_force[static_cast<std::size_t>(d)]));
    }
    const auto constant_body_force =
        FormExpr::asVector(std::move(constant_body_force_components));
    FormExpr f = constant_body_force;
    if (options_.has_body_force_spacetime) {
        std::vector<FormExpr> source_comp;
        source_comp.reserve(static_cast<std::size_t>(dim));
        for (int d = 0; d < dim; ++d) {
            source_comp.push_back(bc::toScalarExpr(
                options_.body_force_spacetime[static_cast<std::size_t>(d)],
                "ns_body_force_spacetime_" + std::to_string(d)));
        }
        f = f + FormExpr::asVector(std::move(source_comp));
    }
    if (body_force_field_id != FE::INVALID_FIELD_ID) {
        f = f + StateField(
                    body_force_field_id,
                    *velocity_space_,
                    options_.body_force_field_name);
    }
    if (options_.rotating_frame_coriolis_enabled) {
        std::vector<FormExpr> omega_comp;
        omega_comp.reserve(3u);
        for (int d = 0; d < 3; ++d) {
            omega_comp.push_back(bc::toScalarExpr(
                options_.rotating_frame_angular_velocity[static_cast<std::size_t>(d)],
                "ns_rotating_frame_angular_velocity_" + std::to_string(d)));
        }
        f = f + FormExpr::constant(-2.0) *
                    cross(FormExpr::asVector(std::move(omega_comp)), u);
    }

    const auto eps_for_mu = sym(grad(u));
    const auto gamma_for_mu =
        sqrt(FormExpr::constant(2.0) * inner(eps_for_mu, eps_for_mu));
    FormExpr mu;
    if (options_.viscosity_model) {
        // Variable viscosity remains a tagged constitutive expression so
        // installFormulation() can publish the law from the residual DAG.
        std::shared_ptr<const FE::forms::ConstitutiveModel> viscosity_model =
            options_.viscosity_model;
        auto viscosity_metadata = FE::analysis::dynamicViscosityMetadata(
            FE::INVALID_FIELD_ID,
            options_.viscosity,
            options_.viscosity_model);
        viscosity_model = FE::constitutive::withConstitutiveLawMetadata(
            std::move(viscosity_model),
            0u,
            std::move(viscosity_metadata));
        mu = constitutive(std::move(viscosity_model), gamma_for_mu).out(0);
    } else {
        mu = FormExpr::constant(options_.viscosity);
    }

    // ALE uses relative convection u - w_mesh. Static/default paths remain unchanged.
    const auto zero = zeroVector(dim);
    const auto w_mesh = meshVelocity();
    const auto mesh_velocity = options_.enable_ale ? w_mesh : zero;
    const auto a = options_.enable_convection
                       ? (options_.enable_ale ? (u - w_mesh) : u)
                       : zero;
    const bool include_mcv =
        options_.enable_ale && options_.include_moving_control_volume_transient;
    const auto moving_volume_strong =
        include_mcv ? (div(mesh_velocity) * u) : zero;

    // Strong momentum residual (full, including dt(u)):
    //   R_m = rho*(dt(u) + grad(u)*a + chi*div(w_mesh)*u - f)
    //         + grad(p) - div(2 mu sym(grad(u)))
    // with a = u - w_mesh for ALE and chi set by the moving-control-volume option.
    const auto stress = FormExpr::constant(2.0) * mu * sym(grad(u));
    const auto pressure_gradient_residual = grad(p);
    const auto pspg_pressure_gradient_pressure =
        pspg_pressure_gradient_form == PspgPressureGradientForm::Incremental
            ? FormExpr::effectiveTimeStep() * dt(p)
            : p;
    const auto pspg_pressure_gradient_residual =
        grad(pspg_pressure_gradient_pressure);
    const auto r_m_without_pressure =
        rho * (dt(u) + grad(u) * a + moving_volume_strong - f) - div(stress);
    const auto r_m = r_m_without_pressure + pressure_gradient_residual;

    // Galerkin terms.
    const auto inertia = rho * inner(dt(u), v);
    const auto moving_volume =
        include_mcv ? rho * div(w_mesh) * inner(u, v) : FormExpr::constant(0.0);
    const auto convection = rho * inner(grad(u) * a, v);
    const auto viscous = FormExpr::constant(2.0) * mu * inner(sym(grad(u)), sym(grad(v)));
    const auto pressure = -p * div(v);
    const auto forcing = -rho * inner(f, v);

    const auto galerkin_momentum_integrand =
        inertia + moving_volume + convection + viscous + pressure + forcing;
    const auto galerkin_continuity_integrand = q * div(u);

    FormExpr active_momentum_integrand = galerkin_momentum_integrand;
    FormExpr active_continuity_integrand = galerkin_continuity_integrand;
    FormExpr vms_pspg_continuity_integrand;
    FormExpr vms_pspg_pressure_gradient_integrand;
    FormExpr vms_pspg_nonpressure_integrand;

    if (enable_vms) {
        // Residual-based VMS with static subscales:
        //   u' = -tau_M * R_m
        //   p' = -tau_C * (div u)
        // and coarse-scale stabilization terms assembled from (u', p').
        const auto eps = FormExpr::constant(options_.stabilization_epsilon);
        const auto dt_step = FormExpr::effectiveTimeStep();
        const auto ct_m = FormExpr::constant(options_.ct_m);
        const auto ct_c = FormExpr::constant(options_.ct_c);

        // Element metric tensor Kxi = J^{-T} J^{-1}. FE Forms exposes Jinv()
        // with the active physical dimension, so 2D contractions do not include
        // a dummy frame-thickness component.
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

        active_momentum_integrand =
            inertia + moving_volume + convection_adv + viscous + pressure_adv +
            forcing + supg + cross_stress;

        // Continuity: Galerkin + VMS (PSPG-like).  The pressure-gradient
        // part is a separate kernel group (installed below).  The unit
        // factors keep the expression trees, and so the kernels, unchanged.
        vms_pspg_pressure_gradient_integrand =
            FormExpr::constant(1.0) * FormExpr::constant(1.0) *
            inner(grad(q), tau_m * pspg_pressure_gradient_residual);
        vms_pspg_nonpressure_integrand =
            FormExpr::constant(1.0) *
            inner(grad(q), tau_m * r_m_without_pressure);
        vms_pspg_continuity_integrand =
            vms_pspg_pressure_gradient_integrand +
            vms_pspg_nonpressure_integrand;
        active_continuity_integrand =
            galerkin_continuity_integrand + vms_pspg_nonpressure_integrand;
    }

    FormExpr momentum_form =
        integrateOnActiveVolume(active_momentum_integrand, active_volume_domain);
    FormExpr continuity_form =
        integrateOnActiveVolume(active_continuity_integrand, active_volume_domain);
    const auto galerkin_momentum_form =
        integrateOnActiveVolume(
            galerkin_momentum_integrand, active_volume_domain);
    const auto galerkin_continuity_form =
        integrateOnActiveVolume(
            galerkin_continuity_integrand, active_volume_domain);
    const auto convection_work_form =
        integrateOnActiveVolume(
            moving_volume + convection, active_volume_domain);
    const auto pressure_work_form =
        integrateOnActiveVolume(pressure, active_volume_domain);
    const auto continuity_work_form = galerkin_continuity_form;
    const bool has_nonconservative_body_force =
        options_.has_body_force_spacetime ||
        body_force_field_id != FE::INVALID_FIELD_ID ||
        options_.rotating_frame_coriolis_enabled;
    FormExpr nonconservative_body_force_work_form;
    if (has_nonconservative_body_force) {
        const auto conservative_body_force_residual =
            -rho * inner(constant_body_force, v);
        nonconservative_body_force_work_form =
            integrateOnActiveVolume(
                forcing - conservative_body_force_residual,
                active_volume_domain);
    }
    FormExpr vms_pspg_pressure_gradient_form;
    if (vms_pspg_pressure_gradient_integrand.isValid()) {
        vms_pspg_pressure_gradient_form =
            integrateOnActiveVolume(
                vms_pspg_pressure_gradient_integrand,
                active_volume_domain);
    }
    FormExpr active_continuity_diagnostic_form = continuity_form;
    if (vms_pspg_pressure_gradient_form.isValid()) {
        active_continuity_diagnostic_form =
            active_continuity_diagnostic_form + vms_pspg_pressure_gradient_form;
    }
    FormExpr vms_pspg_momentum_work_form;
    FormExpr vms_pspg_continuity_work_form;
    if (enable_vms) {
        vms_pspg_momentum_work_form =
            momentum_form - galerkin_momentum_form;
        vms_pspg_continuity_work_form =
            continuity_form - galerkin_continuity_form;
    }
    FormExpr pressure_ghost_penalty_form;
    FormExpr weak_boundary_momentum_work_form;
    FormExpr weak_boundary_continuity_work_form;
    FormExpr free_surface_conservative_pressure_form;
    FormExpr free_surface_conservative_external_pressure_form;
    FormExpr free_surface_conservative_surface_energy_form;
    FormExpr free_surface_conservative_gravitational_potential_form;
    const bool conservative_balance_diagnostic_supported =
        conservativeBalanceDiagnosticSupported(
            effective_free_surfaces, has_exterior_free_surface);
    if (conservative_balance_diagnostic_supported) {
        // The pressure part is the weak pressure work on the active liquid.
        // Prescribed exterior-pressure work is appended on each free surface.
        free_surface_conservative_pressure_form =
            integrateOnActiveVolume(pressure, active_volume_domain);
        // This is exactly the constant body-acceleration term appearing in
        // the production Galerkin momentum residual.  Its sign is the first
        // variation of -rho int(g dot (x-x_ref)) over the active liquid.
        free_surface_conservative_gravitational_potential_form =
            integrateOnActiveVolume(
                -rho * inner(constant_body_force, v),
                active_volume_domain);
    }
    // ---------------------------------------------------------------------
    // Boundary conditions (installer + factories)
    // ---------------------------------------------------------------------

    if (!options_.coupled_outflow_rcr.empty() || !options_.coupled_outflow_rcrcr.empty()) {
        setBoundaryReductionCompilerOptions(system, u_id, options_.jit_policy);
    }

    FE::systems::BoundaryConditionManager bc_manager;

    // Weak velocity Dirichlet is applied directly to the Forms residual (affects both momentum and continuity).
    // Reserve the marker here so validate() catches conflicts with other BC types.
    bc_manager.install(options_.velocity_dirichlet_weak, Factories::reserveMarker);

    auto free_surface_kinematics_install =
        physicsInstallOptions(options_.jit_policy);
    for (const auto& effective_bc : effective_free_surfaces) {
        if (isExteriorOnePhaseBoundary(effective_bc)) {
            applyFreeSurfaceContactLineConstraints(
                constraint_plan, effective_bc, ale_binding, dim);
            registerFreeSurfacePrescribedAngleGeometry(
                system, effective_bc);
            applyDynamicContactAngleResidual(
                momentum_form,
                system,
                effective_bc,
                u,
                v,
                mu,
                dim,
                conservative_balance_diagnostic_supported
                    ? &free_surface_conservative_surface_energy_form
                    : nullptr);
            if (effective_bc.implementation ==
                FreeSurfaceImplementation::FittedALE) {
                bc_manager.add(
                    std::make_unique<FE::forms::bc::ReservedBC>(
                        effective_bc.boundary_marker));
            }
            applyFreeSurfaceBoundary(
                momentum_form,
                continuity_form,
                effective_bc,
                system,
                u,
                p,
                v,
                q,
                mesh_velocity,
                mu,
                options_,
                options_.enable_ale,
                dim,
                conservative_balance_diagnostic_supported
                    ? &free_surface_conservative_pressure_form
                    : nullptr,
                conservative_balance_diagnostic_supported
                    ? &free_surface_conservative_external_pressure_form
                    : nullptr,
                conservative_balance_diagnostic_supported
                    ? &free_surface_conservative_surface_energy_form
                    : nullptr,
                &weak_boundary_momentum_work_form,
                &weak_boundary_continuity_work_form);
            installFittedFreeSurfaceMeshKinematics(
                system,
                effective_bc,
                ale_binding,
                u,
                options_,
                free_surface_kinematics_install,
                u_id);
        }
        applyFreeSurfaceCutCellStabilization(
            momentum_form,
            continuity_form,
            effective_bc,
            u,
            p,
            v,
            q,
            mu,
            options_.density,
            options_.stabilization_epsilon,
            dim,
            velocity_space_->polynomial_order(),
            pressure_space_->polynomial_order(),
            &pressure_ghost_penalty_form);
    }

    bc_manager.install(options_.traction_neumann, [&](const auto& bc) {
        return Factories::toTractionBC(
            bc, dim, generated_active_boundary_for(bc.boundary_marker));
    });
    bc_manager.install(options_.traction_robin, [&](const auto& bc) {
        return Factories::toTractionRobinBC(
            bc, dim, generated_active_boundary_for(bc.boundary_marker));
    });
    bc_manager.install(options_.pressure_outflow, [&](const auto& bc) {
        return Factories::toOutflowBC(
            bc, u, rho, generated_active_boundary_for(bc.boundary_marker));
    });
    bc_manager.install(options_.coupled_outflow_rcr, [&](const auto& bc) {
        return Factories::toCoupledOutflowBC(
            bc,
            system,
            u,
            rho,
            generated_active_boundary_for(bc.boundary_marker));
    });
    bc_manager.install(options_.coupled_outflow_rcrcr, [&](const auto& bc) {
        return Factories::toCoupledOutflowBC(
            bc,
            system,
            u,
            rho,
            generated_active_boundary_for(bc.boundary_marker));
    });
    bc_manager.install(options_.velocity_dirichlet,
                       [&](const auto& bc) {
        return Factories::toVelocityEssentialBC(bc, dim, options_.velocity_field_name);
    });

    constraint_plan.essentials([&](FE::systems::FESystem& target) {
        bc_manager.applyAll(target, momentum_form, u, v, u_id);
    });

    FE::systems::BoundaryConditionManager p_bc_manager;
    p_bc_manager.install(options_.pressure_dirichlet,
                         [&](const auto& bc) { return Factories::toPressureEssentialBC(bc, options_.pressure_field_name); });
    constraint_plan.essentials([&](FE::systems::FESystem& target) {
        p_bc_manager.applyAll(target, p_id);
    });

    if (pending_active_domain_constraints.has_value()) {
        // Physical essential data owns the exterior trace.  The artificial
        // inactive-side extension (declared above) is registered afterward
        // so its existing per-DOF precedence rule fills only otherwise
        // unconstrained support.  Aggregation remains last and therefore
        // sees the complete physical and inactive constraint set.
        const auto& pending = *pending_active_domain_constraints;
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: active-domain inactive constraints enabled") +
            " marker=" + std::to_string(pending.interface_marker) +
            " registration_order=after_physical_essential_constraints"
            " physical_trace_precedence=per_dof");
    }

    if (pending_small_cut_aggregation.has_value()) {
        // Mixed problems must aggregate both spaces: leaving the small-cut
        // pressures free while their conjugate velocities are slaved breaks
        // the local saddle point.  Registration after all essential
        // constraints is deliberate. SmallCutAggregationConstraint gathers
        // preconstrained component facts communicator-wide before installing
        // canonical lines, so even owner/non-owner boundary visibility cannot
        // turn a partial strong condition into an owner-wins aggregation line.
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: small-cut "
                        "aggregation enabled for unfitted free surface") +
            " marker=" +
            std::to_string(
                pending_small_cut_aggregation->interface_marker) +
            " registration_order=after_strong_constraints"
            " component_precedence=per_dof"
            " maximum_root_path_length=" +
            std::to_string(
                pending_small_cut_aggregation->guards
                    .maximum_root_path_length) +
            " maximum_reference_extrapolation_distance=" +
            std::to_string(
                pending_small_cut_aggregation->guards
                    .maximum_reference_extrapolation_distance) +
            " maximum_absolute_coefficient=" +
            std::to_string(
                pending_small_cut_aggregation->guards
                    .maximum_absolute_coefficient) +
            " maximum_row_l1_norm=" +
            std::to_string(
                pending_small_cut_aggregation->guards
                    .maximum_row_l1_norm) +
            " rootless_fallback=" +
            std::string(pending_small_cut_aggregation->guards
                                .rootless_fallback
                            ? "1"
                            : "0") +
            " diagnostic=small_cut_aggregation_registration"
            " velocity_ghost_penalty=skipped_by_aggregation");
    }

    // Strong Ren-E contact-point law (opt-in): registered after every other
    // velocity constraint so that it can expand aggregated DOFs.
    for (const auto& fs_bc : options_.free_surface) {
        for (const auto& contact_line : fs_bc.contact_lines) {
            const auto* dynamic =
                std::get_if<FreeSurfaceContactLine::DynamicRenE>(
                    &contact_line.configuration);
            if (dynamic == nullptr || !dynamic->strong_law) {
                continue;
            }
            const auto wall_normal = normalizedWallNormal(contact_line);
            const FE::Real mobility = constantScalarValueOrThrow(
                contactLineMobility(contact_line),
                "Strong DynamicContactAngle mobility");
            const FE::Real gamma = constantScalarValueOrThrow(
                fs_bc.surface_tension, "Strong DynamicContactAngle surface tension");
            const FE::Real cos_e = std::cos(constantScalarValueOrThrow(
                contactLineAngleRadians(contact_line),
                "Strong DynamicContactAngle equilibrium angle"));
            FE::constraints::ContactPointVelocityLaw law;
            law.wall_boundary_markers = {contactLineWallBoundaryMarker(contact_line)};
            law.wall_normal = {wall_normal[0], wall_normal[1], wall_normal[2]};
            law.level_set_field_name = fs_bc.level_set_field_name;
            law.isovalue = fs_bc.level_set_isovalue;
            law.contact_speed = [mobility, gamma, cos_e](FE::Real cos_d) {
                return mobility * gamma * (cos_e - cos_d);
            };
            law.name = "ren_e_contact_point:" +
                       std::to_string(contactLineWallBoundaryMarker(contact_line));
            constraint_plan.contactPointVelocity(u_id, std::move(law));
            FE_LOG_INFO(
                std::string("IncompressibleNavierStokesVMSModule: registered the strong Ren-E contact-point constraint") +
                " wall_marker=" + std::to_string(contactLineWallBoundaryMarker(contact_line)) +
                " mobility=" + std::to_string(mobility) +
                " registration_order=after_small_cut_aggregation"
                " value_lag=frozen_per_outer_geometry_pass"
                " diagnostic=navier_stokes_strong_contact_point_law");
        }
    }

    // Registers the inactive-side, aggregation and point-law constraints, in
    // that order, after the essential conditions above.
    constraint_plan.commit();

    std::vector<
        FE::forms::bc::
            GeneratedBoundaryNitscheTraceFormBinding>
        generated_nitsche_trace_bindings;
    Factories::applyVelocityNitscheBCs(
        momentum_form,
        continuity_form,
        options_,
        dim,
        u,
        p,
        v,
        q,
        mu,
        generated_active_boundary_for,
        pending_small_cut_aggregation.has_value()
            ? &generated_nitsche_trace_bindings
            : nullptr,
        &weak_boundary_momentum_work_form,
        &weak_boundary_continuity_work_form);
    requireCertifiedNitscheTraces(
        options_,
        generated_active_boundary_for,
        pending_small_cut_aggregation.has_value(),
        generated_nitsche_trace_bindings);

    // Install the complete residual (momentum + continuity) via the unified
    // installFormulation() entry point.  It auto-detects the two-field mixed
    // structure and sets up per-block Jacobian kernels with optimal assembly.
    const auto residual = momentum_form + continuity_form;

    auto install = physicsInstallOptions(options_.jit_policy);
    install.compiler_options.use_symbolic_tangent = true;
    if (!options_.viscosity_model) {
        install.recordDynamicViscosity(u_id, options_.viscosity);
    }
    if (pending_small_cut_aggregation.has_value()) {
        addNitscheTraceRequests(
            install,
            generated_nitsche_trace_bindings,
            pending_small_cut_aggregation->interface_marker,
            options_.generated_boundary_nitsche_minimum_energy_ratio);
    }
    appendUnfittedFreeSurfaceLevelSetTrialFields(
        effective_free_surfaces, system, install);
    appendUnfittedFreeSurfaceCurvatureTrialFields(
        effective_free_surfaces, system, install);
    ale_binding.configureInstallOptions(install);
    (void)FE::systems::installFormulation(
        system, options_.operator_tag, {u_id, p_id}, residual, install);

    if (has_exterior_free_surface) {
        declareFreeSurfaceExternalBoundaryEnergy(system, options_);
        installFreeSurfaceResidualWork(
            system,
            options_,
            u_id,
            p_id,
            install,
            FreeSurfaceResidualWorkForms{
                .convection = convection_work_form,
                .pressure = pressure_work_form,
                .continuity = continuity_work_form,
                .nonconservative_body_force =
                    nonconservative_body_force_work_form,
                .weak_boundary_momentum = weak_boundary_momentum_work_form,
                .weak_boundary_continuity = weak_boundary_continuity_work_form,
                .vms_momentum = vms_pspg_momentum_work_form,
                .vms_continuity = vms_pspg_continuity_work_form,
                .vms_pspg_pressure_gradient = vms_pspg_pressure_gradient_form,
                .ghost_penalty = pressure_ghost_penalty_form,
            },
            FreeSurfaceResidualWorkApplicability{
                .include_mcv = include_mcv,
                .has_nonconservative_body_force =
                    has_nonconservative_body_force,
                .enable_vms = enable_vms,
            });
    }

    if (vms_pspg_pressure_gradient_form.isValid()) {
        auto direct_pspg_install = install;
        direct_pspg_install
            .generated_boundary_nitsche_trace_requests
            .clear();
        direct_pspg_install.source_component_tag =
            "navier_stokes_vms_pspg_pressure_gradient";
        if (std::find(direct_pspg_install.extra_trial_fields.begin(),
                      direct_pspg_install.extra_trial_fields.end(),
                      u_id) == direct_pspg_install.extra_trial_fields.end()) {
            direct_pspg_install.extra_trial_fields.push_back(u_id);
        }
        (void)FE::systems::installFormulation(
            system,
            options_.operator_tag,
            {p_id},
            vms_pspg_pressure_gradient_form,
            direct_pspg_install);
    }

    if (conservative_balance_diagnostic_supported) {
        installConservativeBalanceDiagnosticOperators(
            system,
            u_id,
            p_id,
            install,
            free_surface_conservative_pressure_form,
            free_surface_conservative_external_pressure_form,
            free_surface_conservative_surface_energy_form,
            free_surface_conservative_gravitational_potential_form,
            pressure,
            u,
            q,
            active_volume_domain);
    }

    if (pressureRowContributionDiagnosticEnabled()) {
        installPressureRowContributionDiagnosticOperators(
            system,
            u_id,
            p_id,
            install,
            galerkin_continuity_integrand,
            active_continuity_diagnostic_form,
            vms_pspg_continuity_integrand,
            vms_pspg_pressure_gradient_integrand,
            vms_pspg_nonpressure_integrand,
            pressure_ghost_penalty_form,
            active_volume_domain);
    }

    declareFittedNormalKinematics(
        system,
        options_.operator_tag,
        std::move(normal_declarations),
        effective_free_surfaces);

    effective_configuration_artifact_ = makeEffectiveConfigurationArtifact(
        options_,
        effective_free_surfaces,
        system,
        dim,
        enable_vms,
        pspg_pressure_gradient_form);
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
