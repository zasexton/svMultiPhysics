/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceDiagnostics.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"

#include "FE/Core/Logger.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Forms/CutCellForms.h"
#include "FE/Geometry/FrameGeometry.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <sstream>
#include <string>
#include <variant>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

void logUnfittedFreeSurfaceMode(const FreeSurfaceBoundary& bc)
{
    const bool has_dynamic_stress =
        !FE::forms::bc::isZeroConstantScalarValue(bc.external_pressure) ||
        !FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension);
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
        freeSurfaceCurvatureTangentPolicyName(bc) +
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

void logNaturalUnfittedFreeSurface(const FreeSurfaceBoundary& bc)
{
    FE_LOG_WARNING(
        std::string("IncompressibleNavierStokesVMSModule: unfitted free surface installs no explicit dI boundary residual marker=") +
        std::to_string(bc.interface_marker) +
        " level_set_field='" + bc.level_set_field_name + "'" +
        " dynamic_stress=natural_zero"
        " kinematic_enforcement=None"
        " diagnostic=unfitted_free_surface_natural_mode");
}

void logFreeSurfaceTraction(const FreeSurfaceBoundary& bc)
{
    if (!surfaceStressActive(bc)) {
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: installed pressure-only free-surface stress") +
            " marker=" + std::to_string(freeSurfaceMarker(bc)) +
            " surface_tension_form=SurfaceStress" +
            " normal_source=integration_rule_geometry" +
            " form=p_external_n_dot_v" +
            " surface_energy_form=omitted_literal_zero_gamma" +
            " diagnostic=free_surface_pressure_only_surface_stress");
    } else {
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
    if (usesSemiImplicitNormalIncrement(bc)) {
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
            " qualification=Candidate" +
            " diagnostic=free_surface_semi_implicit_normal_increment");
    }
}

void logContactLineTerms(
    const FreeSurfaceBoundary& bc,
    const FreeSurfaceContactLine& contact_line,
    int contact_marker,
    int active_wall_marker)
{
    if (contactLineKind(contact_line) == ContactLineKind::PrescribedAngle) {
        if (FE::forms::bc::isZeroConstantScalarValue(bc.surface_tension)) {
            return;
        }
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: installed prescribed-angle wall-energy momentum term") +
            " interface_marker=" + std::to_string(bc.interface_marker) +
            " contact_marker=" + std::to_string(contact_marker) +
            " surface_tension_form=" + surfaceTensionFormName(bc) +
            " contact_force_discretization="
            "surface_conormal_plus_equilibrium_wall_energy"
            " diagnostic=navier_stokes_prescribed_contact_wall_energy");
        if (contactLineHasNavierSlip(contact_line)) {
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
        return;
    }
    const bool strong_contact_law =
        std::get<FreeSurfaceContactLine::DynamicRenE>(
            contact_line.configuration).strong_law;
    if (strong_contact_law) {
        FE_LOG_INFO(
            std::string("IncompressibleNavierStokesVMSModule: DynamicContactAngle uses the strong contact-point law") +
            " contact_marker=" + std::to_string(contact_marker) +
            " Contact_line_law_enforcement=Strong line_friction_and_young_term=replaced_by_constraint_reaction" +
            " diagnostic=navier_stokes_dynamic_contact_strong_law");
    }
    FE_LOG_INFO(
        std::string("IncompressibleNavierStokesVMSModule: installed coupled dissipative Ren--E DynamicContactAngle") +
        " interface_marker=" + std::to_string(bc.interface_marker) +
        " contact_marker=" + std::to_string(contact_marker) +
        " wall_marker=" +
        std::to_string(contactLineWallBoundaryMarker(contact_line)) +
        " convention=theta_through_liquid_n_outward_liquid_nw_fluid_to_solid" +
        " law=xi_Vcl_equals_gamma_cos_thetae_minus_cos_thetad" +
        " surface_tension_form=" + surfaceTensionFormName(bc) +
        " contact_force_discretization="
        "surface_conormal_plus_equilibrium_wall_energy"
        " active_wall_marker=" +
        std::to_string(active_wall_marker) +
        " wetted_wall_domain=sharp_generated_active_boundary" +
        " diagnostic=navier_stokes_dynamic_contact_angle");
}

void logCutCellStabilization(
    const FreeSurfaceBoundary& bc,
    int velocity_polynomial_order,
    int pressure_polynomial_order)
{
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
    constexpr FE::Real kCutPenaltyTransientCalibration{0.01};
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
}

void logActiveDomainConstraints(
    const FreeSurfaceBoundary* liquid_boundary,
    bool small_cut_aggregation)
{
    if (liquid_boundary == nullptr) {
        return;
    }
    // Physical essential data owns the exterior trace.  The artificial
    // inactive-side extension is registered afterward so its existing
    // per-DOF precedence rule fills only otherwise unconstrained support.
    // Aggregation remains last and therefore sees the complete physical and
    // inactive constraint set.
    FE_LOG_INFO(
        std::string("IncompressibleNavierStokesVMSModule: active-domain inactive constraints enabled") +
        " marker=" + std::to_string(liquid_boundary->interface_marker) +
        " registration_order=after_physical_essential_constraints"
        " physical_trace_precedence=per_dof");
    if (!small_cut_aggregation) {
        return;
    }
    // Mixed problems must aggregate both spaces: leaving the small-cut
    // pressures free while their conjugate velocities are slaved breaks the
    // local saddle point.  Registration after all essential constraints is
    // deliberate. SmallCutAggregationConstraint gathers preconstrained
    // component facts communicator-wide before installing canonical lines,
    // so even owner/non-owner boundary visibility cannot turn a partial
    // strong condition into an owner-wins aggregation line.
    const auto& guards = liquid_boundary->small_cut_aggregation_guards;
    FE_LOG_INFO(
        std::string("IncompressibleNavierStokesVMSModule: small-cut "
                    "aggregation enabled for unfitted free surface") +
        " marker=" +
        std::to_string(liquid_boundary->interface_marker) +
        " registration_order=after_strong_constraints"
        " component_precedence=per_dof"
        " maximum_root_path_length=" +
        std::to_string(guards.maximum_root_path_length) +
        " maximum_reference_extrapolation_distance=" +
        std::to_string(guards.maximum_reference_extrapolation_distance) +
        " maximum_absolute_coefficient=" +
        std::to_string(guards.maximum_absolute_coefficient) +
        " maximum_row_l1_norm=" +
        std::to_string(guards.maximum_row_l1_norm) +
        " rootless_fallback=" +
        std::string(guards.rootless_fallback ? "1" : "0") +
        " diagnostic=small_cut_aggregation_registration"
        " velocity_ghost_penalty=skipped_by_aggregation");
}

void logStrongContactPointLaw(const FreeSurfaceContactLine& contact_line)
{
    const FE::Real mobility = constantScalarValueOrThrow(
        contactLineMobility(contact_line),
        "Strong DynamicContactAngle mobility");
    FE_LOG_INFO(
        std::string("IncompressibleNavierStokesVMSModule: registered the strong Ren-E contact-point constraint") +
        " wall_marker=" + std::to_string(contactLineWallBoundaryMarker(contact_line)) +
        " mobility=" + std::to_string(mobility) +
        " registration_order=after_small_cut_aggregation"
        " value_lag=frozen_per_outer_geometry_pass"
        " diagnostic=navier_stokes_strong_contact_point_law");
}

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
