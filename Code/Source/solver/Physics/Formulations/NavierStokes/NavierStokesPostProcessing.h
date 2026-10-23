/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_POSTPROCESSING_H
#define SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_POSTPROCESSING_H

/**
 * @file NavierStokesPostProcessing.h
 * @brief What the Navier-Stokes formulation states besides its residual
 *
 * The energy declarations of the free surfaces and of the external boundary,
 * the residual-work channels (exact additive groups of the installed
 * residual, each its own operator) and the opt-in diagnostic operators (the
 * conservative split and the pressure-row parts).  The module and the BC
 * factories collect the groups while they state the terms and pass them in.
 */

#include "Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"

#include "FE/Core/FEException.h"
#include "FE/Core/Logger.h"
#include "FE/Core/Types.h"
#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Geometry/CutQuadrature.h"
#include "FE/Systems/DiagnosticRegistry.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/FunctionalRecords.h"

#include <algorithm>
#include <array>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {
namespace post {

// ---------------------------------------------------------------------------
// The groups the module and the BC factories collect while they state the terms
// ---------------------------------------------------------------------------

/// The forms of the residual-work channels: exact additive groups of the
/// installed residual.
struct FreeSurfaceResidualWorkForms {
    FE::forms::FormExpr convection;                 // moving control volume + convection
    FE::forms::FormExpr pressure;                   // -p div v
    FE::forms::FormExpr continuity;                 // q div u
    FE::forms::FormExpr nonconservative_body_force; // forcing minus its constant part
    FE::forms::FormExpr weak_boundary_momentum;     // weak velocity Nitsche rows
    FE::forms::FormExpr weak_boundary_continuity;
    FE::forms::FormExpr vms_momentum;               // VMS minus Galerkin
    FE::forms::FormExpr vms_continuity;
    FE::forms::FormExpr vms_pspg_pressure_gradient;
    FE::forms::FormExpr ghost_penalty;              // cut-facet pressure penalty
};

struct FreeSurfaceResidualWorkApplicability {
    bool include_mcv{false};
    bool has_nonconservative_body_force{false};
    bool enable_vms{true};
};

/// The forms of the conservative split, collected where the formulation states
/// them (only when the diagnostic is requested and supported).
struct ConservativeBalanceForms {
    FE::forms::FormExpr pressure;                 // -p div v on the liquid, + p_ext n.v on Gamma
    FE::forms::FormExpr external_pressure;        // p_ext n.v on Gamma
    FE::forms::FormExpr surface_energy;           // gamma P:grad v, Young wall energy
    FE::forms::FormExpr gravitational_potential;  // -rho g.v on the liquid
};

/// Adds a free-surface part to the split (@p balance may be null).
inline void addExteriorPressure(ConservativeBalanceForms* balance, const FE::forms::FormExpr& form)
{
    if (balance != nullptr) {
        balance->pressure = balance->pressure.isValid() ? balance->pressure + form : form;
        balance->external_pressure =
            balance->external_pressure.isValid() ? balance->external_pressure + form : form;
    }
}
inline void addSurfaceEnergy(ConservativeBalanceForms* balance, const FE::forms::FormExpr& form)
{
    if (balance != nullptr) {
        balance->surface_energy = balance->surface_energy.isValid() ? balance->surface_energy + form : form;
    }
}

// ---------------------------------------------------------------------------
// Energy declarations of the free surfaces for the accepted-stage energy ledger:
// the discrete surface, wall (Young) and Ren-E functionals of the unfitted free
// surfaces (kind and coefficients) with their preflight, and the applicability
// of the external boundary energy.
// ---------------------------------------------------------------------------

[[nodiscard]] inline bool shouldDeclareFreeSurfaceDiscreteFunctional(
    const FreeSurfaceBoundary& boundary)
{
    return isExteriorOnePhaseBoundary(boundary) &&
           isUnfittedLevelSet(boundary) &&
           boundary.active_domain != FreeSurfaceActiveDomain::None &&
           (usesSurfaceStress(boundary) ||
            hasWallAwareContactLaw(boundary));
}

inline void validateFreeSurfaceDiscreteFunctionalPreflight(
    std::span<const FreeSurfaceBoundary> free_surfaces,
    const FE::systems::FESystem& system)
{
    const bool has_pending_declaration = std::any_of(
        free_surfaces.begin(),
        free_surfaces.end(),
        shouldDeclareFreeSurfaceDiscreteFunctional);
    if (!has_pending_declaration) {
        return;
    }
    if (!svmp::FE::systems::functionalRecords(system).acceptedFunctionalHistory().empty()) {
        throw std::invalid_argument(
            "IncompressibleNavierStokesVMSModule: free-surface discrete-functional declarations cannot be installed after accepted history has begun");
    }

    const auto existing =
        svmp::FE::systems::functionalRecords(system).functionalDeclarations();
    for (const auto& boundary : free_surfaces) {
        if (!shouldDeclareFreeSurfaceDiscreteFunctional(boundary)) {
            continue;
        }
        const auto conflict = std::find_if(
            existing.begin(),
            existing.end(),
            [&](const auto& declaration) {
                return declaration.interface_marker ==
                       boundary.interface_marker;
            });
        if (conflict != existing.end()) {
            throw std::invalid_argument(
                "IncompressibleNavierStokesVMSModule: free-surface interface marker " +
                std::to_string(boundary.interface_marker) +
                " already has discrete-functional owner '" +
                conflict->owner_component + "'");
        }
    }
}

inline void declareFreeSurfaceDiscreteFunctionals(
    const std::vector<FreeSurfaceBoundary>& free_surfaces,
    FE::systems::FESystem& system,
    FE::FieldId velocity_field,
    FE::Real density,
    const std::array<FE::Real, 3>& gravitational_acceleration,
    bool static_conservative_body_force_complete,
    FE::Real dynamic_viscosity,
    bool has_constant_dynamic_viscosity)
{
    for (const auto& bc : free_surfaces) {
        if (!shouldDeclareFreeSurfaceDiscreteFunctional(bc)) {
            continue;
        }
        FE::interfaces::FreeSurfaceDiscreteFunctionalParameters parameters;
        parameters.liquid_side =
            bc.active_domain == FreeSurfaceActiveDomain::LevelSetPositive
                ? FE::geometry::CutIntegrationSide::Positive
                : FE::geometry::CutIntegrationSide::Negative;
        parameters.surface_tension = constantScalarValueOrThrow(
            bc.surface_tension,
            "free-surface discrete-functional surface tension");
        // The pressure unknown is spatially varying and is not the scalar
        // volume multiplier in the capillary functional.  Preserve V_h as a
        // separately reported measure until a scalar multiplier is declared.
        parameters.volume_multiplier = FE::Real{0.0};
        for (const auto& contact_line : bc.contact_lines) {
            const auto kind = contactLineKind(contact_line);
            if (kind != ContactLineKind::PrescribedAngle &&
                kind != ContactLineKind::DynamicRenE) {
                continue;
            }
            parameters.young_wall_coefficients.push_back(
                FE::interfaces::FreeSurfaceYoungWallCoefficient{
                    .boundary_marker =
                        contactLineWallBoundaryMarker(contact_line),
                    .equilibrium_contact_angle_radians =
                        constantScalarValueOrThrow(
                            contactLineAngleRadians(contact_line),
                            "free-surface discrete-functional equilibrium "
                            "contact angle"),
                });
            if (kind == ContactLineKind::DynamicRenE ||
                contactLineHasNavierSlip(contact_line)) {
                const bool dynamic_law =
                    kind == ContactLineKind::DynamicRenE;
                parameters.dynamic_contact_coefficients.push_back(
                    FE::interfaces::FreeSurfaceDynamicContactCoefficient{
                        .boundary_marker =
                            contactLineWallBoundaryMarker(contact_line),
                        .equilibrium_contact_angle_radians =
                            constantScalarValueOrThrow(
                                contactLineAngleRadians(contact_line),
                                "free-surface discrete-functional dynamic "
                                "contact equilibrium angle"),
                        .law = dynamic_law
                            ? FE::interfaces::FreeSurfaceContactLaw::
                                  DynamicRenE
                            : FE::interfaces::FreeSurfaceContactLaw::
                                  PrescribedAngle,
                        .mobility = dynamic_law
                            ? constantScalarValueOrThrow(
                                  contactLineMobility(contact_line),
                                  "free-surface discrete-functional dynamic "
                                  "contact mobility")
                            : FE::Real{0.0},
                        .slip_length = constantScalarValueOrThrow(
                            contactLineSlipLength(contact_line),
                            "free-surface discrete-functional Navier slip "
                            "length"),
                        .dynamic_viscosity = dynamic_viscosity,
                    });
            }
        }
        std::optional<
            FE::interfaces::FreeSurfaceActiveVolumeDissipationParameters>
            active_volume_dissipation_parameters;
        if (has_constant_dynamic_viscosity) {
            active_volume_dissipation_parameters =
                FE::interfaces::
                    FreeSurfaceActiveVolumeDissipationParameters{
                        .liquid_side = parameters.liquid_side,
                        .dynamic_viscosity = dynamic_viscosity,
                    };
        }
        std::optional<
            FE::interfaces::FreeSurfaceExternalPressurePowerParameters>
            external_pressure_power_parameters;
        if (const auto* external_pressure =
                std::get_if<FE::Real>(&bc.external_pressure);
            external_pressure != nullptr) {
            external_pressure_power_parameters =
                FE::interfaces::
                    FreeSurfaceExternalPressurePowerParameters{
                        .liquid_side = parameters.liquid_side,
                        .external_pressure = *external_pressure,
                    };
        }
        // SurfaceStress reads no curvature field (decision D48), so the
        // declaration keeps curvature_field = INVALID_FIELD_ID.
        svmp::FE::systems::functionalRecords(system).declareFunctional(
            FE::systems::FreeSurfaceDiscreteFunctionalDeclaration{
                .interface_marker = bc.interface_marker,
                .level_set_field = resolveLevelSetFieldId(bc, system),
                .curvature_field = FE::INVALID_FIELD_ID,
                .velocity_field = velocity_field,
                .geometry_domain_id = bc.generated_interface_domain_id,
                .parameters = std::move(parameters),
                .active_volume_energy_parameters =
                    FE::interfaces::
                        FreeSurfaceActiveVolumeEnergyParameters{
                            .liquid_side =
                                bc.active_domain ==
                                        FreeSurfaceActiveDomain::
                                            LevelSetPositive
                                    ? FE::geometry::CutIntegrationSide::
                                          Positive
                                    : FE::geometry::CutIntegrationSide::
                                          Negative,
                            .density = density,
                            .gravitational_acceleration =
                                gravitational_acceleration,
                            .gravitational_reference_point =
                                {{FE::Real{0.0},
                                  FE::Real{0.0},
                                  FE::Real{0.0}}},
                        },
                .static_conservative_body_force_complete =
                    static_conservative_body_force_complete,
                .active_volume_dissipation_parameters =
                    active_volume_dissipation_parameters,
                .external_pressure_power_parameters =
                    external_pressure_power_parameters,
                .endpoint_functional_power_enabled = usesSurfaceStress(bc),
                .capillary_balance_method =
                    usesSurfaceStress(bc)
                        ? FE::systems::FreeSurfaceCapillaryBalanceMethod::
                              DiscreteEnergyVolumeStationarity
                        : FE::systems::FreeSurfaceCapillaryBalanceMethod::
                              Unselected,
                .capillary_balance_qualification =
                    usesSurfaceStress(bc)
                        ? FE::systems::
                              FreeSurfaceCapillaryBalanceQualification::
                                  PrerequisiteOnly
                        : FE::systems::
                              FreeSurfaceCapillaryBalanceQualification::
                                  Unselected,
                .owner_component =
                    "IncompressibleNavierStokesVMSModule.FreeSurfaceBoundary",
            });
    }
}

/// Applicability of the external boundary energy of a free-surface system:
/// imposed traction, open-boundary flux, homogeneous velocity data, pressure
/// Dirichlet data.
inline void declareFreeSurfaceExternalBoundaryEnergy(
    FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options)
{
    const auto velocity_data_are_homogeneous = [](const auto& bcs) {
        return std::all_of(
            bcs.begin(), bcs.end(), [](const auto& bc) {
                for (std::size_t component = 0u;
                     component < bc.active_components.size();
                     ++component) {
                    if (bc.active_components[component] &&
                        !FE::forms::bc::isZeroConstantScalarValue(
                            bc.value[component])) {
                        return false;
                    }
                }
                return true;
            });
    };
    const bool homogeneous_velocity_data =
        velocity_data_are_homogeneous(
            options.velocity_dirichlet) &&
        velocity_data_are_homogeneous(
            options.velocity_dirichlet_weak);
    const bool imposed_traction_absent =
        options.traction_neumann.empty() &&
        options.traction_robin.empty();
    const bool open_boundary_flux_absent =
        options.pressure_outflow.empty() &&
        options.coupled_outflow_rcr.empty() &&
        options.coupled_outflow_rcrcr.empty();
    svmp::FE::systems::functionalRecords(system).declareExternalBoundaryEnergy(
        FE::systems::
            FreeSurfaceExternalBoundaryEnergyDeclaration{
                .imposed_traction =
                    imposed_traction_absent
                        ? FE::systems::
                              FreeSurfaceExternalBoundaryEnergyApplicability::
                                  NotApplicable
                        : FE::systems::
                              FreeSurfaceExternalBoundaryEnergyApplicability::
                                  Unresolved,
                .open_boundary_flux =
                    open_boundary_flux_absent
                        ? FE::systems::
                              FreeSurfaceExternalBoundaryEnergyApplicability::
                                  NotApplicable
                        : FE::systems::
                              FreeSurfaceExternalBoundaryEnergyApplicability::
                                  Unresolved,
                .homogeneous_velocity_data =
                    homogeneous_velocity_data,
                .pressure_dirichlet_absent =
                    options.pressure_dirichlet.empty(),
                .owner_component =
                    "IncompressibleNavierStokesVMSModule.boundary_energy_coverage",
            });
}

// ---------------------------------------------------------------------------
// Residual-work channels of the free-surface energy ledger.  Each channel is an
// exact additive group of the installed residual, installed as its own operator
// and declared to the FESystem; pairing its constrained residual with the
// converged state times -dt gives the signed work of that group in one
// accepted step.
// ---------------------------------------------------------------------------

/// The bulk channels from the integrands and the volume forms M, C on @p liquid
/// (before any boundary term is added); @p conservative_body_force_residual is
/// the constant-gravity part -rho g.v of the forcing.
[[nodiscard]] inline FreeSurfaceResidualWorkForms bulkResidualWork(
    const NavierStokesVMSIntegrands& integrands,
    const FE::forms::FormExpr& momentum_form,
    const FE::forms::FormExpr& continuity_form,
    const FE::forms::FormExpr& vms_pspg_pressure_gradient_form,
    const FE::forms::FormExpr& conservative_body_force_residual,
    bool has_nonconservative_body_force,
    bool enable_vms,
    const FE::forms::VolumeDomain& liquid)
{
    FreeSurfaceResidualWorkForms work;
    work.convection = integrands.convection.dx(liquid);
    work.pressure = integrands.pressure.dx(liquid);
    work.continuity = integrands.galerkin_continuity.dx(liquid);
    if (has_nonconservative_body_force) {
        work.nonconservative_body_force =
            (integrands.forcing - conservative_body_force_residual).dx(liquid);
    }
    if (enable_vms) {
        work.vms_momentum = momentum_form - integrands.galerkin_momentum.dx(liquid);
        work.vms_continuity = continuity_form - work.continuity;
        work.vms_pspg_pressure_gradient = vms_pspg_pressure_gradient_form;
    }
    return work;
}

/// Installs every produced channel as its own operator (operator tag + the
/// suffix of FreeSurfaceResidualWorkOperatorSuffixes) with copies of
/// @p install, and declares produced and inapplicable channels.
inline void installFreeSurfaceResidualWork(
    FE::systems::FESystem& system,
    const IncompressibleNavierStokesVMSOptions& options,
    FE::FieldId u_id,
    FE::FieldId p_id,
    const FE::systems::FormInstallOptions& install,
    const FreeSurfaceResidualWorkForms& work,
    const FreeSurfaceResidualWorkApplicability& applicability)
{
    using FE::forms::FormExpr;
    const auto& convection_work_form = work.convection;
    const auto& pressure_work_form = work.pressure;
    const auto& continuity_work_form = work.continuity;
    const auto& nonconservative_body_force_work_form =
        work.nonconservative_body_force;
    const auto& weak_boundary_momentum_work_form = work.weak_boundary_momentum;
    const auto& weak_boundary_continuity_work_form =
        work.weak_boundary_continuity;
    const auto& vms_pspg_momentum_work_form = work.vms_momentum;
    const auto& vms_pspg_continuity_work_form = work.vms_continuity;
    const auto& vms_pspg_pressure_gradient_form =
        work.vms_pspg_pressure_gradient;
    const auto& pressure_ghost_penalty_form = work.ghost_penalty;
    const bool include_mcv = applicability.include_mcv;
    const bool has_nonconservative_body_force =
        applicability.has_nonconservative_body_force;
    const bool enable_vms = applicability.enable_vms;
    const auto work_operator_tag =
        [&](std::string_view suffix) {
            return options.operator_tag + "_" +
                std::string(suffix);
        };
    const auto install_work_operator =
        [&](FE::systems::FreeSurfaceResidualWorkChannel channel,
            std::string_view suffix,
            const FormExpr& momentum_work_form,
            const FormExpr& continuity_work_form,
            const FormExpr& supplemental_continuity_work_form,
            bool retain_generated_nitsche_trace_requests) {
            const auto operator_tag = work_operator_tag(suffix);
            bool installed = false;
            if (momentum_work_form.isValid()) {
                auto momentum_install = install;
                momentum_install.contributes_to_problem_analysis = false;
                if (!retain_generated_nitsche_trace_requests) {
                    momentum_install
                        .generated_boundary_nitsche_trace_requests
                        .clear();
                }
                momentum_install.source_component_tag = operator_tag;
                (void)FE::systems::installFormulation(
                    system,
                    operator_tag,
                    {u_id},
                    momentum_work_form,
                    momentum_install);
                installed = true;
            }
            const auto install_continuity_component =
                [&](const FormExpr& work_form) {
                    if (!work_form.isValid()) {
                        return;
                    }
                    auto continuity_install = install;
                    continuity_install.contributes_to_problem_analysis =
                        false;
                    continuity_install
                        .generated_boundary_nitsche_trace_requests
                        .clear();
                    continuity_install.source_component_tag =
                        operator_tag;
                    if (std::find(
                            continuity_install.extra_trial_fields.begin(),
                            continuity_install.extra_trial_fields.end(),
                            u_id) ==
                        continuity_install.extra_trial_fields.end()) {
                        continuity_install.extra_trial_fields.push_back(
                            u_id);
                    }
                    (void)FE::systems::installFormulation(
                        system,
                        operator_tag,
                        {p_id},
                        work_form,
                        continuity_install);
                    installed = true;
                };
            install_continuity_component(continuity_work_form);
            install_continuity_component(
                supplemental_continuity_work_form);
            FE_THROW_IF(
                !installed,
                FE::InvalidArgumentException,
                "IncompressibleNavierStokesVMSModule: produced free-surface residual-work channel has no form component");
            svmp::FE::systems::functionalRecords(system).declareWorkChannel(
                FE::systems::FreeSurfaceResidualWorkDeclaration{
                    .channel = channel,
                    .applicability = FE::systems::
                        FreeSurfaceResidualWorkApplicability::Produced,
                    .operator_tag = operator_tag,
                    .owner_component =
                        "IncompressibleNavierStokesVMSModule." +
                        std::string(
                            FE::systems::
                                freeSurfaceResidualWorkChannelName(
                                    channel)),
                });
        };
    const auto declare_inapplicable =
        [&](FE::systems::FreeSurfaceResidualWorkChannel channel) {
            svmp::FE::systems::functionalRecords(system).declareWorkChannel(
                FE::systems::FreeSurfaceResidualWorkDeclaration{
                    .channel = channel,
                    .applicability = FE::systems::
                        FreeSurfaceResidualWorkApplicability::
                            NotApplicable,
                    .operator_tag = {},
                    .owner_component =
                        "IncompressibleNavierStokesVMSModule." +
                        std::string(
                            FE::systems::
                                freeSurfaceResidualWorkChannelName(
                                    channel)),
                });
        };

    if (options.enable_convection || include_mcv) {
        install_work_operator(
            FE::systems::FreeSurfaceResidualWorkChannel::Convection,
            FreeSurfaceResidualWorkOperatorSuffixes::convection,
            convection_work_form,
            FormExpr{},
            FormExpr{},
            false);
    } else {
        declare_inapplicable(
            FE::systems::FreeSurfaceResidualWorkChannel::Convection);
    }
    install_work_operator(
        FE::systems::FreeSurfaceResidualWorkChannel::
            PressureContinuity,
        FreeSurfaceResidualWorkOperatorSuffixes::
            pressure_continuity,
        pressure_work_form,
        continuity_work_form,
        FormExpr{},
        false);
    if (has_nonconservative_body_force) {
        install_work_operator(
            FE::systems::FreeSurfaceResidualWorkChannel::
                NonconservativeBodyForce,
            FreeSurfaceResidualWorkOperatorSuffixes::
                nonconservative_body_force,
            nonconservative_body_force_work_form,
            FormExpr{},
            FormExpr{},
            false);
    } else {
        declare_inapplicable(
            FE::systems::FreeSurfaceResidualWorkChannel::
                NonconservativeBodyForce);
    }
    if (weak_boundary_momentum_work_form.isValid() ||
        weak_boundary_continuity_work_form.isValid()) {
        install_work_operator(
            FE::systems::FreeSurfaceResidualWorkChannel::WeakBoundary,
            FreeSurfaceResidualWorkOperatorSuffixes::weak_boundary,
            weak_boundary_momentum_work_form,
            weak_boundary_continuity_work_form,
            FormExpr{},
            true);
    } else {
        declare_inapplicable(
            FE::systems::FreeSurfaceResidualWorkChannel::WeakBoundary);
    }
    if (enable_vms) {
        install_work_operator(
            FE::systems::FreeSurfaceResidualWorkChannel::VmsPspg,
            FreeSurfaceResidualWorkOperatorSuffixes::vms_pspg,
            vms_pspg_momentum_work_form,
            vms_pspg_continuity_work_form,
            vms_pspg_pressure_gradient_form,
            false);
    } else {
        declare_inapplicable(
            FE::systems::FreeSurfaceResidualWorkChannel::VmsPspg);
    }
    declare_inapplicable(
        FE::systems::FreeSurfaceResidualWorkChannel::CutStabilization);
    if (pressure_ghost_penalty_form.isValid()) {
        install_work_operator(
            FE::systems::FreeSurfaceResidualWorkChannel::GhostPenalty,
            FreeSurfaceResidualWorkOperatorSuffixes::ghost_penalty,
            FormExpr{},
            pressure_ghost_penalty_form,
            FormExpr{},
            false);
    } else {
        declare_inapplicable(
            FE::systems::FreeSurfaceResidualWorkChannel::GhostPenalty);
    }
    FE_LOG_INFO(
        "IncompressibleNavierStokesVMSModule: diagnostic=free_surface_residual_work_operators status=declared channels=convection|pressure_continuity|nonconservative_body_force|weak_boundary|vms_pspg|cut_stabilization|ghost_penalty contract=negative_dt_times_constrained_operator_stage_pairing");
}

// ---------------------------------------------------------------------------
// Opt-in diagnostic operators of the residual, installed after the production
// operators, only when requested: the conservative split read by the
// static-capillary certificate, by the options (conservative_balance_diagnostics,
// set for the static-capillary initialization) or the environment switch
// SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC; the continuity-row parts
// used by the cut-stability tests, by the environment switch
// SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC.
// ---------------------------------------------------------------------------

/// The diagnostic sets of the module (FE::systems::DiagnosticRegistry).
inline constexpr std::string_view kPressureRowContributionDiagnostics =
    "navier_stokes_pressure_row_contribution";
inline constexpr std::string_view kConservativeBalanceDiagnostics =
    "navier_stokes_free_surface_conservative_balance";

/// Declares both sets with their environment switches.
inline void declareNavierStokesDiagnosticSets(FE::systems::DiagnosticRegistry& registry)
{
    registry.declareSet(
        std::string(kPressureRowContributionDiagnostics),
        {"SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC",
         "SVMP_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC"});
    // The equilibrium spellings remain explicit compatibility aliases for
    // qualification scripts written while the diagnostic was being designed.
    registry.declareSet(
        std::string(kConservativeBalanceDiagnostics),
        {"SVMP_NS_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC",
         "SVMP_FREE_SURFACE_CONSERVATIVE_BALANCE_DIAGNOSTIC",
         "SVMP_NS_FREE_SURFACE_EQUILIBRIUM_DIAGNOSTIC",
         "SVMP_FREE_SURFACE_EQUILIBRIUM_DIAGNOSTIC"});
}

/// The conservative-balance diagnostic is requested (by the options or the
/// environment switch) and every free surface has a discrete-energy traction
/// with a constant nonzero gamma (warns when it is requested but not
/// supported).
[[nodiscard]] inline bool conservativeBalanceDiagnosticSupported(
    const FE::systems::DiagnosticRegistry& registry,
    const std::vector<FreeSurfaceBoundary>& effective_free_surfaces,
    bool has_exterior_free_surface,
    bool requested_by_options)
{
    // The environment switch only adds observation: it cannot withdraw the
    // operators that the configured static-capillary certificate reads.
    const bool conservative_balance_diagnostic_requested =
        requested_by_options ||
        registry.enabled(kConservativeBalanceDiagnostics);
    const bool conservative_balance_has_surface_energy =
        std::any_of(
            effective_free_surfaces.begin(),
            effective_free_surfaces.end(),
            [](const FreeSurfaceBoundary& bc) {
                return isExteriorOnePhaseBoundary(bc) &&
                       usesSurfaceStress(bc) &&
                       FE::forms::bc::isConstantScalarValue(
                           bc.surface_tension) &&
                       !FE::forms::bc::isZeroConstantScalarValue(
                           bc.surface_tension);
            });
    const bool conservative_balance_all_discrete_energy_tractions =
        has_exterior_free_surface &&
        std::all_of(
            effective_free_surfaces.begin(),
            effective_free_surfaces.end(),
            [](const FreeSurfaceBoundary& bc) {
                return isInternalMaterialInterfaceVolume(bc) ||
                       (usesSurfaceStress(bc) &&
                        FE::forms::bc::isConstantScalarValue(
                            bc.surface_tension));
            });
    const bool conservative_balance_diagnostic_supported =
        conservative_balance_diagnostic_requested &&
        conservative_balance_has_surface_energy &&
        conservative_balance_all_discrete_energy_tractions;
    if (!conservative_balance_diagnostic_supported &&
        conservative_balance_diagnostic_requested) {
        FE_LOG_WARNING(
            "IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_free_surface_conservative_balance_operators"
            " status=skipped"
            " reason=requires_all_free_surfaces_discrete_energy_traction_and_nonzero_constant_surface_tension");
    }
    return conservative_balance_diagnostic_supported;
}

/// The operators of the conservative split
/// (FreeSurfaceConservativeBalanceDiagnosticOperators), on the production rules.
inline void installConservativeBalanceDiagnosticOperators(
    FE::systems::DiagnosticRegistry& registry,
    FE::systems::FESystem& system,
    FE::FieldId u_id,
    FE::FieldId p_id,
    const FE::systems::FormInstallOptions& install,
    const FE::forms::FormExpr& free_surface_conservative_pressure_form,
    const FE::forms::FormExpr& free_surface_conservative_external_pressure_form,
    const FE::forms::FormExpr& free_surface_conservative_surface_energy_form,
    const FE::forms::FormExpr&
        free_surface_conservative_gravitational_potential_form,
    const FE::forms::FormExpr& pressure,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& q,
    const std::optional<ActiveVolumeDomain>& active_volume_domain)
{
    using namespace svmp::FE::forms;
    FE_THROW_IF(
        !free_surface_conservative_pressure_form.isValid() ||
            !free_surface_conservative_external_pressure_form.isValid() ||
            !free_surface_conservative_surface_energy_form.isValid() ||
            !free_surface_conservative_gravitational_potential_form
                 .isValid(),
        FE::InvalidArgumentException,
        "IncompressibleNavierStokesVMSModule: requested free-surface conservative-balance diagnostics did not construct pressure, prescribed exterior-pressure, surface-energy, and gravitational-potential virtual-work forms");

    const auto physical_potential_form =
        free_surface_conservative_surface_energy_form +
        free_surface_conservative_gravitational_potential_form;
    const auto pressure_representability_load_form =
        free_surface_conservative_external_pressure_form +
        physical_potential_form;
    const auto conservative_balance_form =
        free_surface_conservative_pressure_form +
        physical_potential_form;
    // This deliberately contains only the pressure/velocity adjoint pair.
    // With pressure = -p*div(v), the two blocks are
    //   -int p div(v)  and  -int q div(u),
    // hence the assembled matrix is symmetric with identically zero
    // velocity-velocity and pressure-pressure blocks.  That symmetry is
    // the backend-independent transpose action used by the matrix-free
    // LSQR representability diagnostic in NewtonSolver.
    const auto pressure_representability_pair_form =
        integrateOnActiveVolume(pressure, active_volume_domain) -
        integrateOnActiveVolume(q * div(u), active_volume_domain);
    const auto install_diagnostic =
        [&](std::string_view operator_tag,
            const FormExpr& diagnostic_form) {
            (void)registry.install(
                system,
                kConservativeBalanceDiagnostics,
                std::string(operator_tag),
                {u_id},
                diagnostic_form,
                install,
                {.source_component_tag = std::string(operator_tag)});
        };
    install_diagnostic(
        FreeSurfaceConservativeBalanceDiagnosticOperators::
            pressure_virtual_work,
        free_surface_conservative_pressure_form);
    install_diagnostic(
        FreeSurfaceConservativeBalanceDiagnosticOperators::
            surface_energy_virtual_work,
        free_surface_conservative_surface_energy_form);
    install_diagnostic(
        FreeSurfaceConservativeBalanceDiagnosticOperators::
            gravitational_potential_virtual_work,
        free_surface_conservative_gravitational_potential_form);
    install_diagnostic(
        FreeSurfaceConservativeBalanceDiagnosticOperators::
            physical_potential_virtual_work,
        physical_potential_form);
    install_diagnostic(
        FreeSurfaceConservativeBalanceDiagnosticOperators::
            pressure_representability_load_virtual_work,
        pressure_representability_load_form);
    install_diagnostic(
        FreeSurfaceConservativeBalanceDiagnosticOperators::
            conservative_balance,
        conservative_balance_form);

    // The pair is intentionally only a pressure/velocity operator.
    // Generated geometry is frozen external state for this diagnostic;
    // inherited level-set/curvature/ALE extra trial fields would add blocks
    // that are outside [0,G;G^T,0].
    (void)registry.install(
        system,
        kConservativeBalanceDiagnostics,
        std::string(
            FreeSurfaceConservativeBalanceDiagnosticOperators::
                pressure_representability_pair),
        {u_id, p_id},
        pressure_representability_pair_form,
        install,
        {.source_component_tag = std::string(
             FreeSurfaceConservativeBalanceDiagnosticOperators::
                 pressure_representability_pair),
         .clear_extra_trial_fields = true});

    FE_LOG_INFO(
        std::string(
            "IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_free_surface_conservative_balance_operators") +
        " status=installed" +
        " pressure_operator='" +
        std::string(
            FreeSurfaceConservativeBalanceDiagnosticOperators::
                pressure_virtual_work) +
        "' surface_energy_operator='" +
        std::string(
            FreeSurfaceConservativeBalanceDiagnosticOperators::
                surface_energy_virtual_work) +
        "' gravitational_potential_operator='" +
        std::string(
            FreeSurfaceConservativeBalanceDiagnosticOperators::
                gravitational_potential_virtual_work) +
        "' physical_potential_operator='" +
        std::string(
            FreeSurfaceConservativeBalanceDiagnosticOperators::
                physical_potential_virtual_work) +
        "' pressure_representability_load_operator='" +
        std::string(
            FreeSurfaceConservativeBalanceDiagnosticOperators::
                pressure_representability_load_virtual_work) +
        "' balance_operator='" +
        std::string(
            FreeSurfaceConservativeBalanceDiagnosticOperators::
                conservative_balance) +
        "' pressure_representability_pair_operator='" +
        std::string(
            FreeSurfaceConservativeBalanceDiagnosticOperators::
                pressure_representability_pair) +
        "' pressure_terms=active_volume_minus_p_div_v_plus_external_pressure_n_dot_v" +
        " surface_energy_terms=surface_area_variation_plus_young_wall_energy" +
        " gravitational_potential_terms=active_volume_minus_density_g_dot_v" +
        " pressure_representability_load_terms=prescribed_external_pressure_plus_surface_energy_plus_gravitational_potential" +
        " excluded_terms=line_friction_and_wetted_wall_navier_dissipation" +
        " scope=pressure_and_physical_potential_first_variations_only" +
        " total_momentum_equilibrium_claimed=0" +
        " qualification=diagnostic_only_no_time_discrete_energy_claim");
}

/// The continuity-row parts used by the cut-stability tests
/// (equations_diagnostic_ns_*).
inline void installPressureRowContributionDiagnosticOperators(
    FE::systems::DiagnosticRegistry& registry,
    FE::systems::FESystem& system,
    FE::FieldId u_id,
    FE::FieldId p_id,
    const FE::systems::FormInstallOptions& install,
    const FE::forms::FormExpr& galerkin_continuity_integrand,
    const FE::forms::FormExpr& active_continuity_diagnostic_form,
    const FE::forms::FormExpr& vms_pspg_continuity_integrand,
    const FE::forms::FormExpr& vms_pspg_pressure_gradient_integrand,
    const FE::forms::FormExpr& vms_pspg_nonpressure_integrand,
    const FE::forms::FormExpr& pressure_ghost_penalty_form,
    const std::optional<ActiveVolumeDomain>& active_volume_domain)
{
    const FE::systems::DiagnosticOperatorOptions with_velocity{.extra_trial_fields = {u_id}};
    const auto install_row = [&](const char* operator_tag, const FE::forms::FormExpr& form) {
        (void)registry.install(system, kPressureRowContributionDiagnostics, operator_tag, {p_id}, form,
                               install, with_velocity);
    };
    FE_LOG_INFO(
        "IncompressibleNavierStokesVMSModule: diagnostic=navier_stokes_pressure_row_contribution_operators"
        " installing=1"
        " ops=equations_diagnostic_ns_galerkin_continuity|equations_diagnostic_ns_active_continuity|equations_diagnostic_ns_vms_pspg|equations_diagnostic_ns_vms_pspg_pressure_gradient|equations_diagnostic_ns_vms_pspg_nonpressure|equations_diagnostic_ns_pressure_ghost_penalty");
    install_row("equations_diagnostic_ns_galerkin_continuity",
                integrateOnActiveVolume(galerkin_continuity_integrand, active_volume_domain));
    install_row("equations_diagnostic_ns_active_continuity", active_continuity_diagnostic_form);
    if (vms_pspg_continuity_integrand.isValid()) {
        install_row("equations_diagnostic_ns_vms_pspg",
                    integrateOnActiveVolume(vms_pspg_continuity_integrand, active_volume_domain));
    }
    if (vms_pspg_pressure_gradient_integrand.isValid()) {
        install_row("equations_diagnostic_ns_vms_pspg_pressure_gradient",
                    integrateOnActiveVolume(vms_pspg_pressure_gradient_integrand, active_volume_domain));
    }
    if (vms_pspg_nonpressure_integrand.isValid()) {
        install_row("equations_diagnostic_ns_vms_pspg_nonpressure",
                    integrateOnActiveVolume(vms_pspg_nonpressure_integrand, active_volume_domain));
    }
    if (pressure_ghost_penalty_form.isValid()) {
        install_row("equations_diagnostic_ns_pressure_ghost_penalty", pressure_ghost_penalty_form);
    }
}

/// The opt-in diagnostic operators, after the production operators: the
/// conservative split when @p balance is set, and the pressure-row parts
/// when their set is switched on.
inline void installNavierStokesDiagnosticOperators(
    FE::systems::DiagnosticRegistry& registry,
    FE::systems::FESystem& system,
    FE::FieldId u_id,
    FE::FieldId p_id,
    const FE::systems::FormInstallOptions& install,
    const NavierStokesVMSIntegrands& integrands,
    const ConservativeBalanceForms* balance,
    const FE::forms::FormExpr& active_continuity,
    const FE::forms::FormExpr& pressure_ghost_penalty_form,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& q,
    const std::optional<ActiveVolumeDomain>& active_volume_domain)
{
    const auto& r = integrands;
    if (balance != nullptr) {
        installConservativeBalanceDiagnosticOperators(
            registry, system, u_id, p_id, install, balance->pressure, balance->external_pressure,
            balance->surface_energy, balance->gravitational_potential, r.pressure, u, q, active_volume_domain);
    }
    if (registry.enabled(kPressureRowContributionDiagnostics)) {
        const auto vms_pspg = r.pspg_pressure_gradient.isValid()
                                  ? r.pspg_pressure_gradient + r.pspg_without_pressure_gradient
                                  : FE::forms::FormExpr{};
        installPressureRowContributionDiagnosticOperators(
            registry, system, u_id, p_id, install, r.galerkin_continuity, active_continuity, vms_pspg,
            r.pspg_pressure_gradient, r.pspg_without_pressure_gradient, pressure_ghost_penalty_form,
            active_volume_domain);
    }
}

} // namespace post
} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_POSTPROCESSING_H
