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

#include "FE/Core/FEException.h"
#include "FE/Core/LogRecord.h"
#include "FE/Core/Types.h"
#include "FE/Forms/BoundaryConditions.h"
#include "FE/Forms/FormExpr.h"
#include "FE/Forms/IntegrationDomain.h"
#include "FE/Forms/Vocabulary.h"
#include "FE/Geometry/CutQuadrature.h"
#include "FE/Systems/DiagnosticRegistry.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/FormulationBuilder.h"
#include "FE/Systems/FunctionalForms.h"
#include "FE/Systems/FunctionalRecords.h"

#include <algorithm>
#include <array>
#include <cmath>
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

/// A log record of the formulation: "IncompressibleNavierStokesVMSModule: <message> key=value ...".
[[nodiscard]] inline FE::LogRecord navierStokesLog(std::string message = {}, FE::LogLevel level = FE::LogLevel::INFO)
{
    return FE::LogRecord("IncompressibleNavierStokesVMSModule", std::move(message), level);
}

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
        balance->pressure = FE::forms::sumOf(balance->pressure, form);
        balance->external_pressure = FE::forms::sumOf(balance->external_pressure, form);
    }
}
inline void addSurfaceEnergy(ConservativeBalanceForms* balance, const FE::forms::FormExpr& form)
{
    if (balance != nullptr) {
        balance->surface_energy = FE::forms::sumOf(balance->surface_energy, form);
    }
}

// ---------------------------------------------------------------------------
// Energy declarations of the free surfaces for the accepted-stage energy ledger:
// the discrete surface, wall (Young) and Ren-E functionals of the unfitted free
// surfaces (kind and coefficients) with their preflight, and the applicability
// of the external boundary energy.
// ---------------------------------------------------------------------------

/// The reference point x_ref of the gravitational energy -rho g.(x - x_ref):
/// the declaration of the records and the declared form both read it.
inline constexpr std::array<FE::Real, 3> kGravitationalReferencePoint{{FE::Real{0.0}, FE::Real{0.0}, FE::Real{0.0}}};

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
        svmp::FE::systems::functionalRecords(system).declareFunctional(
            FE::systems::FreeSurfaceDiscreteFunctionalDeclaration{
                .interface_marker = bc.interface_marker,
                .level_set_field = resolveLevelSetFieldId(bc, system),
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
                                kGravitationalReferencePoint,
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

/// Installs every produced channel as its own operator (F.workChannel: the
/// operator tag + the suffix of FreeSurfaceResidualWorkOperatorSuffixes, with
/// the builder's options) and declares produced and inapplicable channels.
/// The momentum rows are tested with the velocity, the continuity rows with
/// the pressure (and read the velocity); only the weak-boundary momentum rows
/// keep the generated-boundary Nitsche trace requests.
inline void installResidualWork(FE::systems::FormulationBuilder& F,
                                const IncompressibleNavierStokesVMSOptions& options,
                                FE::FieldId u_id,
                                FE::FieldId p_id,
                                const FreeSurfaceResidualWorkForms& work,
                                const FreeSurfaceResidualWorkApplicability& applicability)
{
    namespace fs = FE::systems;
    using Channel = fs::FreeSurfaceResidualWorkChannel;
    using Suffix = FreeSurfaceResidualWorkOperatorSuffixes;
    using FE::forms::FormExpr;
    const auto declare = [&](Channel channel, fs::OperatorTag operator_tag) {
        const bool produced = !operator_tag.empty();
        fs::functionalRecords(F.system()).declareWorkChannel(fs::FreeSurfaceResidualWorkDeclaration{
            .channel = channel,
            .applicability = produced ? fs::FreeSurfaceResidualWorkApplicability::Produced
                                      : fs::FreeSurfaceResidualWorkApplicability::NotApplicable,
            .operator_tag = std::move(operator_tag),
            .owner_component = "IncompressibleNavierStokesVMSModule." +
                               std::string(fs::freeSurfaceResidualWorkChannelName(channel)),
        });
    };
    const auto produce = [&](Channel channel, std::string_view suffix, const FormExpr& momentum,
                             const FormExpr& continuity, const FormExpr& supplemental_continuity = {},
                             bool keep_trace_requests = false) {
        auto operator_tag =
            F.workChannel(suffix, {{.test_field = u_id, .form = momentum,
                                    .keep_generated_boundary_nitsche_trace_requests = keep_trace_requests},
                                   {.test_field = p_id, .form = continuity, .extra_trial_fields = {u_id}},
                                   {.test_field = p_id, .form = supplemental_continuity, .extra_trial_fields = {u_id}}});
        FE_THROW_IF(
            operator_tag.empty(),
            FE::InvalidArgumentException,
            "IncompressibleNavierStokesVMSModule: produced free-surface residual-work channel has no form component");
        declare(channel, std::move(operator_tag));
    };

    if (options.enable_convection || applicability.include_mcv) {
        produce(Channel::Convection, Suffix::convection, work.convection, {});
    } else {
        declare(Channel::Convection, {});
    }
    produce(Channel::PressureContinuity, Suffix::pressure_continuity, work.pressure, work.continuity);
    if (applicability.has_nonconservative_body_force) {
        produce(Channel::NonconservativeBodyForce, Suffix::nonconservative_body_force,
                work.nonconservative_body_force, {});
    } else {
        declare(Channel::NonconservativeBodyForce, {});
    }
    if (work.weak_boundary_momentum.isValid() || work.weak_boundary_continuity.isValid()) {
        produce(Channel::WeakBoundary, Suffix::weak_boundary, work.weak_boundary_momentum,
                work.weak_boundary_continuity, {}, /*keep_trace_requests=*/true);
    } else {
        declare(Channel::WeakBoundary, {});
    }
    if (applicability.enable_vms) {
        produce(Channel::VmsPspg, Suffix::vms_pspg, work.vms_momentum, work.vms_continuity,
                work.vms_pspg_pressure_gradient);
    } else {
        declare(Channel::VmsPspg, {});
    }
    declare(Channel::CutStabilization, {});
    if (work.ghost_penalty.isValid()) {
        produce(Channel::GhostPenalty, Suffix::ghost_penalty, {}, work.ghost_penalty);
    } else {
        declare(Channel::GhostPenalty, {});
    }
    F.log(navierStokesLog("residual work operators")
              .diagnostic("free_surface_residual_work_operators")
              .text("status", "declared")
              .text("channels", "convection|pressure_continuity|nonconservative_body_force|weak_boundary|vms_pspg|"
                                "cut_stabilization|ghost_penalty")
              .text("contract", "negative_dt_times_constrained_operator_stage_pairing"));
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
inline void declareDiagnosticSets(FE::systems::DiagnosticRegistry& registry)
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
        FE::logRecord(
            navierStokesLog("conservative balance operators", FE::LogLevel::WARNING)
                .diagnostic("navier_stokes_free_surface_conservative_balance_operators")
                .text("status", "skipped")
                .text("reason",
                      "requires_all_free_surfaces_discrete_energy_traction_and_nonzero_constant_surface_tension"));
    }
    return conservative_balance_diagnostic_supported;
}

/// The operators of the conservative split
/// (FreeSurfaceConservativeBalanceDiagnosticOperators), on the production rules.
inline void installConservativeBalanceDiagnosticOperators(
    FE::systems::FormulationBuilder& F,
    FE::FieldId u_id,
    FE::FieldId p_id,
    const FE::forms::FormExpr& free_surface_conservative_pressure_form,
    const FE::forms::FormExpr& free_surface_conservative_external_pressure_form,
    const FE::forms::FormExpr& free_surface_conservative_surface_energy_form,
    const FE::forms::FormExpr&
        free_surface_conservative_gravitational_potential_form,
    const FE::forms::FormExpr& pressure,
    const FE::forms::FormExpr& u,
    const FE::forms::FormExpr& q,
    const FE::forms::VolumeDomain& liquid)
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
        pressure.dx(liquid) - (q * div(u)).dx(liquid);
    const auto install_diagnostic =
        [&](std::string_view operator_tag,
            const FormExpr& diagnostic_form) {
            (void)F.diagnostic(
                kConservativeBalanceDiagnostics,
                std::string(operator_tag),
                {u_id},
                diagnostic_form,
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
    (void)F.diagnostic(
        kConservativeBalanceDiagnostics,
        std::string(
            FreeSurfaceConservativeBalanceDiagnosticOperators::
                pressure_representability_pair),
        {u_id, p_id},
        pressure_representability_pair_form,
        {.source_component_tag = std::string(
             FreeSurfaceConservativeBalanceDiagnosticOperators::
                 pressure_representability_pair),
         .clear_extra_trial_fields = true});

    using Operators = FreeSurfaceConservativeBalanceDiagnosticOperators;
    F.log(navierStokesLog("conservative balance operators")
              .diagnostic("navier_stokes_free_surface_conservative_balance_operators")
              .text("status", "installed")
              .quoted("pressure_operator", Operators::pressure_virtual_work)
              .quoted("surface_energy_operator", Operators::surface_energy_virtual_work)
              .quoted("gravitational_potential_operator", Operators::gravitational_potential_virtual_work)
              .quoted("physical_potential_operator", Operators::physical_potential_virtual_work)
              .quoted("pressure_representability_load_operator",
                      Operators::pressure_representability_load_virtual_work)
              .quoted("balance_operator", Operators::conservative_balance)
              .quoted("pressure_representability_pair_operator", Operators::pressure_representability_pair)
              .text("pressure_terms", "active_volume_minus_p_div_v_plus_external_pressure_n_dot_v")
              .text("surface_energy_terms", "surface_area_variation_plus_young_wall_energy")
              .text("gravitational_potential_terms", "active_volume_minus_density_g_dot_v")
              .text("pressure_representability_load_terms",
                    "prescribed_external_pressure_plus_surface_energy_plus_gravitational_potential")
              .text("excluded_terms", "line_friction_and_wetted_wall_navier_dissipation")
              .text("scope", "pressure_and_physical_potential_first_variations_only")
              .text("total_momentum_equilibrium_claimed", "0")
              .text("qualification", "diagnostic_only_no_time_discrete_energy_claim"));
}

/// The continuity-row parts used by the cut-stability tests
/// (equations_diagnostic_ns_*), on @p liquid.
inline void installPressureRowContributionDiagnosticOperators(
    FE::systems::FormulationBuilder& F,
    FE::FieldId u_id,
    FE::FieldId p_id,
    const FE::forms::FormExpr& galerkin_continuity_integrand,
    const FE::forms::FormExpr& active_continuity_diagnostic_form,
    const FE::forms::FormExpr& vms_pspg_continuity_integrand,
    const FE::forms::FormExpr& vms_pspg_pressure_gradient_integrand,
    const FE::forms::FormExpr& vms_pspg_nonpressure_integrand,
    const FE::forms::FormExpr& pressure_ghost_penalty_form,
    const FE::forms::VolumeDomain& liquid)
{
    const auto install_row = [&](const char* operator_tag, const FE::forms::FormExpr& form) {
        (void)F.diagnostic(kPressureRowContributionDiagnostics, operator_tag, {p_id}, form,
                           {.extra_trial_fields = {u_id}});
    };
    F.log(navierStokesLog("pressure-row contribution operators")
              .diagnostic("navier_stokes_pressure_row_contribution_operators")
              .text("installing", "1")
              .text("ops", "equations_diagnostic_ns_galerkin_continuity|equations_diagnostic_ns_active_continuity|"
                           "equations_diagnostic_ns_vms_pspg|equations_diagnostic_ns_vms_pspg_pressure_gradient|"
                           "equations_diagnostic_ns_vms_pspg_nonpressure|equations_diagnostic_ns_pressure_ghost_penalty"));
    install_row("equations_diagnostic_ns_galerkin_continuity", galerkin_continuity_integrand.dx(liquid));
    install_row("equations_diagnostic_ns_active_continuity", active_continuity_diagnostic_form);
    if (vms_pspg_continuity_integrand.isValid()) {
        install_row("equations_diagnostic_ns_vms_pspg", vms_pspg_continuity_integrand.dx(liquid));
    }
    if (vms_pspg_pressure_gradient_integrand.isValid()) {
        install_row("equations_diagnostic_ns_vms_pspg_pressure_gradient",
                    vms_pspg_pressure_gradient_integrand.dx(liquid));
    }
    if (vms_pspg_nonpressure_integrand.isValid()) {
        install_row("equations_diagnostic_ns_vms_pspg_nonpressure", vms_pspg_nonpressure_integrand.dx(liquid));
    }
    if (pressure_ghost_penalty_form.isValid()) {
        install_row("equations_diagnostic_ns_pressure_ghost_penalty", pressure_ghost_penalty_form);
    }
}

/// The opt-in diagnostic operators, after the production operators, with the
/// builder's options on the system's diagnostic registry: the conservative
/// split when @p balance is set, and the pressure-row parts when their set is
/// switched on.  @p liquid is the volume the residual is integrated on.
inline void installDiagnosticOperators(FE::systems::FormulationBuilder& F,
                                       FE::FieldId u_id,
                                       FE::FieldId p_id,
                                       const NavierStokesVMSIntegrands& integrands,
                                       const ConservativeBalanceForms* balance,
                                       const FE::forms::FormExpr& active_continuity,
                                       const FE::forms::FormExpr& pressure_ghost_penalty_form,
                                       const FE::forms::FormExpr& u,
                                       const FE::forms::FormExpr& q,
                                       const FE::forms::VolumeDomain& liquid)
{
    const auto& r = integrands;
    if (balance != nullptr) {
        installConservativeBalanceDiagnosticOperators(
            F, u_id, p_id, balance->pressure, balance->external_pressure, balance->surface_energy,
            balance->gravitational_potential, r.pressure, u, q, liquid);
    }
    if (F.diagnostics().enabled(kPressureRowContributionDiagnostics)) {
        const auto vms_pspg = r.pspg_pressure_gradient.isValid()
                                  ? r.pspg_pressure_gradient + r.pspg_without_pressure_gradient
                                  : FE::forms::FormExpr{};
        installPressureRowContributionDiagnosticOperators(
            F, u_id, p_id, r.galerkin_continuity, active_continuity, vms_pspg, r.pspg_pressure_gradient,
            r.pspg_without_pressure_gradient, pressure_ghost_penalty_form, liquid);
    }
}

// ---------------------------------------------------------------------------
// D49 step E3: the energies, powers, dissipation rates and measures of an
// unfitted free surface as measured 0-forms, declared on the system with the
// forms and names that E2 calibrated against the pointwise evaluators
// (svmp-dev-refactor-moves/logs/e2-e8dad540).  The accepted-functional records
// read their values (E1's evaluator, exact reduction) instead of the
// pointwise sums; the static-capillary minimizer keeps the pointwise path
// until E4.  The record's other copies come from the same forms:
// active_liquid_volume = liquid_volume, and the backward-Euler kinetic work
// evaluates kinetic_energy on the accepted, the previous and the difference
// states (kinetic_energy_after, kinetic_energy_before_on_endpoint_domain,
// time_discretization_loss).  The per-wall quantities (wetted wall area,
// contact measure, solid-liquid wall energy) live on the generated domains
// of the walls the interface meets, so they are integrated over the
// snapshot's wall domains with the declaration's wall coefficients.  Totals
// stay sums of channels; the volume-constraint potential is zero while no
// multiplier is declared.
//
// The contact-line channels (the contact-line part of the wall power, the
// line friction and the Navier slip) are wall functionals: integrands with
// the rule's wall normal (FormExpr::boundaryNormal()) that the evaluators
// integrate over each wall's domains in the snapshot (freeSurfaceWallFunctionals).
// The records log the wall rates and the contact stage the wall slip from
// them (calibrated by E2b); the line friction stays pointwise until a deck
// with a nonzero friction calibrates it.
// ---------------------------------------------------------------------------

/// What one unfitted free surface declares as its energy functionals (the
/// data of its FreeSurfaceDiscreteFunctionalDeclaration).
struct FreeSurfaceEnergyStatement {
    int interface_marker{-1};
    FE::forms::CutVolumeSide liquid_side{FE::forms::CutVolumeSide::Negative};
    FE::FieldId velocity_field{FE::INVALID_FIELD_ID};
    const FE::spaces::FunctionSpace* velocity_space{nullptr};
    int dimension{0};
    FE::Real surface_tension{0.0};
    FE::Real density{0.0};
    std::array<FE::Real, 3> gravitational_acceleration{{0.0, 0.0, 0.0}};
    std::array<FE::Real, 3> gravitational_reference_point{kGravitationalReferencePoint};
    std::optional<FE::Real> dynamic_viscosity{};   ///< constant: the bulk viscous dissipation
    std::optional<FE::Real> external_pressure{};   ///< constant: the exterior-pressure power
    bool endpoint_functional_power{false};         ///< the area and volume rates
};

/// "free_surface.<interface marker>.<quantity>".
[[nodiscard]] inline std::string freeSurfaceFunctionalName(int interface_marker, std::string_view quantity)
{
    return "free_surface." + std::to_string(interface_marker) + "." + std::string(quantity);
}

/// The energy functionals of one free surface, in this order:
///   liquid_volume                  Measure          int_L 1
///   liquid_gas_area                Measure          int_Gamma 1
///   liquid_gas_surface_energy      Stored           int_Gamma gamma
///   kinetic_energy                 Stored           int_L rho/2 u.u
///   gravitational_energy           Stored           -int_L rho g.(x - x_ref)
///   gravitational_potential_power  Power            -int_L rho g.u
///   bulk_viscous_dissipation_rate  DissipationRate  int_L 2 mu eps(u):eps(u)      (constant mu)
///   external_pressure_power        Power            -p_ext int_Gamma u.n_L        (constant p_ext)
///   liquid_volume_flux             Power            int_Gamma u.n_L               (constant p_ext)
///   liquid_gas_area_rate           Power            int_Gamma div_Gamma u         (SurfaceStress)
///   liquid_volume_rate             Power            int_Gamma u.n_L               (SurfaceStress)
/// with L the liquid side of the interface, n_L = s n the rule normal turned
/// out of the liquid (s = +1 for a negative liquid, -1 for a positive one).
[[nodiscard]] inline std::vector<FE::systems::FunctionalFormDeclaration> freeSurfaceEnergyFunctionals(
    const FreeSurfaceEnergyStatement& statement)
{
    using namespace FE::forms;
    using Kind = FE::systems::FunctionalKind;
    const std::string owner = "IncompressibleNavierStokesVMSModule.FreeSurfaceBoundary";
    const int m = statement.interface_marker;
    const auto side = statement.liquid_side;
    const FE::Real orientation = side == CutVolumeSide::Negative ? FE::Real{1.0} : FE::Real{-1.0};
    const auto one = FormExpr::constant(1.0);
    const auto n = FormExpr::normal();
    const auto u = FormExpr::discreteField(statement.velocity_field, *statement.velocity_space, "u");
    const auto x = FormExpr::coordinate();
    const int dim = statement.dimension;
    const int velocity_components = std::min(dim, statement.velocity_space->value_dimension());
    const auto& g = statement.gravitational_acceleration;
    const auto& x_ref = statement.gravitational_reference_point;
    const FE::Real rho = statement.density;

    std::vector<FE::systems::FunctionalFormDeclaration> out;
    const auto add = [&](std::string_view quantity, Kind kind, FormExpr form) {
        out.push_back(FE::systems::FunctionalFormDeclaration{
            .name = freeSurfaceFunctionalName(m, quantity), .kind = kind, .form = std::move(form), .owner = owner});
    };

    FormExpr potential = FormExpr::constant(0.0);
    FormExpr g_dot_u = FormExpr::constant(0.0);
    for (int i = 0; i < dim; ++i) {
        const auto component = static_cast<std::size_t>(i);
        potential = potential + FormExpr::constant(g[component]) *
                                    (x.component(i) - FormExpr::constant(x_ref[component]));
        if (i < velocity_components) {
            g_dot_u = g_dot_u + FormExpr::constant(g[component]) * u.component(i);
        }
    }
    add("liquid_volume", Kind::Measure, one.dCutVolume(m, side));
    add("liquid_gas_area", Kind::Measure, one.dI(m));
    add("liquid_gas_surface_energy", Kind::Stored, FormExpr::constant(statement.surface_tension).dI(m));
    add("kinetic_energy", Kind::Stored, (FormExpr::constant(0.5 * rho) * inner(u, u)).dCutVolume(m, side));
    add("gravitational_energy", Kind::Stored, (FormExpr::constant(-rho) * potential).dCutVolume(m, side));
    add("gravitational_potential_power", Kind::Power, (FormExpr::constant(-rho) * g_dot_u).dCutVolume(m, side));
    if (statement.dynamic_viscosity.has_value()) {
        const auto strain = symgrad(u);
        add("bulk_viscous_dissipation_rate", Kind::DissipationRate,
            (FormExpr::constant(2.0 * *statement.dynamic_viscosity) * strain.doubleContraction(strain))
                .dCutVolume(m, side));
    }
    if (statement.external_pressure.has_value()) {
        const auto flux = FormExpr::constant(orientation) * inner(u, n);
        add("external_pressure_power", Kind::Power,
            (FormExpr::constant(-*statement.external_pressure) * flux).dI(m));
        add("liquid_volume_flux", Kind::Power, flux.dI(m));
    }
    if (statement.endpoint_functional_power) {
        add("liquid_gas_area_rate", Kind::Power, surfaceDivergence(u, n).dI(m));
        add("liquid_volume_rate", Kind::Power, (FormExpr::constant(orientation) * inner(u, n)).dI(m));
    }
    return out;
}

/// The statement of the free surface that @p declaration declares (the
/// record declaration of declareFreeSurfaceDiscreteFunctionals), with the
/// velocity space of its velocity field.
[[nodiscard]] inline FreeSurfaceEnergyStatement freeSurfaceEnergyStatement(
    const FE::systems::FreeSurfaceDiscreteFunctionalDeclaration& declaration,
    const FE::spaces::FunctionSpace& velocity_space,
    int dimension)
{
    FreeSurfaceEnergyStatement statement;
    statement.interface_marker = declaration.interface_marker;
    statement.liquid_side = declaration.parameters.liquid_side == FE::geometry::CutIntegrationSide::Positive
                                ? FE::forms::CutVolumeSide::Positive
                                : FE::forms::CutVolumeSide::Negative;
    statement.velocity_field = declaration.velocity_field;
    statement.velocity_space = &velocity_space;
    statement.dimension = dimension;
    statement.surface_tension = declaration.parameters.surface_tension;
    if (const auto& energy = declaration.active_volume_energy_parameters; energy.has_value()) {
        statement.density = energy->density;
        statement.gravitational_acceleration = energy->gravitational_acceleration;
        statement.gravitational_reference_point = energy->gravitational_reference_point;
    }
    if (const auto& dissipation = declaration.active_volume_dissipation_parameters; dissipation.has_value()) {
        statement.dynamic_viscosity = dissipation->dynamic_viscosity;
    }
    if (const auto& pressure = declaration.external_pressure_power_parameters; pressure.has_value()) {
        statement.external_pressure = pressure->external_pressure;
    }
    statement.endpoint_functional_power = declaration.endpoint_functional_power_enabled;
    return statement;
}

/// One wall that the contact lines of a free surface meet, with the
/// coefficients of the discrete-functional declaration.
struct FreeSurfaceContactWallStatement {
    int boundary_marker{-1};
    std::optional<FE::Real> equilibrium_contact_angle_radians{};  ///< a Young wall
    std::optional<FE::Real> line_friction{};                      ///< 1/M of a Ren-E wall
    std::optional<FE::Real> slip_friction{};                      ///< mu/l_s of a dynamic-contact wall
};

/// The walls of @p parameters by boundary marker: the Young walls, with the
/// line friction of a Ren-E law and the Navier-slip friction of their
/// dynamic-contact coefficient.
[[nodiscard]] inline std::vector<FreeSurfaceContactWallStatement> freeSurfaceContactWallStatements(
    const FE::interfaces::FreeSurfaceDiscreteFunctionalParameters& parameters)
{
    std::vector<FreeSurfaceContactWallStatement> walls;
    for (const auto& young : parameters.young_wall_coefficients) {
        FreeSurfaceContactWallStatement wall;
        wall.boundary_marker = young.boundary_marker;
        wall.equilibrium_contact_angle_radians = young.equilibrium_contact_angle_radians;
        for (const auto& dynamic : parameters.dynamic_contact_coefficients) {
            if (dynamic.boundary_marker != young.boundary_marker) {
                continue;
            }
            if (dynamic.law == FE::interfaces::FreeSurfaceContactLaw::DynamicRenE) {
                wall.line_friction = FE::Real{1.0} / dynamic.mobility;
            }
            wall.slip_friction = dynamic.dynamic_viscosity / dynamic.slip_length;
        }
        walls.push_back(std::move(wall));
    }
    std::sort(walls.begin(), walls.end(),
              [](const auto& a, const auto& b) { return a.boundary_marker < b.boundary_marker; });
    return walls;
}

/// The wall functionals of one free surface (Systems/FunctionalForms.h),
/// named "free_surface.<m>.wall.<b>.<quantity>" for every wall b the
/// interface meets, in this order:
///   wetted_area_rate                   Power            int_CL u.m_b                        (every wall)
///   young_energy_rate                  Power            -gamma cos(theta_b) int_CL u.m_b    (Young walls)
///   contact_line_friction_dissipation  DissipationRate  (1/M_b) int_CL (u.m_b)^2            (Ren-E walls)
///   wall_slip_dissipation              DissipationRate  (mu/l_s) int_W |u - (u.n_w) n_w|^2 (dynamic-contact walls)
/// with CL the wall's contact lines, W its wetted part, n_w the rule's wall
/// normal (FormExpr::boundaryNormal(): it follows a curved wall; on a flat
/// wall it is the configured normal), n_L the rule normal turned out of the
/// liquid and m_b = conormal(n_L, n_w), the direction along the wall out of
/// the wetted footprint.  The first two are the contact-line part of the wall
/// power, on the accepted state; the dissipations are evaluated on the
/// accepted contact-stage state (its stage velocity), as the contact stage
/// is.  A channel without a wall that states it is not declared.
[[nodiscard]] inline std::vector<FE::systems::WallFunctionalFormDeclaration> freeSurfaceWallFunctionals(
    const FreeSurfaceEnergyStatement& statement,
    std::span<const FreeSurfaceContactWallStatement> walls)
{
    using namespace FE::forms;
    using Kind = FE::systems::FunctionalKind;
    using Measure = FE::systems::WallMeasure;
    const std::string owner = "IncompressibleNavierStokesVMSModule.FreeSurfaceBoundary";
    const std::string prefix = freeSurfaceFunctionalName(statement.interface_marker, "");
    const FE::Real orientation =
        statement.liquid_side == CutVolumeSide::Negative ? FE::Real{1.0} : FE::Real{-1.0};
    const auto u = FormExpr::discreteField(statement.velocity_field, *statement.velocity_space, "u");
    const auto n_wall = FormExpr::boundaryNormal();
    const auto footprint = conormal(FormExpr::constant(orientation) * FormExpr::normal(), n_wall);
    const auto contact_speed = inner(u, footprint);
    const auto slip_velocity = u - inner(u, n_wall) * n_wall;

    std::vector<std::pair<int, FormExpr>> young;
    std::vector<std::pair<int, FormExpr>> friction;
    std::vector<std::pair<int, FormExpr>> slip;
    for (const auto& wall : walls) {
        const int b = wall.boundary_marker;
        if (wall.equilibrium_contact_angle_radians.has_value()) {
            const FE::Real coefficient =
                -statement.surface_tension * std::cos(*wall.equilibrium_contact_angle_radians);
            young.emplace_back(b, FormExpr::constant(coefficient) * contact_speed);
        }
        if (wall.line_friction.has_value()) {
            friction.emplace_back(b, FormExpr::constant(*wall.line_friction) * contact_speed * contact_speed);
        }
        if (wall.slip_friction.has_value()) {
            slip.emplace_back(b, FormExpr::constant(*wall.slip_friction) * inner(slip_velocity, slip_velocity));
        }
    }

    std::vector<FE::systems::WallFunctionalFormDeclaration> out;
    const auto add = [&](std::string quantity, Kind kind, Measure measure, FormExpr every_wall,
                         std::vector<std::pair<int, FormExpr>> by_wall) {
        if (!every_wall.isValid() && by_wall.empty()) {
            return;
        }
        out.push_back(FE::systems::WallFunctionalFormDeclaration{.prefix = prefix,
                                                                 .quantity = std::move(quantity),
                                                                 .kind = kind,
                                                                 .measure = measure,
                                                                 .integrand = std::move(every_wall),
                                                                 .by_wall = std::move(by_wall),
                                                                 .owner = owner});
    };
    add("wetted_area_rate", Kind::Power, Measure::ContactLines, contact_speed, {});
    add("young_energy_rate", Kind::Power, Measure::ContactLines, FormExpr{}, std::move(young));
    add("contact_line_friction_dissipation", Kind::DissipationRate, Measure::ContactLines, FormExpr{},
        std::move(friction));
    add("wall_slip_dissipation", Kind::DissipationRate, Measure::WettedPart, FormExpr{}, std::move(slip));
    return out;
}

/// Declares energies and wall functionals from the registered declarations.
/// Boundary selection and order follow this module's free surfaces. All selected
/// records must already exist; check that invariant before declaring any forms.
inline void declareFreeSurfaceEnergyFunctionals(FE::systems::FormulationBuilder& F,
                                                const std::vector<FreeSurfaceBoundary>& free_surfaces,
                                                const FE::spaces::FunctionSpace& velocity_space,
                                                int dimension)
{
    const auto records =
        FE::systems::functionalRecords(std::as_const(F.system())).functionalDeclarations();
    std::vector<const FE::systems::FreeSurfaceDiscreteFunctionalDeclaration*> selected;
    for (const auto& bc : free_surfaces) {
        if (!shouldDeclareFreeSurfaceDiscreteFunctional(bc)) {
            continue;
        }
        const auto record = std::find_if(records.begin(), records.end(), [&](const auto& declaration) {
            return declaration.interface_marker == bc.interface_marker;
        });
        if (record == records.end()) {
            throw std::logic_error(
                "IncompressibleNavierStokesVMSModule: missing free-surface discrete-functional declaration for interface marker " +
                std::to_string(bc.interface_marker));
        }
        selected.push_back(&*record);
    }
    for (const auto* record : selected) {
        const auto statement = freeSurfaceEnergyStatement(*record, velocity_space, dimension);
        for (auto& declaration : freeSurfaceEnergyFunctionals(statement)) {
            F.functional(std::move(declaration.name), declaration.kind, std::move(declaration.form),
                         std::move(declaration.owner));
        }
        const auto walls = freeSurfaceContactWallStatements(record->parameters);
        for (auto& declaration : freeSurfaceWallFunctionals(statement, walls)) {
            F.wallFunctional(std::move(declaration));
        }
    }
}

} // namespace post
} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp

#endif // SVMP_PHYSICS_FORMULATIONS_NAVIERSTOKES_POSTPROCESSING_H
