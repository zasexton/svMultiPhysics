/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceDiagnosticOperators.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceGeometry.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceQueries.h"

#include "FE/Core/Logger.h"
#include "FE/Dofs/EntityDofMap.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FormsInstaller.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace svmp {
namespace Physics {
namespace formulations {
namespace navier_stokes {

void declareNavierStokesDiagnosticSets(FE::systems::DiagnosticRegistry& registry)
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

bool conservativeBalanceDiagnosticSupported(
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
                       (usesSurfaceStress(bc) ||
                        usesKinematicAreaGradientTraction(bc)) &&
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
                       ((usesSurfaceStress(bc) ||
                         usesKinematicAreaGradientTraction(bc)) &&
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

void installConservativeBalanceDiagnosticOperators(
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

void installPressureRowContributionDiagnosticOperators(
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

void installNavierStokesDiagnosticOperators(
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

} // namespace navier_stokes
} // namespace formulations
} // namespace Physics
} // namespace svmp
