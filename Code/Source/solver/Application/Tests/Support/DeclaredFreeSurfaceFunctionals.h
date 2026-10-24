/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#pragma once

// A free-surface functional declaration together with the energy
// functionals the Navier-Stokes formulation declares for it (D49 steps E3
// and E4), for tests that build an FE system without the formulation.  The
// records and the static minimizer read the declared functionals, as in a
// run, where IncompressibleNavierStokesVMSModule::registerOn declares both.

#include "FE/Forms/IntegrationDomain.h"
#include "FE/Geometry/CutQuadrature.h"
#include "FE/Spaces/FunctionSpace.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FunctionalForms.h"
#include "FE/Systems/FunctionalRecords.h"
#include "Physics/Formulations/NavierStokes/NavierStokesPostProcessing.h"

#include <string>
#include <utility>

namespace application_test::declared_free_surface_functionals {

/// Declares @p declaration in the functional records of @p system and the
/// formulation's energy functionals of the same interface, with the
/// coefficients of the declaration (post::freeSurfaceEnergyFunctionals).
/// Without a velocity space only the functionals of the geometry are
/// declared (liquid volume, liquid-gas area and surface energy, the same
/// forms).  Call before the system is set up.
inline void declareFreeSurfaceFunctionalWithForms(
    svmp::FE::systems::FESystem& system,
    svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration declaration,
    const svmp::FE::spaces::FunctionSpace* velocity_space,
    int dimension)
{
    namespace post = svmp::Physics::formulations::navier_stokes::post;
    if (velocity_space == nullptr) {
        using svmp::FE::forms::FormExpr;
        using Kind = svmp::FE::systems::FunctionalKind;
        const int m = declaration.interface_marker;
        const auto side = declaration.parameters.liquid_side == svmp::FE::geometry::CutIntegrationSide::Positive
                              ? svmp::FE::forms::CutVolumeSide::Positive
                              : svmp::FE::forms::CutVolumeSide::Negative;
        const auto gamma = declaration.parameters.surface_tension;
        const std::string owner = "IncompressibleNavierStokesVMSModule.FreeSurfaceBoundary";
        svmp::FE::systems::functionalRecords(system).declareFunctional(std::move(declaration));
        const auto one = FormExpr::constant(1.0);
        svmp::FE::systems::declareFunctionalForm(
            system, {.name = post::freeSurfaceFunctionalName(m, "liquid_volume"), .kind = Kind::Measure,
                     .form = one.dCutVolume(m, side), .owner = owner});
        svmp::FE::systems::declareFunctionalForm(
            system, {.name = post::freeSurfaceFunctionalName(m, "liquid_gas_area"), .kind = Kind::Measure,
                     .form = one.dI(m), .owner = owner});
        svmp::FE::systems::declareFunctionalForm(
            system, {.name = post::freeSurfaceFunctionalName(m, "liquid_gas_surface_energy"), .kind = Kind::Stored,
                     .form = FormExpr::constant(gamma).dI(m), .owner = owner});
        return;
    }
    post::FreeSurfaceEnergyStatement statement;
    statement.interface_marker = declaration.interface_marker;
    statement.liquid_side =
        declaration.parameters.liquid_side == svmp::FE::geometry::CutIntegrationSide::Positive
            ? svmp::FE::forms::CutVolumeSide::Positive
            : svmp::FE::forms::CutVolumeSide::Negative;
    statement.velocity_field = declaration.velocity_field;
    statement.velocity_space = velocity_space;
    statement.dimension = dimension;
    statement.surface_tension = declaration.parameters.surface_tension;
    if (declaration.active_volume_energy_parameters.has_value()) {
        const auto& energy = *declaration.active_volume_energy_parameters;
        statement.density = energy.density;
        statement.gravitational_acceleration = energy.gravitational_acceleration;
        statement.gravitational_reference_point = energy.gravitational_reference_point;
    }
    if (declaration.active_volume_dissipation_parameters.has_value()) {
        statement.dynamic_viscosity = declaration.active_volume_dissipation_parameters->dynamic_viscosity;
    }
    if (declaration.external_pressure_power_parameters.has_value()) {
        statement.external_pressure = declaration.external_pressure_power_parameters->external_pressure;
    }
    statement.endpoint_functional_power = declaration.endpoint_functional_power_enabled;
    svmp::FE::systems::functionalRecords(system).declareFunctional(std::move(declaration));
    for (auto& functional : post::freeSurfaceEnergyFunctionals(statement)) {
        svmp::FE::systems::declareFunctionalForm(system, std::move(functional));
    }
}

} // namespace application_test::declared_free_surface_functionals
