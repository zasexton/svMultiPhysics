/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/AcceptedFunctionalRecorder.h"

#include "Assembly/Assembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Basis/BasisFunction.h"
#include "Core/RankLog.h"
#include "Elements/Element.h"
#include "Geometry/CutQuadratureMapping.h"
#include "Geometry/GeometryMapping.h"
#include "Geometry/LinearMapping.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Interfaces/SnapshotFieldEvaluator.h"
#include "LevelSet/LevelSetCutTopology.h"
#include "LevelSet/Maintenance/LevelSetMaintenanceRequestSchedule.h"
#include "Math/Matrix.h"
#include "Math/Vector.h"
#include "Spaces/FunctionSpace.h"
#include "Spaces/H1Space.h"
#include "Spaces/ProductSpace.h"
#include "Systems/FESystemHandle.h"
#include "Systems/SnapshotFieldAccess.h"
#include "Systems/SystemCollectives.h"
#include "Systems/FunctionalRecords.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace svmp {
namespace FE {
namespace level_set {

using svmp::FE::systems::FESystemHandle;
using svmp::FE::systems::activeMeshCommunicator;
using svmp::FE::systems::collectiveMaxDouble;
using svmp::FE::systems::collectiveMinDouble;
using svmp::FE::systems::collectiveMinMaxUint64;
using svmp::FE::systems::collectiveSumDouble;
using svmp::FE::systems::collectiveSumSize;
using svmp::FE::interfaces::makeSnapshotVectorFieldEvaluator;
using svmp::FE::systems::snapshotFieldAccess;

std::uint64_t acceptedFunctionalDeclarationContentKey(
    const svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration&
        declaration) noexcept
{
  std::uint64_t key = kCutContextHashOffset;
  const auto mix_signed = [&key](auto value) {
    mixCutContextHash(
        key,
        static_cast<std::uint64_t>(
            static_cast<std::int64_t>(value)));
  };
  const auto mix_real = [&key](svmp::FE::Real value) {
    if (value == svmp::FE::Real{0.0}) {
      value = svmp::FE::Real{0.0};
    }
    std::uint64_t bits = 0u;
    static_assert(sizeof(value) <= sizeof(bits));
    std::memcpy(&bits, &value, sizeof(value));
    mixCutContextHash(key, bits);
  };

  mix_signed(declaration.interface_marker);
  mix_signed(declaration.level_set_field);
  mix_signed(declaration.curvature_field);
  mix_signed(declaration.velocity_field);
  mixCutContextHash(key, declaration.geometry_domain_id);
  mix_signed(declaration.parameters.liquid_side);
  mix_real(declaration.parameters.surface_tension);
  mix_real(declaration.parameters.volume_multiplier);
  mixCutContextHash(
      key,
      static_cast<std::uint64_t>(
          declaration.parameters.young_wall_coefficients.size()));
  for (const auto& coefficient :
       declaration.parameters.young_wall_coefficients) {
    mix_signed(coefficient.boundary_marker);
    mix_real(coefficient.equilibrium_contact_angle_radians);
  }
  mixCutContextHash(
      key,
      static_cast<std::uint64_t>(
          declaration.parameters.dynamic_contact_coefficients.size()));
  for (const auto& coefficient :
       declaration.parameters.dynamic_contact_coefficients) {
    mix_signed(coefficient.boundary_marker);
    mix_real(coefficient.equilibrium_contact_angle_radians);
    mix_signed(coefficient.law);
    mix_real(coefficient.mobility);
    mix_real(coefficient.slip_length);
    mix_real(coefficient.dynamic_viscosity);
  }

  const auto& energy =
      declaration.active_volume_energy_parameters;
  mixCutContextHash(key, energy.has_value() ? 1u : 0u);
  if (energy.has_value()) {
    mix_signed(energy->liquid_side);
    mix_real(energy->density);
    for (const auto value : energy->gravitational_acceleration) {
      mix_real(value);
    }
    for (const auto value : energy->gravitational_reference_point) {
      mix_real(value);
    }
  }
  mixCutContextHash(
      key,
      declaration.static_conservative_body_force_complete ? 1u : 0u);
  const auto& dissipation =
      declaration.active_volume_dissipation_parameters;
  mixCutContextHash(key, dissipation.has_value() ? 1u : 0u);
  if (dissipation.has_value()) {
    mix_signed(dissipation->liquid_side);
    mix_real(dissipation->dynamic_viscosity);
  }
  const auto& pressure =
      declaration.external_pressure_power_parameters;
  mixCutContextHash(key, pressure.has_value() ? 1u : 0u);
  if (pressure.has_value()) {
    mix_signed(pressure->liquid_side);
    mix_real(pressure->external_pressure);
  }
  mixCutContextHash(
      key,
      declaration.endpoint_functional_power_enabled ? 1u : 0u);
  mix_signed(declaration.capillary_balance_method);
  mix_signed(declaration.capillary_balance_qualification);
  mixCutContextHash(key, declaration.owner_component);
  return key == 0u ? 1u : key;
}

std::vector<svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
evaluateCurrentAcceptedFunctionals(
    const FESystemHandle& sim)
{
  const auto declarations =
      systems::functionalRecords(*sim.fe_system).functionalDeclarations();
  const auto comm = activeMeshCommunicator(*sim.fe_system);
  const auto local_declaration_count =
      static_cast<double>(declarations.size());
  if (collectiveMinDouble(local_declaration_count, comm) !=
      collectiveMaxDouble(local_declaration_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Free-surface functional declarations differ across the FE communicator.");
  }
  if (declarations.empty()) {
    return {};
  }
  const auto* context = sim.fe_system->cutIntegrationContext();
  const auto context_available = context != nullptr ? 1.0 : 0.0;
  if (collectiveMinDouble(context_available, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Free-surface functional evaluation requires an authoritative cut-integration context on every rank.");
  }

  std::vector<svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
      states;
  states.reserve(declarations.size());
  for (const auto& declaration : declarations) {
    const auto declaration_content_key =
        acceptedFunctionalDeclarationContentKey(
            declaration);
    const auto [minimum_declaration_content_key,
                maximum_declaration_content_key] =
        collectiveMinMaxUint64(declaration_content_key, comm);
    if (minimum_declaration_content_key !=
        maximum_declaration_content_key) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Free-surface functional declaration content differs across the FE communicator.");
    }
    const auto& active_energy =
        declaration.active_volume_energy_parameters;
    const auto& active_dissipation =
        declaration.active_volume_dissipation_parameters;
    const std::array<double, 23> declaration_metadata{
        static_cast<double>(declaration.interface_marker),
        static_cast<double>(declaration.level_set_field),
        static_cast<double>(declaration.curvature_field),
        static_cast<double>(declaration.velocity_field),
        static_cast<double>(declaration.parameters.liquid_side),
        static_cast<double>(declaration.parameters.surface_tension),
        static_cast<double>(declaration.parameters.volume_multiplier),
        active_energy.has_value() ? 1.0 : 0.0,
        active_energy.has_value()
            ? static_cast<double>(active_energy->density)
            : 0.0,
        active_energy.has_value()
            ? static_cast<double>(
                  active_energy->gravitational_acceleration[0])
            : 0.0,
        active_energy.has_value()
            ? static_cast<double>(
                  active_energy->gravitational_acceleration[1])
            : 0.0,
        active_energy.has_value()
            ? static_cast<double>(
                  active_energy->gravitational_acceleration[2])
            : 0.0,
        active_energy.has_value()
            ? static_cast<double>(
                  active_energy->gravitational_reference_point[0])
            : 0.0,
        active_energy.has_value()
            ? static_cast<double>(
                  active_energy->gravitational_reference_point[1])
            : 0.0,
        active_energy.has_value()
            ? static_cast<double>(
                  active_energy->gravitational_reference_point[2])
            : 0.0,
        declaration.static_conservative_body_force_complete ? 1.0 : 0.0,
        active_dissipation.has_value() ? 1.0 : 0.0,
        active_dissipation.has_value()
            ? static_cast<double>(
                  active_dissipation->dynamic_viscosity)
            : 0.0,
        declaration.external_pressure_power_parameters.has_value()
            ? 1.0
            : 0.0,
        declaration.external_pressure_power_parameters.has_value()
            ? static_cast<double>(
                  declaration.external_pressure_power_parameters
                      ->external_pressure)
            : 0.0,
        declaration.endpoint_functional_power_enabled ? 1.0 : 0.0,
        static_cast<double>(declaration.capillary_balance_method),
        static_cast<double>(
            declaration.capillary_balance_qualification)};
    bool declaration_metadata_consistent = true;
    for (const auto value : declaration_metadata) {
      const bool local_finite = std::isfinite(value);
      if (collectiveMinDouble(local_finite ? 1.0 : 0.0, comm) != 1.0) {
        declaration_metadata_consistent = false;
        continue;
      }
      declaration_metadata_consistent =
          collectiveMinDouble(value, comm) == collectiveMaxDouble(value, comm) &&
          declaration_metadata_consistent;
    }
    if (!declaration_metadata_consistent) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Free-surface functional declaration ordering or scalar parameters differ across the FE communicator.");
    }
    const bool local_marker_available =
        context->hasFreeSurfaceGeometrySnapshotForMarker(
            declaration.interface_marker);
    if (collectiveMinDouble(local_marker_available ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Free-surface functional marker " +
          std::to_string(declaration.interface_marker) +
          " is missing an authoritative geometry snapshot on at least one rank.");
    }
    const auto snapshot_revision =
        context->freeSurfaceGeometrySnapshotRevisionForMarker(
            declaration.interface_marker);
    const auto& snapshots = context->freeSurfaceGeometrySnapshots();
    const auto found = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().snapshot_revision_key ==
                     snapshot_revision;
        });
    const auto [minimum_revision, maximum_revision] =
        collectiveMinMaxUint64(snapshot_revision, comm);
    if (minimum_revision != maximum_revision) {
      std::ostringstream diagnostic;
      diagnostic
          << "[svMultiPhysics::Application] Free-surface geometry revision differs across the FE communicator for marker "
          << declaration.interface_marker
          << ": local=" << snapshot_revision
          << " minimum=" << minimum_revision
          << " maximum=" << maximum_revision;
      if (found != snapshots.end()) {
        const auto& revision = (*found)->revision();
        diagnostic
            << " source_layout=" << revision.source_layout_revision
            << " source_value=" << revision.source_value_revision
            << " mesh_geometry=" << revision.mesh_geometry_revision
            << " mesh_topology=" << revision.mesh_topology_revision
            << " ownership=" << revision.ownership_revision
            << " numbering=" << revision.numbering_revision
            << " quadrature_policy=" << revision.quadrature_policy_key;
      }
      diagnostic << ".";
      throw std::runtime_error(diagnostic.str());
    }
    const bool local_snapshot_available = found != snapshots.end();
    if (collectiveMinDouble(local_snapshot_available ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Free-surface geometry snapshot storage is incomplete across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
        sim.fe_system->meshAccess());
    const auto& snapshot = **found;
    const bool geometry_pruning_occurred =
        collectiveMaxDouble(
            snapshot.ledger().pruned_rule_count != 0u ? 1.0 : 0.0,
            comm) == 1.0;
    auto global_state =
        svmp::FE::interfaces::evaluateFreeSurfaceDiscreteFunctional(
            snapshot, declaration.parameters);

    const auto wall_count = static_cast<double>(global_state.walls.size());
    if (collectiveMinDouble(wall_count, comm) !=
        collectiveMaxDouble(wall_count, comm)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Free-surface functional wall sets differ across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    for (const auto& wall : global_state.walls) {
      const auto marker = static_cast<double>(wall.boundary_marker);
      if (collectiveMinDouble(marker, comm) != collectiveMaxDouble(marker, comm)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Free-surface functional wall ordering differs across the FE communicator for marker " +
            std::to_string(declaration.interface_marker) + ".");
      }
      const auto angle_present =
          wall.equilibrium_contact_angle_radians.has_value() ? 1.0 : 0.0;
      if (collectiveMinDouble(angle_present, comm) !=
          collectiveMaxDouble(angle_present, comm)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Free-surface functional wall coefficient presence differs across the FE communicator for marker " +
            std::to_string(declaration.interface_marker) + ".");
      }
      if (wall.equilibrium_contact_angle_radians.has_value()) {
        const auto angle = static_cast<double>(
            *wall.equilibrium_contact_angle_radians);
        if (collectiveMinDouble(angle, comm) != collectiveMaxDouble(angle, comm)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Free-surface functional wall angles differ across the FE communicator for marker " +
              std::to_string(declaration.interface_marker) + ".");
        }
      }
    }

    const auto global_sum = [&comm](svmp::FE::Real local) {
      return static_cast<svmp::FE::Real>(
          collectiveSumDouble(static_cast<double>(local), comm));
    };
    global_state.owned_liquid_volume =
        global_sum(global_state.owned_liquid_volume);
    global_state.owned_liquid_gas_area =
        global_sum(global_state.owned_liquid_gas_area);
    global_state.owned_wetted_wall_area =
        global_sum(global_state.owned_wetted_wall_area);
    global_state.owned_contact_measure =
        global_sum(global_state.owned_contact_measure);
    global_state.liquid_gas_surface_energy =
        global_sum(global_state.liquid_gas_surface_energy);
    global_state.young_wall_energy =
        global_sum(global_state.young_wall_energy);
    global_state.volume_constraint_potential =
        global_sum(global_state.volume_constraint_potential);
    global_state.total_potential =
        global_sum(global_state.total_potential);
    for (auto& wall : global_state.walls) {
      wall.owned_wetted_wall_area =
          global_sum(wall.owned_wetted_wall_area);
      wall.owned_contact_measure =
          global_sum(wall.owned_contact_measure);
      wall.young_wall_energy = global_sum(wall.young_wall_energy);
    }
    states.push_back(
        svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState{
            .interface_marker = declaration.interface_marker,
            .geometry_revision = snapshot.revision(),
            .geometry_pruning_occurred =
                geometry_pruning_occurred,
            .state = std::move(global_state),
            .contact_stage = std::nullopt,
        });
  }
  return states;
}

namespace {

void attachAcceptedFreeSurfaceEndpointFunctionalPowers(
    const FESystemHandle& sim,
    std::span<const svmp::FE::Real> accepted_solution,
    std::span<
        svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
        accepted_states)
{
  const auto declarations =
      systems::functionalRecords(*sim.fe_system).functionalDeclarations();
  const auto comm = activeMeshCommunicator(*sim.fe_system);
  const auto declared_count = std::count_if(
      declarations.begin(), declarations.end(), [](const auto& declaration) {
        return declaration.endpoint_functional_power_enabled;
      });
  if (declared_count == 0) {
    for (const auto& state : accepted_states) {
      if (state.endpoint_functional_power.has_value()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Endpoint functional power was supplied without a discrete-energy capillary declaration.");
      }
    }
    return;
  }
  const auto local_coverage =
      accepted_states.size() == declarations.size() &&
              !accepted_solution.empty()
          ? 1.0
          : 0.0;
  const auto local_solution_size =
      static_cast<double>(accepted_solution.size());
  if (collectiveMinDouble(local_coverage, comm) != 1.0 ||
      collectiveMinDouble(local_solution_size, comm) !=
          collectiveMaxDouble(local_solution_size, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Endpoint functional power requires a communicator-consistent endpoint solution and complete accepted-state coverage.");
  }
  const auto* context = sim.fe_system->cutIntegrationContext();
  if (collectiveMinDouble(context != nullptr ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Endpoint functional power requires an authoritative geometry snapshot on every rank.");
  }

  for (std::size_t index = 0u; index < declarations.size(); ++index) {
    const auto& declaration = declarations[index];
    auto& accepted = accepted_states[index];
    const bool declared =
        declaration.endpoint_functional_power_enabled;
    if (!declared) {
      if (accepted.endpoint_functional_power.has_value()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Endpoint functional power was supplied for an undeclared interface.");
      }
      continue;
    }
    const auto& snapshots = context->freeSurfaceGeometrySnapshots();
    const auto found = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().snapshot_revision_key ==
                     accepted.geometry_revision.snapshot_revision_key;
        });
    if (collectiveMinDouble(found != snapshots.end() ? 1.0 : 0.0, comm) !=
        1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Endpoint functional-power snapshot storage is incomplete across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    std::optional<
        svmp::FE::interfaces::FreeSurfaceDiscreteFunctionalVariationState>
        local_power;
    std::exception_ptr local_failure;
    try {
      const auto velocity = makeSnapshotVectorFieldEvaluator(
          snapshotFieldAccess(*sim.fe_system, declaration.velocity_field),
          accepted_solution,
          "Accepted endpoint functional power");
      svmp::FE::interfaces::
          FreeSurfaceDiscreteFunctionalDeformationEvaluator
              deformation;
      deformation.value = velocity.value;
      deformation.physical_gradient =
          velocity.physical_gradient;
      local_power =
          svmp::FE::interfaces::
              evaluateFreeSurfaceDiscreteFunctionalFirstVariation(
                  **found,
                  declaration.parameters,
                  deformation);
    } catch (...) {
      local_failure = std::current_exception();
    }
    if (collectiveMinDouble(local_power.has_value() ? 1.0 : 0.0, comm) !=
        1.0) {
      if (local_failure) {
        std::rethrow_exception(local_failure);
      }
      throw std::runtime_error(
          "[svMultiPhysics::Application] Endpoint functional-power evaluation failed on another FE communicator rank for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    auto power = std::move(*local_power);
    const auto global_sum = [&comm](svmp::FE::Real value) {
      return static_cast<svmp::FE::Real>(
          collectiveSumDouble(static_cast<double>(value), comm));
    };
    power.owned_liquid_volume_variation =
        global_sum(power.owned_liquid_volume_variation);
    power.owned_liquid_gas_area_variation =
        global_sum(power.owned_liquid_gas_area_variation);
    power.owned_wetted_wall_area_variation =
        global_sum(power.owned_wetted_wall_area_variation);
    power.liquid_gas_surface_energy_variation =
        global_sum(
            power.liquid_gas_surface_energy_variation);
    power.young_wall_energy_variation =
        global_sum(power.young_wall_energy_variation);
    power.volume_constraint_potential_variation =
        global_sum(
            power.volume_constraint_potential_variation);
    power.total_potential_variation =
        global_sum(power.total_potential_variation);
    for (auto& wall : power.walls) {
      wall.owned_wetted_wall_area_variation =
          global_sum(
              wall.owned_wetted_wall_area_variation);
      wall.young_wall_energy_variation =
          global_sum(wall.young_wall_energy_variation);
    }
    accepted.endpoint_functional_power = std::move(power);
  }
}

} // namespace

void attachAcceptedActiveVolumeEnergies(
    const FESystemHandle& sim,
    std::span<const svmp::FE::Real> accepted_solution,
    std::span<
        svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
        accepted_states)
{
  const auto declarations =
      systems::functionalRecords(*sim.fe_system).functionalDeclarations();
  const auto comm = activeMeshCommunicator(*sim.fe_system);
  const auto local_state_coverage =
      accepted_states.size() == declarations.size() ? 1.0 : 0.0;
  const auto local_declaration_count =
      static_cast<double>(declarations.size());
  if (collectiveMinDouble(local_state_coverage, comm) != 1.0 ||
      collectiveMinDouble(local_declaration_count, comm) !=
          collectiveMaxDouble(local_declaration_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Active-volume energy evaluation requires exactly one accepted state per free-surface declaration.");
  }
  const auto active_count = std::count_if(
      declarations.begin(), declarations.end(), [](const auto& declaration) {
        return declaration.active_volume_energy_parameters.has_value();
      });
  const auto local_active_count = static_cast<double>(active_count);
  if (collectiveMinDouble(local_active_count, comm) !=
      collectiveMaxDouble(local_active_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Active-volume energy declaration coverage differs across the FE communicator.");
  }
  if (active_count == 0) {
    return;
  }
  const auto local_solution_size =
      static_cast<double>(accepted_solution.size());
  const auto all_solutions_nonempty =
      collectiveMinDouble(
          accepted_solution.empty() ? 0.0 : 1.0, comm) == 1.0;
  const auto minimum_solution_size =
      collectiveMinDouble(local_solution_size, comm);
  const auto maximum_solution_size =
      collectiveMaxDouble(local_solution_size, comm);
  if (!all_solutions_nonempty ||
      minimum_solution_size != maximum_solution_size) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Active-volume energy evaluation requires a nonempty communicator-consistent accepted solution.");
  }
  const auto* context = sim.fe_system->cutIntegrationContext();
  if (collectiveMinDouble(context != nullptr ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Active-volume energy evaluation requires an authoritative geometry snapshot on every rank.");
  }

  for (std::size_t index = 0; index < declarations.size(); ++index) {
    const auto& declaration = declarations[index];
    auto& accepted = accepted_states[index];
    const auto local_energy_declared =
        declaration.active_volume_energy_parameters.has_value() ? 1.0 : 0.0;
    if (collectiveMinDouble(local_energy_declared, comm) !=
        collectiveMaxDouble(local_energy_declared, comm)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Active-volume energy declaration ordering differs across the FE communicator.");
    }
    if (!declaration.active_volume_energy_parameters.has_value()) {
      const auto no_undeclared_state =
          accepted.active_volume_energy.has_value() ? 0.0 : 1.0;
      if (collectiveMinDouble(no_undeclared_state, comm) != 1.0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] An undeclared active-volume energy state was supplied.");
      }
      continue;
    }
    const auto& parameters =
        *declaration.active_volume_energy_parameters;
    const std::array<double, 10> metadata{
        static_cast<double>(declaration.interface_marker),
        static_cast<double>(declaration.velocity_field),
        static_cast<double>(parameters.liquid_side),
        static_cast<double>(parameters.density),
        static_cast<double>(parameters.gravitational_acceleration[0]),
        static_cast<double>(parameters.gravitational_acceleration[1]),
        static_cast<double>(parameters.gravitational_acceleration[2]),
        static_cast<double>(parameters.gravitational_reference_point[0]),
        static_cast<double>(parameters.gravitational_reference_point[1]),
        static_cast<double>(parameters.gravitational_reference_point[2])};
    bool metadata_consistent = true;
    for (const auto value : metadata) {
      const auto finite = std::isfinite(value) ? 1.0 : 0.0;
      if (collectiveMinDouble(finite, comm) != 1.0) {
        metadata_consistent = false;
        continue;
      }
      metadata_consistent =
          collectiveMinDouble(value, comm) == collectiveMaxDouble(value, comm) &&
          metadata_consistent;
    }
    if (!metadata_consistent) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Active-volume energy parameters differ across the FE communicator.");
    }
    const auto local_order_matches =
        accepted.interface_marker == declaration.interface_marker &&
                accepted.geometry_revision.interface_marker ==
                    declaration.interface_marker
            ? 1.0
            : 0.0;
    if (collectiveMinDouble(local_order_matches, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Active-volume energy state ordering does not match its free-surface declaration.");
    }
    const auto& snapshots = context->freeSurfaceGeometrySnapshots();
    const auto found = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().snapshot_revision_key ==
                     accepted.geometry_revision.snapshot_revision_key;
        });
    if (collectiveMinDouble(found != snapshots.end() ? 1.0 : 0.0, comm) !=
        1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Active-volume energy snapshot storage is incomplete across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    std::optional<
        svmp::FE::interfaces::FreeSurfaceActiveVolumeEnergyState>
        local_energy;
    std::exception_ptr local_failure;
    try {
      const auto velocity = makeSnapshotVectorFieldEvaluator(
          snapshotFieldAccess(*sim.fe_system, declaration.velocity_field),
          accepted_solution,
          "Accepted active-volume energy");
      local_energy =
          svmp::FE::interfaces::evaluateFreeSurfaceActiveVolumeEnergy(
              **found, parameters, velocity, &sim.fe_system->meshAccess());
    } catch (...) {
      local_failure = std::current_exception();
    }
    if (collectiveMinDouble(local_energy.has_value() ? 1.0 : 0.0, comm) !=
        1.0) {
      if (local_failure) {
        std::rethrow_exception(local_failure);
      }
      throw std::runtime_error(
          "[svMultiPhysics::Application] Active-volume energy evaluation failed on another FE communicator rank for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    auto energy = std::move(*local_energy);
    energy.owned_quadrature_point_count =
        collectiveSumSize(energy.owned_quadrature_point_count, comm);
    energy.owned_liquid_volume = static_cast<svmp::FE::Real>(
        collectiveSumDouble(
            static_cast<double>(energy.owned_liquid_volume), comm));
    energy.kinetic_energy = static_cast<svmp::FE::Real>(
        collectiveSumDouble(
            static_cast<double>(energy.kinetic_energy), comm));
    energy.gravitational_energy = static_cast<svmp::FE::Real>(
        collectiveSumDouble(
            static_cast<double>(energy.gravitational_energy), comm));
    energy.gravitational_potential_power =
        static_cast<svmp::FE::Real>(
            collectiveSumDouble(
                static_cast<double>(
                    energy.gravitational_potential_power),
                comm));
    energy.total_energy =
        energy.kinetic_energy + energy.gravitational_energy;
    accepted.active_volume_energy = std::move(energy);
  }
}

namespace {

void attachAcceptedFreeSurfaceActiveVolumeDissipation(
    const FESystemHandle& sim,
    std::span<const svmp::FE::Real> accepted_solution,
    std::span<
        svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
        accepted_states)
{
  const auto declarations =
      systems::functionalRecords(*sim.fe_system).functionalDeclarations();
  const auto comm = activeMeshCommunicator(*sim.fe_system);
  const auto declared_count = std::count_if(
      declarations.begin(), declarations.end(), [](const auto& declaration) {
        return declaration.active_volume_dissipation_parameters.has_value();
      });
  if (declared_count == 0) {
    for (const auto& state : accepted_states) {
      if (state.active_volume_dissipation.has_value()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Active-volume dissipation was supplied without a declaration.");
      }
    }
    return;
  }
  const auto local_coverage =
      accepted_states.size() == declarations.size() &&
              !accepted_solution.empty()
          ? 1.0
          : 0.0;
  const auto local_solution_size =
      static_cast<double>(accepted_solution.size());
  if (collectiveMinDouble(local_coverage, comm) != 1.0 ||
      collectiveMinDouble(local_solution_size, comm) !=
          collectiveMaxDouble(local_solution_size, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Active-volume dissipation requires a communicator-consistent endpoint solution and complete accepted-state coverage.");
  }
  const auto* context = sim.fe_system->cutIntegrationContext();
  if (collectiveMinDouble(context != nullptr ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Active-volume dissipation requires an authoritative geometry snapshot on every rank.");
  }

  for (std::size_t index = 0; index < declarations.size(); ++index) {
    const auto& declaration = declarations[index];
    auto& accepted = accepted_states[index];
    if (!declaration.active_volume_dissipation_parameters.has_value()) {
      if (accepted.active_volume_dissipation.has_value()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Active-volume dissipation was supplied for an undeclared interface.");
      }
      continue;
    }
    const auto& snapshots = context->freeSurfaceGeometrySnapshots();
    const auto found = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().snapshot_revision_key ==
                     accepted.geometry_revision.snapshot_revision_key;
        });
    if (collectiveMinDouble(found != snapshots.end() ? 1.0 : 0.0, comm) !=
        1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Active-volume dissipation snapshot storage is incomplete across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    std::optional<
        svmp::FE::interfaces::FreeSurfaceActiveVolumeDissipationState>
        local_dissipation;
    std::exception_ptr local_failure;
    try {
      const auto velocity = makeSnapshotVectorFieldEvaluator(
          snapshotFieldAccess(*sim.fe_system, declaration.velocity_field),
          accepted_solution,
          "Accepted active-volume dissipation");
      local_dissipation =
          svmp::FE::interfaces::
              evaluateFreeSurfaceActiveVolumeDissipation(
                  **found,
                  *declaration.active_volume_dissipation_parameters,
                  velocity,
                  &sim.fe_system->meshAccess());
    } catch (...) {
      local_failure = std::current_exception();
    }
    if (collectiveMinDouble(
            local_dissipation.has_value() ? 1.0 : 0.0,
            comm) != 1.0) {
      if (local_failure) {
        std::rethrow_exception(local_failure);
      }
      throw std::runtime_error(
          "[svMultiPhysics::Application] Active-volume dissipation evaluation failed on another FE communicator rank for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    auto dissipation = std::move(*local_dissipation);
    dissipation.owned_quadrature_point_count =
        collectiveSumSize(
            dissipation.owned_quadrature_point_count, comm);
    dissipation.owned_liquid_volume =
        static_cast<svmp::FE::Real>(
            collectiveSumDouble(
                static_cast<double>(
                    dissipation.owned_liquid_volume),
                comm));
    dissipation.bulk_viscous_dissipation_rate =
        static_cast<svmp::FE::Real>(
            collectiveSumDouble(
                static_cast<double>(
                    dissipation.bulk_viscous_dissipation_rate),
                comm));
    accepted.active_volume_dissipation =
        std::move(dissipation);
  }
}

void attachAcceptedFreeSurfaceExternalPressurePowers(
    const FESystemHandle& sim,
    std::span<const svmp::FE::Real> accepted_solution,
    std::span<
        svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
        accepted_states)
{
  const auto declarations =
      systems::functionalRecords(*sim.fe_system).functionalDeclarations();
  const auto comm = activeMeshCommunicator(*sim.fe_system);
  const auto declared_count = std::count_if(
      declarations.begin(), declarations.end(), [](const auto& declaration) {
        return declaration.external_pressure_power_parameters.has_value();
      });
  if (declared_count == 0) {
    for (const auto& state : accepted_states) {
      if (state.external_pressure_power.has_value()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Exterior-pressure power was supplied without a declaration.");
      }
    }
    return;
  }
  const auto local_coverage =
      accepted_states.size() == declarations.size() &&
              !accepted_solution.empty()
          ? 1.0
          : 0.0;
  const auto local_solution_size =
      static_cast<double>(accepted_solution.size());
  if (collectiveMinDouble(local_coverage, comm) != 1.0 ||
      collectiveMinDouble(local_solution_size, comm) !=
          collectiveMaxDouble(local_solution_size, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Exterior-pressure power requires a communicator-consistent endpoint solution and complete accepted-state coverage.");
  }
  const auto* context = sim.fe_system->cutIntegrationContext();
  if (collectiveMinDouble(context != nullptr ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Exterior-pressure power requires an authoritative geometry snapshot on every rank.");
  }

  for (std::size_t index = 0u; index < declarations.size(); ++index) {
    const auto& declaration = declarations[index];
    auto& accepted = accepted_states[index];
    if (!declaration.external_pressure_power_parameters.has_value()) {
      if (accepted.external_pressure_power.has_value()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Exterior-pressure power was supplied for an undeclared interface.");
      }
      continue;
    }
    const auto& snapshots = context->freeSurfaceGeometrySnapshots();
    const auto found = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().snapshot_revision_key ==
                     accepted.geometry_revision.snapshot_revision_key;
        });
    if (collectiveMinDouble(found != snapshots.end() ? 1.0 : 0.0, comm) !=
        1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Exterior-pressure-power snapshot storage is incomplete across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    std::optional<
        svmp::FE::interfaces::FreeSurfaceExternalPressurePowerState>
        local_power;
    std::exception_ptr local_failure;
    try {
      const auto velocity = makeSnapshotVectorFieldEvaluator(
          snapshotFieldAccess(*sim.fe_system, declaration.velocity_field),
          accepted_solution,
          "Accepted exterior-pressure power");
      local_power =
          svmp::FE::interfaces::
              evaluateFreeSurfaceExternalPressurePower(
                  **found,
                  *declaration.external_pressure_power_parameters,
                  velocity);
    } catch (...) {
      local_failure = std::current_exception();
    }
    if (collectiveMinDouble(local_power.has_value() ? 1.0 : 0.0, comm) !=
        1.0) {
      if (local_failure) {
        std::rethrow_exception(local_failure);
      }
      throw std::runtime_error(
          "[svMultiPhysics::Application] Exterior-pressure-power evaluation failed on another FE communicator rank for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    auto power = std::move(*local_power);
    power.owned_quadrature_point_count =
        collectiveSumSize(
            power.owned_quadrature_point_count, comm);
    const auto global_sum = [&comm](svmp::FE::Real value) {
      return static_cast<svmp::FE::Real>(
          collectiveSumDouble(static_cast<double>(value), comm));
    };
    power.owned_liquid_gas_area =
        global_sum(power.owned_liquid_gas_area);
    power.outward_liquid_volume_flux_rate =
        global_sum(
            power.outward_liquid_volume_flux_rate);
    power.external_pressure_power =
        global_sum(power.external_pressure_power);
    accepted.external_pressure_power = std::move(power);
  }
}

} // namespace

std::uint64_t collectiveFieldRevision(
    const svmp::FE::systems::FESystem& system,
    svmp::FE::FieldId field,
    std::span<const svmp::FE::Real> solution,
    const svmp::MeshComm& comm,
    std::string_view diagnostic_context)
{
  const auto offset = system.fieldDofOffset(field);
  const auto count = system.fieldDofHandler(field).getNumDofs();
  if (offset < 0 || count < 0 ||
      static_cast<std::size_t>(offset) > solution.size() ||
      static_cast<std::size_t>(count) >
          solution.size() - static_cast<std::size_t>(offset)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] " +
        std::string(diagnostic_context) +
        " field slice is outside the FE-ordered solution.");
  }
  auto revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          solution.subspan(
              static_cast<std::size_t>(offset),
              static_cast<std::size_t>(count)),
          comm);
  mixCutContextHash(
      revision,
      static_cast<std::uint64_t>(
          static_cast<std::int64_t>(field)));
  return revision == 0u ? 1u : revision;
}

namespace {

void attachAcceptedFreeSurfaceBackwardEulerKineticWork(
    const FESystemHandle& sim,
    std::span<const svmp::FE::Real> previous_solution,
    std::span<const svmp::FE::Real> accepted_solution,
    std::span<
        svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
        accepted_states)
{
  const auto declarations =
      systems::functionalRecords(*sim.fe_system).functionalDeclarations();
  const auto comm = activeMeshCommunicator(*sim.fe_system);
  const auto local_previous_present =
      previous_solution.empty() ? 0.0 : 1.0;
  if (collectiveMinDouble(local_previous_present, comm) !=
      collectiveMaxDouble(local_previous_present, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Backward-Euler kinetic-work endpoint coverage differs across the FE communicator.");
  }
  if (previous_solution.empty()) {
    for (const auto& state : accepted_states) {
      if (state.backward_euler_kinetic_work.has_value()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Backward-Euler kinetic work was supplied without a previous endpoint solution.");
      }
    }
    return;
  }
  const auto local_coverage =
      accepted_states.size() == declarations.size() &&
              !accepted_solution.empty() &&
              previous_solution.size() == accepted_solution.size()
          ? 1.0
          : 0.0;
  const auto local_solution_size =
      static_cast<double>(accepted_solution.size());
  if (collectiveMinDouble(local_coverage, comm) != 1.0 ||
      collectiveMinDouble(local_solution_size, comm) !=
          collectiveMaxDouble(local_solution_size, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Backward-Euler kinetic work requires communicator-consistent previous and endpoint solutions with complete accepted-state coverage.");
  }
  const auto* context = sim.fe_system->cutIntegrationContext();
  if (collectiveMinDouble(context != nullptr ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Backward-Euler kinetic work requires an authoritative endpoint geometry snapshot on every rank.");
  }

  std::map<
      svmp::FE::FieldId,
      std::pair<std::uint64_t, std::uint64_t>>
      velocity_revisions;
  for (std::size_t index = 0; index < declarations.size(); ++index) {
    const auto& declaration = declarations[index];
    auto& accepted = accepted_states[index];
    if (!declaration.active_volume_energy_parameters.has_value()) {
      if (accepted.backward_euler_kinetic_work.has_value()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Backward-Euler kinetic work was supplied for a declaration without active-volume energy.");
      }
      continue;
    }
    if (!accepted.active_volume_energy.has_value()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Backward-Euler kinetic work requires the accepted active-volume stored energy.");
    }
    const auto [velocity_revisions_it, inserted] =
        velocity_revisions.try_emplace(
            declaration.velocity_field,
            std::uint64_t{0u},
            std::uint64_t{0u});
    if (inserted) {
      velocity_revisions_it->second.first =
          collectiveFieldRevision(
              *sim.fe_system,
              declaration.velocity_field,
              previous_solution,
              comm,
              "Backward-Euler previous velocity revision");
      velocity_revisions_it->second.second =
          collectiveFieldRevision(
              *sim.fe_system,
              declaration.velocity_field,
              accepted_solution,
              comm,
              "Backward-Euler endpoint velocity revision");
    }
    const auto previous_velocity_revision =
        velocity_revisions_it->second.first;
    const auto endpoint_velocity_revision =
        velocity_revisions_it->second.second;
    const auto& snapshots = context->freeSurfaceGeometrySnapshots();
    const auto found = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().snapshot_revision_key ==
                     accepted.geometry_revision.snapshot_revision_key;
        });
    if (collectiveMinDouble(found != snapshots.end() ? 1.0 : 0.0, comm) !=
        1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Backward-Euler kinetic-work endpoint snapshot storage is incomplete across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }

    std::optional<
        svmp::FE::interfaces::FreeSurfaceBackwardEulerKineticWorkState>
        local_work;
    std::exception_ptr local_failure;
    try {
      const auto previous_velocity = makeSnapshotVectorFieldEvaluator(
          snapshotFieldAccess(*sim.fe_system, declaration.velocity_field),
          previous_solution,
          "Backward-Euler previous-endpoint kinetic work");
      const auto endpoint_velocity = makeSnapshotVectorFieldEvaluator(
          snapshotFieldAccess(*sim.fe_system, declaration.velocity_field),
          accepted_solution,
          "Backward-Euler accepted-endpoint kinetic work");
      const auto& parameters =
          *declaration.active_volume_energy_parameters;
      local_work =
          svmp::FE::interfaces::
              evaluateFreeSurfaceBackwardEulerKineticWork(
                  **found,
                  parameters.liquid_side,
                  parameters.density,
                  previous_velocity_revision,
                  endpoint_velocity_revision,
                  previous_velocity,
                  endpoint_velocity,
                  &sim.fe_system->meshAccess());
    } catch (...) {
      local_failure = std::current_exception();
    }
    if (collectiveMinDouble(local_work.has_value() ? 1.0 : 0.0, comm) !=
        1.0) {
      if (local_failure) {
        std::rethrow_exception(local_failure);
      }
      throw std::runtime_error(
          "[svMultiPhysics::Application] Backward-Euler kinetic-work evaluation failed on another FE communicator rank for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }

    auto work = std::move(*local_work);
    work.owned_quadrature_point_count =
        collectiveSumSize(work.owned_quadrature_point_count, comm);
    const auto global_sum = [&](svmp::FE::Real value) {
      return static_cast<svmp::FE::Real>(
          collectiveSumDouble(static_cast<double>(value), comm));
    };
    work.owned_liquid_volume =
        global_sum(work.owned_liquid_volume);
    work.kinetic_energy_before_on_endpoint_domain =
        global_sum(
            work.kinetic_energy_before_on_endpoint_domain);
    work.kinetic_energy_after =
        global_sum(work.kinetic_energy_after);
    work.step_integrated_inertia_work =
        global_sum(work.step_integrated_inertia_work);
    work.time_discretization_loss =
        global_sum(work.time_discretization_loss);
    work.kinetic_energy_change_on_endpoint_domain =
        work.kinetic_energy_after -
        work.kinetic_energy_before_on_endpoint_domain;
    work.identity_residual =
        work.step_integrated_inertia_work -
        work.kinetic_energy_change_on_endpoint_domain -
        work.time_discretization_loss;

    const auto& energy = *accepted.active_volume_energy;
    const auto values_near = [](svmp::FE::Real left,
                                svmp::FE::Real right) {
      const auto scale = std::max(
          {svmp::FE::Real{1.0}, std::abs(left), std::abs(right)});
      return std::abs(left - right) <=
          svmp::FE::Real{256.0} *
              std::numeric_limits<svmp::FE::Real>::epsilon() * scale;
    };
    if (!values_near(
            work.owned_liquid_volume,
            energy.owned_liquid_volume) ||
        !values_near(
            work.kinetic_energy_after,
            energy.kinetic_energy) ||
        !values_near(work.identity_residual, svmp::FE::Real{0.0})) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Backward-Euler kinetic work is inconsistent with the accepted stored energy or its endpoint-domain identity for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    accepted.backward_euler_kinetic_work = std::move(work);
  }
}

std::uint64_t levelSetMaintenanceCutTopologyRevision(
    const svmp::FE::interfaces::FreeSurfaceGeometrySnapshot& snapshot,
    const svmp::MeshComm& comm)
{
  CutTopologyFingerprintMultiset local_digest;
  for (const auto& rule : snapshot.rules()) {
    if (!rule.locally_owned) {
      continue;
    }
    std::uint64_t revision = kCutContextHashOffset;
    mixCutContextHash(
        revision, static_cast<std::uint64_t>(rule.role));
    mixCutContextHash(
        revision, static_cast<std::uint64_t>(rule.retention));
    mixCutContextHash(
        revision,
        static_cast<std::uint64_t>(
            static_cast<std::int64_t>(rule.physical_boundary_marker)));
    mixCutContextHash(
        revision,
        static_cast<std::uint64_t>(
            rule.reference_rule.geometric_dimension));
    mixCutContextHash(
        revision,
        static_cast<std::uint64_t>(
            rule.reference_rule.provenance.parent_entity_global_id));
    mixCutContextHash(
        revision,
        rule.reference_rule.provenance.cut_topology_revision);
    mixCutContextHash(
        revision,
        rule.physical_rule.cut_topology_revision);
    mixCutContextHash(
        revision,
        static_cast<std::uint64_t>(rule.component_id));
    mixCutContextHash(revision, rule.topology_id);
    mixCutContextHash(
        revision,
        static_cast<std::uint64_t>(
            rule.source_fragment_stable_ids.size()));
    for (const auto stable_id : rule.source_fragment_stable_ids) {
      mixCutContextHash(revision, stable_id);
    }
    local_digest.insert(revision);
  }

  const auto global_digest =
      collectiveCutTopologyFingerprintMultiset(local_digest, comm);
  std::uint64_t revision = kCutContextHashOffset;
  mixCutContextHash(
      revision,
      static_cast<std::uint64_t>(
          snapshot.revision().interface_marker));
  for (const auto word : global_digest.sums) {
    mixCutContextHash(revision, word);
  }
  for (const auto word : global_digest.xors) {
    mixCutContextHash(revision, word);
  }
  return revision == 0u ? 1u : revision;
}

} // namespace

std::vector<svmp::FE::level_set::LevelSetAuthoritativeFunctionalValue>
levelSetMaintenanceFunctionalValues(
    const FESystemHandle& sim,
    std::span<const svmp::FE::systems::
                        AcceptedFreeSurfaceDiscreteFunctionalState> states,
    std::span<const svmp::FE::Real> solution)
{
  std::vector<svmp::FE::level_set::LevelSetAuthoritativeFunctionalValue>
      values;
  values.reserve(states.size());
  if (!sim.fe_system) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set maintenance work provenance requires an FE system.");
  }
  const auto comm = activeMeshCommunicator(*sim.fe_system);
  const auto local_state_count =
      static_cast<std::uint64_t>(states.size());
  const auto [minimum_state_count, maximum_state_count] =
      collectiveMinMaxUint64(local_state_count, comm);
  if (minimum_state_count != maximum_state_count) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set maintenance functional coverage differs across the FE communicator.");
  }
  if (states.empty()) {
    return values;
  }
  const auto* context = sim.fe_system->cutIntegrationContext();
  if (collectiveMinDouble(context != nullptr ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set maintenance work provenance requires an authoritative cut-integration context.");
  }
  std::vector<
      svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
      enriched_states(states.begin(), states.end());
  const auto declarations =
      systems::functionalRecords(*sim.fe_system).functionalDeclarations();
  const bool solution_channel_declared = std::any_of(
      declarations.begin(),
      declarations.end(),
      [](const auto& declaration) {
        return declaration.active_volume_energy_parameters.has_value() ||
            declaration.active_volume_dissipation_parameters.has_value() ||
            declaration.external_pressure_power_parameters.has_value() ||
            declaration.endpoint_functional_power_enabled;
      });
  const auto [minimum_solution_channel_declared,
              maximum_solution_channel_declared] =
      collectiveMinMaxUint64(
          solution_channel_declared ? 1u : 0u, comm);
  if (minimum_solution_channel_declared !=
      maximum_solution_channel_declared) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set maintenance endpoint-channel declarations differ across the FE communicator.");
  }
  if (solution_channel_declared) {
    if (collectiveMinDouble(
            solution.empty() ? 0.0 : 1.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set maintenance modeled-energy, endpoint-power, and dissipation work requires the complete FE-ordered candidate solution.");
    }
    attachAcceptedFreeSurfaceEndpointFunctionalPowers(
        sim, solution, enriched_states);
    attachAcceptedActiveVolumeEnergies(
        sim, solution, enriched_states);
    attachAcceptedFreeSurfaceActiveVolumeDissipation(
        sim, solution, enriched_states);
    attachAcceptedFreeSurfaceExternalPressurePowers(
        sim, solution, enriched_states);
  }
  const auto& snapshots = context->freeSurfaceGeometrySnapshots();
  for (const auto& state : enriched_states) {
    const auto marker = static_cast<double>(state.interface_marker);
    if (collectiveMinDouble(marker, comm) !=
        collectiveMaxDouble(marker, comm)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set maintenance functional ordering differs across the FE communicator.");
    }
    const auto found = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& snapshot) {
          return snapshot &&
                 snapshot->revision().snapshot_revision_key ==
                     state.geometry_revision.snapshot_revision_key;
        });
    if (collectiveMinDouble(
            found != snapshots.end() ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Level-set maintenance work could not resolve its authoritative geometry snapshot.");
    }
    const auto* active_energy =
        state.active_volume_energy.has_value()
            ? &*state.active_volume_energy
            : nullptr;
    const auto* endpoint_power =
        state.endpoint_functional_power.has_value()
            ? &*state.endpoint_functional_power
            : nullptr;
    const auto* active_dissipation =
        state.active_volume_dissipation.has_value()
            ? &*state.active_volume_dissipation
            : nullptr;
    const auto* external_pressure =
        state.external_pressure_power.has_value()
            ? &*state.external_pressure_power
            : nullptr;
    const auto modeled_stored_energy =
        active_energy != nullptr
            ? std::optional<double>{
                  static_cast<double>(active_energy->total_energy) +
                  static_cast<double>(
                      state.state.liquid_gas_surface_energy) +
                  static_cast<double>(state.state.young_wall_energy)}
            : std::nullopt;
    values.push_back(
        svmp::FE::level_set::LevelSetAuthoritativeFunctionalValue{
            .interface_marker = state.interface_marker,
            .snapshot_revision =
                state.geometry_revision.snapshot_revision_key,
            .mesh_topology_revision =
                state.geometry_revision.mesh_topology_revision,
            .cut_topology_revision =
                levelSetMaintenanceCutTopologyRevision(
                    **found, comm),
            .liquid_volume = state.state.owned_liquid_volume,
            .liquid_gas_area = state.state.owned_liquid_gas_area,
            .wetted_wall_area = state.state.owned_wetted_wall_area,
            .contact_measure = state.state.owned_contact_measure,
            .surface_energy = state.state.liquid_gas_surface_energy,
            .young_wall_energy = state.state.young_wall_energy,
            .volume_constraint_potential =
                state.state.volume_constraint_potential,
            .total_potential = state.state.total_potential,
            .kinetic_energy =
                active_energy != nullptr
                    ? std::optional<double>{
                          static_cast<double>(
                              active_energy->kinetic_energy)}
                    : std::nullopt,
            .gravitational_energy =
                active_energy != nullptr
                    ? std::optional<double>{
                          static_cast<double>(
                              active_energy->gravitational_energy)}
                    : std::nullopt,
            .gravitational_potential_power =
                active_energy != nullptr
                    ? std::optional<double>{
                          static_cast<double>(
                              active_energy
                                  ->gravitational_potential_power)}
                    : std::nullopt,
            .surface_wall_potential_power =
                endpoint_power != nullptr
                    ? std::optional<double>{
                          static_cast<double>(
                              endpoint_power
                                  ->liquid_gas_surface_energy_variation +
                              endpoint_power
                                  ->young_wall_energy_variation)}
                    : std::nullopt,
            .volume_constraint_potential_power =
                endpoint_power != nullptr
                    ? std::optional<double>{
                          static_cast<double>(
                              endpoint_power
                                  ->volume_constraint_potential_variation)}
                    : std::nullopt,
            .bulk_viscous_dissipation_rate =
                active_dissipation != nullptr
                    ? std::optional<double>{
                          static_cast<double>(
                              active_dissipation
                                  ->bulk_viscous_dissipation_rate)}
                    : std::nullopt,
            .external_pressure_power =
                external_pressure != nullptr
                    ? std::optional<double>{
                          static_cast<double>(
                              external_pressure
                                  ->external_pressure_power)}
                    : std::nullopt,
            .modeled_stored_energy = modeled_stored_energy,
        });
  }
  std::sort(
      values.begin(), values.end(), [](const auto& left, const auto& right) {
        return left.interface_marker < right.interface_marker;
      });
  return values;
}

AcceptedFunctionalCoverage
requireCommunicatorConsistentAcceptedFunctionalCoverage(
    std::span<
        const svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    const svmp::MeshComm& comm)
{
  const auto local_declaration_count =
      static_cast<double>(declarations.size());
  if (collectiveMinDouble(local_declaration_count, comm) !=
      collectiveMaxDouble(local_declaration_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Free-surface functional declaration coverage differs across the FE communicator.");
  }
  for (const auto& declaration : declarations) {
    const auto declaration_content_key =
        acceptedFunctionalDeclarationContentKey(
            declaration);
    const auto [minimum_content_key, maximum_content_key] =
        collectiveMinMaxUint64(declaration_content_key, comm);
    if (minimum_content_key != maximum_content_key) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Free-surface functional declaration content differs across the FE communicator.");
    }
    const std::array<double, 3> method_metadata{
        static_cast<double>(declaration.interface_marker),
        static_cast<double>(declaration.capillary_balance_method),
        static_cast<double>(
            declaration.capillary_balance_qualification)};
    for (const auto value : method_metadata) {
      if (collectiveMinDouble(value, comm) !=
          collectiveMaxDouble(value, comm)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Free-surface capillary-balance method declarations differ across the FE communicator.");
      }
    }
  }
  const auto local_dynamic_contact_count =
      static_cast<double>(std::count_if(
          declarations.begin(),
          declarations.end(),
          [](const auto& declaration) {
            return !declaration.parameters.dynamic_contact_coefficients
                        .empty();
          }));
  if (collectiveMinDouble(local_dynamic_contact_count, comm) !=
      collectiveMaxDouble(local_dynamic_contact_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Dynamic contact-stage declaration coverage differs across the FE communicator.");
  }
  return AcceptedFunctionalCoverage{
      .has_functional = !declarations.empty(),
      .has_dynamic_contact = local_dynamic_contact_count > 0.0,
  };
}

std::vector<
    svmp::FE::systems::AcceptedFreeSurfaceDiscreteFunctionalState>
recordAcceptedFunctionals(
    const FESystemHandle& sim,
    std::uint64_t accepted_step,
    svmp::FE::Real accepted_time,
    svmp::FE::Real dt,
    std::uint64_t pre_maintenance_endpoint_state_revision,
    std::uint64_t state_revision,
    std::span<const svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
        contact_stages,
    std::span<const svmp::FE::Real> accepted_solution,
    std::span<const svmp::FE::Real>
        backward_euler_previous_solution,
    std::optional<std::uint64_t>
        accepted_extension_map_revision,
    bool* backward_euler_kinetic_work_bound)
{
  if (backward_euler_kinetic_work_bound != nullptr) {
    *backward_euler_kinetic_work_bound = true;
  }
  const auto declarations =
      systems::functionalRecords(*sim.fe_system).functionalDeclarations();
  const auto comm = activeMeshCommunicator(*sim.fe_system);
  const auto coverage =
      requireCommunicatorConsistentAcceptedFunctionalCoverage(
          declarations, comm);
  if (!coverage.has_functional) {
    if (!contact_stages.empty()) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Contact-stage states were supplied without free-surface functional declarations.");
    }
    return {};
  }
  const auto [minimum_pre_maintenance_revision,
              maximum_pre_maintenance_revision] =
      collectiveMinMaxUint64(
          pre_maintenance_endpoint_state_revision, comm);
  if (pre_maintenance_endpoint_state_revision == 0u ||
      minimum_pre_maintenance_revision !=
          maximum_pre_maintenance_revision) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] The pre-maintenance endpoint state revision must be nonzero and identical across the FE communicator.");
  }
  const auto [minimum_state_revision, maximum_state_revision] =
      collectiveMinMaxUint64(state_revision, comm);
  if (state_revision == 0u ||
      minimum_state_revision != maximum_state_revision) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] The accepted state revision must be nonzero and identical across the FE communicator.");
  }
  const double local_contact_stage_count =
      static_cast<double>(contact_stages.size());
  if (collectiveMinDouble(local_contact_stage_count, comm) !=
      collectiveMaxDouble(local_contact_stage_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage coverage differs across the FE communicator.");
  }
  std::map<
      int,
      const svmp::FE::systems::FreeSurfaceAcceptedContactStageState*>
      contact_stage_by_marker;
  bool local_contact_stages_valid = true;
  for (const auto& stage : contact_stages) {
    const int marker = stage.geometry_revision.interface_marker;
    const double local_marker = static_cast<double>(marker);
    if (collectiveMinDouble(local_marker, comm) !=
        collectiveMaxDouble(local_marker, comm)) {
      local_contact_stages_valid = false;
    }
    local_contact_stages_valid =
        marker >= 0 &&
        stage.endpoint_state_revision ==
            pre_maintenance_endpoint_state_revision &&
        contact_stage_by_marker.emplace(marker, &stage).second &&
        local_contact_stages_valid;
  }
  if (collectiveMinDouble(local_contact_stages_valid ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage states must have communicator-consistent unique nonnegative interface markers and must bind to the pre-maintenance endpoint revision.");
  }
  auto accepted_states = evaluateCurrentAcceptedFunctionals(sim);
  if (accepted_states.size() != declarations.size()) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted free-surface functional evaluation returned incomplete declaration coverage.");
  }
  attachAcceptedFreeSurfaceEndpointFunctionalPowers(
      sim, accepted_solution, accepted_states);
  attachAcceptedActiveVolumeEnergies(
      sim, accepted_solution, accepted_states);
  attachAcceptedFreeSurfaceActiveVolumeDissipation(
      sim, accepted_solution, accepted_states);
  attachAcceptedFreeSurfaceExternalPressurePowers(
      sim, accepted_solution, accepted_states);
  const auto existing_history =
      systems::functionalRecords(*sim.fe_system).acceptedFunctionalHistory();
  attachAcceptedFreeSurfaceBackwardEulerKineticWork(
      sim,
      backward_euler_previous_solution,
      accepted_solution,
      accepted_states);
  const bool has_backward_euler_work = std::any_of(
      accepted_states.begin(),
      accepted_states.end(),
      [](const auto& state) {
        return state.backward_euler_kinetic_work.has_value();
      });
  // The backward-Euler kinetic work of this step is computed from the actual
  // previous and endpoint solutions.  Binding it to the previously recorded
  // endpoint is energy-ledger bookkeeping (diagnostic under D1/D6): when the
  // binding cannot be established (for example after a cut-topology change,
  // whose new constraints re-project the previous state), the binding is
  // reported unavailable and the run continues; the caller then treats the
  // complete energy history as non-contiguous.
  std::string local_binding_failure;
  if (has_backward_euler_work) {
    if (dt == svmp::FE::Real{0.0}) {
      for (const auto& state : accepted_states) {
        if (!state.backward_euler_kinetic_work.has_value()) {
          continue;
        }
        const auto& work =
            *state.backward_euler_kinetic_work;
        if (work.previous_velocity_revision !=
            work.endpoint_velocity_revision) {
          local_binding_failure =
              "zero_duration_baseline_with_distinct_velocity_endpoints";
        }
      }
    } else if (existing_history.empty()) {
      local_binding_failure = "no_previously_accepted_endpoint";
    } else if (accepted_step >
               existing_history.back().accepted_step) {
      if (existing_history.size() < declarations.size()) {
        local_binding_failure = "incomplete_accepted_functional_history";
      } else {
        const auto previous_group_begin =
            existing_history.size() - declarations.size();
        const auto previous_group_step =
            existing_history[previous_group_begin].accepted_step;
        const auto previous_group_time =
            existing_history[previous_group_begin].accepted_time;
        for (std::size_t index = 0u;
             index < accepted_states.size();
             ++index) {
          const auto& current =
              accepted_states[index].backward_euler_kinetic_work;
          if (!current.has_value()) {
            continue;
          }
          const auto& previous =
              existing_history[previous_group_begin + index];
          if (previous.accepted_step != previous_group_step ||
              previous.accepted_time != previous_group_time ||
              acceptedFunctionalDeclarationContentKey(
                  previous.declaration) !=
                  acceptedFunctionalDeclarationContentKey(
                      declarations[index])) {
            local_binding_failure =
                "latest_accepted_group_not_declaration_aligned";
          } else if (!previous.backward_euler_kinetic_work.has_value()) {
            local_binding_failure =
                "previous_endpoint_without_kinetic_work";
          } else if (current->previous_velocity_revision !=
                     previous.backward_euler_kinetic_work
                         ->endpoint_velocity_revision) {
            local_binding_failure =
                "previous_velocity_differs_from_recorded_endpoint";
          }
        }
      }
    }
  }
  const bool binding_available =
      collectiveMinDouble(local_binding_failure.empty() ? 1.0 : 0.0, comm) ==
      1.0;
  if (backward_euler_kinetic_work_bound != nullptr) {
    *backward_euler_kinetic_work_bound = binding_available;
  }
  if (!binding_available) {
    svmp::FE::rank_log::oopCout()
        << "[svMultiPhysics::Application] Backward-Euler kinetic work"
        << " diagnostic=backward_euler_kinetic_work_binding"
        << " status=unavailable"
        << " accepted_step=" << accepted_step
        << " reason="
        << (local_binding_failure.empty() ? "failure_on_another_rank"
                                          : local_binding_failure)
        << " action=record_unbound_and_continue" << std::endl;
  }
  const auto* accepted_cut_context =
      sim.fe_system->cutIntegrationContext();
  if (collectiveMinDouble(
          accepted_cut_context != nullptr ? 1.0 : 0.0,
          comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted free-surface functional history requires the authoritative cut-integration context.");
  }
  const auto& accepted_snapshots =
      accepted_cut_context->freeSurfaceGeometrySnapshots();
  for (std::size_t index = 0; index < declarations.size(); ++index) {
    const auto& declaration = declarations[index];
    auto& accepted_state = accepted_states[index];
    if (accepted_state.interface_marker != declaration.interface_marker) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted free-surface functional evaluation changed declaration order.");
    }
    const auto accepted_snapshot = std::find_if(
        accepted_snapshots.begin(),
        accepted_snapshots.end(),
        [&](const auto& candidate) {
          return candidate &&
              candidate->revision().snapshot_revision_key ==
                  accepted_state.geometry_revision
                      .snapshot_revision_key;
        });
    if (collectiveMinDouble(
            accepted_snapshot != accepted_snapshots.end()
                ? 1.0
                : 0.0,
            comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted free-surface functional history cannot resolve its authoritative geometry snapshot.");
    }
    accepted_state.cut_topology_revision =
        levelSetMaintenanceCutTopologyRevision(
            **accepted_snapshot, comm);
    const auto contact_stage =
        contact_stage_by_marker.find(declaration.interface_marker);
    const bool dynamic_contact_declared =
        !declaration.parameters.dynamic_contact_coefficients.empty();
    if (dynamic_contact_declared !=
        (contact_stage != contact_stage_by_marker.end())) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage coverage does not match the free-surface declaration for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    if (contact_stage != contact_stage_by_marker.end()) {
      accepted_state.contact_stage = *contact_stage->second;
    }
  }

  const auto declared_dynamic_count = std::count_if(
      declarations.begin(), declarations.end(), [](const auto& declaration) {
        return !declaration.parameters.dynamic_contact_coefficients.empty();
      });
  if (contact_stage_by_marker.size() !=
      static_cast<std::size_t>(declared_dynamic_count)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage states include an undeclared interface marker.");
  }

  if (!existing_history.empty()) {
    bool local_latest_group_valid =
        existing_history.size() >= declarations.size() &&
        existing_history.size() % declarations.size() == 0u;
    std::size_t previous_group_begin = 0u;
    if (local_latest_group_valid) {
      previous_group_begin =
          existing_history.size() - declarations.size();
      const auto& latest = existing_history[previous_group_begin];
      for (std::size_t index = 0u;
           index < declarations.size();
           ++index) {
        const auto& record =
            existing_history[previous_group_begin + index];
        local_latest_group_valid =
            record.accepted_step == latest.accepted_step &&
            record.accepted_time == latest.accepted_time &&
            record.dt == latest.dt &&
            record.pre_maintenance_endpoint_state_revision ==
                latest.pre_maintenance_endpoint_state_revision &&
            record.state_revision == latest.state_revision &&
            record.extension_map_revision ==
                latest.extension_map_revision &&
            acceptedFunctionalDeclarationContentKey(
                record.declaration) ==
                acceptedFunctionalDeclarationContentKey(
                    declarations[index]) &&
            local_latest_group_valid;
      }
    }
    if (collectiveMinDouble(
            local_latest_group_valid ? 1.0 : 0.0,
            comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Latest accepted free-surface history is not one complete declaration-aligned provenance group on every rank.");
    }
    const auto& latest = existing_history[previous_group_begin];
    const auto [minimum_latest_step, maximum_latest_step] =
        collectiveMinMaxUint64(latest.accepted_step, comm);
    const auto [minimum_latest_pre_maintenance_revision,
                maximum_latest_pre_maintenance_revision] =
        collectiveMinMaxUint64(
            latest.pre_maintenance_endpoint_state_revision,
            comm);
    const auto [minimum_latest_state_revision,
                maximum_latest_state_revision] =
        collectiveMinMaxUint64(latest.state_revision, comm);
    const double extension_available =
        latest.extension_map_revision.has_value() ? 1.0 : 0.0;
    const auto [minimum_latest_extension_revision,
                maximum_latest_extension_revision] =
        collectiveMinMaxUint64(
            latest.extension_map_revision.value_or(0u),
            comm);
    if (minimum_latest_step != maximum_latest_step ||
        collectiveMinDouble(latest.accepted_time, comm) !=
            collectiveMaxDouble(latest.accepted_time, comm) ||
        collectiveMinDouble(latest.dt, comm) !=
            collectiveMaxDouble(latest.dt, comm) ||
        minimum_latest_pre_maintenance_revision !=
            maximum_latest_pre_maintenance_revision ||
        minimum_latest_state_revision !=
            maximum_latest_state_revision ||
        collectiveMinDouble(extension_available, comm) !=
            collectiveMaxDouble(extension_available, comm) ||
        minimum_latest_extension_revision !=
            maximum_latest_extension_revision) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Latest accepted free-surface history provenance differs across the FE communicator.");
    }
    for (std::size_t index = 0u; index < accepted_states.size(); ++index) {
      auto& accepted = accepted_states[index];
      if (!accepted.contact_stage.has_value()) {
        continue;
      }
      const auto& previous =
          existing_history[previous_group_begin + index];
      if (accepted_step == latest.accepted_step) {
        accepted.contact_line_kinematics =
            previous.contact_line_kinematics;
      } else if (accepted_step > latest.accepted_step) {
        accepted.contact_line_kinematics =
            systems::deriveAcceptedContactLineKinematics(
                previous,
                accepted_step,
                *accepted.contact_stage);
      }
    }
  }

  systems::functionalRecords(*sim.fe_system).recordAcceptedFunctionals(
      accepted_step,
      accepted_time,
      dt,
      pre_maintenance_endpoint_state_revision,
      state_revision,
      accepted_states,
      accepted_extension_map_revision);
  return accepted_states;
}

void recordInitialAcceptedFunctionalBaseline(
    const FESystemHandle& sim,
    std::uint64_t initial_step,
    svmp::FE::Real initial_time,
    std::span<const svmp::FE::Real> initial_solution,
    const AcceptedContactStageEvaluator& evaluate_contact_stages,
    bool record_backward_euler_kinetic_baseline,
    std::optional<std::uint64_t>
        initial_extension_map_revision)
{
  const auto initial_state_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          initial_solution,
          activeMeshCommunicator(*sim.fe_system));
  const auto initial_contact_stages =
      evaluate_contact_stages(initial_state_revision);
  (void)recordAcceptedFunctionals(
      sim,
      initial_step,
      initial_time,
      svmp::FE::Real{0.0},
      initial_state_revision,
      initial_state_revision,
      initial_contact_stages,
      initial_solution,
      record_backward_euler_kinetic_baseline
          ? initial_solution
          : std::span<const svmp::FE::Real>{},
      initial_extension_map_revision);
}

} // namespace level_set
} // namespace FE
} // namespace svmp
