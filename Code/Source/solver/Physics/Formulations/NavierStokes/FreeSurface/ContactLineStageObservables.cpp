/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/SnapshotFunctionals.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/ContactLineStageObservables.h"
#include "FE/LevelSet/Maintenance/LevelSetWallAwareMaintenance.h"
#include <map>
#include <set>

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Interfaces/LevelSetGeometrySnapshot.h"
#include "FE/Interfaces/SnapshotFieldEvaluator.h"
#include "FE/Systems/FESystem.h"
#include "FE/Systems/FunctionalFormEvaluator.h"
#include "FE/Systems/FunctionalForms.h"
#include "FE/Systems/SnapshotFieldAccess.h"
#include "FE/Systems/SystemState.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

using namespace svmp::FE;
using namespace svmp::FE::interfaces;


namespace {

using svmp::FE::systems::collectiveAnyTrue;
using svmp::FE::systems::collectiveMaxDouble;
using svmp::FE::systems::collectiveMinDouble;
using svmp::FE::systems::collectiveMinMaxUint64;
using svmp::FE::systems::collectiveSumDouble;
using svmp::FE::systems::collectiveSumSize;
using svmp::FE::systems::requireCollectivePhasePreparation;
using svmp::FE::systems::snapshotFieldAccess;

// The active communicator of the FE system (the Mesh communicator of the
// callers when FE is built with the Mesh library).
svmp::FE::systems::CollectiveCommunicator activeSystemCommunicator(
    const svmp::FE::systems::FESystem& system)
{
#if FE_HAS_MPI
  return svmp::FE::systems::CollectiveCommunicator(
      system.activeMpiCommunicator());
#else
  (void)system;
  return {};
#endif
}

/**
 * The wall-slip dissipation of each wall of an accepted contact stage as the
 * declared wall functional "wall.<b>.wall_slip_dissipation" of its interface
 * (D49 E3, contact-line switch), evaluated at the stage solution on the
 * stage's snapshot with the exact reduction; the totals stay sums of the
 * walls.  The line friction stays pointwise until a deck with a nonzero
 * friction calibrates its form.  Without a declared wall-slip functional the
 * stage keeps its pointwise values.  Collective.
 */
void applyDeclaredStageWallSlipDissipation(
    const svmp::FE::systems::FESystem& system,
    const FreeSurfaceDiscreteFunctionalDeclaration& declaration,
    const LevelSetGeometrySnapshot& snapshot,
    std::span<const svmp::FE::Real> stage_solution,
    FreeSurfaceAcceptedContactStageState& stage,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  auto& state = stage.state;
  std::vector<int> walls;
  walls.reserve(state.walls.size());
  for (const auto& wall : state.walls) {
    walls.push_back(wall.boundary_marker);
  }
  constexpr std::string_view kWallSlip = "wall_slip_dissipation";
  const std::string prefix =
      "free_surface." + std::to_string(declaration.interface_marker) + ".";
  const auto declared = FE::systems::declaredWallFunctionalForms(
      system, prefix, snapshot, walls, declaration.parameters.active_side,
      std::span<const std::string_view>(&kWallSlip, 1u));
  const auto [minimum_count, maximum_count] = collectiveMinMaxUint64(
      static_cast<std::uint64_t>(declared.size()), comm);
  if (minimum_count != maximum_count) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] The declared contact-stage wall functionals of interface " +
        std::to_string(declaration.interface_marker) +
        " differ across the FE communicator.");
  }
  if (declared.empty()) {
    return;
  }
  std::vector<svmp::FE::forms::FormExpr> forms;
  forms.reserve(declared.size());
  for (const auto& entry : declared) {
    forms.push_back(entry.form);
  }
  svmp::FE::systems::SystemStateView view;
  view.u = stage_solution;
  const auto values =
      svmp::FE::systems::evaluateFunctionalForms(system, forms, view, comm);
  for (std::size_t k = 0; k < declared.size(); ++k) {
    for (auto& wall : state.walls) {
      if (wall.boundary_marker == declared[k].wall) {
        wall.wall_slip_dissipation = values[k].value;
      }
    }
    stage.declared_functional_masses.emplace_back(declared[k].name,
                                                  values[k].absolute_mass);
  }
  // As finalizeFreeSurfaceDynamicContactState forms them.
  state.wall_slip_dissipation = svmp::FE::Real{0.0};
  for (const auto& wall : state.walls) {
    state.wall_slip_dissipation += wall.wall_slip_dissipation;
  }
  state.total_dissipation =
      state.line_friction_dissipation + state.wall_slip_dissipation;
}

} // namespace

bool firstOrderGeneralizedAlphaContactProvenanceValid(
    const FreeSurfaceFirstOrderGeneralizedAlphaProvenance& provenance) noexcept
{
  if (!std::isfinite(provenance.alpha_m) ||
      !std::isfinite(provenance.alpha_f) ||
      !std::isfinite(provenance.gamma) ||
      !std::isfinite(provenance.dt) ||
      provenance.alpha_f < svmp::FE::Real{0.5} ||
      provenance.alpha_f > svmp::FE::Real{1.0} ||
      provenance.alpha_m < svmp::FE::Real{0.5} ||
      provenance.alpha_m > svmp::FE::Real{1.5} ||
      !(provenance.gamma > svmp::FE::Real{0.0}) ||
      !(provenance.dt > svmp::FE::Real{0.0})) {
    return false;
  }
  // This tuple records the exact Jansen--Whiting--Hulbert rho-infinity family
  // produced by TimeLoop, not arbitrary generalized-alpha coefficients.
  // alpha_m is not a convex weight: it exceeds one for rho_inf < 1/3.
  const auto expected_alpha_m =
      svmp::FE::Real{2.0} * provenance.alpha_f - svmp::FE::Real{0.5};
  const auto expected_gamma = provenance.alpha_f;
  const auto scale = std::max(
      {svmp::FE::Real{1.0},
       std::abs(provenance.alpha_m),
       std::abs(provenance.alpha_f),
       std::abs(provenance.gamma),
       std::abs(expected_alpha_m),
       std::abs(expected_gamma)});
  const auto tolerance = svmp::FE::Real{512.0} *
                         std::numeric_limits<svmp::FE::Real>::epsilon() *
                         scale;
  return std::abs(provenance.alpha_m - expected_alpha_m) <= tolerance &&
         std::abs(provenance.gamma - expected_gamma) <= tolerance;
}

bool contactStageRealValuesMatch(
    svmp::FE::Real left, svmp::FE::Real right) noexcept
{
  if (!std::isfinite(left) || !std::isfinite(right)) {
    return false;
  }
  const auto scale = std::max(
      {svmp::FE::Real{1.0}, std::abs(left), std::abs(right)});
  return std::abs(left - right) <=
         svmp::FE::Real{512.0} *
             std::numeric_limits<svmp::FE::Real>::epsilon() * scale;
}

std::uint64_t acceptedContactStageRevision(
    std::uint64_t previous_state_revision,
    std::uint64_t endpoint_state_revision,
    std::uint64_t snapshot_revision,
    svmp::FE::Real stage_time,
    svmp::FE::Real stage_alpha_f,
    std::span<const svmp::FE::Real> stage_solution,
    const std::optional<FreeSurfaceFirstOrderGeneralizedAlphaProvenance>&
        first_order_generalized_alpha)
{
  constexpr std::uint64_t offset = 1469598103934665603ull;
  constexpr std::uint64_t prime = 1099511628211ull;
  std::uint64_t hash = offset;
  const auto mix = [&hash](std::uint64_t value) {
    hash ^= value;
    hash *= prime;
  };
  const auto real_bits = [](svmp::FE::Real value) {
    if (value == svmp::FE::Real{0.0}) {
      value = svmp::FE::Real{0.0};
    }
    std::uint64_t bits = 0u;
    static_assert(sizeof(value) <= sizeof(bits));
    std::memcpy(&bits, &value, sizeof(value));
    return bits;
  };
  mix(previous_state_revision);
  mix(endpoint_state_revision);
  mix(snapshot_revision);
  mix(real_bits(stage_time));
  mix(real_bits(stage_alpha_f));
  mix(first_order_generalized_alpha.has_value() ? 1u : 0u);
  if (first_order_generalized_alpha.has_value()) {
    mix(real_bits(first_order_generalized_alpha->alpha_m));
    mix(real_bits(first_order_generalized_alpha->alpha_f));
    mix(real_bits(first_order_generalized_alpha->gamma));
    mix(real_bits(first_order_generalized_alpha->dt));
  }
  mix(static_cast<std::uint64_t>(stage_solution.size()));
  for (const auto value : stage_solution) {
    mix(real_bits(value));
  }
  return hash == 0u ? 1u : hash;
}

std::vector<FreeSurfaceAcceptedContactStageState>
evaluateAcceptedFreeSurfaceContactStages(
    const svmp::FE::systems::FESystem& system,
    std::span<const FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    svmp::FE::Real stage_time,
    svmp::FE::Real stage_alpha_f,
    std::uint64_t previous_state_revision,
    std::uint64_t endpoint_state_revision,
    std::span<const svmp::FE::Real> stage_solution,
    const std::optional<FreeSurfaceFirstOrderGeneralizedAlphaProvenance>&
        first_order_generalized_alpha)
{
  const auto comm = activeSystemCommunicator(system);
  const auto dynamic_count = static_cast<std::size_t>(std::count_if(
      declarations.begin(),
      declarations.end(),
      [](const auto& declaration) {
        return !declaration.parameters.dynamic_contact_coefficients.empty();
      }));
  std::vector<FreeSurfaceAcceptedContactStageState>
      stages;
  std::exception_ptr local_preparation_failure;
  try {
    stages.reserve(dynamic_count);
  } catch (...) {
    local_preparation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_preparation_failure,
      comm,
      "accepted_contact_stage_evaluation");
  const double local_dynamic_count =
      static_cast<double>(dynamic_count);
  if (collectiveMinDouble(local_dynamic_count, comm) !=
      collectiveMaxDouble(local_dynamic_count, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Dynamic contact-stage declarations differ across the FE communicator.");
  }
  if (dynamic_count == 0) {
    return {};
  }
  const bool local_provenance_incomplete =
      !std::isfinite(stage_time) || !std::isfinite(stage_alpha_f) ||
      !(stage_alpha_f > svmp::FE::Real{0.0}) ||
      stage_alpha_f > svmp::FE::Real{1.0} ||
      previous_state_revision == 0u || endpoint_state_revision == 0u ||
      (first_order_generalized_alpha.has_value() &&
       (!firstOrderGeneralizedAlphaContactProvenanceValid(
            *first_order_generalized_alpha) ||
        !contactStageRealValuesMatch(
            stage_alpha_f,
            first_order_generalized_alpha->alpha_f)));
  if (collectiveAnyTrue(local_provenance_incomplete, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage provenance is incomplete.");
  }
  const double local_solution_size =
      static_cast<double>(stage_solution.size());
  if (collectiveMinDouble(local_solution_size, comm) !=
      collectiveMaxDouble(local_solution_size, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Contact-stage solution layouts differ across the FE communicator.");
  }
  const auto* context = system.cutIntegrationContext();
  if (collectiveMinDouble(context != nullptr ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage recording requires an authoritative geometry snapshot on every rank.");
  }

  for (const auto& declaration : declarations) {
    if (declaration.parameters.dynamic_contact_coefficients.empty()) {
      continue;
    }
    const double local_marker =
        static_cast<double>(declaration.interface_marker);
    const double local_velocity_field =
        static_cast<double>(declaration.velocity_field);
    const double local_contact_count = static_cast<double>(
        declaration.parameters.dynamic_contact_coefficients.size());
    if (collectiveMinDouble(local_marker, comm) !=
            collectiveMaxDouble(local_marker, comm) ||
        collectiveMinDouble(local_velocity_field, comm) !=
            collectiveMaxDouble(local_velocity_field, comm) ||
        collectiveMinDouble(local_contact_count, comm) !=
            collectiveMaxDouble(local_contact_count, comm)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Dynamic contact-stage declaration ordering differs across the FE communicator.");
    }
    for (const auto& coefficient :
         declaration.parameters.dynamic_contact_coefficients) {
      const double boundary_marker =
          static_cast<double>(coefficient.boundary_marker);
      const double equilibrium_angle =
          static_cast<double>(
              coefficient.equilibrium_contact_angle_radians);
      const double contact_law = static_cast<double>(coefficient.law);
      const double mobility = static_cast<double>(coefficient.mobility);
      const double slip_length =
          static_cast<double>(coefficient.slip_length);
      const double dynamic_viscosity =
          static_cast<double>(coefficient.dynamic_viscosity);
      if (collectiveMinDouble(boundary_marker, comm) !=
              collectiveMaxDouble(boundary_marker, comm) ||
          collectiveMinDouble(equilibrium_angle, comm) !=
              collectiveMaxDouble(equilibrium_angle, comm) ||
          collectiveMinDouble(contact_law, comm) !=
              collectiveMaxDouble(contact_law, comm) ||
          collectiveMinDouble(mobility, comm) !=
              collectiveMaxDouble(mobility, comm) ||
          collectiveMinDouble(slip_length, comm) !=
              collectiveMaxDouble(slip_length, comm) ||
          collectiveMinDouble(dynamic_viscosity, comm) !=
              collectiveMaxDouble(dynamic_viscosity, comm)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Dynamic contact-stage wall coefficients differ across the FE communicator for marker " +
            std::to_string(declaration.interface_marker) + ".");
      }
    }
    const bool local_marker_available =
        context->hasLevelSetGeometrySnapshotForMarker(
            declaration.interface_marker);
    if (collectiveMinDouble(local_marker_available ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage marker " +
          std::to_string(declaration.interface_marker) +
          " is missing an authoritative geometry snapshot on at least one rank.");
    }
    const auto snapshot_revision =
        context->levelSetGeometrySnapshotRevisionForMarker(
            declaration.interface_marker);
    const auto [minimum_revision, maximum_revision] =
        collectiveMinMaxUint64(snapshot_revision, comm);
    if (minimum_revision != maximum_revision) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage geometry revision differs across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    const auto& snapshots = context->levelSetGeometrySnapshots();
    const auto found = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().snapshot_revision_key ==
                     snapshot_revision;
        });
    if (collectiveMinDouble(found != snapshots.end() ? 1.0 : 0.0, comm) !=
        1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage snapshot storage is incomplete across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    bool local_snapshot_current = true;
    try {
      context->assertAllLevelSetGeometrySnapshotsCurrent(
          system.meshAccess());
    } catch (const std::exception&) {
      local_snapshot_current = false;
    }
    if (collectiveMinDouble(local_snapshot_current ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage snapshot is stale or incomplete on at least one rank for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    const auto& snapshot = **found;
    std::optional<
        FreeSurfaceDynamicContactState>
        local_state;
    try {
      const auto velocity = makeSnapshotVectorFieldEvaluator(
          snapshotFieldAccess(system, declaration.velocity_field),
          stage_solution,
          "[svMultiPhysics::Application] Accepted contact-stage",
          /*allow_scalar_product_fast_path=*/true, "velocity");
      local_state =
          evaluateFreeSurfaceDynamicContactState(
              snapshot, declaration.parameters, velocity);
    } catch (const std::exception&) {
      local_state.reset();
    }
    if (collectiveMinDouble(local_state.has_value() ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Dynamic contact-stage evaluation failed on at least one rank for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    auto state = std::move(*local_state);
    const double local_wall_count =
        static_cast<double>(state.walls.size());
    if (collectiveMinDouble(local_wall_count, comm) !=
        collectiveMaxDouble(local_wall_count, comm)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Dynamic contact-stage wall ordering differs across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }

    const auto global_sum = [&comm](svmp::FE::Real local) {
      return static_cast<svmp::FE::Real>(
          collectiveSumDouble(static_cast<double>(local), comm));
    };
    for (auto& wall : state.walls) {
      wall.owned_quadrature_point_count = collectiveSumSize(
          wall.owned_quadrature_point_count, comm);
      wall.owned_advancing_point_count = collectiveSumSize(
          wall.owned_advancing_point_count, comm);
      wall.owned_receding_point_count = collectiveSumSize(
          wall.owned_receding_point_count, comm);
      wall.owned_stationary_point_count = collectiveSumSize(
          wall.owned_stationary_point_count, comm);
      wall.owned_contact_measure =
          global_sum(wall.owned_contact_measure);
      wall.dynamic_angle_integral =
          global_sum(wall.dynamic_angle_integral);
      wall.dynamic_cosine_integral =
          global_sum(wall.dynamic_cosine_integral);
      wall.contact_speed_integral =
          global_sum(wall.contact_speed_integral);
      wall.contact_speed_squared_integral =
          global_sum(wall.contact_speed_squared_integral);
      wall.constitutive_residual_integral =
          global_sum(wall.constitutive_residual_integral);
      wall.absolute_constitutive_residual_integral =
          global_sum(wall.absolute_constitutive_residual_integral);
      wall.line_friction_dissipation =
          global_sum(wall.line_friction_dissipation);
      wall.owned_wetted_wall_quadrature_point_count = collectiveSumSize(
          wall.owned_wetted_wall_quadrature_point_count, comm);
      wall.owned_wetted_wall_measure =
          global_sum(wall.owned_wetted_wall_measure);
      wall.wall_slip_speed_integral =
          global_sum(wall.wall_slip_speed_integral);
      wall.wall_slip_speed_squared_integral =
          global_sum(wall.wall_slip_speed_squared_integral);
      wall.wall_slip_dissipation =
          global_sum(wall.wall_slip_dissipation);
      for (std::size_t component = 0; component < 3u; ++component) {
        wall.wall_tangential_velocity_integral[component] =
            global_sum(
                wall.wall_tangential_velocity_integral[component]);
        wall.contact_position_integral[component] =
            global_sum(wall.contact_position_integral[component]);
        wall.wall_normal_integral[component] =
            global_sum(wall.wall_normal_integral[component]);
        wall.footprint_direction_integral[component] =
            global_sum(wall.footprint_direction_integral[component]);
        wall.contact_line_tangent_integral[component] =
            global_sum(wall.contact_line_tangent_integral[component]);
      }
    }
    std::exception_ptr local_completion_failure;
    try {
      finalizeFreeSurfaceDynamicContactState(state);
      stages.push_back(
          FreeSurfaceAcceptedContactStageState{
              .stage_time = stage_time,
              .stage_alpha_f = stage_alpha_f,
              .first_order_generalized_alpha =
                  first_order_generalized_alpha,
              .previous_state_revision = previous_state_revision,
              .endpoint_state_revision = endpoint_state_revision,
              .stage_state_revision = acceptedContactStageRevision(
                  previous_state_revision,
                  endpoint_state_revision,
                  snapshot_revision,
                  stage_time,
                  stage_alpha_f,
                  stage_solution,
                  first_order_generalized_alpha),
              .geometry_revision = snapshot.revision(),
              .state = std::move(state),
          });
    } catch (...) {
      local_completion_failure = std::current_exception();
    }
    requireCollectivePhasePreparation(
        local_completion_failure,
        comm,
        "accepted_contact_stage_evaluation_completion");
    applyDeclaredStageWallSlipDissipation(
        system, declaration, snapshot, stage_solution, stages.back(), comm);
  }
  return stages;
}

void bindAcceptedFreeSurfaceContactStagesToEndpointRevision(
    std::span<
        FreeSurfaceAcceptedContactStageState> stages,
    std::uint64_t endpoint_state_revision,
    std::span<const svmp::FE::Real> stage_solution,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  const auto [minimum_revision, maximum_revision] =
      collectiveMinMaxUint64(endpoint_state_revision, comm);
  if (endpoint_state_revision == 0u ||
      minimum_revision != maximum_revision) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] The pre-maintenance endpoint state revision must be nonzero and identical across the FE communicator.");
  }
  const auto local_stage_count =
      static_cast<std::uint64_t>(stages.size());
  const auto [minimum_stage_count, maximum_stage_count] =
      collectiveMinMaxUint64(local_stage_count, comm);
  if (minimum_stage_count != maximum_stage_count) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage binding coverage differs across the FE communicator.");
  }

  std::vector<std::uint64_t> bound_stage_revisions;
  std::exception_ptr local_revision_storage_failure;
  try {
    bound_stage_revisions.assign(stages.size(), 0u);
  } catch (...) {
    local_revision_storage_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_revision_storage_failure,
      comm,
      "accepted_contact_stage_bound_revision_storage");
  for (std::size_t index = 0u; index < stages.size(); ++index) {
    const auto& stage = stages[index];
    const auto bound_stage_revision = acceptedContactStageRevision(
        stage.previous_state_revision,
        endpoint_state_revision,
        stage.geometry_revision.snapshot_revision_key,
        stage.stage_time,
        stage.stage_alpha_f,
        stage_solution,
        stage.first_order_generalized_alpha);
    const auto [minimum_stage_revision, maximum_stage_revision] =
        collectiveMinMaxUint64(bound_stage_revision, comm);
    if (minimum_stage_revision != maximum_stage_revision) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage composite revision differs across the FE communicator.");
    }
    bound_stage_revisions[index] = bound_stage_revision;
  }

  for (std::size_t index = 0u; index < stages.size(); ++index) {
    stages[index].endpoint_state_revision = endpoint_state_revision;
    stages[index].stage_state_revision = bound_stage_revisions[index];
  }
}

std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
captureAcceptedContactStageWallConstraints(
    const svmp::FE::systems::FESystem& system,
    std::span<const FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    std::span<const FreeSurfaceAcceptedContactStageState>
        accepted_contact_stages)
{
  std::vector<svmp::FE::level_set::LevelSetWallContactConstraint> captured;
#if FE_HAS_MPI
  const svmp::FE::systems::CollectiveCommunicator comm(system.activeMpiCommunicator());
#else
  const svmp::FE::systems::CollectiveCommunicator comm{};
#endif
  const auto* context = system.cutIntegrationContext();
  if (collectiveMinDouble(context != nullptr ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage constraint capture requires an authoritative geometry snapshot on every rank.");
  }
  bool local_snapshots_current = true;
  try {
    context->assertAllLevelSetGeometrySnapshotsCurrent(
        system.meshAccess());
  } catch (const std::exception&) {
    local_snapshots_current = false;
  }
  if (collectiveMinDouble(local_snapshots_current ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage constraint capture rejected a stale authoritative geometry snapshot.");
  }
  std::map<
      int,
      const FreeSurfaceAcceptedContactStageState*>
      stage_by_interface;
  bool local_stage_map_valid = true;
  for (const auto& stage : accepted_contact_stages) {
    const int marker = stage.geometry_revision.interface_marker;
    local_stage_map_valid =
        marker >= 0 && stage_by_interface.emplace(marker, &stage).second &&
        local_stage_map_valid;
  }
  if (collectiveMinDouble(local_stage_map_valid ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage constraint capture requires unique nonnegative interface markers.");
  }

  for (const auto& declaration : declarations) {
    const bool has_dynamic_contact_law = std::any_of(
        declaration.parameters.dynamic_contact_coefficients.begin(),
        declaration.parameters.dynamic_contact_coefficients.end(),
        [](const auto& coefficient) {
          return coefficient.law ==
                 FreeSurfaceContactLaw::DynamicRenE;
        });
    if (!has_dynamic_contact_law) {
      continue;
    }
    const auto stage_it = stage_by_interface.find(declaration.interface_marker);
    const bool local_stage_available = stage_it != stage_by_interface.end();
    if (collectiveMinDouble(local_stage_available ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage constraint capture is missing interface marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    const auto& stage = *stage_it->second;
    const auto& snapshots = context->levelSetGeometrySnapshots();
    const auto snapshot_it = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().snapshot_revision_key ==
                     stage.geometry_revision.snapshot_revision_key;
        });
    if (collectiveMinDouble(snapshot_it != snapshots.end() ? 1.0 : 0.0, comm) !=
        1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage constraint capture cannot retain the stage geometry for interface marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    const auto& snapshot = **snapshot_it;
    const bool local_revision_valid =
        stage.geometry_revision.complete() &&
        snapshot.revision().snapshot_revision_key ==
            stage.geometry_revision.snapshot_revision_key &&
        snapshot.revision().interface_marker == declaration.interface_marker;
    if (collectiveMinDouble(local_revision_valid ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage constraint capture found mismatched stage geometry provenance.");
    }

    std::set<int> dynamic_contact_walls;
    for (const auto& coefficient :
         declaration.parameters.dynamic_contact_coefficients) {
      if (coefficient.law !=
          FreeSurfaceContactLaw::DynamicRenE) {
        continue;
      }
      dynamic_contact_walls.insert(coefficient.boundary_marker);
    }
    bool local_parent_ids_valid = true;
    bool local_rules_valid = true;
    std::string local_rule_diagnostic;
    std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
        declaration_constraints;
    for (const auto& record : snapshot.rules()) {
      if (record.role != svmp::FE::interfaces::
                             LevelSetGeometryRuleRole::Contact ||
          record.retention != svmp::FE::interfaces::
                                  LevelSetGeometryRetention::Retained ||
          !record.locally_owned) {
        continue;
      }
      if (dynamic_contact_walls.find(record.physical_boundary_marker) ==
          dynamic_contact_walls.end()) {
        continue;
      }
      const auto parent =
          record.reference_rule.provenance.parent_entity_global_id;
      local_parent_ids_valid =
          parent != svmp::FE::INVALID_GLOBAL_INDEX &&
          local_parent_ids_valid;
      if (parent == svmp::FE::INVALID_GLOBAL_INDEX) {
        continue;
      }
      try {
        declaration_constraints.push_back(svmp::FE::level_set::makeAcceptedSnapshotWallConstraint(
            record,
            declaration.interface_marker,
            stage.geometry_revision.snapshot_revision_key));
      } catch (const std::exception& error) {
        local_rules_valid = false;
        local_rule_diagnostic = error.what();
      }
    }
    if (collectiveMinDouble(local_parent_ids_valid ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage constraint capture found a rule without a partition-independent parent identity.");
    }
    if (collectiveMinDouble(local_rules_valid ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage constraint capture rejected a contact rule with incomplete or stale provenance" +
          (local_rule_diagnostic.empty()
               ? std::string{"."}
               : std::string{": "} + local_rule_diagnostic));
    }
    declaration_constraints = svmp::FE::level_set::canonicalizeAcceptedWallConstraints(
        std::move(declaration_constraints),
        comm,
        "Accepted contact-stage constraint capture");
    captured.insert(captured.end(),
                    declaration_constraints.begin(),
                    declaration_constraints.end());
  }
  return captured;
}

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
