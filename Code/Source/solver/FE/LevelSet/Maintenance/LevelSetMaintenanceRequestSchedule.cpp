/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/Maintenance/LevelSetMaintenanceRequestSchedule.h"

#include "Assembly/CutIntegrationContext.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "LevelSet/LevelSetGeometry.h"
#include "LevelSet/LevelSetOptions.h"
#include "Systems/FESystem.h"
#include "Systems/SystemCollectives.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace svmp {
namespace FE {
namespace level_set {

namespace {

void appendMaintenanceScheduleOptionalInt(
    std::vector<std::uint64_t>& words,
    const std::optional<int>& value)
{
  appendMaintenanceScheduleBool(words, value.has_value());
  appendMaintenanceScheduleSigned(
      words, static_cast<std::int64_t>(value.value_or(0)));
}

void appendMaintenanceScheduleOptionalReal(
    std::vector<std::uint64_t>& words,
    const std::optional<svmp::FE::Real>& value)
{
  appendMaintenanceScheduleBool(words, value.has_value());
  appendMaintenanceScheduleReal(
      words, static_cast<double>(value.value_or(0.0)));
}

void appendMaintenanceScheduleActiveCutRequest(
    std::vector<std::uint64_t>& words,
    const std::optional<ActiveCutVolumeRequest>& request)
{
  appendMaintenanceScheduleBool(words, request.has_value());
  if (!request.has_value()) {
    return;
  }
  appendMaintenanceScheduleString(
      words, request->level_set_field_name);
  appendMaintenanceScheduleString(words, request->domain_id);
  appendMaintenanceScheduleString(words, request->equation_type);
  appendMaintenanceScheduleEnum(words, request->origin);
  appendMaintenanceScheduleSigned(
      words, request->requested_interface_marker);
  appendMaintenanceScheduleReal(words, request->isovalue);
  appendMaintenanceScheduleOptionalInt(
      words, request->quadrature_order);
  appendMaintenanceScheduleOptionalInt(
      words, request->interface_quadrature_order);
  appendMaintenanceScheduleOptionalInt(
      words, request->volume_quadrature_order);
  appendMaintenanceScheduleEnum(words, request->geometry_mode);
  appendMaintenanceScheduleEnum(
      words, request->implicit_cut_backend);
  appendMaintenanceScheduleEnum(
      words, request->implicit_cut_fallback_policy);
  appendMaintenanceScheduleEnum(
      words, request->geometry_tangent_policy);
  appendMaintenanceScheduleBool(
      words, request->geometry_tangent_policy_explicit);
  appendMaintenanceScheduleReal(
      words, request->implicit_cut_root_tolerance);
  appendMaintenanceScheduleReal(
      words, request->implicit_cut_root_coordinate_tolerance);
  appendMaintenanceScheduleSigned(
      words, request->implicit_cut_root_max_iterations);
  appendMaintenanceScheduleSigned(
      words, request->implicit_cut_max_subdivision_depth);
  appendMaintenanceScheduleSigned(
      words, request->affected_cell_neighborhood_layers);
  appendMaintenanceScheduleEnum(words, request->active_side);
  appendMaintenanceScheduleEnum(words, request->volume_retention);
  appendMaintenanceScheduleBool(
      words, request->allow_corner_linearized_geometry);
  appendMaintenanceScheduleBool(
      words,
      request->require_production_qualified_implicit_cut_backend);
}

void appendCanonicalFreeSurfaceGeometryRevision(
    std::vector<std::uint64_t>& words,
    const svmp::FE::interfaces::FreeSurfaceGeometryRevision& revision)
{
  appendMaintenanceScheduleString(words, revision.source_id);
  appendMaintenanceScheduleString(words, revision.domain_id);
  appendMaintenanceScheduleSigned(words, revision.interface_marker);
  appendMaintenanceScheduleReal(words, revision.isovalue);
  appendMaintenanceScheduleWord(
      words, revision.source_layout_revision);
  appendMaintenanceScheduleWord(
      words, revision.source_value_revision);
  appendMaintenanceScheduleWord(
      words, revision.mesh_geometry_revision);
  appendMaintenanceScheduleWord(
      words, revision.mesh_topology_revision);
  appendMaintenanceScheduleWord(
      words, revision.ownership_revision);
  appendMaintenanceScheduleWord(
      words, revision.numbering_revision);
  appendMaintenanceScheduleWord(
      words, revision.quadrature_policy_key);
  appendMaintenanceScheduleWord(
      words, revision.snapshot_revision_key);
}

} // namespace

void appendMaintenanceScheduleGraphMetadata(
    std::vector<std::uint64_t>& words,
    const std::optional<
        svmp::FE::level_set::LevelSetP1PhaseTransportGraph>& graph)
{
  appendMaintenanceScheduleBool(words, graph.has_value());
  if (!graph.has_value()) {
    return;
  }
  appendMaintenanceScheduleBool(
      words, static_cast<bool>(graph->collective_state));
  appendMaintenanceScheduleBool(words, graph->success);
  appendMaintenanceScheduleBool(
      words, graph->partition_of_unity_satisfied);
  appendMaintenanceScheduleBool(
      words, graph->gradient_partition_satisfied);
  appendMaintenanceScheduleBool(
      words, graph->positive_control_volumes_satisfied);
  appendMaintenanceScheduleBool(
      words, graph->gradient_row_sum_satisfied);
  appendMaintenanceScheduleBool(
      words, graph->measure_closure_satisfied);
  appendMaintenanceScheduleBool(
      words, graph->edge_ownership_satisfied);
  appendMaintenanceScheduleBool(words, graph->distributed);
  appendMaintenanceScheduleBool(
      words, graph->replicated_sparse_graph);
  appendMaintenanceScheduleSigned(words, graph->dimension);
  appendMaintenanceScheduleSigned(words, graph->parallel_size);
  appendMaintenanceScheduleSigned(
      words, graph->maximum_quadrature_order);
  appendMaintenanceScheduleWord(
      words, static_cast<std::uint64_t>(graph->cells));
  appendMaintenanceScheduleWord(
      words, static_cast<std::uint64_t>(graph->nodes));
  appendMaintenanceScheduleWord(
      words,
      static_cast<std::uint64_t>(
          graph->lumped_control_volume.size()));
  appendMaintenanceScheduleWord(
      words,
      static_cast<std::uint64_t>(
          graph->diagonal_gradient.size()));
  appendMaintenanceScheduleWord(
      words,
      static_cast<std::uint64_t>(
          graph->boundary_column_sum.size()));
  appendMaintenanceScheduleWord(
      words, static_cast<std::uint64_t>(graph->edges.size()));
  // The graph caches rank-local mesh event counters so each partition can
  // decide when its own graph is stale.  Equally valid partitions can advance
  // those counters a different number of times, so they are not replicated
  // request semantics and must not participate in exact schedule consensus.
  // The graph's replicated shape, invariant flags, and FE DOF-layout identity
  // above/below remain part of the collective preflight.
  appendMaintenanceScheduleWord(words, graph->dof_layout_revision);
}

std::uint64_t levelSetMaintenanceRequestActionBits(
    const LevelSetMaintenanceConfiguration& configuration,
    int completed_step)
{
  const bool accepted_kinematic_due =
      !configuration.transport.conservative_phase.enabled &&
      svmp::FE::level_set::shouldApplyLevelSetKinematicReconciliation(
          configuration.transport.kinematic_reconciliation,
          completed_step);
  const bool accepted_patch_bounds_due =
      !configuration.transport.conservative_phase.enabled &&
      svmp::FE::level_set::shouldApplyLevelSetSignDefinitePatchBounds(
          configuration.transport.sign_definite_patch_bounds,
          completed_step);
  const bool accepted_redistancing_due =
      !configuration.transport.conservative_phase.enabled &&
      svmp::FE::level_set::shouldApplyLevelSetInterfacePreservingRedistancing(
          configuration.transport.interface_preserving_redistancing,
          completed_step);
  const bool artifact_due =
      configuration.transport.conservative_phase.enabled &&
      configuration.transport.conservative_phase.write_flux_artifacts &&
      configuration.transport.conservative_phase.flux_artifact_cadence_steps > 0 &&
      completed_step >= 0 &&
      static_cast<std::uint64_t>(completed_step) %
              static_cast<std::uint64_t>(
                  configuration.transport.conservative_phase
                      .flux_artifact_cadence_steps) ==
          0u;
  const bool functional_due =
      configuration.transport.conservative_phase.enabled ||
      accepted_kinematic_due ||
      accepted_patch_bounds_due ||
      accepted_redistancing_due;
  std::uint64_t bits = 0u;
  bits |= configuration.transport.conservative_phase.enabled ? (1ull << 0u) : 0u;
  // Bit 1 described the retired bound-preserving limiter and bit 2 the
  // retired conservative-phase reinitialization.
  bits |=
      configuration.transport.velocity.source ==
              svmp::FE::level_set::LevelSetVelocitySource::
                  ConstantVector
          ? (1ull << 3u)
          : (1ull << 4u);
  // Bit 5 described the retired projection reinitialization and bits 6 and
  // 7 the retired global volume correction.
  bits |= functional_due ? (1ull << 8u) : 0u;
  bits |= artifact_due ? (1ull << 9u) : 0u;
  bits |= configuration.transport.conservative_phase.reconcile_geometry
              ? (1ull << 10u)
              : 0u;
  // Bit 11 is the application's (static capillary equilibrium).
  bits |= accepted_kinematic_due ? (1ull << 12u) : 0u;
  bits |= accepted_patch_bounds_due ? (1ull << 13u) : 0u;
  bits |= accepted_redistancing_due ? (1ull << 14u) : 0u;
  return bits;
}

void appendLevelSetMaintenanceScheduleHeader(
    std::vector<std::uint64_t>& words,
    LevelSetMaintenanceScheduleStage stage,
    int completed_step,
    std::size_t request_count,
    std::uint64_t aggregate_action_bits)
{
  appendMaintenanceScheduleWord(words, 4u);
  appendMaintenanceScheduleEnum(words, stage);
  appendMaintenanceScheduleSigned(words, completed_step);
  appendMaintenanceScheduleWord(
      words, static_cast<std::uint64_t>(request_count));
  appendMaintenanceScheduleWord(words, aggregate_action_bits);
}

bool appendLevelSetMaintenanceConfigurationScheduleWords(
    std::vector<std::uint64_t>& words,
    const LevelSetMaintenanceConfiguration& configuration)
{
  appendMaintenanceScheduleString(
      words, configuration.transport.level_set.field_name);
  appendMaintenanceScheduleReal(words, configuration.isovalue);
  appendMaintenanceScheduleEnum(words, configuration.transport.transport_form);

  appendMaintenanceScheduleString(
      words, configuration.transport.velocity.field_name);
  appendMaintenanceScheduleEnum(words, configuration.transport.velocity.source);
  appendMaintenanceScheduleBool(
      words, configuration.transport.velocity.auto_register_field);
  appendMaintenanceScheduleBool(
      words, static_cast<bool>(configuration.transport.velocity.space));
  const bool supported = !configuration.transport.velocity.space;
  for (const auto value : configuration.transport.velocity.constant_value) {
    appendMaintenanceScheduleReal(
        words, static_cast<double>(value));
  }
  appendMaintenanceScheduleString(
      words,
      configuration.transport.velocity
          .algebraic_extension_source_field_name);
  appendMaintenanceScheduleSigned(
      words, configuration.transport.velocity.material_interface_marker);

  appendMaintenanceScheduleBool(
      words, configuration.transport.conservative_phase.enabled);
  appendMaintenanceScheduleString(
      words,
      configuration.transport.conservative_phase.liquid_indicator.field_name);
  appendMaintenanceScheduleEnum(
      words,
      configuration.transport.conservative_phase.liquid_indicator.source);
  appendMaintenanceScheduleBool(
      words,
      configuration.transport.conservative_phase.liquid_indicator
          .auto_register_field);
  appendMaintenanceScheduleEnum(
      words, configuration.transport.conservative_phase.liquid_side);
  appendMaintenanceScheduleReal(
      words, configuration.transport.conservative_phase.invariant_tolerance);
  appendMaintenanceScheduleReal(
      words,
      configuration.transport.conservative_phase.component_activity_tolerance);
  appendMaintenanceScheduleReal(
      words, configuration.transport.conservative_phase.maximum_courant);
  appendMaintenanceScheduleBool(
      words,
      configuration.transport.conservative_phase.enforce_courant_limit);
  appendMaintenanceScheduleBool(
      words,
      configuration.transport.conservative_phase.require_constant_preservation);
  appendMaintenanceScheduleReal(
      words,
      configuration.transport.conservative_phase.momentum_relative_tolerance);
  appendMaintenanceScheduleBool(
      words, configuration.transport.conservative_phase.write_flux_artifacts);
  appendMaintenanceScheduleSigned(
      words,
      configuration.transport.conservative_phase
          .flux_artifact_cadence_steps);
  appendMaintenanceScheduleBool(
      words,
      configuration.transport.conservative_phase
          .classify_nonprimary_components_as_satellites);
  appendMaintenanceScheduleEnum(
      words,
      configuration.transport.conservative_phase.boundary_flux_policy);
  appendMaintenanceScheduleWord(
      words,
      static_cast<std::uint64_t>(
          configuration.transport.conservative_phase.fixed_flux_regions.size()));
  for (const auto& region :
       configuration.transport.conservative_phase.fixed_flux_regions) {
    appendMaintenanceScheduleString(words, region.name);
    appendMaintenanceScheduleEnum(words, region.kind);
    for (const auto value : region.minimum) {
      appendMaintenanceScheduleReal(words, value);
    }
    for (const auto value : region.maximum) {
      appendMaintenanceScheduleReal(words, value);
    }
  }
  appendMaintenanceScheduleReal(
      words,
      configuration.transport.conservative_phase
          .impermeable_normal_velocity_tolerance);
  appendMaintenanceScheduleBool(
      words,
      configuration.transport.conservative_phase
          .pointwise_impermeable_velocity_tolerance_explicitly_requested);
  appendMaintenanceScheduleBool(
      words, configuration.transport.conservative_phase.reconcile_geometry);
  appendMaintenanceScheduleReal(
      words,
      configuration.transport.conservative_phase.geometry_measure_tolerance);
  appendMaintenanceScheduleSigned(
      words,
      configuration.transport.conservative_phase
          .geometry_correction_max_iterations);
  appendMaintenanceScheduleReal(
      words,
      configuration.transport.conservative_phase
          .maximum_geometry_displacement_fraction);

  appendMaintenanceScheduleWord(
      words,
      static_cast<std::uint64_t>(
          configuration.open_boundaries.size()));
  for (const auto& boundary : configuration.open_boundaries) {
    appendMaintenanceScheduleString(
        words, boundary.face_name);
    appendMaintenanceScheduleBool(words, boundary.inflow);
    appendMaintenanceScheduleOptionalReal(
        words, boundary.literal_inflow_value);
  }

  appendMaintenanceScheduleBool(
      words,
      configuration.transport.kinematic_reconciliation.enabled);
  appendMaintenanceScheduleBool(
      words,
      configuration.transport.sign_definite_patch_bounds.enabled);
  if (configuration.transport.interface_preserving_redistancing.enabled) {
    // Appended only when enabled so that existing schedules keep their words.
    appendMaintenanceScheduleBool(words, true);
  }
  appendMaintenanceScheduleActiveCutRequest(
      words, configuration.volume_cut_request);
  return supported;
}

void appendLevelSetCurvatureProjectionScheduleWords(
    std::vector<std::uint64_t>& words,
    const LevelSetMaintenanceConfiguration& configuration,
    const LevelSetCurvatureProjectionOptions& curvature_options)
{
  appendMaintenanceScheduleBool(
      words, configuration.curvature_projection_enabled);
  appendMaintenanceScheduleString(
      words, configuration.curvature_field_name);
  appendMaintenanceScheduleSigned(
      words, configuration.curvature_projection_cadence_steps);
  appendMaintenanceScheduleReal(
      words, curvature_options.isovalue);
  appendMaintenanceScheduleReal(
      words,
      curvature_options.gradient_tolerance);
  appendMaintenanceScheduleReal(
      words,
      curvature_options.normal_equation_tolerance);
  appendMaintenanceScheduleReal(
      words,
      curvature_options
          .max_normalized_fit_residual);
  appendMaintenanceScheduleSigned(
      words,
      curvature_options.max_neighbor_rings);
  appendMaintenanceScheduleSigned(
      words,
      curvature_options
          .max_neighbor_fallback_vertices);
  appendMaintenanceScheduleSigned(
      words,
      curvature_options
          .max_zero_fallback_vertices);
  appendMaintenanceScheduleReal(
      words,
      curvature_options
          .supplemental_sample_weight);
  appendMaintenanceScheduleEnum(
      words, curvature_options.recovery_mode);
  appendMaintenanceScheduleReal(
      words,
      curvature_options
          .kinematic_area_gradient_filter_coefficient);
  appendMaintenanceScheduleEnum(
      words, curvature_options.kinematic_area_gradient_mass);
  appendMaintenanceScheduleBool(
      words,
      curvature_options
          .kinematic_area_gradient_negative_liquid_side);
  appendMaintenanceScheduleSigned(
      words,
      static_cast<std::int64_t>(
          curvature_options
              .kinematic_area_gradient_young_walls.size()));
  for (const auto& wall :
       curvature_options
           .kinematic_area_gradient_young_walls) {
    appendMaintenanceScheduleSigned(words, wall.boundary_marker);
    appendMaintenanceScheduleReal(
        words, wall.equilibrium_contact_angle_radians);
  }
  appendMaintenanceScheduleReal(
      words,
      curvature_options.narrow_band_width);
  appendMaintenanceScheduleSigned(
      words,
      curvature_options.smoothing_iterations);
  appendMaintenanceScheduleReal(
      words,
      curvature_options.smoothing_relaxation);
  appendMaintenanceScheduleEnum(
      words, curvature_options.smoothing_mode);
}

void appendCanonicalLevelSetMaintenanceScheduleSection(
    std::vector<std::uint64_t>& words,
    const CanonicalLevelSetMaintenanceSchedule& schedule)
{
  // A distinct section tag and length make the enclosing commit-state
  // serialization unambiguous even when the request schema grows.
  appendMaintenanceScheduleWord(
      words, 0x4c534d5343484544ull);
  appendMaintenanceScheduleBool(words, schedule.supported);
  appendMaintenanceScheduleWord(
      words,
      static_cast<std::uint64_t>(schedule.words.size()));
  words.insert(
      words.end(),
      schedule.words.begin(),
      schedule.words.end());
}

CanonicalLevelSetMaintenanceGeometryState
canonicalLevelSetMaintenanceGeometryState(
    const systems::FESystem* system,
    std::span<const LevelSetMaintenanceRequest> requests)
{
  CanonicalLevelSetMaintenanceGeometryState state;
  auto& words = state.words;
  appendMaintenanceScheduleWord(words, 1u);
  appendMaintenanceScheduleBool(
      words, system != nullptr);
  if (system == nullptr) {
    state.supported = false;
    return state;
  }

  const auto operator_revision =
      system->operatorRevisionSnapshot();
  appendMaintenanceScheduleBool(words, operator_revision.valid);
  // Raw operator mesh event counters are partition-local: equally valid
  // partitions can process different numbers of local geometry, topology,
  // ownership, or numbering events.  The full authoritative snapshot
  // revisions serialized below carry the communicator-replicated mesh
  // geometry/topology/ownership/numbering identities instead.
  appendMaintenanceScheduleWord(
      words, operator_revision.fe_layout.space);
  appendMaintenanceScheduleWord(
      words, operator_revision.fe_layout.dof_layout);
  appendMaintenanceScheduleWord(
      words, operator_revision.fe_layout.constraint_layout);
  appendMaintenanceScheduleWord(
      words, operator_revision.fe_layout.block_layout);
  appendMaintenanceScheduleWord(
      words, operator_revision.system_layout_key);
  appendMaintenanceScheduleEnum(
      words, operator_revision.geometry_state);
  appendMaintenanceScheduleEnum(
      words, operator_revision.geometry_use);

  const auto* context =
      system->cutIntegrationContext();
  appendMaintenanceScheduleBool(words, context != nullptr);
  appendMaintenanceScheduleWord(
      words, static_cast<std::uint64_t>(requests.size()));
  for (const auto& request : requests) {
    appendMaintenanceScheduleBool(
        words, request.configuration->volume_cut_request.has_value());
    if (!request.configuration->volume_cut_request.has_value()) {
      continue;
    }

    const auto marker =
        resolvedActiveCutVolumeInterfaceMarker(
            *system, *request.configuration->volume_cut_request);
    appendMaintenanceScheduleBool(words, marker.has_value());
    appendMaintenanceScheduleSigned(
        words, marker.value_or(-1));
    if (!marker.has_value() || context == nullptr) {
      state.supported = false;
      appendMaintenanceScheduleBool(words, false);
      appendMaintenanceScheduleBool(words, false);
      appendMaintenanceScheduleBool(words, false);
      continue;
    }

    const bool has_snapshot_binding =
        context->hasFreeSurfaceGeometrySnapshotForMarker(*marker);
    appendMaintenanceScheduleBool(
        words, has_snapshot_binding);
    const auto bound_snapshot_revision =
        has_snapshot_binding
            ? context->freeSurfaceGeometrySnapshotRevisionForMarker(
                  *marker)
            : 0u;
    appendMaintenanceScheduleWord(
        words, bound_snapshot_revision);

    const bool has_expected_source_revision =
        context->hasExpectedGeneratedSourceValueRevision(*marker);
    appendMaintenanceScheduleBool(
        words, has_expected_source_revision);
    appendMaintenanceScheduleWord(
        words,
        has_expected_source_revision
            ? context->expectedGeneratedSourceValueRevision(*marker)
            : 0u);

    const auto& snapshots =
        context->freeSurfaceGeometrySnapshots();
    const auto found = std::find_if(
        snapshots.begin(), snapshots.end(),
        [&](const auto& snapshot) {
          return snapshot &&
                 snapshot->revision().interface_marker == *marker &&
                 snapshot->revision().snapshot_revision_key ==
                     bound_snapshot_revision;
        });
    const bool has_bound_snapshot =
        has_snapshot_binding && found != snapshots.end();
    appendMaintenanceScheduleBool(
        words, has_bound_snapshot);
    if (has_bound_snapshot) {
      appendCanonicalFreeSurfaceGeometryRevision(
          words, (*found)->revision());
    } else {
      state.supported = false;
    }
  }

  using GeometrySnapshot =
      svmp::FE::interfaces::FreeSurfaceGeometrySnapshot;
  std::vector<const GeometrySnapshot*> ordered_snapshots;
  if (context != nullptr) {
    const auto& snapshots =
        context->freeSurfaceGeometrySnapshots();
    ordered_snapshots.reserve(snapshots.size());
    for (const auto& snapshot : snapshots) {
      ordered_snapshots.push_back(snapshot.get());
    }
    std::sort(
        ordered_snapshots.begin(), ordered_snapshots.end(),
        [](const GeometrySnapshot* left,
           const GeometrySnapshot* right) {
          if (left == nullptr || right == nullptr) {
            return left == nullptr && right != nullptr;
          }
          const auto& lhs = left->revision();
          const auto& rhs = right->revision();
          return std::tuple{
                     lhs.interface_marker,
                     lhs.source_id,
                     lhs.domain_id,
                     std::bit_cast<std::uint64_t>(
                         static_cast<double>(lhs.isovalue)),
                     lhs.source_layout_revision,
                     lhs.source_value_revision,
                     lhs.snapshot_revision_key} <
                 std::tuple{
                     rhs.interface_marker,
                     rhs.source_id,
                     rhs.domain_id,
                     std::bit_cast<std::uint64_t>(
                         static_cast<double>(rhs.isovalue)),
                     rhs.source_layout_revision,
                     rhs.source_value_revision,
                     rhs.snapshot_revision_key};
        });
  }
  appendMaintenanceScheduleWord(
      words,
      static_cast<std::uint64_t>(
          ordered_snapshots.size()));
  for (const auto* snapshot : ordered_snapshots) {
    appendMaintenanceScheduleBool(
        words, snapshot != nullptr);
    if (snapshot == nullptr) {
      state.supported = false;
      continue;
    }
    appendCanonicalFreeSurfaceGeometryRevision(
        words, snapshot->revision());
  }
  return state;
}

void appendCanonicalLevelSetMaintenanceGeometrySection(
    std::vector<std::uint64_t>& words,
    const CanonicalLevelSetMaintenanceGeometryState& geometry)
{
  appendMaintenanceScheduleWord(
      words, 0x4c534d47454f4d31ull);
  appendMaintenanceScheduleBool(words, geometry.supported);
  appendMaintenanceScheduleWord(
      words,
      static_cast<std::uint64_t>(geometry.words.size()));
  words.insert(
      words.end(),
      geometry.words.begin(),
      geometry.words.end());
}

void requireCollectiveLevelSetMaintenanceRequestSchedule(
    const CanonicalLevelSetMaintenanceSchedule& schedule,
    const systems::CollectiveCommunicator& comm)
{
  const bool identical =
      collectiveLevelSetMaintenanceCanonicalWordsAgree(
          schedule.words, comm);
  if (!identical) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Ordered level-set maintenance "
        "request schedule differs across the FE communicator.");
  }
  if (!schedule.supported) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Maintenance request schedule "
        "preflight does not support an embedded velocity FunctionSpace; "
        "production requests must resolve velocity fields through the FE "
        "system.");
  }
}

std::uint64_t levelSetMaintenanceAlgebraicRevision(
    std::span<const Real> state) noexcept
{
  std::uint64_t revision = 1469598103934665603ull;
  constexpr std::uint64_t prime = 1099511628211ull;
  const auto mix = [&](std::uint64_t value) {
    for (std::size_t byte = 0u; byte < sizeof(value); ++byte) {
      revision ^= (value >> (byte * 8u)) & 0xffu;
      revision *= prime;
    }
  };
  mix(static_cast<std::uint64_t>(state.size()));
  for (const auto value : state) {
    std::uint64_t bits = 0u;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    mix(bits);
  }
  return revision == 0u ? 1u : revision;
}

std::uint64_t collectiveLevelSetMaintenanceAlgebraicRevision(
    std::span<const Real> state,
    const systems::CollectiveCommunicator& comm)
{
  const auto revision = levelSetMaintenanceAlgebraicRevision(state);
  const auto [minimum, maximum] =
      systems::collectiveMinMaxUint64(revision, comm);
  if (minimum != maximum) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Level-set maintenance algebraic state revision differs across the FE communicator.");
  }
  return revision;
}

} // namespace level_set
} // namespace FE
} // namespace svmp
