/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactStages.h"

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceFieldEvaluators.h"

#include "FE/Assembly/CutIntegrationContext.h"
#include "FE/Assembly/GlobalSystemView.h"
#include "FE/Geometry/CutQuadrature.h"
#include "FE/LevelSet/Maintenance/LevelSetWallAwareMaintenance.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceRecords.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

using namespace detail;

// The accepted wall-contact constraint helpers live in FE (wall-aware level-set
// maintenance).
using svmp::FE::level_set::canonicalizeAcceptedWallConstraints;
using svmp::FE::level_set::makeAcceptedSnapshotWallConstraint;

bool firstOrderGeneralizedAlphaContactProvenanceValid(
    const svmp::FE::systems::
        FreeSurfaceFirstOrderGeneralizedAlphaProvenance& provenance) noexcept
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

bool dynamicContactStageMeshRevisionMatches(
    const svmp::FE::timestepping::CandidateStageMeshRevision& captured,
    const svmp::FE::assembly::IMeshAccess& live_mesh)
{
  // Generalized-alpha retains the exact operator-stage phi/velocity slices,
  // then the Application deliberately publishes endpoint cut geometry before
  // reconstructing that stage. Generated-domain publication may attach mesh
  // fields and rewrite generated labels, so those two global mesh epochs are
  // expected to advance. The retained slices are checked independently
  // against the current FE field layout before overlay. The immutable base
  // mesh and selected coordinate frame must still match exactly.
  return live_mesh.revisionTrackingAvailable() &&
         captured.geometry_revision == live_mesh.geometryRevision() &&
         captured.topology_revision == live_mesh.topologyRevision() &&
         captured.ownership_revision == live_mesh.ownershipRevision() &&
         captured.numbering_revision == live_mesh.numberingRevision() &&
         captured.active_configuration_epoch ==
             live_mesh.activeConfigurationEpoch() &&
         captured.coordinate_configuration_key ==
             live_mesh.coordinateConfigurationKey();
}

std::uint64_t dynamicContactWallBoundaryFingerprint(
    const svmp::FE::assembly::IMeshAccess& mesh,
    std::span<const svmp::FE::systems::
                        FreeSurfaceDiscreteFunctionalDeclaration>
        declarations)
{
  std::vector<int> wall_markers;
  for (const auto& declaration : declarations) {
    for (const auto& coefficient :
         declaration.parameters.dynamic_contact_coefficients) {
      if (coefficient.boundary_marker < 0) {
        throw std::invalid_argument(
            "Dynamic-contact wall provenance contains an invalid boundary "
            "marker.");
      }
      wall_markers.push_back(coefficient.boundary_marker);
    }
  }
  std::sort(wall_markers.begin(), wall_markers.end());
  wall_markers.erase(
      std::unique(wall_markers.begin(), wall_markers.end()),
      wall_markers.end());
  if (wall_markers.empty()) {
    throw std::invalid_argument(
        "Dynamic-contact wall provenance requires at least one wall marker.");
  }

  std::uint64_t fingerprint = kCutContextHashOffset;
  constexpr std::uint64_t contract_tag =
      0x44594e57414c4c31ull; // "DYNWALL1"
  mixCutContextHash(fingerprint, contract_tag);
  mixCutContextHash(
      fingerprint, static_cast<std::uint64_t>(wall_markers.size()));
  for (const auto marker : wall_markers) {
    std::vector<
        std::pair<svmp::FE::GlobalIndex, svmp::FE::GlobalIndex>> faces;
    mesh.forEachBoundaryFace(
        marker,
        [&](svmp::FE::GlobalIndex face,
            svmp::FE::GlobalIndex parent_cell) {
          if (face < 0 || parent_cell < 0 ||
              mesh.getBoundaryFaceMarker(face) != marker) {
            throw std::runtime_error(
                "Dynamic-contact wall provenance encountered invalid "
                "boundary incidence.");
          }
          // This digest is rank-local. The base numbering/ownership epochs
          // above make local face and parent identifiers stable across the
          // generated-domain publication being admitted here.
          faces.emplace_back(face, parent_cell);
        });
    std::sort(faces.begin(), faces.end());
    mixCutContextHash(
        fingerprint,
        static_cast<std::uint64_t>(static_cast<std::int64_t>(marker)));
    mixCutContextHash(
        fingerprint, static_cast<std::uint64_t>(faces.size()));
    for (const auto& [face, parent_cell] : faces) {
      mixCutContextHash(
          fingerprint, static_cast<std::uint64_t>(face));
      mixCutContextHash(
          fingerprint, static_cast<std::uint64_t>(parent_cell));
    }
  }
  return fingerprint == 0u ? 1u : fingerprint;
}

bool dynamicContactStageProvenanceMatches(
    const DynamicContactFirstOrderGeneralizedAlphaObservation& captured,
    const svmp::FE::assembly::IMeshAccess& live_mesh,
    std::span<const svmp::FE::systems::
                        FreeSurfaceDiscreteFunctionalDeclaration>
        declarations)
{
  return captured.contact_wall_boundary_fingerprint != 0u &&
         dynamicContactStageMeshRevisionMatches(
             captured.mesh_revision, live_mesh) &&
         captured.contact_wall_boundary_fingerprint ==
             dynamicContactWallBoundaryFingerprint(
                 live_mesh, declarations);
}

DynamicContactFirstOrderGeneralizedAlphaObservation
makeDynamicContactFirstOrderGeneralizedAlphaObservation(
    const svmp::FE::timestepping::CandidateStageObservation& observation)
{
  if (observation.scheme !=
          svmp::FE::timestepping::SchemeKind::GeneralizedAlpha ||
      observation.temporal_order != 1 || observation.step_index <= 0 ||
      observation.attempt_index < 0 ||
      observation.state_vector == nullptr ||
      observation.rate_vector == nullptr ||
      !observation.generalized_alpha.has_value() ||
      !observation.mesh_revision.has_value() ||
      !std::isfinite(observation.step_start_time) ||
      !std::isfinite(observation.step_end_time) ||
      !std::isfinite(observation.state_time) ||
      !std::isfinite(observation.rate_time) ||
      !std::isfinite(observation.dt) || !(observation.dt > 0.0)) {
    throw std::invalid_argument(
        "Dynamic-contact generalized-alpha provenance requires one complete "
        "temporal-order-one candidate-stage observation.");
  }

  const auto& generalized_alpha = *observation.generalized_alpha;
  DynamicContactFirstOrderGeneralizedAlphaObservation captured{
      .step_index = observation.step_index,
      .attempt_index = observation.attempt_index,
      .step_start_time =
          static_cast<svmp::FE::Real>(observation.step_start_time),
      .step_end_time =
          static_cast<svmp::FE::Real>(observation.step_end_time),
      .state_time = static_cast<svmp::FE::Real>(observation.state_time),
      .rate_time = static_cast<svmp::FE::Real>(observation.rate_time),
      .mesh_revision = *observation.mesh_revision,
      .provenance = {
          .alpha_m = static_cast<svmp::FE::Real>(generalized_alpha.alpha_m),
          .alpha_f = static_cast<svmp::FE::Real>(generalized_alpha.alpha_f),
          .gamma = static_cast<svmp::FE::Real>(generalized_alpha.gamma),
          .dt = static_cast<svmp::FE::Real>(observation.dt),
      },
  };
  const auto& provenance = captured.provenance;
  if (!firstOrderGeneralizedAlphaContactProvenanceValid(provenance) ||
      !contactStageRealValuesMatch(
          captured.step_end_time,
          captured.step_start_time + provenance.dt) ||
      !contactStageRealValuesMatch(
          captured.state_time,
          captured.step_start_time + provenance.alpha_f * provenance.dt) ||
      !contactStageRealValuesMatch(
          captured.rate_time,
          captured.step_start_time + provenance.alpha_m * provenance.dt)) {
    throw std::invalid_argument(
        "Dynamic-contact generalized-alpha provenance has inconsistent "
        "scheme parameters or stage times.");
  }
  return captured;
}

namespace {

std::uint64_t dynamicContactFirstOrderGeneralizedAlphaObservationRevision(
    const DynamicContactFirstOrderGeneralizedAlphaObservation& observation)
{
  constexpr std::uint64_t offset = 1469598103934665603ull;
  constexpr std::uint64_t prime = 1099511628211ull;
  std::uint64_t hash = offset;
  const auto mix = [&hash](std::uint64_t value) {
    hash ^= value;
    hash *= prime;
  };
  const auto mix_real = [&](svmp::FE::Real value) {
    std::uint64_t bits = 0u;
    static_assert(sizeof(value) <= sizeof(bits));
    std::memcpy(&bits, &value, sizeof(value));
    mix(bits);
  };
  mix(static_cast<std::uint64_t>(observation.step_index));
  mix(static_cast<std::uint64_t>(observation.attempt_index));
  mix_real(observation.step_start_time);
  mix_real(observation.step_end_time);
  mix_real(observation.state_time);
  mix_real(observation.rate_time);
  mix_real(observation.provenance.alpha_m);
  mix_real(observation.provenance.alpha_f);
  mix_real(observation.provenance.gamma);
  mix_real(observation.provenance.dt);
  mix(static_cast<std::uint64_t>(
      observation.operator_stage_state.full_state_size));
  mix(static_cast<std::uint64_t>(
      observation.operator_stage_state.fields.size()));
  for (const auto& field : observation.operator_stage_state.fields) {
    mix(static_cast<std::uint64_t>(field.field));
    mix(static_cast<std::uint64_t>(field.offset));
    mix(static_cast<std::uint64_t>(field.count));
    mix(static_cast<std::uint64_t>(field.values.size()));
    for (const auto value : field.values) {
      mix_real(value);
    }
  }
  // Mesh revision stamps and contact-wall incidence fingerprints are
  // deliberately excluded: they are rank-local and are checked against each
  // rank's live mesh instead of compared across ranks.
  return hash == 0u ? 1u : hash;
}

std::vector<DynamicContactOperatorStageFieldSlice>
canonicalDynamicContactOperatorStageFieldSlices(
    const svmp::FE::systems::FESystem& system,
    std::span<const svmp::FE::systems::
                        FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    svmp::FE::GlobalIndex full_state_size)
{
  if (full_state_size < 0 ||
      full_state_size != system.dofHandler().getNumDofs()) {
    throw std::invalid_argument(
        "Dynamic-contact operator-stage retention requires the current full "
        "FE state layout.");
  }

  std::vector<svmp::FE::FieldId> fields;
  fields.reserve(declarations.size() * 2u);
  for (const auto& declaration : declarations) {
    if (declaration.parameters.dynamic_contact_coefficients.empty()) {
      continue;
    }
    fields.push_back(declaration.level_set_field);
    fields.push_back(declaration.velocity_field);
  }
  std::sort(fields.begin(), fields.end());
  fields.erase(std::unique(fields.begin(), fields.end()), fields.end());
  if (fields.empty()) {
    throw std::invalid_argument(
        "Dynamic-contact operator-stage retention requires at least one "
        "declared level-set/velocity field pair.");
  }

  std::vector<DynamicContactOperatorStageFieldSlice> slices;
  slices.reserve(fields.size());
  for (const auto field : fields) {
    if (field == svmp::FE::INVALID_FIELD_ID) {
      throw std::invalid_argument(
          "Dynamic-contact operator-stage retention found an invalid field.");
    }
    const auto offset = system.fieldDofOffset(field);
    const auto count = system.fieldDofHandler(field).getNumDofs();
    if (offset < 0 || count <= 0 || offset > full_state_size ||
        count > full_state_size - offset) {
      throw std::invalid_argument(
          "Dynamic-contact operator-stage field slice exceeds the current FE "
          "state layout.");
    }
    slices.push_back(DynamicContactOperatorStageFieldSlice{
        .field = field,
        .offset = offset,
        .count = count,
    });
  }
  auto by_offset = slices;
  std::sort(
      by_offset.begin(), by_offset.end(), [](const auto& left, const auto& right) {
        return std::tie(left.offset, left.count, left.field) <
               std::tie(right.offset, right.count, right.field);
      });
  for (std::size_t index = 1u; index < by_offset.size(); ++index) {
    const auto& previous = by_offset[index - 1u];
    const auto& current = by_offset[index];
    if (current.offset < previous.offset + previous.count) {
      throw std::invalid_argument(
          "Dynamic-contact operator-stage field slices overlap in the current "
          "FE state layout.");
    }
  }
  return slices;
}

std::uint64_t dynamicContactOperatorStageLayoutRevision(
    const DynamicContactOperatorStageState& state) noexcept
{
  constexpr std::uint64_t offset = 1469598103934665603ull;
  constexpr std::uint64_t prime = 1099511628211ull;
  std::uint64_t hash = offset;
  const auto mix = [&hash](std::uint64_t value) {
    hash ^= value;
    hash *= prime;
  };
  mix(static_cast<std::uint64_t>(state.full_state_size));
  mix(static_cast<std::uint64_t>(state.fields.size()));
  for (const auto& field : state.fields) {
    mix(static_cast<std::uint64_t>(field.field));
    mix(static_cast<std::uint64_t>(field.offset));
    mix(static_cast<std::uint64_t>(field.count));
  }
  return hash == 0u ? 1u : hash;
}

} // namespace

DynamicContactOperatorStageState captureDynamicContactOperatorStageState(
    const svmp::FE::systems::FESystem& system,
    std::span<const svmp::FE::systems::
                        FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    const svmp::FE::backends::GenericVector& state,
    const svmp::MeshComm& comm)
{
  DynamicContactOperatorStageState captured;
  std::exception_ptr local_layout_failure;
  try {
    captured.full_state_size = state.size();
    captured.fields = canonicalDynamicContactOperatorStageFieldSlices(
        system, declarations, captured.full_state_size);
  } catch (...) {
    local_layout_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_layout_failure,
      comm,
      "dynamic_contact_operator_stage_layout");
  const auto [minimum_full_state_size, maximum_full_state_size] =
      globalMinMaxUint64(
          static_cast<std::uint64_t>(captured.full_state_size), comm);
  const auto [minimum_field_count, maximum_field_count] =
      globalMinMaxUint64(
          static_cast<std::uint64_t>(captured.fields.size()), comm);
  bool exact_layout_agrees =
      minimum_full_state_size == maximum_full_state_size &&
      minimum_field_count == maximum_field_count;
  if (minimum_field_count == maximum_field_count) {
    // Field count is authoritative for the following collective schedule.
    // Finish every fixed per-index metadata comparison before diagnosing a
    // mismatch so all ranks enter the same reductions in the same order.
    for (std::size_t index = 0u; index < captured.fields.size(); ++index) {
      const auto& field = captured.fields[index];
      const auto [minimum_field, maximum_field] = globalMinMaxUint64(
          static_cast<std::uint64_t>(field.field), comm);
      const auto [minimum_offset, maximum_offset] = globalMinMaxUint64(
          static_cast<std::uint64_t>(field.offset), comm);
      const auto [minimum_count, maximum_count] = globalMinMaxUint64(
          static_cast<std::uint64_t>(field.count), comm);
      exact_layout_agrees =
          minimum_field == maximum_field &&
          minimum_offset == maximum_offset &&
          minimum_count == maximum_count && exact_layout_agrees;
    }
  }
  if (!exact_layout_agrees) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Dynamic-contact operator-stage field "
        "layout differs across the active FE communicator.");
  }
  const auto layout_revision =
      dynamicContactOperatorStageLayoutRevision(captured);
  const auto [minimum_layout_revision, maximum_layout_revision] =
      globalMinMaxUint64(layout_revision, comm);
  if (minimum_layout_revision != maximum_layout_revision) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Dynamic-contact operator-stage field "
        "layout digest differs after exact metadata consensus.");
  }

  std::vector<svmp::FE::GlobalIndex> owned_rows;
  std::unique_ptr<svmp::FE::assembly::GlobalSystemView> read_view;
  std::exception_ptr local_read_preparation_failure;
  try {
    owned_rows = state.ownedGlobalRows();
    if (std::adjacent_find(
            owned_rows.begin(),
            owned_rows.end(),
            [](svmp::FE::GlobalIndex left, svmp::FE::GlobalIndex right) {
              return left >= right;
            }) != owned_rows.end() ||
        !std::all_of(
            owned_rows.begin(),
            owned_rows.end(),
            [&](svmp::FE::GlobalIndex row) {
              return row >= 0 && row < state.size();
            })) {
      throw std::runtime_error(
          "Dynamic-contact operator-stage retention requires strictly "
          "ordered, unique, in-range backend-owned rows.");
    }
    read_view =
        const_cast<svmp::FE::backends::GenericVector&>(state)
            .createGhostedReadView();
    if (!read_view) {
      throw std::runtime_error(
          "Dynamic-contact operator-stage retention requires an owned-row "
          "read view.");
    }
    for (auto& field : captured.fields) {
      field.values.resize(
          static_cast<std::size_t>(field.count), svmp::FE::Real{0.0});
    }
  } catch (...) {
    local_read_preparation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_read_preparation_failure,
      comm,
      "dynamic_contact_operator_stage_owned_read_preparation");

  constexpr std::size_t chunk_capacity = 4096u;
  static_assert(
      sizeof(svmp::FE::Real) == sizeof(std::uint64_t),
      "Exact dynamic-contact operator-stage retention requires a 64-bit FE scalar.");
  std::array<std::uint64_t, chunk_capacity> local_value_bits{};
  std::array<std::uint64_t, chunk_capacity> global_value_bits{};
  std::array<svmp::FE::Real, chunk_capacity> owned_values{};
  std::array<std::uint32_t, chunk_capacity> local_counts{};
  std::array<std::uint32_t, chunk_capacity> global_counts{};
  for (auto& field : captured.fields) {
    for (svmp::FE::GlobalIndex relative_begin = 0;
         relative_begin < field.count;
         relative_begin += static_cast<svmp::FE::GlobalIndex>(chunk_capacity)) {
      const auto chunk_size = static_cast<std::size_t>(
          std::min<svmp::FE::GlobalIndex>(
              static_cast<svmp::FE::GlobalIndex>(chunk_capacity),
              field.count - relative_begin));
      std::fill_n(local_value_bits.begin(), chunk_size, std::uint64_t{0u});
      std::fill_n(global_value_bits.begin(), chunk_size, std::uint64_t{0u});
      std::fill_n(local_counts.begin(), chunk_size, std::uint32_t{0u});
      std::fill_n(global_counts.begin(), chunk_size, std::uint32_t{0u});

      const auto global_begin = field.offset + relative_begin;
      const auto global_end =
          global_begin + static_cast<svmp::FE::GlobalIndex>(chunk_size);
      const auto first =
          std::lower_bound(owned_rows.begin(), owned_rows.end(), global_begin);
      const auto last =
          std::lower_bound(first, owned_rows.end(), global_end);
      const auto first_index = static_cast<std::size_t>(
          std::distance(owned_rows.begin(), first));
      const auto owned_count =
          static_cast<std::size_t>(std::distance(first, last));

      std::exception_ptr local_read_failure;
      try {
        if (owned_count != 0u) {
          read_view->getVectorEntries(
              std::span<const svmp::FE::GlobalIndex>(
                  owned_rows.data() +
                      static_cast<std::ptrdiff_t>(first_index),
                  owned_count),
              std::span<svmp::FE::Real>(owned_values.data(), owned_count));
        }
        for (std::size_t index = 0u; index < owned_count; ++index) {
          const auto row = owned_rows[first_index + index];
          const auto relative = static_cast<std::size_t>(row - global_begin);
          if (relative >= chunk_size) {
            throw std::runtime_error(
                "Dynamic-contact operator-stage owned read returned an "
                "out-of-range value.");
          }
          local_value_bits[relative] =
              std::bit_cast<std::uint64_t>(owned_values[index]);
          local_counts[relative] = 1u;
        }
      } catch (...) {
        local_read_failure = std::current_exception();
      }
      requireCollectivePhasePreparation(
          local_read_failure,
          comm,
          "dynamic_contact_operator_stage_owned_read");

#ifdef MESH_HAS_MPI
      if (comm.is_parallel()) {
#ifdef MPI_UINT32_T
        const MPI_Datatype count_type = MPI_UINT32_T;
#else
        static_assert(sizeof(std::uint32_t) == sizeof(unsigned));
        const MPI_Datatype count_type = MPI_UNSIGNED;
#endif
#ifdef MPI_UINT64_T
        const MPI_Datatype value_bits_type = MPI_UINT64_T;
#else
        static_assert(sizeof(std::uint64_t) == sizeof(unsigned long long));
        const MPI_Datatype value_bits_type = MPI_UNSIGNED_LONG_LONG;
#endif
        MPI_Allreduce(
            local_counts.data(),
            global_counts.data(),
            static_cast<int>(chunk_size),
            count_type,
            MPI_SUM,
            comm.native());
        MPI_Allreduce(
            local_value_bits.data(),
            global_value_bits.data(),
            static_cast<int>(chunk_size),
            value_bits_type,
            MPI_BOR,
            comm.native());
      } else {
        std::copy_n(local_counts.begin(), chunk_size, global_counts.begin());
        std::copy_n(
            local_value_bits.begin(), chunk_size, global_value_bits.begin());
      }
#else
      if (comm.is_parallel()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Distributed dynamic-contact "
            "operator-stage retention requires MPI.");
      }
      std::copy_n(local_counts.begin(), chunk_size, global_counts.begin());
      std::copy_n(
          local_value_bits.begin(), chunk_size, global_value_bits.begin());
#endif
      if (std::any_of(
              global_counts.begin(),
              global_counts.begin() +
                  static_cast<std::ptrdiff_t>(chunk_size),
              [](std::uint32_t count) { return count != 1u; })) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Dynamic-contact operator-stage "
            "retention requires every requested FE row to have exactly one "
            "backend owner.");
      }
      for (std::size_t index = 0u; index < chunk_size; ++index) {
        const auto value =
            std::bit_cast<svmp::FE::Real>(global_value_bits[index]);
        if (!std::isfinite(value)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Dynamic-contact operator-stage "
              "retention reconstructed a non-finite owned value.");
        }
        field.values[static_cast<std::size_t>(relative_begin) + index] = value;
      }
    }
  }
  return captured;
}

void overlayDynamicContactOperatorStageState(
    const svmp::FE::systems::FESystem& system,
    std::span<const svmp::FE::systems::
                        FreeSurfaceDiscreteFunctionalDeclaration>
        declarations,
    const DynamicContactOperatorStageState& exact_state,
    std::span<svmp::FE::Real> stage_solution)
{
  if (exact_state.full_state_size < 0 ||
      static_cast<std::size_t>(exact_state.full_state_size) !=
          stage_solution.size()) {
    throw std::invalid_argument(
        "Dynamic-contact operator-stage snapshot does not match the current "
        "full FE state size.");
  }
  const auto expected = canonicalDynamicContactOperatorStageFieldSlices(
      system, declarations, exact_state.full_state_size);
  if (expected.size() != exact_state.fields.size()) {
    throw std::invalid_argument(
        "Dynamic-contact operator-stage snapshot has stale field coverage.");
  }
  for (std::size_t index = 0u; index < expected.size(); ++index) {
    const auto& expected_field = expected[index];
    const auto& captured_field = exact_state.fields[index];
    if (captured_field.field != expected_field.field ||
        captured_field.offset != expected_field.offset ||
        captured_field.count != expected_field.count ||
        captured_field.values.size() !=
            static_cast<std::size_t>(captured_field.count) ||
        !std::all_of(
            captured_field.values.begin(),
            captured_field.values.end(),
            [](svmp::FE::Real value) { return std::isfinite(value); })) {
      throw std::invalid_argument(
          "Dynamic-contact operator-stage snapshot has stale field layout or "
          "invalid retained values.");
    }
    std::copy(
        captured_field.values.begin(),
        captured_field.values.end(),
        stage_solution.begin() +
            static_cast<std::ptrdiff_t>(captured_field.offset));
  }
}

void requireDynamicContactOperatorStageObservationConsensus(
    const DynamicContactFirstOrderGeneralizedAlphaObservation& observation,
    const svmp::MeshComm& comm)
{
  static_assert(sizeof(svmp::FE::Real) == sizeof(std::uint64_t));
  bool exact_metadata_agrees = true;
  const auto compare_word = [&](std::uint64_t word) {
    const auto [minimum, maximum] = globalMinMaxUint64(word, comm);
    exact_metadata_agrees = minimum == maximum && exact_metadata_agrees;
  };
  const auto real_word = [](svmp::FE::Real value) {
    return std::bit_cast<std::uint64_t>(value);
  };
  compare_word(static_cast<std::uint64_t>(observation.step_index));
  compare_word(static_cast<std::uint64_t>(observation.attempt_index));
  compare_word(real_word(observation.step_start_time));
  compare_word(real_word(observation.step_end_time));
  compare_word(real_word(observation.state_time));
  compare_word(real_word(observation.rate_time));
  compare_word(real_word(observation.provenance.alpha_m));
  compare_word(real_word(observation.provenance.alpha_f));
  compare_word(real_word(observation.provenance.gamma));
  compare_word(real_word(observation.provenance.dt));
  compare_word(static_cast<std::uint64_t>(
      observation.operator_stage_state.full_state_size));

  const auto local_field_count = static_cast<std::uint64_t>(
      observation.operator_stage_state.fields.size());
  const auto [minimum_field_count, maximum_field_count] =
      globalMinMaxUint64(local_field_count, comm);
  exact_metadata_agrees =
      minimum_field_count == maximum_field_count && exact_metadata_agrees;
  if (minimum_field_count == maximum_field_count) {
    for (const auto& field : observation.operator_stage_state.fields) {
      compare_word(static_cast<std::uint64_t>(field.field));
      compare_word(static_cast<std::uint64_t>(field.offset));
      compare_word(static_cast<std::uint64_t>(field.count));
      compare_word(static_cast<std::uint64_t>(field.values.size()));
      exact_metadata_agrees =
          field.count >= 0 &&
          static_cast<std::uint64_t>(field.count) == field.values.size() &&
          exact_metadata_agrees;
    }
  }
  if (!exact_metadata_agrees) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Dynamic-contact generalized-alpha "
        "operator-stage metadata differs across the active FE communicator.");
  }

  constexpr std::size_t chunk_capacity = 4096u;
  std::array<std::uint64_t, chunk_capacity> local_bits{};
  std::array<std::uint64_t, chunk_capacity> minimum_bits{};
  std::array<std::uint64_t, chunk_capacity> maximum_bits{};
  bool exact_values_agree = true;
  for (const auto& field : observation.operator_stage_state.fields) {
    for (std::size_t offset = 0u; offset < field.values.size();
         offset += chunk_capacity) {
      const auto count =
          std::min(chunk_capacity, field.values.size() - offset);
      for (std::size_t index = 0u; index < count; ++index) {
        local_bits[index] = real_word(field.values[offset + index]);
      }
#ifdef MESH_HAS_MPI
      if (comm.is_parallel()) {
#ifdef MPI_UINT64_T
        const MPI_Datatype datatype = MPI_UINT64_T;
#else
        static_assert(sizeof(std::uint64_t) == sizeof(unsigned long long));
        const MPI_Datatype datatype = MPI_UNSIGNED_LONG_LONG;
#endif
        MPI_Allreduce(
            local_bits.data(),
            minimum_bits.data(),
            static_cast<int>(count),
            datatype,
            MPI_MIN,
            comm.native());
        MPI_Allreduce(
            local_bits.data(),
            maximum_bits.data(),
            static_cast<int>(count),
            datatype,
            MPI_MAX,
            comm.native());
      } else {
        std::copy_n(local_bits.begin(), count, minimum_bits.begin());
        std::copy_n(local_bits.begin(), count, maximum_bits.begin());
      }
#else
      if (comm.is_parallel()) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Distributed dynamic-contact "
            "operator-stage consensus requires MPI.");
      }
      std::copy_n(local_bits.begin(), count, minimum_bits.begin());
      std::copy_n(local_bits.begin(), count, maximum_bits.begin());
#endif
      exact_values_agree =
          std::equal(
              minimum_bits.begin(),
              minimum_bits.begin() + static_cast<std::ptrdiff_t>(count),
              maximum_bits.begin()) &&
          exact_values_agree;
    }
  }
  if (!exact_values_agree) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Dynamic-contact generalized-alpha "
        "operator-stage values differ across the active FE communicator.");
  }

  const auto local_revision =
      dynamicContactFirstOrderGeneralizedAlphaObservationRevision(observation);
  const auto [minimum_revision, maximum_revision] =
      globalMinMaxUint64(local_revision, comm);
  if (minimum_revision != maximum_revision) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Dynamic-contact generalized-alpha "
        "operator-stage digest differs after exact consensus.");
  }
}

FreeSurfaceContactStageTemporalCoordinates
resolveFreeSurfaceContactStageTemporalCoordinates(
    svmp::FE::timestepping::SchemeKind scheme,
    const svmp::FE::timestepping::TimeHistory& history,
    int expected_attempt_index,
    const std::optional<DynamicContactFirstOrderGeneralizedAlphaObservation>&
        generalized_alpha_observation)
{
  const bool endpoint_stage_scheme =
      scheme == svmp::FE::timestepping::SchemeKind::BackwardEuler ||
      scheme == svmp::FE::timestepping::SchemeKind::BDF2 ||
      scheme == svmp::FE::timestepping::SchemeKind::VSVO_BDF;
  if (endpoint_stage_scheme) {
    if (generalized_alpha_observation.has_value()) {
      throw std::logic_error(
          "Endpoint contact-stage construction received generalized-alpha "
          "provenance.");
    }
    return FreeSurfaceContactStageTemporalCoordinates{
        .stage_time = static_cast<svmp::FE::Real>(
            history.time() + history.dt()),
        .stage_alpha_f = svmp::FE::Real{1.0},
    };
  }
  if (scheme != svmp::FE::timestepping::SchemeKind::GeneralizedAlpha ||
      !generalized_alpha_observation.has_value()) {
    throw std::logic_error(
        "Dynamic-contact generalized-alpha stage construction is missing its "
        "authentic candidate-stage provenance.");
  }
  const auto& observation = *generalized_alpha_observation;
  const auto expected_step = history.stepIndex() + 1;
  const auto expected_start = static_cast<svmp::FE::Real>(history.time());
  const auto expected_dt = static_cast<svmp::FE::Real>(history.dt());
  if (observation.step_index != expected_step ||
      observation.attempt_index != expected_attempt_index ||
      !contactStageRealValuesMatch(
          observation.step_start_time, expected_start) ||
      !contactStageRealValuesMatch(observation.provenance.dt, expected_dt) ||
      !contactStageRealValuesMatch(
          observation.step_end_time, expected_start + expected_dt)) {
    throw std::logic_error(
        "Dynamic-contact generalized-alpha stage provenance is stale for the "
        "current candidate step.");
  }
  return FreeSurfaceContactStageTemporalCoordinates{
      .stage_time = observation.state_time,
      .stage_alpha_f = observation.provenance.alpha_f,
      .first_order_generalized_alpha = observation.provenance,
  };
}

std::uint64_t acceptedContactStageRevision(
    std::uint64_t previous_state_revision,
    std::uint64_t endpoint_state_revision,
    std::uint64_t snapshot_revision,
    svmp::FE::Real stage_time,
    svmp::FE::Real stage_alpha_f,
    std::span<const svmp::FE::Real> stage_solution,
    const std::optional<svmp::FE::systems::
                            FreeSurfaceFirstOrderGeneralizedAlphaProvenance>&
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

std::vector<svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
evaluateAcceptedFreeSurfaceContactStages(
    const FreeSurfaceSystemHandle& sim,
    svmp::FE::Real stage_time,
    svmp::FE::Real stage_alpha_f,
    std::uint64_t previous_state_revision,
    std::uint64_t endpoint_state_revision,
    std::span<const svmp::FE::Real> stage_solution,
    const std::optional<svmp::FE::systems::
                            FreeSurfaceFirstOrderGeneralizedAlphaProvenance>&
        first_order_generalized_alpha)
{
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  const auto declarations =
      free_surface::records(*sim.fe_system).freeSurfaceDiscreteFunctionalDeclarations();
  const auto dynamic_count = static_cast<std::size_t>(std::count_if(
      declarations.begin(),
      declarations.end(),
      [](const auto& declaration) {
        return !declaration.parameters.dynamic_contact_coefficients.empty();
      }));
  std::vector<svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
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
  if (globalMinDouble(local_dynamic_count, comm) !=
      globalMaxDouble(local_dynamic_count, comm)) {
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
  if (globalAnyBool(local_provenance_incomplete, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage provenance is incomplete.");
  }
  const double local_solution_size =
      static_cast<double>(stage_solution.size());
  if (globalMinDouble(local_solution_size, comm) !=
      globalMaxDouble(local_solution_size, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Contact-stage solution layouts differ across the FE communicator.");
  }
  const auto* context = sim.fe_system->cutIntegrationContext();
  if (globalMinDouble(context != nullptr ? 1.0 : 0.0, comm) != 1.0) {
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
    if (globalMinDouble(local_marker, comm) !=
            globalMaxDouble(local_marker, comm) ||
        globalMinDouble(local_velocity_field, comm) !=
            globalMaxDouble(local_velocity_field, comm) ||
        globalMinDouble(local_contact_count, comm) !=
            globalMaxDouble(local_contact_count, comm)) {
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
      if (globalMinDouble(boundary_marker, comm) !=
              globalMaxDouble(boundary_marker, comm) ||
          globalMinDouble(equilibrium_angle, comm) !=
              globalMaxDouble(equilibrium_angle, comm) ||
          globalMinDouble(contact_law, comm) !=
              globalMaxDouble(contact_law, comm) ||
          globalMinDouble(mobility, comm) !=
              globalMaxDouble(mobility, comm) ||
          globalMinDouble(slip_length, comm) !=
              globalMaxDouble(slip_length, comm) ||
          globalMinDouble(dynamic_viscosity, comm) !=
              globalMaxDouble(dynamic_viscosity, comm)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Dynamic contact-stage wall coefficients differ across the FE communicator for marker " +
            std::to_string(declaration.interface_marker) + ".");
      }
    }
    const bool local_marker_available =
        context->hasFreeSurfaceGeometrySnapshotForMarker(
            declaration.interface_marker);
    if (globalMinDouble(local_marker_available ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage marker " +
          std::to_string(declaration.interface_marker) +
          " is missing an authoritative geometry snapshot on at least one rank.");
    }
    const auto snapshot_revision =
        context->freeSurfaceGeometrySnapshotRevisionForMarker(
            declaration.interface_marker);
    const auto [minimum_revision, maximum_revision] =
        globalMinMaxUint64(snapshot_revision, comm);
    if (minimum_revision != maximum_revision) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage geometry revision differs across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    const auto& snapshots = context->freeSurfaceGeometrySnapshots();
    const auto found = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().snapshot_revision_key ==
                     snapshot_revision;
        });
    if (globalMinDouble(found != snapshots.end() ? 1.0 : 0.0, comm) !=
        1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage snapshot storage is incomplete across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    bool local_snapshot_current = true;
    try {
      context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
          sim.fe_system->meshAccess());
    } catch (const std::exception&) {
      local_snapshot_current = false;
    }
    if (globalMinDouble(local_snapshot_current ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage snapshot is stale or incomplete on at least one rank for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    const auto& snapshot = **found;
    std::optional<
        svmp::FE::interfaces::FreeSurfaceDynamicContactState>
        local_state;
    try {
      const auto velocity = makeFreeSurfaceVelocityEvaluator(
          sim,
          declaration.velocity_field,
          stage_solution,
          "Accepted contact-stage");
      local_state =
          svmp::FE::interfaces::evaluateFreeSurfaceDynamicContactState(
              snapshot, declaration.parameters, velocity);
    } catch (const std::exception&) {
      local_state.reset();
    }
    if (globalMinDouble(local_state.has_value() ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Dynamic contact-stage evaluation failed on at least one rank for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    auto state = std::move(*local_state);
    const double local_wall_count =
        static_cast<double>(state.walls.size());
    if (globalMinDouble(local_wall_count, comm) !=
        globalMaxDouble(local_wall_count, comm)) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Dynamic contact-stage wall ordering differs across the FE communicator for marker " +
          std::to_string(declaration.interface_marker) + ".");
    }

    const auto global_sum = [&comm](svmp::FE::Real local) {
      return static_cast<svmp::FE::Real>(
          globalSumDouble(static_cast<double>(local), comm));
    };
    for (auto& wall : state.walls) {
      wall.owned_quadrature_point_count = globalSumSize(
          wall.owned_quadrature_point_count, comm);
      wall.owned_advancing_point_count = globalSumSize(
          wall.owned_advancing_point_count, comm);
      wall.owned_receding_point_count = globalSumSize(
          wall.owned_receding_point_count, comm);
      wall.owned_stationary_point_count = globalSumSize(
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
      wall.owned_wetted_wall_quadrature_point_count = globalSumSize(
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
      svmp::FE::interfaces::finalizeFreeSurfaceDynamicContactState(state);
      stages.push_back(
          svmp::FE::systems::FreeSurfaceAcceptedContactStageState{
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
  }
  return stages;
}

std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
captureAcceptedContactStageWallConstraints(
    const FreeSurfaceSystemHandle& sim,
    std::span<const svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
        accepted_contact_stages)
{
  std::vector<svmp::FE::level_set::LevelSetWallContactConstraint> captured;
  const auto declarations =
      free_surface::records(*sim.fe_system).freeSurfaceDiscreteFunctionalDeclarations();
  const auto comm = activeFESystemCommunicator(*sim.fe_system);
  const auto* context = sim.fe_system->cutIntegrationContext();
  if (globalMinDouble(context != nullptr ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage constraint capture requires an authoritative geometry snapshot on every rank.");
  }
  bool local_snapshots_current = true;
  try {
    context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
        sim.fe_system->meshAccess());
  } catch (const std::exception&) {
    local_snapshots_current = false;
  }
  if (globalMinDouble(local_snapshots_current ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage constraint capture rejected a stale authoritative geometry snapshot.");
  }
  std::map<
      int,
      const svmp::FE::systems::FreeSurfaceAcceptedContactStageState*>
      stage_by_interface;
  bool local_stage_map_valid = true;
  for (const auto& stage : accepted_contact_stages) {
    const int marker = stage.geometry_revision.interface_marker;
    local_stage_map_valid =
        marker >= 0 && stage_by_interface.emplace(marker, &stage).second &&
        local_stage_map_valid;
  }
  if (globalMinDouble(local_stage_map_valid ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact-stage constraint capture requires unique nonnegative interface markers.");
  }

  for (const auto& declaration : declarations) {
    const bool has_dynamic_contact_law = std::any_of(
        declaration.parameters.dynamic_contact_coefficients.begin(),
        declaration.parameters.dynamic_contact_coefficients.end(),
        [](const auto& coefficient) {
          return coefficient.law ==
                 svmp::FE::interfaces::FreeSurfaceContactLaw::DynamicRenE;
        });
    if (!has_dynamic_contact_law) {
      continue;
    }
    const auto stage_it = stage_by_interface.find(declaration.interface_marker);
    const bool local_stage_available = stage_it != stage_by_interface.end();
    if (globalMinDouble(local_stage_available ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage constraint capture is missing interface marker " +
          std::to_string(declaration.interface_marker) + ".");
    }
    const auto& stage = *stage_it->second;
    const auto& snapshots = context->freeSurfaceGeometrySnapshots();
    const auto snapshot_it = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().snapshot_revision_key ==
                     stage.geometry_revision.snapshot_revision_key;
        });
    if (globalMinDouble(snapshot_it != snapshots.end() ? 1.0 : 0.0, comm) !=
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
    if (globalMinDouble(local_revision_valid ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage constraint capture found mismatched stage geometry provenance.");
    }

    std::set<int> dynamic_contact_walls;
    for (const auto& coefficient :
         declaration.parameters.dynamic_contact_coefficients) {
      if (coefficient.law !=
          svmp::FE::interfaces::FreeSurfaceContactLaw::DynamicRenE) {
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
                             FreeSurfaceGeometryRuleRole::Contact ||
          record.retention != svmp::FE::interfaces::
                                  FreeSurfaceGeometryRetention::Retained ||
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
        declaration_constraints.push_back(makeAcceptedSnapshotWallConstraint(
            record,
            declaration.interface_marker,
            stage.geometry_revision.snapshot_revision_key));
      } catch (const std::exception& error) {
        local_rules_valid = false;
        local_rule_diagnostic = error.what();
      }
    }
    if (globalMinDouble(local_parent_ids_valid ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage constraint capture found a rule without a partition-independent parent identity.");
    }
    if (globalMinDouble(local_rules_valid ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Accepted contact-stage constraint capture rejected a contact rule with incomplete or stale provenance" +
          (local_rule_diagnostic.empty()
               ? std::string{"."}
               : std::string{": "} + local_rule_diagnostic));
    }
    declaration_constraints = canonicalizeAcceptedWallConstraints(
        std::move(declaration_constraints),
        feCollectiveCommunicator(comm),
        "Accepted contact-stage constraint capture");
    captured.insert(captured.end(),
                    declaration_constraints.begin(),
                    declaration_constraints.end());
  }
  return captured;
}

void bindAcceptedFreeSurfaceContactStagesToEndpointRevision(
    std::span<
        svmp::FE::systems::FreeSurfaceAcceptedContactStageState> stages,
    std::uint64_t endpoint_state_revision,
    std::span<const svmp::FE::Real> stage_solution,
    const svmp::MeshComm& comm)
{
  const auto [minimum_revision, maximum_revision] =
      globalMinMaxUint64(endpoint_state_revision, comm);
  if (endpoint_state_revision == 0u ||
      minimum_revision != maximum_revision) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] The pre-maintenance endpoint state revision must be nonzero and identical across the FE communicator.");
  }
  const auto local_stage_count =
      static_cast<std::uint64_t>(stages.size());
  const auto [minimum_stage_count, maximum_stage_count] =
      globalMinMaxUint64(local_stage_count, comm);
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
        globalMinMaxUint64(bound_stage_revision, comm);
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

void assertAcceptedFreeSurfaceContactStageEndpointUnchanged(
    std::uint64_t expected_endpoint_algebraic_revision,
    std::span<const svmp::FE::Real> endpoint_solution,
    const svmp::MeshComm& comm)
{
  const auto endpoint_algebraic_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          endpoint_solution, comm);
  if (expected_endpoint_algebraic_revision == 0u ||
      endpoint_algebraic_revision !=
          expected_endpoint_algebraic_revision) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] The finalized endpoint state differs from the endpoint used to reconstruct accepted contact-stage provenance.");
  }
}

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
