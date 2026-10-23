/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/OperatorStageCapture.h"

#include "Assembly/GlobalSystemView.h"
#include "Assembly/Assembler.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Interfaces/ContactLineStageObservables.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>

#if FE_HAS_MPI
#include <mpi.h>
#endif

namespace svmp {
namespace FE {
namespace timestepping {

namespace {

using svmp::FE::interfaces::contactStageRealValuesMatch;
using svmp::FE::interfaces::firstOrderGeneralizedAlphaContactProvenanceValid;
using svmp::FE::systems::collectiveMinMaxUint64;
using svmp::FE::systems::requireCollectivePhasePreparation;

// 64-bit FNV-1a word mixing, as the cut-context fingerprints of
// FE/LevelSet/LevelSetCutTopology.
constexpr std::uint64_t kStageHashOffset = 1469598103934665603ull;
constexpr std::uint64_t kStageHashPrime = 1099511628211ull;

void mixStageHash(std::uint64_t& h, std::uint64_t value) noexcept
{
  h ^= value;
  h *= kStageHashPrime;
}

} // namespace

// Moved from Physics/Formulations/NavierStokes/FreeSurface/
// FreeSurfaceContactStages.cpp with bodies, messages and collective order
// unchanged.

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

  std::uint64_t fingerprint = kStageHashOffset;
  constexpr std::uint64_t contract_tag =
      0x44594e57414c4c31ull; // "DYNWALL1"
  mixStageHash(fingerprint, contract_tag);
  mixStageHash(
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
    mixStageHash(
        fingerprint,
        static_cast<std::uint64_t>(static_cast<std::int64_t>(marker)));
    mixStageHash(
        fingerprint, static_cast<std::uint64_t>(faces.size()));
    for (const auto& [face, parent_cell] : faces) {
      mixStageHash(
          fingerprint, static_cast<std::uint64_t>(face));
      mixStageHash(
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
    const svmp::FE::systems::CollectiveCommunicator& comm)
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
      collectiveMinMaxUint64(
          static_cast<std::uint64_t>(captured.full_state_size), comm);
  const auto [minimum_field_count, maximum_field_count] =
      collectiveMinMaxUint64(
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
      const auto [minimum_field, maximum_field] = collectiveMinMaxUint64(
          static_cast<std::uint64_t>(field.field), comm);
      const auto [minimum_offset, maximum_offset] = collectiveMinMaxUint64(
          static_cast<std::uint64_t>(field.offset), comm);
      const auto [minimum_count, maximum_count] = collectiveMinMaxUint64(
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
      collectiveMinMaxUint64(layout_revision, comm);
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

#if FE_HAS_MPI
      if (comm.isParallel()) {
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
      if (comm.isParallel()) {
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
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  static_assert(sizeof(svmp::FE::Real) == sizeof(std::uint64_t));
  bool exact_metadata_agrees = true;
  const auto compare_word = [&](std::uint64_t word) {
    const auto [minimum, maximum] = collectiveMinMaxUint64(word, comm);
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
      collectiveMinMaxUint64(local_field_count, comm);
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
#if FE_HAS_MPI
      if (comm.isParallel()) {
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
      if (comm.isParallel()) {
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
      collectiveMinMaxUint64(local_revision, comm);
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

} // namespace timestepping
} // namespace FE
} // namespace svmp
