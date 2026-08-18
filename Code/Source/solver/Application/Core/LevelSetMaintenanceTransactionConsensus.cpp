#include "Application/Core/LevelSetMaintenanceTransactionConsensus.h"

#include "Application/Core/ApplicationDriver.h"
#include "Mesh/Core/MeshComm.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#ifdef MESH_HAS_MPI
#include <mpi.h>
#endif

namespace application::core {
namespace {

using CanonicalWords = std::vector<std::uint64_t>;

void appendWord(CanonicalWords& words, std::uint64_t value)
{
  words.push_back(value);
}

void appendDouble(CanonicalWords& words, double value)
{
  appendWord(words, std::bit_cast<std::uint64_t>(value));
}

void appendOptionalRevision(
    CanonicalWords& words,
    const std::optional<std::uint64_t>& revision)
{
  appendWord(words, revision.has_value() ? 1u : 0u);
  appendWord(words, revision.value_or(0u));
}

void appendOptionalDouble(
    CanonicalWords& words,
    const std::optional<double>& value)
{
  appendWord(words, value.has_value() ? 1u : 0u);
  appendDouble(words, value.value_or(0.0));
}

void appendFunctional(
    CanonicalWords& words,
    const LevelSetAuthoritativeFunctionalValue& value)
{
  appendWord(
      words,
      static_cast<std::uint64_t>(
          static_cast<std::int64_t>(value.interface_marker)));
  appendWord(words, value.snapshot_revision);
  appendWord(words, value.mesh_topology_revision);
  appendWord(words, value.cut_topology_revision);
  appendDouble(words, value.liquid_volume);
  appendDouble(words, value.liquid_gas_area);
  appendDouble(words, value.wetted_wall_area);
  appendDouble(words, value.contact_measure);
  appendDouble(words, value.surface_energy);
  appendDouble(words, value.young_wall_energy);
  appendDouble(words, value.volume_constraint_potential);
  appendDouble(words, value.total_potential);
  appendOptionalDouble(words, value.kinetic_energy);
  appendOptionalDouble(words, value.gravitational_energy);
  appendOptionalDouble(
      words, value.gravitational_potential_power);
  appendOptionalDouble(
      words, value.surface_wall_potential_power);
  appendOptionalDouble(
      words, value.volume_constraint_potential_power);
  appendOptionalDouble(
      words, value.bulk_viscous_dissipation_rate);
  appendOptionalDouble(words, value.external_pressure_power);
  appendOptionalDouble(words, value.modeled_stored_energy);
}

void appendFunctionals(
    CanonicalWords& words,
    const std::vector<LevelSetAuthoritativeFunctionalValue>& values)
{
  appendWord(words, static_cast<std::uint64_t>(values.size()));
  for (const auto& value : values) {
    appendFunctional(words, value);
  }
}

void appendTransaction(
    CanonicalWords& words,
    const LevelSetMaintenanceWorkTransaction& transaction)
{
  appendWord(words, transaction.transaction_id);
  appendWord(words, transaction.step);
  appendWord(words, transaction.attempt);
  appendDouble(words, transaction.time);
  appendDouble(words, transaction.dt);
  appendWord(
      words, static_cast<std::uint64_t>(transaction.declared_stage));
  appendOptionalRevision(words, transaction.extension_map_revision);
}

void appendRow(
    CanonicalWords& words,
    const LevelSetMaintenanceWorkRow& row)
{
  appendWord(words, row.transaction_id);
  appendWord(words, static_cast<std::uint64_t>(row.status));
  appendWord(words, static_cast<std::uint64_t>(row.substage));
  appendWord(words, row.step);
  appendWord(words, row.attempt);
  appendDouble(words, row.time);
  appendDouble(words, row.dt);
  appendWord(words, row.algebraic_state_revision_before);
  appendWord(words, row.algebraic_state_revision_after);
  appendWord(words, row.snapshot_set_revision_before);
  appendWord(words, row.snapshot_set_revision_after);
  appendWord(words, row.mesh_topology_set_revision_before);
  appendWord(words, row.mesh_topology_set_revision_after);
  appendWord(words, row.cut_topology_set_revision_before);
  appendWord(words, row.cut_topology_set_revision_after);
  appendOptionalRevision(words, row.extension_map_revision_before);
  appendOptionalRevision(words, row.extension_map_revision_after);
  appendWord(words, static_cast<std::uint64_t>(row.declared_stage));
  appendFunctionals(words, row.before);
  appendFunctionals(words, row.after);
  appendDouble(words, row.numerical_work);
  appendDouble(words, row.accepted_numerical_work);
  appendOptionalDouble(
      words, row.modeled_energy_numerical_work);
  appendOptionalDouble(
      words, row.accepted_modeled_energy_numerical_work);
}

CanonicalWords canonicalActiveTransaction(
    const LevelSetMaintenanceWorkLedger& ledger,
    std::span<const std::uint64_t> commit_state_words)
{
  CanonicalWords words;
  words.reserve(
      32u + 80u * ledger.trialRows().size() +
      commit_state_words.size());
  appendWord(words, 1u);
  const auto* transaction = ledger.activeTransaction();
  appendWord(words, transaction != nullptr ? 1u : 0u);
  if (transaction != nullptr) {
    appendTransaction(words, *transaction);
  }
  appendWord(
      words, static_cast<std::uint64_t>(ledger.trialRows().size()));
  for (const auto& row : ledger.trialRows()) {
    appendRow(words, row);
  }
  appendWord(
      words,
      static_cast<std::uint64_t>(commit_state_words.size()));
  words.insert(
      words.end(),
      commit_state_words.begin(),
      commit_state_words.end());
  return words;
}

} // namespace

bool collectiveLevelSetMaintenanceCanonicalWordsAgree(
    std::span<const std::uint64_t> local_words,
    const svmp::MeshComm& comm)
{
  if (!comm.is_parallel()) {
    return true;
  }

#ifdef MESH_HAS_MPI
  const auto local_word_count =
      static_cast<std::uint64_t>(local_words.size());
  std::uint64_t minimum_word_count{0u};
  std::uint64_t maximum_word_count{0u};
  MPI_Allreduce(
      &local_word_count,
      &minimum_word_count,
      1,
      MPI_UINT64_T,
      MPI_MIN,
      comm.native());
  MPI_Allreduce(
      &local_word_count,
      &maximum_word_count,
      1,
      MPI_UINT64_T,
      MPI_MAX,
      comm.native());

  // Keep the comparison scratch bounded and allocation-free after ranks have
  // entered the collective schedule. A rank-local allocation failure here
  // would otherwise strand peers inside the first value reduction.
  constexpr std::size_t chunk_words = 4096u;
  std::array<std::uint64_t, chunk_words> words{};
  std::array<std::uint64_t, chunk_words> minimum_words{};
  std::array<std::uint64_t, chunk_words> maximum_words{};
  bool values_agree = true;
  const auto global_word_count =
      static_cast<std::size_t>(maximum_word_count);
  for (std::size_t offset = 0u; offset < global_word_count;
       offset += chunk_words) {
    const auto words_this_chunk = std::min(
        chunk_words, global_word_count - offset);
    std::fill_n(words.begin(), words_this_chunk, 0u);
    const auto local_words_this_chunk = offset < local_words.size()
        ? std::min(words_this_chunk, local_words.size() - offset)
        : 0u;
    std::copy_n(
        local_words.begin() + static_cast<std::ptrdiff_t>(
                                  std::min(offset, local_words.size())),
        local_words_this_chunk,
        words.begin());
    const auto count = static_cast<int>(words_this_chunk);
    MPI_Allreduce(
        words.data(),
        minimum_words.data(),
        count,
        MPI_UINT64_T,
        MPI_MIN,
        comm.native());
    MPI_Allreduce(
        words.data(),
        maximum_words.data(),
        count,
        MPI_UINT64_T,
        MPI_MAX,
        comm.native());
    values_agree = values_agree && std::equal(
        minimum_words.begin(),
        minimum_words.begin() +
            static_cast<std::ptrdiff_t>(words_this_chunk),
        maximum_words.begin());
  }

  return minimum_word_count == maximum_word_count &&
         values_agree;
#else
  (void)local_words;
  return true;
#endif
}

LevelSetMaintenanceTransactionDecision
collectiveLevelSetMaintenanceTransactionDecision(
    const LevelSetMaintenanceWorkLedger& ledger,
    bool local_invariants_satisfied,
    const svmp::MeshComm& comm,
    std::span<const std::uint64_t> local_commit_state_words)
{
  const bool local_transaction_active =
      ledger.activeTransaction() != nullptr;
  if (!comm.is_parallel()) {
    return local_transaction_active && local_invariants_satisfied
        ? LevelSetMaintenanceTransactionDecision::Commit
        : LevelSetMaintenanceTransactionDecision::Reject;
  }

#ifdef MESH_HAS_MPI
  std::vector<std::uint64_t> words;
  bool local_canonical_prepared = true;
  try {
    words = canonicalActiveTransaction(
        ledger, local_commit_state_words);
  } catch (...) {
    local_canonical_prepared = false;
  }

  // Prepare every variable-size canonical payload before the first
  // collective.  Allocating it after ranks have entered the flag reduction
  // can strand peers in the subsequent word comparison if one rank throws.
  const int local_flags[3]{
      local_transaction_active ? 1 : 0,
      local_invariants_satisfied ? 1 : 0,
      local_canonical_prepared ? 1 : 0,
  };
  int global_flags[3]{0, 0, 0};
  MPI_Allreduce(
      local_flags,
      global_flags,
      3,
      MPI_INT,
      MPI_MIN,
      comm.native());

  if (global_flags[0] == 0 || global_flags[1] == 0 ||
      global_flags[2] == 0) {
    return LevelSetMaintenanceTransactionDecision::Reject;
  }
  const bool identical =
      collectiveLevelSetMaintenanceCanonicalWordsAgree(words, comm);
  return identical
      ? LevelSetMaintenanceTransactionDecision::Commit
      : LevelSetMaintenanceTransactionDecision::Reject;
#else
  return local_transaction_active && local_invariants_satisfied
      ? LevelSetMaintenanceTransactionDecision::Commit
      : LevelSetMaintenanceTransactionDecision::Reject;
#endif
}

} // namespace application::core
