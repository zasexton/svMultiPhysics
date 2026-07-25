#include "Application/Core/LevelSetMaintenanceTransactionConsensus.h"

#include "Application/Core/ApplicationDriver.h"
#include "Mesh/Core/MeshComm.h"

#include <algorithm>
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
}

CanonicalWords canonicalActiveTransaction(
    const LevelSetMaintenanceWorkLedger& ledger,
    std::span<const std::uint64_t> commit_state_words)
{
  CanonicalWords words;
  words.reserve(
      32u + 48u * ledger.trialRows().size() +
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

  CanonicalWords words(local_words.begin(), local_words.end());
  words.resize(static_cast<std::size_t>(maximum_word_count), 0u);
  CanonicalWords minimum_words(words.size(), 0u);
  CanonicalWords maximum_words(words.size(), 0u);
  std::size_t offset = 0u;
  while (offset < words.size()) {
    const auto remaining = words.size() - offset;
    const auto count = static_cast<int>(std::min<std::size_t>(
        remaining,
        static_cast<std::size_t>(std::numeric_limits<int>::max())));
    MPI_Allreduce(
        words.data() + offset,
        minimum_words.data() + offset,
        count,
        MPI_UINT64_T,
        MPI_MIN,
        comm.native());
    MPI_Allreduce(
        words.data() + offset,
        maximum_words.data() + offset,
        count,
        MPI_UINT64_T,
        MPI_MAX,
        comm.native());
    offset += static_cast<std::size_t>(count);
  }

  return minimum_word_count == maximum_word_count &&
         minimum_words == maximum_words;
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
  const int local_flags[2]{
      local_transaction_active ? 1 : 0,
      local_invariants_satisfied ? 1 : 0,
  };
  int global_flags[2]{0, 0};
  MPI_Allreduce(
      local_flags,
      global_flags,
      2,
      MPI_INT,
      MPI_MIN,
      comm.native());

  const auto words = canonicalActiveTransaction(
      ledger, local_commit_state_words);
  const bool identical =
      collectiveLevelSetMaintenanceCanonicalWordsAgree(words, comm);
  return global_flags[0] != 0 &&
             global_flags[1] != 0 &&
             identical
      ? LevelSetMaintenanceTransactionDecision::Commit
      : LevelSetMaintenanceTransactionDecision::Reject;
#else
  return local_transaction_active && local_invariants_satisfied
      ? LevelSetMaintenanceTransactionDecision::Commit
      : LevelSetMaintenanceTransactionDecision::Reject;
#endif
}

} // namespace application::core
