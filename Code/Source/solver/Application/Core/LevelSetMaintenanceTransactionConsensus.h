#pragma once

#include <cstdint>
#include <span>

namespace svmp {
class MeshComm;
}

namespace application::core {

class LevelSetMaintenanceWorkLedger;

enum class LevelSetMaintenanceTransactionDecision : std::uint8_t {
  Commit,
  Reject
};

/**
 * @brief Exact all-rank agreement for a canonical sequence of 64-bit words.
 *
 * Length and every word are compared directly.  The routine deliberately
 * does not hash the input, so agreement cannot be produced by a digest
 * collision.  All ranks execute the same padded collective schedule even
 * when their local sequence lengths differ.
 */
[[nodiscard]] bool collectiveLevelSetMaintenanceCanonicalWordsAgree(
    std::span<const std::uint64_t> local_words,
    const svmp::MeshComm& comm);

[[nodiscard]] LevelSetMaintenanceTransactionDecision
collectiveLevelSetMaintenanceTransactionDecision(
    const LevelSetMaintenanceWorkLedger& ledger,
    bool local_invariants_satisfied,
    const svmp::MeshComm& comm,
    std::span<const std::uint64_t> local_commit_state_words = {});

} // namespace application::core
