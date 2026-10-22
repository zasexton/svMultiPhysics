/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_LEVELSET_LEVELSETMAINTENANCESCHEDULE_H
#define SVMP_FE_LEVELSET_LEVELSETMAINTENANCESCHEDULE_H

/**
 * @file LevelSetMaintenanceSchedule.h
 * @brief Canonical words of level-set maintenance schedules and their exact
 *        agreement across ranks.
 *
 * Maintenance decisions that every rank must take identically (the request
 * schedule, the phase-graph identity, the staged candidate provenance) are
 * written as sequences of 64-bit words in a fixed order and compared exactly
 * across the communicator; nothing is hashed.  Reals enter by their bit
 * patterns, strings by their length and bytes.
 */

#include "Core/Types.h"

#include <bit>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace svmp {
namespace FE {

namespace systems {
class CollectiveCommunicator;
} // namespace systems

namespace level_set {

/// Canonical words of a maintenance request schedule; supported is false
/// when a request cannot be represented.
// Moved from Application/Core/ApplicationDriver.cpp (lines 3615-3618 at 7d9ddc8a)
// with bodies, messages and collective order unchanged.
struct CanonicalLevelSetMaintenanceSchedule {
  std::vector<std::uint64_t> words{};
  bool supported{true};
};

// Appending canonical words.
// Moved from Application/Core/ApplicationDriver.cpp (lines 3625-3681 at 7d9ddc8a)
// with bodies, messages and collective order unchanged.
inline void appendMaintenanceScheduleWord(
    std::vector<std::uint64_t>& words,
    std::uint64_t value)
{
  words.push_back(value);
}

inline void appendMaintenanceScheduleBool(
    std::vector<std::uint64_t>& words,
    bool value)
{
  appendMaintenanceScheduleWord(words, value ? 1u : 0u);
}

inline void appendMaintenanceScheduleSigned(
    std::vector<std::uint64_t>& words,
    std::int64_t value)
{
  appendMaintenanceScheduleWord(
      words, static_cast<std::uint64_t>(value));
}

template <typename Enum>
inline void appendMaintenanceScheduleEnum(
    std::vector<std::uint64_t>& words,
    Enum value)
{
  static_assert(std::is_enum_v<Enum>);
  using Underlying = std::underlying_type_t<Enum>;
  if constexpr (std::is_signed_v<Underlying>) {
    appendMaintenanceScheduleSigned(
        words, static_cast<std::int64_t>(value));
  } else {
    appendMaintenanceScheduleWord(
        words, static_cast<std::uint64_t>(value));
  }
}

inline void appendMaintenanceScheduleReal(
    std::vector<std::uint64_t>& words,
    double value)
{
  appendMaintenanceScheduleWord(
      words, std::bit_cast<std::uint64_t>(value));
}

inline void appendMaintenanceScheduleString(
    std::vector<std::uint64_t>& words,
    std::string_view value)
{
  appendMaintenanceScheduleWord(
      words, static_cast<std::uint64_t>(value.size()));
  for (const unsigned char byte : value) {
    appendMaintenanceScheduleWord(
        words, static_cast<std::uint64_t>(byte));
  }
}

/**
 * @brief Exact all-rank agreement of canonical maintenance words.
 *
 * systems::collectiveCanonicalWordsAgree with the level-set maintenance
 * report ("[svMultiPhysics::Application] ... Level-set maintenance canonical
 * words disagree diagnostic=maintenance_consensus_word_disagreement").
 * Returns true without communicating on a single rank.  Collective over
 * @p comm.
 */
[[nodiscard]] bool collectiveLevelSetMaintenanceCanonicalWordsAgree(
    std::span<const std::uint64_t> local_words,
    const systems::CollectiveCommunicator& comm,
    std::string_view context = "canonical_words");

} // namespace level_set
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_LEVELSET_LEVELSETMAINTENANCESCHEDULE_H
