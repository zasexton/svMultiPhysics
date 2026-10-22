/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/LevelSetMaintenanceSchedule.h"

#include "Systems/SystemCollectives.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace svmp {
namespace FE {
namespace level_set {

namespace {

// Moved from Application/Core/LevelSetMaintenanceTransactionConsensus.cpp
// (kMaintenanceCanonicalWordsReport and canonicalWordsAgree at 7d9ddc8a)
// with the text and body unchanged.

// Report text of a failed agreement; FE writes the line rank-locally before
// the caller acts on the collective result.
constexpr systems::CanonicalWordsDisagreementReport
    kMaintenanceCanonicalWordsReport{
        "[svMultiPhysics::Application]",
        "Level-set maintenance canonical words disagree",
        "maintenance_consensus_word_disagreement"};

} // namespace

bool collectiveLevelSetMaintenanceCanonicalWordsAgree(
    std::span<const std::uint64_t> local_words,
    const systems::CollectiveCommunicator& comm,
    std::string_view context)
{
  if (!comm.isParallel()) {
    return true;
  }

#if FE_HAS_MPI
  return systems::collectiveCanonicalWordsAgree(
      local_words,
      comm,
      context,
      kMaintenanceCanonicalWordsReport);
#else
  (void)local_words;
  (void)context;
  return true;
#endif
}

} // namespace level_set
} // namespace FE
} // namespace svmp
