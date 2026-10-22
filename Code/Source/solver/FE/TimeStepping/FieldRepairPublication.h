/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_FIELDREPAIRPUBLICATION_H
#define SVMP_FE_TIMESTEPPING_FIELDREPAIRPUBLICATION_H

/**
 * @file FieldRepairPublication.h
 * @brief Publication of a field repair (for example level-set maintenance)
 *        into an accepted TimeHistory.
 *
 * A repair changes one field after TimeHistory::acceptStep().  The caller
 * states, field by field, the delta of every stored state and optionally of
 * the rate; the staging layer applies them to replicated copies; the
 * collective publication checks that every rank staged the same thing and
 * assigns it to the history vectors.
 */

#include "Core/Types.h"
#include "Systems/SystemCollectives.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace timestepping {

class TimeHistory;

/**
 * Non-owning field-slice mapping into an already accepted TimeHistory.  This
 * layer selects no time-integration policy: it stages the explicit deltas it
 * is given (for example the common representation shift of
 * makeCommonRepresentationMaintainedFieldPublication).
 */
struct MaintainedFieldPostAcceptPublicationView {
  svmp::FE::FieldId field{svmp::FE::INVALID_FIELD_ID};
  std::size_t field_offset{0u};
  std::span<const svmp::FE::Real> u_delta{};
  std::span<const svmp::FE::Real> u_prev_delta{};
  std::span<const svmp::FE::Real> u_prev2_and_deeper_delta{};
  std::optional<std::span<const svmp::FE::Real>> u_dot_delta{};
  bool maintained_first_order_u_ddot_unchanged{true};
};

struct MaintainedFieldPostAcceptPublicationScheduleEntry {
  svmp::FE::FieldId field{svmp::FE::INVALID_FIELD_ID};
  std::size_t field_offset{0u};
  std::size_t field_count{0u};
  bool publishes_u_dot{false};
};

struct StagedPostacceptMaintenanceHistoryPublication {
  std::vector<svmp::FE::Real> current{};
  std::vector<std::vector<svmp::FE::Real>> history{};
  std::optional<std::vector<svmp::FE::Real>> u_dot{};
  std::vector<MaintainedFieldPostAcceptPublicationScheduleEntry> schedule{};
};

/// The same delta on every stored state and no rate change (endpoint schemes).
MaintainedFieldPostAcceptPublicationView
makeCommonRepresentationMaintainedFieldPublication(
    svmp::FE::FieldId field,
    std::size_t field_offset,
    std::span<const svmp::FE::Real> delta);

/// Replicated copies of the accepted state, history and rate to stage into.
StagedPostacceptMaintenanceHistoryPublication
makeStagedPostacceptMaintenanceHistoryPublication(
    std::vector<svmp::FE::Real> current,
    std::vector<std::vector<svmp::FE::Real>> history,
    std::optional<std::vector<svmp::FE::Real>> u_dot = std::nullopt);

/**
 * Add one field slice's deltas to the staged copies.  Fields must be staged
 * in increasing field order; every check precedes the first change.
 */
void stageMaintainedFieldPostAcceptPublication(
    const MaintainedFieldPostAcceptPublicationView& publication,
    StagedPostacceptMaintenanceHistoryPublication& staged);

/// Collective: the staged layouts and field schedules agree on every rank.
bool postacceptMaintenanceHistoryPublicationScheduleAgrees(
    const StagedPostacceptMaintenanceHistoryPublication& staged,
    const systems::CollectiveCommunicator& comm);

/// Collective: the staged values agree bitwise on every rank.
bool postacceptMaintenanceHistoryPublicationContentAgrees(
    const StagedPostacceptMaintenanceHistoryPublication& staged,
    const systems::CollectiveCommunicator& comm);

/**
 * Collective: check that the staged publication is complete and identical
 * on every rank, then assign it to the accepted TimeHistory (u, every
 * uPrevK, and uDot when staged).  *rate_publication_started is set just
 * before uDot is first written.
 */
void publishStagedPostacceptMaintenanceHistoryCollectively(
    svmp::FE::timestepping::TimeHistory& history,
    const StagedPostacceptMaintenanceHistoryPublication& staged,
    const systems::CollectiveCommunicator& comm,
    bool* rate_publication_started = nullptr);

} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_FIELDREPAIRPUBLICATION_H
