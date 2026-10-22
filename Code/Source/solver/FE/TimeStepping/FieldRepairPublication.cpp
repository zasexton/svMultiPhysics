/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/FieldRepairPublication.h"

#include "Backends/Interfaces/GenericVector.h"
#include "TimeStepping/TimeHistory.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace svmp {
namespace FE {
namespace timestepping {

MaintainedFieldPostAcceptPublicationView
makeCommonRepresentationMaintainedFieldPublication(
    svmp::FE::FieldId field,
    std::size_t field_offset,
    std::span<const svmp::FE::Real> delta)
{
  return MaintainedFieldPostAcceptPublicationView{
      .field = field,
      .field_offset = field_offset,
      .u_delta = delta,
      .u_prev_delta = delta,
      .u_prev2_and_deeper_delta = delta,
      .u_dot_delta = std::nullopt,
      .maintained_first_order_u_ddot_unchanged = true,
  };
}

StagedPostacceptMaintenanceHistoryPublication
makeStagedPostacceptMaintenanceHistoryPublication(
    std::vector<svmp::FE::Real> current,
    std::vector<std::vector<svmp::FE::Real>> history,
    std::optional<std::vector<svmp::FE::Real>> u_dot)
{
  if (current.empty() || history.empty()) {
    throw std::invalid_argument(
        "Postaccept maintained-field publication requires a nonempty current "
        "state and at least one stored history level.");
  }
  for (const auto& level : history) {
    if (level.size() != current.size()) {
      throw std::invalid_argument(
          "Postaccept maintained-field publication history layouts differ.");
    }
  }
  if (u_dot.has_value() && u_dot->size() != current.size()) {
    throw std::invalid_argument(
        "Postaccept maintained-field publication rate layout differs from "
        "the solution layout.");
  }
  return StagedPostacceptMaintenanceHistoryPublication{
      .current = std::move(current),
      .history = std::move(history),
      .u_dot = std::move(u_dot),
  };
}

void stageMaintainedFieldPostAcceptPublication(
    const MaintainedFieldPostAcceptPublicationView& publication,
    StagedPostacceptMaintenanceHistoryPublication& staged)
{
  const auto field_count = publication.u_delta.size();
  const bool common_delta_layout =
      field_count != 0u &&
      publication.u_prev_delta.size() == field_count &&
      publication.u_prev2_and_deeper_delta.size() == field_count &&
      (!publication.u_dot_delta.has_value() ||
       publication.u_dot_delta->size() == field_count);
  const bool field_slice_fits =
      publication.field_offset <= staged.current.size() &&
      field_count <= staged.current.size() - publication.field_offset;
  const bool canonical_field_order =
      staged.schedule.empty() ||
      staged.schedule.back().field < publication.field;
  if (publication.field == svmp::FE::INVALID_FIELD_ID ||
      !common_delta_layout || !field_slice_fits ||
      !canonical_field_order || staged.history.empty() ||
      !publication.maintained_first_order_u_ddot_unchanged ||
      (publication.u_dot_delta.has_value() && !staged.u_dot.has_value())) {
    throw std::invalid_argument(
        "Postaccept maintained-field publication received an invalid, "
        "duplicate, unordered, or unsupported field mapping.");
  }
  for (const auto& level : staged.history) {
    if (level.size() != staged.current.size()) {
      throw std::invalid_argument(
          "Postaccept maintained-field publication history layout changed "
          "during staging.");
    }
  }
  if (staged.u_dot.has_value() &&
      staged.u_dot->size() != staged.current.size()) {
    throw std::invalid_argument(
        "Postaccept maintained-field publication rate layout changed during "
        "staging.");
  }

  const auto finite = [](std::span<const svmp::FE::Real> values) {
    return std::all_of(values.begin(), values.end(), [](svmp::FE::Real value) {
      return std::isfinite(value);
    });
  };
  if (!finite(publication.u_delta) ||
      !finite(publication.u_prev_delta) ||
      !finite(publication.u_prev2_and_deeper_delta) ||
      (publication.u_dot_delta.has_value() &&
       !finite(*publication.u_dot_delta))) {
    throw std::invalid_argument(
        "Postaccept maintained-field publication deltas must be finite.");
  }

  // Validation is complete before any staged coefficient is changed.
  for (std::size_t i = 0u; i < field_count; ++i) {
    const auto index = publication.field_offset + i;
    staged.current[index] += publication.u_delta[i];
    staged.history.front()[index] += publication.u_prev_delta[i];
    for (std::size_t level = 1u; level < staged.history.size(); ++level) {
      staged.history[level][index] +=
          publication.u_prev2_and_deeper_delta[i];
    }
    if (publication.u_dot_delta.has_value()) {
      (*staged.u_dot)[index] += (*publication.u_dot_delta)[i];
    }
  }
  staged.schedule.push_back(
      MaintainedFieldPostAcceptPublicationScheduleEntry{
          .field = publication.field,
          .field_offset = publication.field_offset,
          .field_count = field_count,
          .publishes_u_dot = publication.u_dot_delta.has_value(),
      });
}

bool postacceptMaintenanceHistoryPublicationScheduleAgrees(
    const StagedPostacceptMaintenanceHistoryPublication& staged,
    const systems::CollectiveCommunicator& comm)
{
  bool agrees = true;
  const std::array<std::uint64_t, 4> header{{
      static_cast<std::uint64_t>(staged.current.size()),
      static_cast<std::uint64_t>(staged.history.size()),
      staged.u_dot.has_value() ? 1u : 0u,
      static_cast<std::uint64_t>(staged.schedule.size()),
  }};
  for (const auto word : header) {
    const auto [minimum_word, maximum_word] =
        systems::collectiveMinMaxUint64(word, comm);
    agrees = minimum_word == maximum_word && agrees;
  }
  const auto [minimum_entry_count, maximum_entry_count] =
      systems::collectiveMinMaxUint64(
          static_cast<std::uint64_t>(staged.schedule.size()), comm);
  if (minimum_entry_count != maximum_entry_count) {
    return false;
  }
  for (const auto& entry : staged.schedule) {
    const std::array<std::uint64_t, 4> words{{
        static_cast<std::uint64_t>(
            static_cast<std::int64_t>(entry.field)),
        static_cast<std::uint64_t>(entry.field_offset),
        static_cast<std::uint64_t>(entry.field_count),
        entry.publishes_u_dot ? 1u : 0u,
    }};
    for (const auto word : words) {
      const auto [minimum_word, maximum_word] =
          systems::collectiveMinMaxUint64(word, comm);
      agrees = minimum_word == maximum_word && agrees;
    }
  }
  return agrees;
}

bool postacceptMaintenanceHistoryPublicationContentAgrees(
    const StagedPostacceptMaintenanceHistoryPublication& staged,
    const systems::CollectiveCommunicator& comm)
{
  static_assert(
      sizeof(svmp::FE::Real) == sizeof(std::uint64_t),
      "Exact maintained-field publication consensus requires a 64-bit FE scalar.");
  constexpr std::size_t chunk_capacity = 4096u;
  std::array<std::uint64_t, chunk_capacity> local_bits{};
  std::array<std::uint64_t, chunk_capacity> minimum_bits{};
  std::array<std::uint64_t, chunk_capacity> maximum_bits{};
  bool agrees = true;
  const auto compare = [&](std::span<const svmp::FE::Real> values) {
    for (std::size_t offset = 0u; offset < values.size();
         offset += chunk_capacity) {
      const auto count = std::min(chunk_capacity, values.size() - offset);
      for (std::size_t i = 0u; i < count; ++i) {
        local_bits[i] =
            std::bit_cast<std::uint64_t>(values[offset + i]);
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
            "[svMultiPhysics::Application] Distributed maintained-field "
            "publication consensus requires MPI.");
      }
      std::copy_n(local_bits.begin(), count, minimum_bits.begin());
      std::copy_n(local_bits.begin(), count, maximum_bits.begin());
#endif
      for (std::size_t i = 0u; i < count; ++i) {
        agrees = minimum_bits[i] == maximum_bits[i] && agrees;
      }
    }
  };

  compare(staged.current);
  for (const auto& level : staged.history) {
    compare(level);
  }
  if (staged.u_dot.has_value()) {
    compare(*staged.u_dot);
  }
  return agrees;
}

void publishStagedPostacceptMaintenanceHistoryCollectively(
    svmp::FE::timestepping::TimeHistory& history,
    const StagedPostacceptMaintenanceHistoryPublication& staged,
    const systems::CollectiveCommunicator& comm,
    bool* rate_publication_started)
{
  if (rate_publication_started != nullptr) {
    *rate_publication_started = false;
  }
  const bool schedule_has_rate_target = std::any_of(
      staged.schedule.begin(),
      staged.schedule.end(),
      [](const auto& entry) { return entry.publishes_u_dot; });
  const bool schedule_layout_valid =
      !staged.schedule.empty() &&
      schedule_has_rate_target == staged.u_dot.has_value() &&
      std::all_of(
          staged.schedule.begin(), staged.schedule.end(),
          [&](const auto& entry) {
            return entry.field != svmp::FE::INVALID_FIELD_ID &&
                   entry.field_count != 0u &&
                   entry.field_offset <= staged.current.size() &&
                   entry.field_count <=
                       staged.current.size() - entry.field_offset;
          }) &&
      std::adjacent_find(
          staged.schedule.begin(), staged.schedule.end(),
          [](const auto& left, const auto& right) {
            return left.field >= right.field;
          }) == staged.schedule.end();
  const bool local_layout_valid =
      schedule_layout_valid && !staged.current.empty() &&
      !staged.history.empty() &&
      staged.history.size() ==
          static_cast<std::size_t>(history.historyDepth()) &&
      staged.current.size() == static_cast<std::size_t>(history.u().size()) &&
      std::all_of(
          staged.history.begin(), staged.history.end(), [&](const auto& level) {
            return level.size() == staged.current.size();
          }) &&
      (!staged.u_dot.has_value() ||
       (history.hasUDotState() &&
        staged.u_dot->size() == staged.current.size()));
  if (systems::collectiveMinDouble(local_layout_valid ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Postaccept maintained-field "
        "publication layouts are incomplete on at least one rank.");
  }
  if (!postacceptMaintenanceHistoryPublicationScheduleAgrees(staged, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Postaccept maintained-field "
        "publication schedules differ across the FE communicator.");
  }
  if (!postacceptMaintenanceHistoryPublicationContentAgrees(staged, comm)) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Postaccept maintained-field "
        "publication values differ across the FE communicator.");
  }

  systems::assignFeOrderedVectorCollectively(
      history.u(), staged.current, comm);
  for (int level = 1; level <= history.historyDepth(); ++level) {
    systems::assignFeOrderedVectorCollectively(
        history.uPrevK(level),
        staged.history[static_cast<std::size_t>(level - 1)],
        comm);
  }
  if (staged.u_dot.has_value()) {
    // Set the explicit flag before the first operation that may mutate uDot.
    // Recovery therefore never relies on backend valueRevision behavior.
    if (rate_publication_started != nullptr) {
      *rate_publication_started = true;
    }
    systems::assignFeOrderedVectorCollectively(
        history.uDot(), *staged.u_dot, comm);
  }
}

} // namespace timestepping
} // namespace FE
} // namespace svmp
