/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file LevelSetMaintenanceRequestSchedule.h
 * @ingroup fe_level_set
 * @brief The canonical schedule of the maintained level-set fields, the
 *        live cut geometry it is committed against, and their exact
 *        agreement across ranks.
 *
 * Every rank must take the same maintenance decisions.  Before each
 * maintenance stage the ordered requests are written as canonical words
 * (LevelSetMaintenanceSchedule.h) and compared exactly across the FE
 * communicator; a commit consensus adds the words of the live cut geometry.
 *
 * The words of a schedule, in this order: the header
 * (appendLevelSetMaintenanceScheduleHeader), then for each request its
 * action bits, its configuration words
 * (appendLevelSetMaintenanceConfigurationScheduleWords), the words the
 * application adds for its own maintenance, and its conservative-phase run
 * state (initialized flag and appendMaintenanceScheduleGraphMetadata).  The
 * curvature-projection words went with the projection (decision D48).
 *
 * The functions were moved from Application/Core/ApplicationDriver.cpp with
 * the words, messages and collective order unchanged.
 */

#include "Core/Types.h"
#include "LevelSet/LevelSetConservativePhaseOperator.h"
#include "LevelSet/LevelSetMaintenanceSchedule.h"
#include "LevelSet/Maintenance/LevelSetMaintenanceConfiguration.h"
#include "LevelSet/Maintenance/LevelSetMaintenancePipeline.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace svmp {
namespace FE {

namespace systems {
class CollectiveCommunicator;
class FESystem;
} // namespace systems

namespace level_set {

/// The maintenance stage a schedule is agreed for.
enum class LevelSetMaintenanceScheduleStage : std::uint64_t {
  SteadyInitialization = 1u,
  TransientInitialization = 2u,
  ProspectiveAcceptedEndpoint = 3u,
  AcceptedEndpointPostStep = 4u,
};

/// Canonical words of the live cut geometry a maintenance candidate is
/// committed against; supported is false when a request's geometry cannot
/// be represented.
struct CanonicalLevelSetMaintenanceGeometryState {
  std::vector<std::uint64_t> words{};
  bool supported{true};
};

/**
 * @brief Bits of the maintenance actions of @p configuration configured and
 *        due after @p completed_step.
 *
 * Bit 0 conservative phase, 3 constant or 4 field velocity, 8 any
 * functional maintenance due, 9 conservative flux artifact due, 10
 * conservative geometry reconciliation, 12 kinematic reconciliation due, 13
 * sign-definite patch bounds due, 14 interface-preserving redistancing due.
 * Bits 1, 2, 5, 6 and 7 described retired operations and stay unused; bit
 * 11 is the application's.
 */
[[nodiscard]] std::uint64_t levelSetMaintenanceRequestActionBits(
    const LevelSetMaintenanceConfiguration& configuration,
    int completed_step);

/// The schedule header: the schema version, @p stage, @p completed_step, the
/// number of requests and the union of their action bits.
void appendLevelSetMaintenanceScheduleHeader(
    std::vector<std::uint64_t>& words,
    LevelSetMaintenanceScheduleStage stage,
    int completed_step,
    std::size_t request_count,
    std::uint64_t aggregate_action_bits);

/**
 * @brief The words of a request's configuration: transport, velocity,
 *        conservative phase, open boundaries, accepted-step stages and the
 *        active cut-volume request.
 *
 * Returns false when the configuration cannot be represented (an embedded
 * velocity FunctionSpace); the words are appended either way.
 */
[[nodiscard]] bool appendLevelSetMaintenanceConfigurationScheduleWords(
    std::vector<std::uint64_t>& words,
    const LevelSetMaintenanceConfiguration& configuration);

/// The replicated identity of a conservative-phase graph.  The graph's
/// rank-local mesh event counters are left out: equally valid partitions can
/// advance them a different number of times.
void appendMaintenanceScheduleGraphMetadata(
    std::vector<std::uint64_t>& words,
    const std::optional<LevelSetP1PhaseTransportGraph>& graph);

/// Append @p schedule as a tagged section of an enclosing commit state.
void appendCanonicalLevelSetMaintenanceScheduleSection(
    std::vector<std::uint64_t>& words,
    const CanonicalLevelSetMaintenanceSchedule& schedule);

/**
 * @brief Words of the live cut geometry of @p requests in @p system: the
 *        operator layout revisions and, for each request with an active
 *        cut-volume request, its interface marker and bound geometry
 *        snapshot; then every snapshot of the cut integration context in a
 *        canonical order.  Unsupported without a system.  Rank-local.
 */
[[nodiscard]] CanonicalLevelSetMaintenanceGeometryState
canonicalLevelSetMaintenanceGeometryState(
    const systems::FESystem* system,
    std::span<const LevelSetMaintenanceRequest> requests);

/// Append @p geometry as a tagged section of an enclosing commit state.
void appendCanonicalLevelSetMaintenanceGeometrySection(
    std::vector<std::uint64_t>& words,
    const CanonicalLevelSetMaintenanceGeometryState& geometry);

/**
 * @brief Require @p schedule to agree exactly on every rank and to be
 *        supported; throws std::runtime_error otherwise.  Collective over
 *        @p comm.
 */
void requireCollectiveLevelSetMaintenanceRequestSchedule(
    const CanonicalLevelSetMaintenanceSchedule& schedule,
    const systems::CollectiveCommunicator& comm);

/// Rank-local revision of an algebraic state: a 64-bit FNV-1a digest of its
/// length and value bit patterns, never zero.
[[nodiscard]] std::uint64_t levelSetMaintenanceAlgebraicRevision(
    std::span<const Real> state) noexcept;

/// levelSetMaintenanceAlgebraicRevision, required to agree on every rank
/// (throws std::runtime_error otherwise).  Collective over @p comm.
[[nodiscard]] std::uint64_t collectiveLevelSetMaintenanceAlgebraicRevision(
    std::span<const Real> state,
    const systems::CollectiveCommunicator& comm);

} // namespace level_set
} // namespace FE
} // namespace svmp
