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
 * A repair changes one field after TimeHistory::acceptStep().  For a
 * first-order generalized-alpha field the planner decides, from the stage
 * and endpoint deltas and an explicit closure, which stored states and
 * rates change; the staging layer applies the planned deltas to replicated
 * copies field by field; the collective publication checks that every rank
 * staged the same thing and assigns it to the history vectors.
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
 * Algebraic closure for planning a first-order generalized-alpha level-set
 * repair publication after TimeHistory::acceptStep().
 *
 * Neither policy infers a past-rate change from two repaired state slices.
 * The choice is explicit because the generalized-alpha state identities do
 * not determine both rate corrections.
 */
enum class FirstOrderGeneralizedAlphaMaintenanceClosure : std::uint8_t {
  /** Apply one representation delta to every stored state; rates are fixed. */
  SameRepresentationDelta,
  /** Keep the prior accepted state and rate fixed; update the endpoint rate. */
  PreservePriorStateAndRate
};

enum class FirstOrderGeneralizedAlphaMaintenancePlanStatus : std::uint8_t {
  AlgebraicallyComplete,
  ClosureMismatch,
  NearSingularStageInversion,
  EndpointPolicyMismatch,
  MissingRateParameters
};

struct FirstOrderGeneralizedAlphaMaintenanceScheme {
  std::optional<svmp::FE::Real> alpha_m{};
  svmp::FE::Real alpha_f{1.0};
  std::optional<svmp::FE::Real> gamma{};
  std::optional<svmp::FE::Real> dt{};
};

/**
 * Explicit post-accept TimeHistory mapping for one maintained first-order
 * field slice.  `u_prev2_and_deeper_delta` is applied to uPrevK(2) and every
 * deeper stored solution state; u and uPrevK(1) are both endpoint copies
 * after acceptStep().  uDDot is intentionally not a publication target for
 * this first-order field policy.
 */
struct FirstOrderGeneralizedAlphaPostAcceptPublication {
  std::vector<svmp::FE::Real> u_delta{};
  std::vector<svmp::FE::Real> u_prev_delta{};
  std::vector<svmp::FE::Real> u_prev2_and_deeper_delta{};
  std::vector<svmp::FE::Real> prior_rate_delta{};
  std::vector<svmp::FE::Real> u_dot_delta{};
  std::vector<svmp::FE::Real> accepted_stage_rate_delta{};
  bool maintained_first_order_u_ddot_unchanged{true};
};

struct FirstOrderGeneralizedAlphaMaintenancePublicationPlan {
  FirstOrderGeneralizedAlphaMaintenanceScheme scheme{};
  FirstOrderGeneralizedAlphaMaintenanceClosure closure{
      FirstOrderGeneralizedAlphaMaintenanceClosure::SameRepresentationDelta};
  FirstOrderGeneralizedAlphaMaintenancePlanStatus status{
      FirstOrderGeneralizedAlphaMaintenancePlanStatus::ClosureMismatch};
  std::vector<svmp::FE::Real> requested_stage_state_delta{};
  std::vector<svmp::FE::Real> requested_endpoint_state_delta{};
  std::optional<std::vector<svmp::FE::Real>>
      implied_prior_state_delta{};
  std::optional<FirstOrderGeneralizedAlphaPostAcceptPublication>
      post_accept{};
  svmp::FE::Real max_stage_state_identity_residual{0.0};
  svmp::FE::Real max_endpoint_update_identity_residual{0.0};
  svmp::FE::Real max_stage_rate_identity_residual{0.0};
  svmp::FE::Real identity_tolerance{0.0};
  bool requires_separate_geometric_motion_account{false};
  std::string diagnostic{};
};

/// Scheme with explicit, validated parameters (gamma = 1/2 + alpha_m - alpha_f).
FirstOrderGeneralizedAlphaMaintenanceScheme
makeFirstOrderGeneralizedAlphaMaintenanceScheme(
    svmp::FE::Real alpha_m,
    svmp::FE::Real alpha_f,
    svmp::FE::Real gamma,
    svmp::FE::Real dt);

/**
 * Plan the post-accept publication of one maintained first-order field
 * repair: the stage-state and endpoint-state deltas of the repair, and the
 * closure that fixes what the deltas do not determine.  Returns a plan whose
 * status says whether the publication is algebraically complete.
 */
FirstOrderGeneralizedAlphaMaintenancePublicationPlan
planFirstOrderGeneralizedAlphaMaintenancePublication(
    const FirstOrderGeneralizedAlphaMaintenanceScheme& scheme,
    FirstOrderGeneralizedAlphaMaintenanceClosure closure,
    std::span<const svmp::FE::Real> stage_state_delta,
    std::span<const svmp::FE::Real> endpoint_state_delta);

/**
 * Non-owning field-slice mapping into an already accepted TimeHistory.  This
 * layer performs no generalized-alpha policy selection: it can only stage the
 * explicit deltas supplied by a validated planner (or the existing common
 * representation shift used by endpoint schemes).
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

/// A complete plan's publication plus one common delta on every stored state.
FirstOrderGeneralizedAlphaPostAcceptPublication
combineFirstOrderGeneralizedAlphaPublicationWithCommonDelta(
    const FirstOrderGeneralizedAlphaMaintenancePublicationPlan& plan,
    std::span<const svmp::FE::Real> common_delta);

/// Field-slice view of a complete plan's publication.
MaintainedFieldPostAcceptPublicationView
makeFirstOrderGeneralizedAlphaMaintainedFieldPublication(
    svmp::FE::FieldId field,
    std::size_t field_offset,
    const FirstOrderGeneralizedAlphaPostAcceptPublication& publication,
    bool publish_u_dot,
    bool maintained_field_has_time_derivative);

MaintainedFieldPostAcceptPublicationView
makeFirstOrderGeneralizedAlphaMaintainedFieldPublication(
    svmp::FE::FieldId field,
    std::size_t field_offset,
    const FirstOrderGeneralizedAlphaMaintenancePublicationPlan& plan,
    bool publish_u_dot,
    bool maintained_field_has_time_derivative);

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
