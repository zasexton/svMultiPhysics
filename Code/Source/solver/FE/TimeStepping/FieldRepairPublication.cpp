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

FirstOrderGeneralizedAlphaMaintenanceScheme
makeFirstOrderGeneralizedAlphaMaintenanceScheme(
    svmp::FE::Real alpha_m,
    svmp::FE::Real alpha_f,
    svmp::FE::Real gamma,
    svmp::FE::Real dt)
{
  if (!std::isfinite(alpha_m) ||
      !std::isfinite(alpha_f) ||
      !std::isfinite(gamma) ||
      !std::isfinite(dt) ||
      !(alpha_f > svmp::FE::Real{0.0}) ||
      alpha_f > svmp::FE::Real{1.0} ||
      !(gamma > svmp::FE::Real{0.0}) ||
      !(dt > svmp::FE::Real{0.0})) {
    throw std::invalid_argument(
        "First-order generalized-alpha maintenance requires finite explicit "
        "parameters with alpha_f in (0,1], gamma > 0, and dt > 0.");
  }
  const auto expected_gamma =
      svmp::FE::Real{0.5} + alpha_m - alpha_f;
  const auto parameter_tolerance =
      svmp::FE::Real{512.0} *
      std::numeric_limits<svmp::FE::Real>::epsilon() *
      std::max(
          {svmp::FE::Real{1.0},
           std::abs(alpha_m),
           std::abs(alpha_f),
           std::abs(gamma)});
  if (std::abs(gamma - expected_gamma) > parameter_tolerance) {
    throw std::invalid_argument(
        "First-order generalized-alpha maintenance requires explicit gamma "
        "consistent with 1/2 + alpha_m - alpha_f.");
  }
  return FirstOrderGeneralizedAlphaMaintenanceScheme{
      .alpha_m = alpha_m,
      .alpha_f = alpha_f,
      .gamma = gamma,
      .dt = dt,
  };
}

FirstOrderGeneralizedAlphaMaintenancePublicationPlan
planFirstOrderGeneralizedAlphaMaintenancePublication(
    const FirstOrderGeneralizedAlphaMaintenanceScheme& scheme,
    FirstOrderGeneralizedAlphaMaintenanceClosure closure,
    std::span<const svmp::FE::Real> stage_state_delta,
    std::span<const svmp::FE::Real> endpoint_state_delta)
{
  const bool has_any_rate_parameter =
      scheme.alpha_m.has_value() || scheme.gamma.has_value() ||
      scheme.dt.has_value();
  const bool has_complete_rate_parameters =
      scheme.alpha_m.has_value() && scheme.gamma.has_value() &&
      scheme.dt.has_value();
  const auto parameter_scale = std::max(
      {svmp::FE::Real{1.0},
       std::abs(scheme.alpha_f),
       scheme.alpha_m.has_value() ? std::abs(*scheme.alpha_m)
                                  : svmp::FE::Real{0.0},
       scheme.gamma.has_value() ? std::abs(*scheme.gamma)
                                : svmp::FE::Real{0.0}});
  const auto parameter_tolerance =
      svmp::FE::Real{512.0} *
      std::numeric_limits<svmp::FE::Real>::epsilon() * parameter_scale;
  const auto expected_gamma = has_complete_rate_parameters
      ? svmp::FE::Real{0.5} + *scheme.alpha_m - scheme.alpha_f
      : svmp::FE::Real{0.0};
  if (!std::isfinite(scheme.alpha_f) ||
      !(scheme.alpha_f > svmp::FE::Real{0.0}) ||
      scheme.alpha_f > svmp::FE::Real{1.0} ||
      (has_any_rate_parameter != has_complete_rate_parameters) ||
      (has_complete_rate_parameters &&
       (!std::isfinite(*scheme.alpha_m) ||
        !std::isfinite(*scheme.gamma) ||
        !std::isfinite(*scheme.dt) ||
        !(*scheme.gamma > svmp::FE::Real{0.0}) ||
        !(*scheme.dt > svmp::FE::Real{0.0}) ||
        std::abs(*scheme.gamma - expected_gamma) >
            parameter_tolerance))) {
    throw std::invalid_argument(
        "First-order generalized-alpha maintenance received inconsistent "
        "scheme parameters.");
  }
  if (stage_state_delta.empty() ||
      stage_state_delta.size() != endpoint_state_delta.size()) {
    throw std::invalid_argument(
        "First-order generalized-alpha maintenance repair deltas require "
        "one nonempty common field layout.");
  }
  for (std::size_t i = 0u; i < stage_state_delta.size(); ++i) {
    if (!std::isfinite(stage_state_delta[i]) ||
        !std::isfinite(endpoint_state_delta[i])) {
      throw std::invalid_argument(
          "First-order generalized-alpha maintenance repair deltas must be "
          "finite.");
    }
  }

  FirstOrderGeneralizedAlphaMaintenancePublicationPlan plan;
  plan.scheme = scheme;
  plan.closure = closure;
  plan.requested_stage_state_delta.assign(
      stage_state_delta.begin(), stage_state_delta.end());
  plan.requested_endpoint_state_delta.assign(
      endpoint_state_delta.begin(), endpoint_state_delta.end());

  const auto scaled_tolerance = [](svmp::FE::Real left,
                                   svmp::FE::Real right) {
    return svmp::FE::Real{1024.0} *
           std::numeric_limits<svmp::FE::Real>::epsilon() *
           std::max(
               {svmp::FE::Real{1.0}, std::abs(left), std::abs(right)});
  };
  const auto deltas_match = [&](auto expected) {
    bool match = true;
    for (std::size_t i = 0u; i < stage_state_delta.size(); ++i) {
      const auto expected_stage = expected(endpoint_state_delta[i]);
      const auto tolerance =
          scaled_tolerance(stage_state_delta[i], expected_stage);
      plan.identity_tolerance = std::max(plan.identity_tolerance, tolerance);
      plan.max_stage_state_identity_residual = std::max(
          plan.max_stage_state_identity_residual,
          std::abs(stage_state_delta[i] - expected_stage));
      match =
          std::abs(stage_state_delta[i] - expected_stage) <= tolerance &&
          match;
    }
    return match;
  };

  const bool endpoint_stage =
      scheme.alpha_f == svmp::FE::Real{1.0};
  bool closure_satisfied = false;
  if (closure == FirstOrderGeneralizedAlphaMaintenanceClosure::
                     SameRepresentationDelta) {
    closure_satisfied = deltas_match([](svmp::FE::Real endpoint_delta) {
      return endpoint_delta;
    });
  } else if (!endpoint_stage) {
    closure_satisfied = deltas_match([&](svmp::FE::Real endpoint_delta) {
      return scheme.alpha_f * endpoint_delta;
    });
  }

  if (endpoint_stage) {
    if (closure != FirstOrderGeneralizedAlphaMaintenanceClosure::
                       SameRepresentationDelta) {
      plan.status =
          FirstOrderGeneralizedAlphaMaintenancePlanStatus::
              EndpointPolicyMismatch;
      plan.diagnostic =
          "alpha_f=1 intentionally remains on the supported common-"
          "representation publication policy.";
      return plan;
    }
    if (!closure_satisfied) {
      plan.status =
          FirstOrderGeneralizedAlphaMaintenancePlanStatus::
              EndpointPolicyMismatch;
      plan.diagnostic =
          "An alpha_f=1 endpoint stage requires matching stage and endpoint "
          "repair deltas for the supported common-representation policy.";
      return plan;
    }
  }

  if (closure == FirstOrderGeneralizedAlphaMaintenanceClosure::
                     PreservePriorStateAndRate &&
      !has_complete_rate_parameters) {
    plan.status =
        FirstOrderGeneralizedAlphaMaintenancePlanStatus::
            MissingRateParameters;
    plan.diagnostic =
        "The fixed-prior-state/rate closure requires authentic alpha_m, "
        "gamma, and dt metadata; alpha_f alone is insufficient.";
    return plan;
  }

  if (!closure_satisfied) {
    const auto stage_complement =
        svmp::FE::Real{1.0} - scheme.alpha_f;
    const auto minimum_bounded_complement =
        std::sqrt(std::numeric_limits<svmp::FE::Real>::epsilon());
    // Cap state-inversion coefficient amplification at 1/sqrt(epsilon).
    if (!(stage_complement > minimum_bounded_complement)) {
      plan.status =
          FirstOrderGeneralizedAlphaMaintenancePlanStatus::
              NearSingularStageInversion;
      plan.requires_separate_geometric_motion_account = true;
      plan.diagnostic =
          "The requested closure failed and inferring a prior-state delta "
          "would exceed the planner's bounded 1/(1-alpha_f) amplification.";
      return plan;
    }

    std::vector<svmp::FE::Real> implied_prior(stage_state_delta.size());
    for (std::size_t i = 0u; i < implied_prior.size(); ++i) {
      implied_prior[i] =
          (stage_state_delta[i] -
           scheme.alpha_f * endpoint_state_delta[i]) /
          stage_complement;
      if (!std::isfinite(implied_prior[i])) {
        throw std::overflow_error(
            "First-order generalized-alpha maintenance prior-state "
            "inference overflowed.");
      }
    }
    plan.implied_prior_state_delta = std::move(implied_prior);
    plan.status =
        FirstOrderGeneralizedAlphaMaintenancePlanStatus::ClosureMismatch;
    plan.requires_separate_geometric_motion_account = true;
    plan.diagnostic =
        "The repair pair does not satisfy the requested publication "
        "closure. Its implied prior-state rewrite has no selected prior-rate "
        "or geometric-motion accounting policy.";
    return plan;
  }

  FirstOrderGeneralizedAlphaPostAcceptPublication publication;
  publication.u_delta.assign(
      endpoint_state_delta.begin(), endpoint_state_delta.end());
  publication.u_prev_delta = publication.u_delta;
  publication.u_prev2_and_deeper_delta.assign(
      endpoint_state_delta.size(), svmp::FE::Real{0.0});
  publication.prior_rate_delta.assign(
      endpoint_state_delta.size(), svmp::FE::Real{0.0});
  publication.u_dot_delta.assign(
      endpoint_state_delta.size(), svmp::FE::Real{0.0});
  publication.accepted_stage_rate_delta.assign(
      endpoint_state_delta.size(), svmp::FE::Real{0.0});

  if (closure == FirstOrderGeneralizedAlphaMaintenanceClosure::
                     SameRepresentationDelta) {
    publication.u_prev2_and_deeper_delta = publication.u_delta;
    plan.implied_prior_state_delta = publication.u_delta;
  } else {
    plan.implied_prior_state_delta =
        publication.u_prev2_and_deeper_delta;
    plan.requires_separate_geometric_motion_account = true;
    const auto rate_scale = *scheme.gamma * *scheme.dt;
    for (std::size_t i = 0u; i < publication.u_dot_delta.size(); ++i) {
      publication.u_dot_delta[i] =
          endpoint_state_delta[i] / rate_scale;
      publication.accepted_stage_rate_delta[i] =
          *scheme.alpha_m * publication.u_dot_delta[i];
      if (!std::isfinite(publication.u_dot_delta[i]) ||
          !std::isfinite(publication.accepted_stage_rate_delta[i])) {
        throw std::overflow_error(
            "First-order generalized-alpha maintenance rate publication "
            "overflowed.");
      }
    }
  }

  for (std::size_t i = 0u; i < stage_state_delta.size(); ++i) {
    const auto reconstructed_stage =
        (svmp::FE::Real{1.0} - scheme.alpha_f) *
            publication.u_prev2_and_deeper_delta[i] +
        scheme.alpha_f * publication.u_delta[i];
    const auto reconstructed_endpoint_increment =
        has_complete_rate_parameters
            ? publication.u_prev2_and_deeper_delta[i] +
                  *scheme.dt *
                      ((svmp::FE::Real{1.0} - *scheme.gamma) *
                           publication.prior_rate_delta[i] +
                       *scheme.gamma * publication.u_dot_delta[i])
            : publication.u_prev2_and_deeper_delta[i];
    const auto reconstructed_stage_rate =
        has_complete_rate_parameters
            ? (svmp::FE::Real{1.0} - *scheme.alpha_m) *
                      publication.prior_rate_delta[i] +
                  *scheme.alpha_m * publication.u_dot_delta[i]
            : svmp::FE::Real{0.0};
    const auto stage_residual =
        std::abs(stage_state_delta[i] - reconstructed_stage);
    const auto endpoint_residual = std::abs(
        endpoint_state_delta[i] - reconstructed_endpoint_increment);
    const auto stage_rate_residual = std::abs(
        publication.accepted_stage_rate_delta[i] -
        reconstructed_stage_rate);
    const auto stage_tolerance =
        scaled_tolerance(stage_state_delta[i], reconstructed_stage);
    const auto endpoint_tolerance = scaled_tolerance(
        endpoint_state_delta[i], reconstructed_endpoint_increment);
    const auto stage_rate_tolerance = scaled_tolerance(
        publication.accepted_stage_rate_delta[i],
        reconstructed_stage_rate);
    plan.max_stage_state_identity_residual = std::max(
        plan.max_stage_state_identity_residual, stage_residual);
    plan.max_endpoint_update_identity_residual = std::max(
        plan.max_endpoint_update_identity_residual, endpoint_residual);
    plan.max_stage_rate_identity_residual = std::max(
        plan.max_stage_rate_identity_residual, stage_rate_residual);
    plan.identity_tolerance = std::max(
        {plan.identity_tolerance,
         stage_tolerance,
         endpoint_tolerance,
         stage_rate_tolerance});
    if (stage_residual > stage_tolerance ||
        endpoint_residual > endpoint_tolerance ||
        stage_rate_residual > stage_rate_tolerance) {
      throw std::logic_error(
          "First-order generalized-alpha maintenance publication failed "
          "its state/rate identity validation.");
    }
  }

  plan.status =
      FirstOrderGeneralizedAlphaMaintenancePlanStatus::
          AlgebraicallyComplete;
  plan.post_accept = std::move(publication);
  plan.diagnostic =
      closure == FirstOrderGeneralizedAlphaMaintenanceClosure::
                     SameRepresentationDelta
          ? "Common representation delta preserves every stored temporal "
            "increment and rate."
          : "Fixed prior accepted state/rate closure is algebraically "
            "complete but requires separate maintenance/geometric-motion "
            "accounting before production publication.";
  return plan;
}

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

FirstOrderGeneralizedAlphaPostAcceptPublication
combineFirstOrderGeneralizedAlphaPublicationWithCommonDelta(
    const FirstOrderGeneralizedAlphaMaintenancePublicationPlan& plan,
    std::span<const svmp::FE::Real> common_delta)
{
  if (plan.status !=
          FirstOrderGeneralizedAlphaMaintenancePlanStatus::
              AlgebraicallyComplete ||
      !plan.post_accept.has_value()) {
    throw std::invalid_argument(
        "Combining a generalized-alpha maintained-field publication "
        "requires an algebraically complete planner result.");
  }
  auto combined = *plan.post_accept;
  const auto count = combined.u_delta.size();
  const bool common_layout =
      count != 0u && common_delta.size() == count &&
      combined.u_prev_delta.size() == count &&
      combined.u_prev2_and_deeper_delta.size() == count &&
      combined.prior_rate_delta.size() == count &&
      combined.u_dot_delta.size() == count &&
      combined.accepted_stage_rate_delta.size() == count;
  if (!common_layout) {
    throw std::invalid_argument(
        "Combining a generalized-alpha maintained-field publication "
        "requires one nonempty common field layout.");
  }
  for (std::size_t i = 0u; i < count; ++i) {
    if (!std::isfinite(common_delta[i])) {
      throw std::invalid_argument(
          "A generalized-alpha maintained-field common delta must be "
          "finite.");
    }
    combined.u_delta[i] += common_delta[i];
    combined.u_prev_delta[i] += common_delta[i];
    combined.u_prev2_and_deeper_delta[i] += common_delta[i];
    if (!std::isfinite(combined.u_delta[i]) ||
        !std::isfinite(combined.u_prev_delta[i]) ||
        !std::isfinite(combined.u_prev2_and_deeper_delta[i])) {
      throw std::overflow_error(
          "A generalized-alpha maintained-field combined state "
          "publication overflowed.");
    }
  }
  return combined;
}

MaintainedFieldPostAcceptPublicationView
makeFirstOrderGeneralizedAlphaMaintainedFieldPublication(
    svmp::FE::FieldId field,
    std::size_t field_offset,
    const FirstOrderGeneralizedAlphaPostAcceptPublication& publication,
    bool publish_u_dot,
    bool maintained_field_has_time_derivative)
{
  if (publish_u_dot && !maintained_field_has_time_derivative) {
    throw std::invalid_argument(
        "First-order generalized-alpha maintained-field rate publication "
        "requires a time-derivative field target.");
  }
  return MaintainedFieldPostAcceptPublicationView{
      .field = field,
      .field_offset = field_offset,
      .u_delta = publication.u_delta,
      .u_prev_delta = publication.u_prev_delta,
      .u_prev2_and_deeper_delta =
          publication.u_prev2_and_deeper_delta,
      .u_dot_delta = publish_u_dot
          ? std::optional<std::span<const svmp::FE::Real>>{
                publication.u_dot_delta}
          : std::nullopt,
      .maintained_first_order_u_ddot_unchanged =
          publication.maintained_first_order_u_ddot_unchanged,
  };
}

MaintainedFieldPostAcceptPublicationView
makeFirstOrderGeneralizedAlphaMaintainedFieldPublication(
    svmp::FE::FieldId field,
    std::size_t field_offset,
    const FirstOrderGeneralizedAlphaMaintenancePublicationPlan& plan,
    bool publish_u_dot,
    bool maintained_field_has_time_derivative)
{
  if (plan.status !=
          FirstOrderGeneralizedAlphaMaintenancePlanStatus::
              AlgebraicallyComplete ||
      !plan.post_accept.has_value()) {
    throw std::invalid_argument(
        "First-order generalized-alpha maintained-field publication requires "
        "an algebraically complete planner result.");
  }
  return makeFirstOrderGeneralizedAlphaMaintainedFieldPublication(
      field,
      field_offset,
      *plan.post_accept,
      publish_u_dot,
      maintained_field_has_time_derivative);
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
