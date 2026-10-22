/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file GeneratedState.h
 * @brief Generated state of a nonlinear solve: the synchronization points
 *        at which NewtonSolver asks its owner to regenerate state that the
 *        residual depends on (cut geometry, projected fields, extensions,
 *        lagged references), and the policy predicates of those points.
 *
 * The predicates were moved from Application/Core/ApplicationDriver.cpp with
 * bodies and environment names unchanged.
 */

#include "TimeStepping/NewtonSolver.h"

namespace svmp {
namespace FE {
namespace timestepping {

/**
 * @brief Provenance token of a synchronization point, as the generated-state
 *        owner logs it and passes it to its refreshes (for example
 *        "outer_fixed_point", "endpoint_candidate", "restored_time_step").
 *
 * The tokens are part of log formats and of the cut-topology attempt
 * protocol, which matches on them; they differ from NewtonSolver's own
 * trace tokens and must not be unified with them.
 */
[[nodiscard]] const char* generatedStateSyncPointName(
    NewtonOptions::StateSynchronizationPoint point) noexcept;

/// True for the points that restore the accepted time-step state
/// (RestoredTimeStepState, RestoredProjectedTimeStepState).
[[nodiscard]] bool restoresAcceptedTimeStepGeneratedState(
    NewtonOptions::StateSynchronizationPoint point) noexcept;

/// True when a frozen algebraic level-set extension map is refreshed at
/// @p point: with the outer fixed point, and at the projected restored
/// time-step state.
[[nodiscard]] bool refreshesFrozenLevelSetExtensionAtStateSync(
    NewtonOptions::StateSynchronizationPoint point,
    bool use_external_state_fixed_point) noexcept;

/// True when a line-search trial may reuse the cached curvature after a
/// failed projection (opt-in: SVMP_CURVATURE_REUSE_CACHE_ON_TRIAL_FAILURE).
[[nodiscard]] bool allowCachedCurvatureAfterProjectionFailure(
    NewtonOptions::StateSynchronizationPoint point);

/// Whether line-search trials synchronize generated state; defaults to
/// @p residual_defining_state_changes (override: SVMP_SYNC_LINE_SEARCH_TRIALS).
[[nodiscard]] bool synchronizeTransientLineSearchTrials(
    bool residual_defining_state_changes);

} // namespace timestepping
} // namespace FE
} // namespace svmp
