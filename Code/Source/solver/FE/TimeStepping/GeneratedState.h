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
 *
 * GeneratedStateRegistry composes the owners of generated state
 * (participants) into NewtonSolver's synchronization hook.  At a
 * synchronization point it runs, in this order:
 *   1. the point filter of the policy (with the outer fixed point only the
 *      outer, endpoint and restore points synchronize);
 *   2. prepare() of every participant (rank-local resets, for example of a
 *      cache at a restore point);
 *   3. the solve gates of a transient policy (line-search trials, refresh
 *      within the solve, per-step-only state);
 *   4. synchronize() of the Geometry participants, which publish the source
 *      value revisions of the geometry they generated;
 *   5. nothing more on a constraint-construction pass (the outer fixed
 *      point's constraint pass, the endpoint candidate, the restored
 *      time-step state);
 *   6. synchronize() of the DerivedField, Extension and Reference
 *      participants.
 * Participants run by role and, within a role, in registration order.  This
 * is the order of the generated-state synchronization of the Application's
 * steady and transient solves.
 */

#include "TimeStepping/NewtonSolver.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

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

/// Whether line-search trials synchronize generated state; defaults to
/// @p residual_defining_state_changes (override: SVMP_SYNC_LINE_SEARCH_TRIALS).
[[nodiscard]] bool synchronizeTransientLineSearchTrials(
    bool residual_defining_state_changes);

// ---------------------------------------------------------------------------
// Generated-state registry
// ---------------------------------------------------------------------------

/// What a participant generates; participants run in this order.
enum class GeneratedStateRole : std::uint8_t {
    Geometry,      ///< generated geometry (cut integration context)
    DerivedField,  ///< fields projected from it (for example curvature)
    Extension,     ///< extended fields (for example an advection velocity)
    Reference,     ///< lagged copies of the state (semi-implicit terms)
};

[[nodiscard]] const char* generatedStateRoleName(GeneratedStateRole role) noexcept;

enum class GeneratedStateSolve : std::uint8_t { Steady, Transient };

/// When the generated state is synchronized.
struct GeneratedStatePolicy {
    GeneratedStateSolve solve{GeneratedStateSolve::Transient};
    /// The outer fixed point regenerates the state between frozen inner
    /// Newton solves.
    bool outer_fixed_point{false};
    /// Transient without the outer fixed point: refresh at the solve's
    /// synchronization points.
    bool refresh_within_solve{false};
    /// Transient without the outer fixed point: synchronize line-search
    /// trials (synchronizeTransientLineSearchTrials).
    bool synchronize_line_search_trials{true};
    /// Transient without the outer fixed point: refresh once per step only.
    bool per_step_only{false};
};

/// One synchronization: the point and what the Geometry role produced.
struct GeneratedStateSync {
    NewtonOptions::StateSynchronizationPoint point{};
    /// RestoredOuterFixedPointState or RestoredTimeStepState.
    bool restoring_generated_state{false};
    /// restoresAcceptedTimeStepGeneratedState(point).
    bool restoring_time_step_state{false};
    /// Source value revisions of the generated geometry, by interface marker
    /// (written by the Geometry participants).
    std::map<int, std::uint64_t> evaluated_state_source_revisions{};
};

/**
 * @brief An owner of generated state.
 *
 * synchronize() regenerates the state from a solution state; it is called on
 * every rank at the same points and may be collective.  prepare() runs at
 * every synchronized point before the solve gates and must be rank-local.
 * invalidate() drops cached generated state (for example after a rejected
 * topology change).  One participant may provide the cut-topology epochs of
 * the outer fixed point (NewtonOptions::external_state_* hooks).
 */
class IGeneratedStateParticipant {
public:
    virtual ~IGeneratedStateParticipant() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual GeneratedStateRole role() const noexcept = 0;

    virtual void prepare(const GeneratedStateSync& /*sync*/) {}
    virtual void synchronize(const systems::SystemStateView& state,
                             GeneratedStateSync& sync) = 0;
    virtual void invalidate() {}

    /// True when the participant provides the epoch hooks below.
    [[nodiscard]] virtual bool providesEpochs() const noexcept { return false; }
    [[nodiscard]] virtual bool epochDiscontinuity(
        NewtonOptions::StateSynchronizationPoint /*point*/) const
    {
        return false;
    }
    virtual void acknowledgeEpochDiscontinuity(
        NewtonOptions::StateSynchronizationPoint /*point*/)
    {
    }
    [[nodiscard]] virtual bool epochRevisited(
        NewtonOptions::StateSynchronizationPoint /*point*/) const
    {
        return false;
    }
    virtual void acknowledgeFrozenEpoch() {}
};

/// A participant made of callbacks, for owners that are not FE services yet.
class FunctionGeneratedStateParticipant final : public IGeneratedStateParticipant {
public:
    struct Callbacks {
        std::function<void(const GeneratedStateSync&)> prepare{};
        std::function<void(const systems::SystemStateView&, GeneratedStateSync&)>
            synchronize{};
        std::function<void()> invalidate{};
    };

    FunctionGeneratedStateParticipant(std::string name,
                                      GeneratedStateRole role,
                                      Callbacks callbacks);

    [[nodiscard]] std::string_view name() const noexcept override { return name_; }
    [[nodiscard]] GeneratedStateRole role() const noexcept override { return role_; }
    void prepare(const GeneratedStateSync& sync) override;
    void synchronize(const systems::SystemStateView& state,
                     GeneratedStateSync& sync) override;
    void invalidate() override;

private:
    std::string name_;
    GeneratedStateRole role_;
    Callbacks callbacks_;
};

/**
 * @brief The generated state of a solve, synchronized in a fixed order.
 *
 * Create with std::make_shared: bind() keeps the registry alive in the
 * Newton options it fills.
 */
class GeneratedStateRegistry
    : public std::enable_shared_from_this<GeneratedStateRegistry> {
public:
    explicit GeneratedStateRegistry(GeneratedStatePolicy policy);

    /// Adds a participant; it runs after the earlier ones of its role.
    void add(std::shared_ptr<IGeneratedStateParticipant> participant);

    [[nodiscard]] const GeneratedStatePolicy& policy() const noexcept { return policy_; }

    /// The participants in run order.
    [[nodiscard]] const std::vector<std::shared_ptr<IGeneratedStateParticipant>>&
    participants() const noexcept
    {
        return participants_;
    }

    /// Whether @p point is synchronized at all (step 1).
    [[nodiscard]] bool synchronizes(NewtonOptions::StateSynchronizationPoint point) const noexcept;

    /// Whether only the Geometry role runs at @p point (step 5).
    [[nodiscard]] bool constraintConstructionPass(
        NewtonOptions::StateSynchronizationPoint point) const noexcept;

    /// Steps 1-6 at @p point.
    void synchronize(const systems::SystemStateView& state,
                     NewtonOptions::StateSynchronizationPoint point) const;

    /// invalidate() of every participant, in run order.
    void invalidate() const;

    /// Installs synchronize() as NewtonOptions::synchronize_state and, for a
    /// transient outer fixed point with an epoch-providing participant, the
    /// four external-state epoch hooks.  At most one participant may provide
    /// epochs.
    void bind(NewtonOptions& options) const;

private:
    GeneratedStatePolicy policy_;
    std::vector<std::shared_ptr<IGeneratedStateParticipant>> participants_{};
};

} // namespace timestepping
} // namespace FE
} // namespace svmp
