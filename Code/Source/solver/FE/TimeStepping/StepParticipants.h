/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_TIMESTEPPING_STEP_PARTICIPANTS_H
#define SVMP_FE_TIMESTEPPING_STEP_PARTICIPANTS_H

/**
 * @file StepParticipants.h
 * @brief Components that take part in every time step through TimeLoop hooks.
 *
 * A step participant implements the TimeLoop hooks it declares in hooks().
 * StepParticipantList::callbacks(base) composes the registered participants
 * with a base set of callbacks:
 *
 * - A composed hook exists exactly when the base hook exists or a participant
 *   declares it.  TimeLoop behaves differently with some hooks present (an
 *   on_candidate_stage hook arms the candidate transaction), so a participant
 *   that declares nothing changes nothing.
 * - Participants registered BeforeBase run, in registration order, before
 *   the base hook; AfterBase participants run, in registration order, after
 *   it.  The candidate-discarded and step-rejected hooks unwind in reverse:
 *   AfterBase participants in reverse registration order, the base, then
 *   BeforeBase participants in reverse registration order.
 * - A hook that can refuse (on_before_physics_solve, on_before_step_accept,
 *   on_step_candidate_ready) stops at the first refusal and returns it.
 * - Exceptions propagate unchanged; the hooks after the throwing one do not
 *   run.
 *
 * The composed callbacks share ownership of the participants, so they stay
 * valid after the list is destroyed.
 */

#include "TimeStepping/StepController.h"
#include "TimeStepping/TimeLoop.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace svmp {
namespace FE {
namespace timestepping {

class TimeHistory;
struct NewtonReport;

/// TimeLoop hooks, as a bit set; each bit names one TimeLoopCallbacks member.
enum class StepHook : std::uint32_t {
    None = 0u,
    StepStart = 1u << 0,           ///< on_step_start
    BeforePhysicsSolve = 1u << 1,  ///< on_before_physics_solve
    NonlinearDone = 1u << 2,       ///< on_nonlinear_done
    BeforeStepAccept = 1u << 3,    ///< on_before_step_accept
    CandidateStage = 1u << 4,      ///< on_candidate_stage
    CandidateReady = 1u << 5,      ///< on_step_candidate_ready
    CandidateDiscarded = 1u << 6,  ///< on_step_candidate_discarded
    CommitReady = 1u << 7,         ///< on_step_commit_ready
    Accepted = 1u << 8,            ///< on_step_accepted
    Rejected = 1u << 9,            ///< on_step_rejected
    DtUpdated = 1u << 10,          ///< on_dt_updated
};

[[nodiscard]] constexpr StepHook operator|(StepHook left, StepHook right) noexcept
{
    return static_cast<StepHook>(static_cast<std::uint32_t>(left) |
                                 static_cast<std::uint32_t>(right));
}

/// True when @p set contains every hook of @p hook.
[[nodiscard]] constexpr bool hasStepHook(StepHook set, StepHook hook) noexcept
{
    return (static_cast<std::uint32_t>(set) & static_cast<std::uint32_t>(hook)) ==
           static_cast<std::uint32_t>(hook);
}

/// Where a participant's hooks run relative to the base callbacks.
enum class StepParticipantOrder : std::uint8_t {
    BeforeBase,
    AfterBase
};

/**
 * @brief A component that takes part in each time step.
 *
 * Only the hooks named by hooks() are called; the others are never installed
 * on its behalf.  The default implementations do nothing and accept.
 */
class IStepParticipant {
public:
    virtual ~IStepParticipant() = default;

    /// The hooks this participant takes part in; fixed for its lifetime.
    [[nodiscard]] virtual StepHook hooks() const noexcept = 0;

    virtual void onStepStart(const TimeHistory& history);
    /// Return false to reject the step before the physics solve.
    virtual bool onBeforePhysicsSolve(TimeHistory& history, double solve_time, double dt);
    virtual void onNonlinearDone(const TimeHistory& history, const NewtonReport& report);
    /// Return false to reject the converged candidate.
    virtual bool onBeforeStepAccept(TimeHistory& history, const NewtonReport& report);
    virtual void onCandidateStage(const CandidateStageObservation& observation);
    /// Return a reason to reject the projected endpoint.
    virtual std::optional<StepRejectReason> onStepCandidateReady(TimeHistory& history,
                                                                 const NewtonReport& report);
    virtual void onStepCandidateDiscarded(TimeHistory& history);
    virtual void onStepCommitReady(TimeHistory& history);
    virtual void onStepAccepted(TimeHistory& history);
    virtual void onStepRejected(const TimeHistory& history,
                                StepRejectReason reason,
                                const NewtonReport& report);
    virtual void onDtUpdated(double old_dt, double new_dt, int step_index, int attempt_index);
};

/// Ordered step participants and their composition with base callbacks.
class StepParticipantList {
public:
    /// Registers @p participant (non-null) at @p order.
    void add(std::shared_ptr<IStepParticipant> participant,
             StepParticipantOrder order = StepParticipantOrder::BeforeBase);

    [[nodiscard]] bool empty() const noexcept { return before_.empty() && after_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return before_.size() + after_.size(); }

    /// @p base with every registered participant composed into it (see the file comment).
    [[nodiscard]] TimeLoopCallbacks callbacks(TimeLoopCallbacks base = {}) const;

private:
    std::vector<std::shared_ptr<IStepParticipant>> before_{};
    std::vector<std::shared_ptr<IStepParticipant>> after_{};
};

} // namespace timestepping
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_TIMESTEPPING_STEP_PARTICIPANTS_H
