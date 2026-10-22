/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/StepParticipants.h"

#include "TimeStepping/NewtonSolver.h"
#include "TimeStepping/TimeHistory.h"

#include <stdexcept>
#include <utility>

namespace svmp {
namespace FE {
namespace timestepping {

void IStepParticipant::onStepStart(const TimeHistory& /*history*/) {}

bool IStepParticipant::onBeforePhysicsSolve(TimeHistory& /*history*/,
                                            double /*solve_time*/,
                                            double /*dt*/)
{
    return true;
}

void IStepParticipant::onNonlinearDone(const TimeHistory& /*history*/,
                                       const NewtonReport& /*report*/)
{
}

bool IStepParticipant::onBeforeStepAccept(TimeHistory& /*history*/,
                                          const NewtonReport& /*report*/)
{
    return true;
}

void IStepParticipant::onCandidateStage(const CandidateStageObservation& /*observation*/) {}

std::optional<StepRejectReason> IStepParticipant::onStepCandidateReady(
    TimeHistory& /*history*/, const NewtonReport& /*report*/)
{
    return std::nullopt;
}

void IStepParticipant::onStepCandidateDiscarded(TimeHistory& /*history*/) {}

void IStepParticipant::onStepCommitReady(TimeHistory& /*history*/) {}

void IStepParticipant::onStepAccepted(TimeHistory& /*history*/) {}

void IStepParticipant::onStepRejected(const TimeHistory& /*history*/,
                                      StepRejectReason /*reason*/,
                                      const NewtonReport& /*report*/)
{
}

void IStepParticipant::onDtUpdated(double /*old_dt*/,
                                   double /*new_dt*/,
                                   int /*step_index*/,
                                   int /*attempt_index*/)
{
}

void StepParticipantList::add(std::shared_ptr<IStepParticipant> participant,
                              StepParticipantOrder order)
{
    if (!participant) {
        throw std::invalid_argument("StepParticipantList::add: null participant");
    }
    (order == StepParticipantOrder::BeforeBase ? before_ : after_)
        .push_back(std::move(participant));
}

namespace {

using Participants = std::vector<std::shared_ptr<IStepParticipant>>;

/// The participants of @p list that declare @p hook, in order.
Participants declaring(const Participants& list, StepHook hook)
{
    Participants out;
    for (const auto& participant : list) {
        if (hasStepHook(participant->hooks(), hook)) {
            out.push_back(participant);
        }
    }
    return out;
}

Participants reversed(Participants list)
{
    return Participants(list.rbegin(), list.rend());
}

} // namespace

TimeLoopCallbacks StepParticipantList::callbacks(TimeLoopCallbacks base) const
{
    TimeLoopCallbacks out = base;
    const auto present = [](const auto& base_hook, const Participants& before,
                            const Participants& after) {
        return static_cast<bool>(base_hook) || !before.empty() || !after.empty();
    };

    {
        auto before = declaring(before_, StepHook::StepStart);
        auto after = declaring(after_, StepHook::StepStart);
        if (present(base.on_step_start, before, after)) {
            out.on_step_start = [before, after, hook = base.on_step_start](
                                    const TimeHistory& history) {
                for (const auto& p : before) p->onStepStart(history);
                if (hook) hook(history);
                for (const auto& p : after) p->onStepStart(history);
            };
        }
    }
    {
        auto before = declaring(before_, StepHook::BeforePhysicsSolve);
        auto after = declaring(after_, StepHook::BeforePhysicsSolve);
        if (present(base.on_before_physics_solve, before, after)) {
            out.on_before_physics_solve = [before, after, hook = base.on_before_physics_solve](
                                              TimeHistory& history, double solve_time,
                                              double dt) {
                for (const auto& p : before) {
                    if (!p->onBeforePhysicsSolve(history, solve_time, dt)) return false;
                }
                if (hook && !hook(history, solve_time, dt)) return false;
                for (const auto& p : after) {
                    if (!p->onBeforePhysicsSolve(history, solve_time, dt)) return false;
                }
                return true;
            };
        }
    }
    {
        auto before = declaring(before_, StepHook::NonlinearDone);
        auto after = declaring(after_, StepHook::NonlinearDone);
        if (present(base.on_nonlinear_done, before, after)) {
            out.on_nonlinear_done = [before, after, hook = base.on_nonlinear_done](
                                        const TimeHistory& history, const NewtonReport& report) {
                for (const auto& p : before) p->onNonlinearDone(history, report);
                if (hook) hook(history, report);
                for (const auto& p : after) p->onNonlinearDone(history, report);
            };
        }
    }
    {
        auto before = declaring(before_, StepHook::BeforeStepAccept);
        auto after = declaring(after_, StepHook::BeforeStepAccept);
        if (present(base.on_before_step_accept, before, after)) {
            out.on_before_step_accept = [before, after, hook = base.on_before_step_accept](
                                            TimeHistory& history, const NewtonReport& report) {
                for (const auto& p : before) {
                    if (!p->onBeforeStepAccept(history, report)) return false;
                }
                if (hook && !hook(history, report)) return false;
                for (const auto& p : after) {
                    if (!p->onBeforeStepAccept(history, report)) return false;
                }
                return true;
            };
        }
    }
    {
        auto before = declaring(before_, StepHook::CandidateStage);
        auto after = declaring(after_, StepHook::CandidateStage);
        if (present(base.on_candidate_stage, before, after)) {
            out.on_candidate_stage = [before, after, hook = base.on_candidate_stage](
                                         const CandidateStageObservation& observation) {
                for (const auto& p : before) p->onCandidateStage(observation);
                if (hook) hook(observation);
                for (const auto& p : after) p->onCandidateStage(observation);
            };
        }
    }
    {
        auto before = declaring(before_, StepHook::CandidateReady);
        auto after = declaring(after_, StepHook::CandidateReady);
        if (present(base.on_step_candidate_ready, before, after)) {
            out.on_step_candidate_ready = [before, after, hook = base.on_step_candidate_ready](
                                              TimeHistory& history, const NewtonReport& report)
                -> std::optional<StepRejectReason> {
                for (const auto& p : before) {
                    if (auto reason = p->onStepCandidateReady(history, report)) return reason;
                }
                if (hook) {
                    if (auto reason = hook(history, report)) return reason;
                }
                for (const auto& p : after) {
                    if (auto reason = p->onStepCandidateReady(history, report)) return reason;
                }
                return std::nullopt;
            };
        }
    }
    {
        auto before = reversed(declaring(before_, StepHook::CandidateDiscarded));
        auto after = reversed(declaring(after_, StepHook::CandidateDiscarded));
        if (present(base.on_step_candidate_discarded, before, after)) {
            out.on_step_candidate_discarded = [before, after,
                                               hook = base.on_step_candidate_discarded](
                                                  TimeHistory& history) {
                for (const auto& p : after) p->onStepCandidateDiscarded(history);
                if (hook) hook(history);
                for (const auto& p : before) p->onStepCandidateDiscarded(history);
            };
        }
    }
    {
        auto before = declaring(before_, StepHook::CommitReady);
        auto after = declaring(after_, StepHook::CommitReady);
        if (present(base.on_step_commit_ready, before, after)) {
            out.on_step_commit_ready = [before, after, hook = base.on_step_commit_ready](
                                           TimeHistory& history) {
                for (const auto& p : before) p->onStepCommitReady(history);
                if (hook) hook(history);
                for (const auto& p : after) p->onStepCommitReady(history);
            };
        }
    }
    {
        auto before = declaring(before_, StepHook::Accepted);
        auto after = declaring(after_, StepHook::Accepted);
        if (present(base.on_step_accepted, before, after)) {
            out.on_step_accepted = [before, after, hook = base.on_step_accepted](
                                       TimeHistory& history) {
                for (const auto& p : before) p->onStepAccepted(history);
                if (hook) hook(history);
                for (const auto& p : after) p->onStepAccepted(history);
            };
        }
    }
    {
        auto before = reversed(declaring(before_, StepHook::Rejected));
        auto after = reversed(declaring(after_, StepHook::Rejected));
        if (present(base.on_step_rejected, before, after)) {
            out.on_step_rejected = [before, after, hook = base.on_step_rejected](
                                       const TimeHistory& history, StepRejectReason reason,
                                       const NewtonReport& report) {
                for (const auto& p : after) p->onStepRejected(history, reason, report);
                if (hook) hook(history, reason, report);
                for (const auto& p : before) p->onStepRejected(history, reason, report);
            };
        }
    }
    {
        auto before = declaring(before_, StepHook::DtUpdated);
        auto after = declaring(after_, StepHook::DtUpdated);
        if (present(base.on_dt_updated, before, after)) {
            out.on_dt_updated = [before, after, hook = base.on_dt_updated](
                                    double old_dt, double new_dt, int step_index,
                                    int attempt_index) {
                for (const auto& p : before) p->onDtUpdated(old_dt, new_dt, step_index, attempt_index);
                if (hook) hook(old_dt, new_dt, step_index, attempt_index);
                for (const auto& p : after) p->onDtUpdated(old_dt, new_dt, step_index, attempt_index);
            };
        }
    }
    return out;
}

} // namespace timestepping
} // namespace FE
} // namespace svmp
