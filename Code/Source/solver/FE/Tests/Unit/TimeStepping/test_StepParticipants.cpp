/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Systems/FESystemHandle.h"
#include "TimeStepping/NewtonSolver.h"
#include "TimeStepping/StepContext.h"
#include "TimeStepping/StepController.h"
#include "TimeStepping/StepParticipants.h"
#include "TimeStepping/TimeHistory.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace ts = svmp::FE::timestepping;

/// Records every hook call as "<name>.<hook>" and refuses on request.
class RecordingParticipant final : public ts::IStepParticipant {
public:
    RecordingParticipant(std::string name, ts::StepHook hooks, std::vector<std::string>& calls)
        : name_(std::move(name)), hooks_(hooks), calls_(calls)
    {
    }

    [[nodiscard]] ts::StepHook hooks() const noexcept override { return hooks_; }

    void onStepStart(const ts::TimeHistory&) override { record("start"); }
    bool onBeforePhysicsSolve(ts::TimeHistory&, double, double) override
    {
        record("before_solve");
        return !refuse;
    }
    bool onBeforeStepAccept(ts::TimeHistory&, const ts::NewtonReport&) override
    {
        record("before_accept");
        return !refuse;
    }
    void onCandidateStage(const ts::CandidateStageObservation&) override { record("stage"); }
    std::optional<ts::StepRejectReason> onStepCandidateReady(ts::TimeHistory&,
                                                             const ts::NewtonReport&) override
    {
        record("ready");
        return reason;
    }
    void onStepCandidateDiscarded(ts::TimeHistory&) override { record("discarded"); }
    void onStepAccepted(ts::TimeHistory&) override { record("accepted"); }
    void onStepRejected(const ts::TimeHistory&, ts::StepRejectReason,
                        const ts::NewtonReport&) override
    {
        record("rejected");
    }
    void onDtUpdated(double, double, int, int) override { record("dt"); }

    bool refuse{false};
    std::optional<ts::StepRejectReason> reason{};

private:
    void record(const char* hook) { calls_.push_back(name_ + "." + hook); }

    std::string name_;
    ts::StepHook hooks_;
    std::vector<std::string>& calls_;
};

constexpr ts::StepHook kAllHooks =
    ts::StepHook::StepStart | ts::StepHook::BeforePhysicsSolve | ts::StepHook::NonlinearDone |
    ts::StepHook::BeforeStepAccept | ts::StepHook::CandidateStage | ts::StepHook::CandidateReady |
    ts::StepHook::CandidateDiscarded | ts::StepHook::CommitReady | ts::StepHook::Accepted |
    ts::StepHook::Rejected | ts::StepHook::DtUpdated;

using Calls = std::vector<std::string>;

} // namespace

TEST(StepParticipants, HooksRunBeforeAndAfterTheBaseInRegistrationOrder)
{
    Calls calls;
    ts::StepParticipantList list;
    list.add(std::make_shared<RecordingParticipant>("a", kAllHooks, calls));
    list.add(std::make_shared<RecordingParticipant>("c", kAllHooks, calls),
             ts::StepParticipantOrder::AfterBase);
    list.add(std::make_shared<RecordingParticipant>("b", kAllHooks, calls));
    ASSERT_EQ(list.size(), 3u);

    ts::TimeLoopCallbacks base;
    base.on_step_start = [&](const ts::TimeHistory&) { calls.push_back("base.start"); };
    base.on_step_accepted = [&](ts::TimeHistory&) { calls.push_back("base.accepted"); };
    base.on_candidate_stage = [&](const ts::CandidateStageObservation&) {
        calls.push_back("base.stage");
    };
    const auto composed = list.callbacks(base);

    ts::TimeHistory history;
    composed.on_step_start(history);
    EXPECT_EQ(calls, (Calls{"a.start", "b.start", "base.start", "c.start"}));

    calls.clear();
    composed.on_candidate_stage(ts::CandidateStageObservation{});
    EXPECT_EQ(calls, (Calls{"a.stage", "b.stage", "base.stage", "c.stage"}));

    calls.clear();
    composed.on_step_accepted(history);
    EXPECT_EQ(calls, (Calls{"a.accepted", "b.accepted", "base.accepted", "c.accepted"}));

    calls.clear();
    composed.on_dt_updated(1.0, 0.5, 3, 1);
    EXPECT_EQ(calls, (Calls{"a.dt", "b.dt", "c.dt"}));
}

TEST(StepParticipants, DiscardAndRejectionUnwindInReverseOrder)
{
    Calls calls;
    ts::StepParticipantList list;
    list.add(std::make_shared<RecordingParticipant>("a", kAllHooks, calls));
    list.add(std::make_shared<RecordingParticipant>("b", kAllHooks, calls));
    list.add(std::make_shared<RecordingParticipant>("c", kAllHooks, calls),
             ts::StepParticipantOrder::AfterBase);
    list.add(std::make_shared<RecordingParticipant>("d", kAllHooks, calls),
             ts::StepParticipantOrder::AfterBase);

    ts::TimeLoopCallbacks base;
    base.on_step_candidate_discarded = [&](ts::TimeHistory&) { calls.push_back("base.discarded"); };
    base.on_step_rejected = [&](const ts::TimeHistory&, ts::StepRejectReason,
                                const ts::NewtonReport&) { calls.push_back("base.rejected"); };
    const auto composed = list.callbacks(base);

    ts::TimeHistory history;
    composed.on_step_candidate_discarded(history);
    EXPECT_EQ(calls, (Calls{"d.discarded", "c.discarded", "base.discarded", "b.discarded",
                            "a.discarded"}));

    calls.clear();
    composed.on_step_rejected(history, ts::StepRejectReason::ErrorTooLarge, ts::NewtonReport{});
    EXPECT_EQ(calls, (Calls{"d.rejected", "c.rejected", "base.rejected", "b.rejected",
                            "a.rejected"}));
}

TEST(StepParticipants, ComposedHookExistsOnlyWhenTheBaseOrAParticipantHasIt)
{
    Calls calls;
    ts::TimeLoopCallbacks base;
    base.on_step_start = [&](const ts::TimeHistory&) { calls.push_back("base.start"); };

    // No participants: the callbacks are the base callbacks.
    {
        const auto composed = ts::StepParticipantList{}.callbacks(base);
        EXPECT_TRUE(static_cast<bool>(composed.on_step_start));
        EXPECT_FALSE(static_cast<bool>(composed.on_candidate_stage));
        EXPECT_FALSE(static_cast<bool>(composed.on_before_step_accept));
        EXPECT_FALSE(static_cast<bool>(composed.on_step_candidate_ready));
        EXPECT_FALSE(static_cast<bool>(composed.on_step_rejected));
    }

    // A participant installs only the hooks it declares.
    ts::StepParticipantList list;
    list.add(std::make_shared<RecordingParticipant>("a", ts::StepHook::CandidateStage, calls));
    const auto composed = list.callbacks(base);
    EXPECT_TRUE(static_cast<bool>(composed.on_candidate_stage));
    EXPECT_TRUE(static_cast<bool>(composed.on_step_start));
    EXPECT_FALSE(static_cast<bool>(composed.on_step_accepted));
    EXPECT_FALSE(static_cast<bool>(composed.on_step_candidate_discarded));
    EXPECT_FALSE(static_cast<bool>(composed.on_dt_updated));

    // Undeclared hooks of a participant are never called.
    ts::TimeHistory history;
    composed.on_step_start(history);
    EXPECT_EQ(calls, (Calls{"base.start"}));
    calls.clear();
    composed.on_candidate_stage(ts::CandidateStageObservation{});
    EXPECT_EQ(calls, (Calls{"a.stage"}));

    EXPECT_FALSE(static_cast<bool>(ts::StepParticipantList{}.callbacks().on_step_start));
}

TEST(StepParticipants, RefusingHooksStopAtTheFirstRefusal)
{
    Calls calls;
    auto a = std::make_shared<RecordingParticipant>("a", kAllHooks, calls);
    auto b = std::make_shared<RecordingParticipant>("b", kAllHooks, calls);
    ts::StepParticipantList list;
    list.add(a);
    list.add(b, ts::StepParticipantOrder::AfterBase);

    ts::TimeLoopCallbacks base;
    bool base_accepts = true;
    base.on_before_physics_solve = [&](ts::TimeHistory&, double, double) {
        calls.push_back("base.before_solve");
        return base_accepts;
    };
    base.on_step_candidate_ready = [&](ts::TimeHistory&, const ts::NewtonReport&)
        -> std::optional<ts::StepRejectReason> {
        calls.push_back("base.ready");
        return std::nullopt;
    };
    const auto composed = list.callbacks(base);
    ts::TimeHistory history;
    const ts::NewtonReport report{};

    EXPECT_TRUE(composed.on_before_physics_solve(history, 1.0, 0.1));
    EXPECT_EQ(calls, (Calls{"a.before_solve", "base.before_solve", "b.before_solve"}));

    calls.clear();
    a->refuse = true;
    EXPECT_FALSE(composed.on_before_physics_solve(history, 1.0, 0.1));
    EXPECT_EQ(calls, (Calls{"a.before_solve"}));

    calls.clear();
    a->refuse = false;
    base_accepts = false;
    EXPECT_FALSE(composed.on_before_physics_solve(history, 1.0, 0.1));
    EXPECT_EQ(calls, (Calls{"a.before_solve", "base.before_solve"}));

    // on_before_step_accept has no base hook here: the participants alone decide.
    calls.clear();
    b->refuse = true;
    EXPECT_FALSE(composed.on_before_step_accept(history, report));
    EXPECT_EQ(calls, (Calls{"a.before_accept", "b.before_accept"}));

    calls.clear();
    b->reason = ts::StepRejectReason::CutTopologyChanged;
    const auto reason = composed.on_step_candidate_ready(history, report);
    ASSERT_TRUE(reason.has_value());
    EXPECT_EQ(*reason, ts::StepRejectReason::CutTopologyChanged);
    EXPECT_EQ(calls, (Calls{"a.ready", "base.ready", "b.ready"}));

    calls.clear();
    a->reason = ts::StepRejectReason::ErrorTooLarge;
    const auto first = composed.on_step_candidate_ready(history, report);
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(*first, ts::StepRejectReason::ErrorTooLarge);
    EXPECT_EQ(calls, (Calls{"a.ready"}));
}

TEST(StepParticipants, ComposedCallbacksKeepTheirParticipantsAlive)
{
    Calls calls;
    std::weak_ptr<RecordingParticipant> observer;
    ts::TimeLoopCallbacks composed;
    {
        auto participant =
            std::make_shared<RecordingParticipant>("a", ts::StepHook::Accepted, calls);
        observer = participant;
        ts::StepParticipantList list;
        list.add(std::move(participant));
        composed = list.callbacks();
    }
    ASSERT_FALSE(observer.expired());
    ts::TimeHistory history;
    composed.on_step_accepted(history);
    EXPECT_EQ(calls, (Calls{"a.accepted"}));
    composed = {};
    EXPECT_TRUE(observer.expired());
}

TEST(StepParticipants, ExceptionsPropagateAndStopLaterHooks)
{
    Calls calls;
    class Throwing final : public ts::IStepParticipant {
    public:
        [[nodiscard]] ts::StepHook hooks() const noexcept override
        {
            return ts::StepHook::StepStart;
        }
        void onStepStart(const ts::TimeHistory&) override
        {
            throw std::runtime_error("participant failure");
        }
    };
    ts::StepParticipantList list;
    list.add(std::make_shared<Throwing>());
    list.add(std::make_shared<RecordingParticipant>("b", ts::StepHook::StepStart, calls));
    ts::TimeLoopCallbacks base;
    base.on_step_start = [&](const ts::TimeHistory&) { calls.push_back("base.start"); };
    const auto composed = list.callbacks(base);
    ts::TimeHistory history;
    EXPECT_THROW(composed.on_step_start(history), std::runtime_error);
    EXPECT_TRUE(calls.empty());
}

TEST(StepParticipants, NullParticipantIsRejected)
{
    ts::StepParticipantList list;
    EXPECT_THROW(list.add(nullptr), std::invalid_argument);
    EXPECT_TRUE(list.empty());
}

namespace {

/// Holds the run pointers the way a host's simulation components do.
struct HostComponents {
    std::unique_ptr<svmp::FE::systems::FESystem> fe_system{};
    std::unique_ptr<ts::TimeHistory> time_history{};
    std::unique_ptr<svmp::FE::backends::BackendFactory> backend{};
};

const std::unique_ptr<svmp::FE::systems::FESystem>& systemOf(
    const svmp::FE::systems::FESystemHandle& handle)
{
    return handle.fe_system;
}

} // namespace

TEST(StepContext, RunComponentsAndSystemHandleReferToTheHostPointers)
{
    HostComponents host;
    const ts::RunComponents run(host);
    EXPECT_EQ(&run.fe_system, &host.fe_system);
    EXPECT_EQ(&run.time_history, &host.time_history);
    EXPECT_EQ(&run.backend, &host.backend);
    EXPECT_EQ(&systemOf(host), &host.fe_system);
    EXPECT_EQ(&systemOf(run), &host.fe_system);
}

TEST(StepController, RejectReasonNamesAreTheEnumeratorNames)
{
    EXPECT_EQ(ts::stepRejectReasonName(ts::StepRejectReason::NonlinearSolveFailed),
              "NonlinearSolveFailed");
    EXPECT_EQ(ts::stepRejectReasonName(ts::StepRejectReason::ErrorTooLarge), "ErrorTooLarge");
    EXPECT_EQ(ts::stepRejectReasonName(ts::StepRejectReason::CutTopologyChanged),
              "CutTopologyChanged");
    EXPECT_EQ(ts::stepRejectReasonName(static_cast<ts::StepRejectReason>(7)), "7");
}
