/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Core/FEException.h"

#include "TimeStepping/StepController.h"
#include "TimeStepping/VSVO_BDF_Controller.h"

#include <cmath>

TEST(SimpleStepController, ValidatesOptions)
{
    using svmp::FE::timestepping::SimpleStepController;
    using svmp::FE::timestepping::SimpleStepControllerOptions;

    {
        SimpleStepControllerOptions o;
        o.max_retries = -1;
        EXPECT_THROW((void)SimpleStepController{o}, svmp::FE::InvalidArgumentException);
    }
    {
        SimpleStepControllerOptions o;
        o.decrease_factor = 1.0;
        EXPECT_THROW((void)SimpleStepController{o}, svmp::FE::InvalidArgumentException);
    }
    {
        SimpleStepControllerOptions o;
        o.increase_factor = 0.9;
        EXPECT_THROW((void)SimpleStepController{o}, svmp::FE::InvalidArgumentException);
    }
    {
        SimpleStepControllerOptions o;
        o.target_newton_iterations = 0;
        EXPECT_THROW((void)SimpleStepController{o}, svmp::FE::InvalidArgumentException);
    }
    {
        SimpleStepControllerOptions o;
        o.min_dt = 2.0;
        o.max_dt = 1.0;
        EXPECT_THROW((void)SimpleStepController{o}, svmp::FE::InvalidArgumentException);
    }
}

TEST(SimpleStepController, AdaptsDtAndClamps)
{
    using svmp::FE::timestepping::SimpleStepController;
    using svmp::FE::timestepping::SimpleStepControllerOptions;
    using svmp::FE::timestepping::StepAttemptInfo;

    SimpleStepControllerOptions o;
    o.min_dt = 0.05;
    o.max_dt = 0.2;
    o.max_retries = 3;
    o.decrease_factor = 0.5;
    o.increase_factor = 2.0;
    o.target_newton_iterations = 6;
    SimpleStepController ctrl(o);

    StepAttemptInfo info;
    info.dt = 0.1;

    info.newton.iterations = 20;
    {
        const auto d = ctrl.onAccepted(info);
        EXPECT_TRUE(d.accept);
        EXPECT_FALSE(d.retry);
        EXPECT_NEAR(d.next_dt, 0.05, 1e-15); // 0.1*0.5 clamped to min_dt
    }

    info.newton.iterations = 1;
    {
        const auto d = ctrl.onAccepted(info);
        EXPECT_TRUE(d.accept);
        EXPECT_FALSE(d.retry);
        EXPECT_NEAR(d.next_dt, 0.2, 1e-15); // 0.1*2 clamped to max_dt
    }
}

TEST(SimpleStepController, RejectedStopsWhenAtMinDtAndRetriesExceeded)
{
    using svmp::FE::timestepping::SimpleStepController;
    using svmp::FE::timestepping::SimpleStepControllerOptions;
    using svmp::FE::timestepping::StepAttemptInfo;
    using svmp::FE::timestepping::StepRejectReason;

    SimpleStepControllerOptions o;
    o.min_dt = 0.1;
    o.max_retries = 2;
    o.decrease_factor = 0.5;
    SimpleStepController ctrl(o);

    StepAttemptInfo info;
    info.dt = 0.1;
    info.attempt_index = 2;

    const auto d = ctrl.onRejected(info, StepRejectReason::NonlinearSolveFailed);
    EXPECT_FALSE(d.accept);
    EXPECT_FALSE(d.retry);
    EXPECT_NEAR(d.next_dt, 0.1, 1e-15);
}

TEST(VSVO_BDF_Controller, ValidatesOptions)
{
    using svmp::FE::timestepping::VSVO_BDF_Controller;
    using svmp::FE::timestepping::VSVO_BDF_ControllerOptions;

    {
        VSVO_BDF_ControllerOptions o;
        o.abs_tol = 0.0;
        EXPECT_THROW((void)VSVO_BDF_Controller{o}, svmp::FE::InvalidArgumentException);
    }
    {
        VSVO_BDF_ControllerOptions o;
        o.min_order = 3;
        o.max_order = 2;
        EXPECT_THROW((void)VSVO_BDF_Controller{o}, svmp::FE::InvalidArgumentException);
    }
    {
        VSVO_BDF_ControllerOptions o;
        o.initial_order = 6;
        EXPECT_THROW((void)VSVO_BDF_Controller{o}, svmp::FE::InvalidArgumentException);
    }
}

TEST(VSVO_BDF_Controller, ValidatesAdditionalOptions)
{
    using svmp::FE::timestepping::VSVO_BDF_Controller;
    using svmp::FE::timestepping::VSVO_BDF_ControllerOptions;

    {
        VSVO_BDF_ControllerOptions o;
        o.rel_tol = -0.1;
        EXPECT_THROW((void)VSVO_BDF_Controller{o}, svmp::FE::InvalidArgumentException);
    }
    {
        VSVO_BDF_ControllerOptions o;
        o.pi_alpha = -0.1;
        EXPECT_THROW((void)VSVO_BDF_Controller{o}, svmp::FE::InvalidArgumentException);
    }
    {
        VSVO_BDF_ControllerOptions o;
        o.pi_beta = -0.1;
        EXPECT_THROW((void)VSVO_BDF_Controller{o}, svmp::FE::InvalidArgumentException);
    }
    {
        VSVO_BDF_ControllerOptions o;
        o.increase_order_threshold = -0.1;
        EXPECT_THROW((void)VSVO_BDF_Controller{o}, svmp::FE::InvalidArgumentException);
    }
}

TEST(VSVO_BDF_Controller, AcceptedWithoutErrorEstimateKeepsDtAndOrder)
{
    using svmp::FE::timestepping::StepAttemptInfo;
    using svmp::FE::timestepping::VSVO_BDF_Controller;
    using svmp::FE::timestepping::VSVO_BDF_ControllerOptions;

    VSVO_BDF_ControllerOptions o;
    o.initial_order = 2;
    VSVO_BDF_Controller ctrl(o);

    StepAttemptInfo info;
    info.dt = 0.1;
    info.scheme_order = 2;
    info.error_norm = -1.0;

    const auto d = ctrl.onAccepted(info);
    EXPECT_TRUE(d.accept);
    EXPECT_FALSE(d.retry);
    EXPECT_NEAR(d.next_dt, 0.1, 1e-15);
    EXPECT_EQ(d.next_order, 2);
}

TEST(VSVO_BDF_Controller, SelectsMostEfficientOrderAmongCandidatesAndRespectsIncreaseGate)
{
    using svmp::FE::timestepping::StepAttemptInfo;
    using svmp::FE::timestepping::VSVO_BDF_Controller;
    using svmp::FE::timestepping::VSVO_BDF_ControllerOptions;

    VSVO_BDF_ControllerOptions o;
    o.initial_order = 2;
    o.min_order = 1;
    o.max_order = 3;
    o.max_retries = 2;
    o.safety = 0.9;
    o.min_factor = 1e-6;
    o.max_factor = 100.0;
    o.pi_alpha = 0.7;
    o.pi_beta = 0.4;
    o.increase_order_threshold = 0.05;
    VSVO_BDF_Controller ctrl(o);

    auto expectedFactor = [&o](int q, double err_q) {
        const double inv_q1 = 1.0 / static_cast<double>(q + 1);
        const double k_i = o.pi_alpha * inv_q1;
        const double k_p = o.pi_beta * inv_q1;
        const double fac = o.safety *
            std::pow(1.0 / std::max(err_q, 1e-16), k_i) *
            std::pow(1.0 / std::max(err_q, 1e-16), k_p);
        return std::min(o.max_factor, std::max(o.min_factor, fac));
    };

    StepAttemptInfo info;
    info.dt = 1.0;
    info.scheme_order = 2;
    info.step_index = 0;
    info.attempt_index = 0;
    info.error_norm_low = 0.4;
    info.error_norm = 0.5;

    // Case 1: gate blocks p+1 (error_norm_high >= increase_order_threshold).
    info.error_norm_high = 0.1;
    {
        const auto d = ctrl.onAccepted(info);
        EXPECT_TRUE(d.accept);
        EXPECT_FALSE(d.retry);
        EXPECT_EQ(d.next_order, 1);
        EXPECT_NEAR(d.next_dt, info.dt * expectedFactor(/*q=*/1, info.error_norm_low), 1e-12);
    }

    // Case 2: gate allows p+1 and it wins the efficiency comparison.
    info.error_norm_high = 1e-6;
    {
        const auto d = ctrl.onAccepted(info);
        EXPECT_TRUE(d.accept);
        EXPECT_FALSE(d.retry);
        EXPECT_EQ(d.next_order, 3);
        EXPECT_NEAR(d.next_dt, info.dt * expectedFactor(/*q=*/3, info.error_norm_high), 1e-12);
    }
}

TEST(VSVO_BDF_Controller, RejectsOnErrorGreaterThanOneAndReducesDt)
{
    using svmp::FE::timestepping::StepAttemptInfo;
    using svmp::FE::timestepping::VSVO_BDF_Controller;
    using svmp::FE::timestepping::VSVO_BDF_ControllerOptions;

    VSVO_BDF_ControllerOptions o;
    o.initial_order = 3;
    o.min_order = 1;
    o.max_order = 5;
    o.max_retries = 5;
    o.safety = 0.9;
    o.min_factor = 0.2;
    o.max_factor = 5.0;
    VSVO_BDF_Controller ctrl(o);

    StepAttemptInfo info;
    info.dt = 1.0;
    info.scheme_order = 3;
    info.error_norm = 2.0;
    info.step_index = 0;
    info.attempt_index = 0;

    const auto d = ctrl.onAccepted(info);
    EXPECT_FALSE(d.accept);
    EXPECT_TRUE(d.retry);
    EXPECT_EQ(d.next_order, 2);
    EXPECT_GT(d.next_dt, 0.0);
    EXPECT_LT(d.next_dt, 1.0);

    // Expect PI-like dt update with clamping to <= 1.0*dt.
    const double inv_p1 = 1.0 / 4.0;
    const double k_i = o.pi_alpha * inv_p1;
    const double k_p = o.pi_beta * inv_p1;
    const double fac = o.safety * std::pow(1.0 / info.error_norm, k_i) * std::pow(1.0 / info.error_norm, k_p);
    const double fac_clamped = std::min(1.0, std::max(o.min_factor, fac));
    EXPECT_NEAR(d.next_dt, info.dt * fac_clamped, 1e-12);
}

TEST(VSVO_BDF_Controller, UsesPreviousAcceptedErrorForPiControlWhenAvailable)
{
    using svmp::FE::timestepping::StepAttemptInfo;
    using svmp::FE::timestepping::VSVO_BDF_Controller;
    using svmp::FE::timestepping::VSVO_BDF_ControllerOptions;

    VSVO_BDF_ControllerOptions o;
    o.initial_order = 2;
    o.min_order = 1;
    o.max_order = 3;
    o.max_retries = 2;
    o.safety = 0.9;
    o.min_factor = 0.1;
    o.max_factor = 10.0;
    o.pi_alpha = 0.7;
    o.pi_beta = 0.4;
    o.increase_order_threshold = 0.0;
    VSVO_BDF_Controller ctrl(o);

    StepAttemptInfo info;
    info.dt = 1.0;
    info.scheme_order = 2;
    info.error_norm_low = -1.0;
    info.error_norm_high = -1.0;

    // Step 0 seeds prev_error_norm_.
    info.step_index = 0;
    info.error_norm = 0.5;
    (void)ctrl.onAccepted(info);

    // Step 1 should use prev_error_norm_ in the PI controller.
    info.step_index = 1;
    info.error_norm = 0.25;
    const auto d = ctrl.onAccepted(info);
    EXPECT_TRUE(d.accept);
    EXPECT_FALSE(d.retry);
    EXPECT_EQ(d.next_order, 2);

    const double inv_p1 = 1.0 / 3.0;
    const double k_i = o.pi_alpha * inv_p1;
    const double k_p = o.pi_beta * inv_p1;
    const double expected_fac =
        o.safety *
        std::pow(1.0 / info.error_norm, k_i) *
        std::pow(1.0 / 0.5, k_p);
    const double expected_fac_clamped = std::min(o.max_factor, std::max(o.min_factor, expected_fac));
    EXPECT_NEAR(d.next_dt, info.dt * expected_fac_clamped, 1e-12);
}

TEST(VSVO_BDF_Controller, RejectedNonlinearFailureUsesNonlinearDecreaseFactor)
{
    using svmp::FE::timestepping::StepAttemptInfo;
    using svmp::FE::timestepping::StepRejectReason;
    using svmp::FE::timestepping::VSVO_BDF_Controller;
    using svmp::FE::timestepping::VSVO_BDF_ControllerOptions;

    VSVO_BDF_ControllerOptions o;
    o.initial_order = 2;
    o.nonlinear_decrease_factor = 0.5;
    VSVO_BDF_Controller ctrl(o);

    StepAttemptInfo info;
    info.dt = 0.4;
    info.scheme_order = 2;
    info.attempt_index = 0;

    const auto d = ctrl.onRejected(info, StepRejectReason::NonlinearSolveFailed);
    EXPECT_FALSE(d.accept);
    EXPECT_TRUE(d.retry);
    EXPECT_EQ(d.next_order, 1);
    EXPECT_NEAR(d.next_dt, 0.2, 1e-15);
}

TEST(VSVO_BDF_Controller, InvalidDtAfterUpdateStopsRetrying)
{
    using svmp::FE::timestepping::StepAttemptInfo;
    using svmp::FE::timestepping::VSVO_BDF_Controller;
    using svmp::FE::timestepping::VSVO_BDF_ControllerOptions;

    VSVO_BDF_ControllerOptions o;
    o.initial_order = 2;
    o.min_order = 1;
    o.max_order = 3;
    o.max_retries = 5;
    VSVO_BDF_Controller ctrl(o);

    StepAttemptInfo info;
    info.dt = 0.0;
    info.scheme_order = 2;
    info.error_norm = 2.0;
    info.step_index = 0;
    info.attempt_index = 0;

    const auto d = ctrl.onAccepted(info);
    EXPECT_FALSE(d.accept);
    EXPECT_FALSE(d.retry);
    EXPECT_EQ(d.next_order, 1);
    EXPECT_DOUBLE_EQ(d.next_dt, 0.0);
}
