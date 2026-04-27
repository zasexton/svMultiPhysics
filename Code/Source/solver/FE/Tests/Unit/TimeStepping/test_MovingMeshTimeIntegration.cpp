/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Systems/SystemState.h"
#include "Systems/TimeIntegrator.h"
#include "TimeStepping/GeneralizedAlpha.h"
#include "TimeStepping/MovingMeshTimeIntegration.h"

#include <span>
#include <vector>

namespace {

using svmp::FE::Real;
namespace moving_mesh = svmp::FE::timestepping::moving_mesh;

struct StateStorage {
    explicit StateStorage(double dt_in, double dt_prev_in = 0.0)
        : dt(dt_in)
        , dt_prev(dt_prev_in > 0.0 ? dt_prev_in : dt_in)
        , dt_history{dt, dt_prev}
        , u_history{u_prev, u_prev2}
    {
    }

    [[nodiscard]] svmp::FE::systems::SystemStateView view() const
    {
        svmp::FE::systems::SystemStateView state;
        state.dt = dt;
        state.dt_prev = dt_prev;
        state.u = u;
        state.u_prev = u_prev;
        state.u_prev2 = u_prev2;
        state.u_history = u_history;
        state.dt_history = dt_history;
        return state;
    }

    double dt{0.0};
    double dt_prev{0.0};
    std::vector<Real> u{0.0};
    std::vector<Real> u_prev{0.0};
    std::vector<Real> u_prev2{0.0};
    std::vector<double> dt_history{};
    std::vector<std::span<const Real>> u_history{};
};

} // namespace

TEST(MovingMeshTimeIntegration, HistoryStoresAndAcceptsCoordinateAndDisplacementStates)
{
    moving_mesh::MovingMeshTimeHistory history(/*dimension=*/2, /*entity_count=*/2, /*history_depth=*/2);

    const std::vector<Real> current_coordinates = {1.0, 2.0, 3.0, 4.0};
    const std::vector<Real> current_displacements = {0.1, 0.2, 0.3, 0.4};
    const std::vector<Real> previous_coordinates = {-1.0, -2.0, -3.0, -4.0};
    const std::vector<Real> previous_displacements = {-0.1, -0.2, -0.3, -0.4};

    history.setCurrentCoordinates(current_coordinates);
    history.setCurrentMeshDisplacements(current_displacements);
    history.setPreviousCoordinates(1, previous_coordinates);
    history.setPreviousMeshDisplacements(1, previous_displacements);
    history.setDt(0.5);
    history.acceptStep(0.5);

    EXPECT_EQ(history.stepIndex(), 1);
    EXPECT_DOUBLE_EQ(history.dtPrev(), 0.5);

    const auto x_prev = history.previousCoordinates(1);
    const auto d_prev = history.previousMeshDisplacements(1);
    const auto x_prev2 = history.previousCoordinates(2);
    const auto d_prev2 = history.previousMeshDisplacements(2);
    for (std::size_t i = 0; i < current_coordinates.size(); ++i) {
        EXPECT_DOUBLE_EQ(x_prev[i], current_coordinates[i]);
        EXPECT_DOUBLE_EQ(d_prev[i], current_displacements[i]);
        EXPECT_DOUBLE_EQ(x_prev2[i], previous_coordinates[i]);
        EXPECT_DOUBLE_EQ(d_prev2[i], previous_displacements[i]);
    }
}

TEST(MovingMeshTimeIntegration, TrialCoordinateRollbackLeavesAcceptedHistoryAndTimeLevels)
{
    moving_mesh::MovingMeshTimeHistory history(/*dimension=*/2, /*entity_count=*/2, /*history_depth=*/2);

    const std::vector<Real> accepted_coordinates = {0.0, 0.0, 1.0, 0.0};
    const std::vector<Real> accepted_displacements = {0.0, 0.0, 0.1, 0.0};
    const std::vector<Real> previous_coordinates = {-0.2, 0.0, 0.8, 0.0};
    const std::vector<Real> previous_displacements = {-0.2, 0.0, -0.1, 0.0};
    const std::vector<Real> previous2_coordinates = {-0.5, 0.0, 0.5, 0.0};
    const std::vector<Real> previous2_displacements = {-0.5, 0.0, -0.4, 0.0};

    history.setCurrentCoordinates(accepted_coordinates);
    history.setCurrentMeshDisplacements(accepted_displacements);
    history.setPreviousCoordinates(1, previous_coordinates);
    history.setPreviousMeshDisplacements(1, previous_displacements);
    history.setPreviousCoordinates(2, previous2_coordinates);
    history.setPreviousMeshDisplacements(2, previous2_displacements);
    history.setStepIndex(7);
    history.setDt(0.25);
    history.setPrevDt(0.5);

    const std::vector<Real> trial_coordinates = {0.4, 0.1, 1.6, -0.2};
    const std::vector<Real> trial_displacements = {0.4, 0.1, 0.6, -0.2};
    history.setCurrentCoordinates(trial_coordinates);
    history.setCurrentMeshDisplacements(trial_displacements);

    const svmp::FE::assembly::TimeDerivativeStencil backward_euler{
        .order = 1,
        .a = {4.0, -4.0},
    };
    const auto trial_velocity =
        moving_mesh::computeMeshVelocityFromCoordinates(history, backward_euler);
    ASSERT_EQ(trial_velocity.size(), trial_coordinates.size());
    EXPECT_NEAR(trial_velocity[0], 2.4, 1.0e-14);
    EXPECT_NEAR(trial_velocity[2], 3.2, 1.0e-14);

    history.setCurrentCoordinates(accepted_coordinates);
    history.setCurrentMeshDisplacements(accepted_displacements);

    EXPECT_EQ(history.stepIndex(), 7);
    EXPECT_DOUBLE_EQ(history.dt(), 0.25);
    EXPECT_DOUBLE_EQ(history.dtPrev(), 0.5);
    for (std::size_t i = 0; i < accepted_coordinates.size(); ++i) {
        EXPECT_DOUBLE_EQ(history.currentCoordinates()[i], accepted_coordinates[i]);
        EXPECT_DOUBLE_EQ(history.currentMeshDisplacements()[i], accepted_displacements[i]);
        EXPECT_DOUBLE_EQ(history.previousCoordinates(1)[i], previous_coordinates[i]);
        EXPECT_DOUBLE_EQ(history.previousMeshDisplacements(1)[i], previous_displacements[i]);
        EXPECT_DOUBLE_EQ(history.previousCoordinates(2)[i], previous2_coordinates[i]);
        EXPECT_DOUBLE_EQ(history.previousMeshDisplacements(2)[i], previous2_displacements[i]);
    }
}

TEST(MovingMeshTimeIntegration, BDFStencilsComputeMeshVelocityAndAcceleration)
{
    moving_mesh::MovingMeshTimeHistory history(/*dimension=*/1, /*entity_count=*/1, /*history_depth=*/2);
    const std::vector<Real> x_current = {4.0};
    const std::vector<Real> x_prev = {1.0};
    const std::vector<Real> x_prev2 = {0.0};
    history.setCurrentCoordinates(x_current);
    history.setPreviousCoordinates(1, x_prev);
    history.setPreviousCoordinates(2, x_prev2);

    svmp::FE::systems::BDF2Integrator bdf2;
    const StateStorage bdf2_state(/*dt=*/1.0, /*dt_prev=*/1.0);
    const auto bdf2_ctx = bdf2.buildContext(/*max_time_derivative_order=*/1,
                                            bdf2_state.view());
    ASSERT_TRUE(bdf2_ctx.dt1.has_value());
    const auto velocity = moving_mesh::computeMeshVelocityFromCoordinates(history, *bdf2_ctx.dt1);
    ASSERT_EQ(velocity.size(), 1u);
    EXPECT_NEAR(velocity[0], 4.0, 1e-14);

    svmp::FE::systems::BackwardDifferenceIntegrator bdf;
    const StateStorage bdf_state(/*dt=*/1.0);
    const auto bdf_ctx = bdf.buildContext(/*max_time_derivative_order=*/2, bdf_state.view());
    ASSERT_TRUE(bdf_ctx.dt2.has_value());
    const auto acceleration = moving_mesh::computeMeshAccelerationFromCoordinates(history, *bdf_ctx.dt2);
    ASSERT_EQ(acceleration.size(), 1u);
    EXPECT_NEAR(acceleration[0], 2.0, 1e-14);
}

TEST(MovingMeshTimeIntegration, GeneralizedAlphaStencilsApplyThroughTheSameHelper)
{
    moving_mesh::MovingMeshTimeHistory first_order_history(/*dimension=*/1,
                                                           /*entity_count=*/1,
                                                           /*history_depth=*/2);
    const std::vector<Real> d_current_first = {1.0};
    const std::vector<Real> d_prev_first = {0.0};
    const std::vector<Real> d_prev2_first = {0.0};
    first_order_history.setCurrentMeshDisplacements(d_current_first);
    first_order_history.setPreviousMeshDisplacements(1, d_prev_first);
    first_order_history.setPreviousMeshDisplacements(2, d_prev2_first);

    svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator first_order({
        .alpha_m = 0.5,
        .alpha_f = 0.5,
        .gamma = 0.5,
        .history_rate_order = 1,
    });
    const StateStorage first_state(/*dt=*/0.1);
    const auto first_ctx = first_order.buildContext(/*max_time_derivative_order=*/1,
                                                    first_state.view());
    ASSERT_TRUE(first_ctx.dt1.has_value());
    const auto first_velocity =
        moving_mesh::computeMeshVelocityFromDisplacements(first_order_history, *first_ctx.dt1);
    ASSERT_EQ(first_velocity.size(), 1u);
    EXPECT_NEAR(first_velocity[0], first_ctx.dt1->coeff(0), 1e-14);

    moving_mesh::MovingMeshTimeHistory second_order_history(/*dimension=*/1,
                                                            /*entity_count=*/1,
                                                            /*history_depth=*/2);
    const std::vector<Real> d_current_second = {2.0};
    const std::vector<Real> d_prev_second = {3.0};
    const std::vector<Real> d_prev2_second = {5.0};
    second_order_history.setCurrentMeshDisplacements(d_current_second);
    second_order_history.setPreviousMeshDisplacements(1, d_prev_second);
    second_order_history.setPreviousMeshDisplacements(2, d_prev2_second);

    svmp::FE::timestepping::GeneralizedAlphaSecondOrderIntegrator second_order({
        .alpha_m = 0.5,
        .alpha_f = 0.5,
        .beta = 0.25,
        .gamma = 0.5,
    });
    const StateStorage second_state(/*dt=*/0.1);
    const auto second_ctx = second_order.buildContext(/*max_time_derivative_order=*/2,
                                                      second_state.view());
    ASSERT_TRUE(second_ctx.dt1.has_value());
    ASSERT_TRUE(second_ctx.dt2.has_value());

    const auto second_velocity =
        moving_mesh::computeMeshVelocityFromDisplacements(second_order_history, *second_ctx.dt1);
    ASSERT_EQ(second_velocity.size(), 1u);
    EXPECT_NEAR(second_velocity[0],
                second_ctx.dt1->coeff(0) * 2.0 + second_ctx.dt1->coeff(2) * 5.0,
                1e-14);

    const auto second_acceleration =
        moving_mesh::computeMeshAccelerationFromDisplacements(second_order_history, *second_ctx.dt2);
    ASSERT_EQ(second_acceleration.size(), 1u);
    EXPECT_NEAR(second_acceleration[0],
                second_ctx.dt2->coeff(0) * 2.0 + second_ctx.dt2->coeff(1) * 3.0,
                1e-10);
}

TEST(MovingMeshTimeIntegration, PredictedVelocityBlendIsPhysicsNeutral)
{
    const std::vector<Real> previous = {1.0, 2.0, 3.0};
    const std::vector<Real> current = {3.0, 6.0, 9.0};
    const auto predicted = moving_mesh::blendMeshVelocity(previous, current, 0.25);

    ASSERT_EQ(predicted.size(), 3u);
    EXPECT_NEAR(predicted[0], 1.5, 1e-14);
    EXPECT_NEAR(predicted[1], 3.0, 1e-14);
    EXPECT_NEAR(predicted[2], 4.5, 1e-14);
}

TEST(MovingMeshTimeIntegration, GCLDiagnosticComparesMeasureChangeToSuppliedRate)
{
    const auto exact = moving_mesh::evaluateControlVolumeGCL(/*current_measure=*/1.1,
                                                             /*previous_measure=*/1.0,
                                                             /*dt=*/0.1,
                                                             /*supplied_mesh_measure_rate=*/1.0);
    EXPECT_NEAR(exact.discrete_measure_rate, 1.0, 1e-14);
    EXPECT_NEAR(exact.residual, 0.0, 1e-14);
    EXPECT_NEAR(exact.relative_residual, 0.0, 1e-14);

    const std::vector<Real> current = {1.0, 1.2};
    const std::vector<Real> previous = {1.0, 1.0};
    const std::vector<Real> supplied = {0.0, 1.0};
    const auto diagnostics = moving_mesh::evaluateControlVolumeGCL(current, previous, 0.1, supplied);
    ASSERT_EQ(diagnostics.size(), 2u);
    EXPECT_NEAR(diagnostics[0].residual, 0.0, 1e-14);
    EXPECT_NEAR(diagnostics[1].residual, 1.0, 1e-14);
}

TEST(MovingMeshTimeIntegration, GCLDiagnosticHandlesMultipleNonuniformSteps)
{
    const std::vector<Real> step0_previous = {1.0, 2.0};
    const std::vector<Real> step0_current = {1.1, 2.3};
    const Real dt0 = 0.2;
    const std::vector<Real> step0_rates = {
        (step0_current[0] - step0_previous[0]) / dt0,
        (step0_current[1] - step0_previous[1]) / dt0,
    };
    const auto first =
        moving_mesh::evaluateControlVolumeGCL(step0_current, step0_previous, dt0, step0_rates);
    ASSERT_EQ(first.size(), 2u);
    EXPECT_NEAR(first[0].residual, 0.0, 1.0e-14);
    EXPECT_NEAR(first[1].residual, 0.0, 1.0e-14);

    const std::vector<Real> step1_current = {1.35, 2.0};
    const Real dt1 = 0.5;
    const std::vector<Real> step1_rates = {
        (step1_current[0] - step0_current[0]) / dt1,
        (step1_current[1] - step0_current[1]) / dt1,
    };
    const auto second =
        moving_mesh::evaluateControlVolumeGCL(step1_current, step0_current, dt1, step1_rates);
    ASSERT_EQ(second.size(), 2u);
    EXPECT_NEAR(second[0].residual, 0.0, 1.0e-14);
    EXPECT_NEAR(second[1].residual, 0.0, 1.0e-14);
}
