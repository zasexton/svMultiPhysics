/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Interfaces/ContactLineStageObservables.h"
#include "TimeStepping/OperatorStageCapture.h"

#include "Backends/Interfaces/BackendFactory.h"
#include "Backends/Interfaces/GenericVector.h"
#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ts = svmp::FE::timestepping;
namespace interfaces = svmp::FE::interfaces;
using svmp::FE::Real;
using Provenance =
    svmp::FE::systems::FreeSurfaceFirstOrderGeneralizedAlphaProvenance;

namespace {

/// The first-order generalized-alpha parameters of rho_inf = 0.5:
/// alpha_f = 1 / (1 + rho_inf), alpha_m = 2 alpha_f - 1/2, gamma = alpha_f.
Provenance rhoInfinityHalf(Real dt)
{
    const Real alpha_f = Real{1.0} / Real{1.5};
    return Provenance{
        .alpha_m = Real{2.0} * alpha_f - Real{0.5},
        .alpha_f = alpha_f,
        .gamma = alpha_f,
        .dt = dt,
    };
}

ts::CandidateStageObservation generalizedAlphaStage(
    const svmp::FE::backends::GenericVector* state,
    const svmp::FE::backends::GenericVector* rate)
{
    const auto provenance = rhoInfinityHalf(0.1);
    ts::CandidateStageObservation observation;
    observation.scheme = ts::SchemeKind::GeneralizedAlpha;
    observation.temporal_order = 1;
    observation.step_index = 3;
    observation.attempt_index = 1;
    observation.step_start_time = 0.2;
    observation.dt = 0.1;
    observation.step_end_time = 0.2 + 0.1;
    observation.state_time = 0.2 + provenance.alpha_f * 0.1;
    observation.rate_time = 0.2 + provenance.alpha_m * 0.1;
    observation.generalized_alpha = ts::GeneralizedAlphaStageMetadata{
        .alpha_f = provenance.alpha_f,
        .alpha_m = provenance.alpha_m,
        .gamma = provenance.gamma,
    };
    observation.mesh_revision = ts::CandidateStageMeshRevision{
        .geometry_revision = 11u,
        .topology_revision = 12u,
        .ownership_revision = 13u,
        .numbering_revision = 14u,
    };
    observation.state_vector = state;
    observation.rate_vector = rate;
    return observation;
}

} // namespace

TEST(OperatorStageCapture, GeneralizedAlphaProvenanceIsTheRhoInfinityFamily)
{
    const auto valid = rhoInfinityHalf(0.1);
    EXPECT_TRUE(interfaces::firstOrderGeneralizedAlphaContactProvenanceValid(valid));

    auto other_alpha_m = valid;
    other_alpha_m.alpha_m += Real{1.0e-6};
    EXPECT_FALSE(
        interfaces::firstOrderGeneralizedAlphaContactProvenanceValid(other_alpha_m));
    auto small_alpha_f = valid;
    small_alpha_f.alpha_f = Real{0.4};
    EXPECT_FALSE(
        interfaces::firstOrderGeneralizedAlphaContactProvenanceValid(small_alpha_f));
    auto no_step = valid;
    no_step.dt = Real{0.0};
    EXPECT_FALSE(interfaces::firstOrderGeneralizedAlphaContactProvenanceValid(no_step));

    EXPECT_TRUE(interfaces::contactStageRealValuesMatch(1.0, 1.0 + 1.0e-14));
    EXPECT_FALSE(interfaces::contactStageRealValuesMatch(1.0, 1.0 + 1.0e-12));
    EXPECT_FALSE(interfaces::contactStageRealValuesMatch(
        1.0, std::numeric_limits<Real>::quiet_NaN()));
}

TEST(OperatorStageCapture, AcceptedStageRevisionCoversItsInputsAndNotTheSignOfZero)
{
    const std::vector<Real> solution{0.0, 1.5, -2.0};
    const std::vector<Real> negative_zero{-0.0, 1.5, -2.0};
    const auto base = interfaces::acceptedContactStageRevision(
        3u, 5u, 7u, 0.25, 1.0, solution);
    EXPECT_NE(base, 0u);
    EXPECT_EQ(base, interfaces::acceptedContactStageRevision(
                        3u, 5u, 7u, 0.25, 1.0, negative_zero));
    EXPECT_NE(base, interfaces::acceptedContactStageRevision(
                        3u, 6u, 7u, 0.25, 1.0, solution));
    EXPECT_NE(base, interfaces::acceptedContactStageRevision(
                        3u, 5u, 7u, 0.5, 1.0, solution));
    EXPECT_NE(base, interfaces::acceptedContactStageRevision(
                        3u, 5u, 7u, 0.25, 1.0, solution, rhoInfinityHalf(0.1)));
}

TEST(OperatorStageCapture, ObservesACompleteFirstOrderGeneralizedAlphaStage)
{
    const auto factory = ts::test::createTestFactory();
    if (!factory) {
        GTEST_SKIP() << "Requires a serial backend.";
    }
    const auto state = factory->createVector(4);
    const auto rate = factory->createVector(4);
    const auto observation = generalizedAlphaStage(state.get(), rate.get());

    const auto captured =
        ts::makeDynamicContactFirstOrderGeneralizedAlphaObservation(observation);
    EXPECT_EQ(captured.step_index, 3);
    EXPECT_EQ(captured.attempt_index, 1);
    EXPECT_EQ(captured.mesh_revision, *observation.mesh_revision);
    EXPECT_DOUBLE_EQ(captured.provenance.dt, 0.1);
    EXPECT_DOUBLE_EQ(captured.state_time, observation.state_time);
    EXPECT_EQ(captured.contact_wall_boundary_fingerprint, 0u);
    EXPECT_TRUE(captured.operator_stage_state.fields.empty());

    auto stale_time = observation;
    stale_time.state_time += 1.0e-3;
    EXPECT_THROW(
        (void)ts::makeDynamicContactFirstOrderGeneralizedAlphaObservation(stale_time),
        std::invalid_argument);
    auto endpoint_scheme = observation;
    endpoint_scheme.scheme = ts::SchemeKind::BackwardEuler;
    EXPECT_THROW(
        (void)ts::makeDynamicContactFirstOrderGeneralizedAlphaObservation(
            endpoint_scheme),
        std::invalid_argument);
    auto no_state = observation;
    no_state.state_vector = nullptr;
    EXPECT_THROW(
        (void)ts::makeDynamicContactFirstOrderGeneralizedAlphaObservation(no_state),
        std::invalid_argument);
}

TEST(OperatorStageCapture, ResolvesEndpointAndAuthenticGeneralizedAlphaStages)
{
    ts::TimeHistory history;
    history.setTime(0.2);
    history.setDt(0.1);
    history.setStepIndex(2);

    const auto endpoint = ts::resolveFreeSurfaceContactStageTemporalCoordinates(
        ts::SchemeKind::BDF2, history, 0, std::nullopt);
    EXPECT_DOUBLE_EQ(endpoint.stage_time, 0.2 + 0.1);
    EXPECT_DOUBLE_EQ(endpoint.stage_alpha_f, 1.0);
    EXPECT_FALSE(endpoint.first_order_generalized_alpha.has_value());

    const auto factory = ts::test::createTestFactory();
    if (!factory) {
        GTEST_SKIP() << "Requires a serial backend.";
    }
    const auto state = factory->createVector(4);
    const auto rate = factory->createVector(4);
    const auto observation =
        ts::makeDynamicContactFirstOrderGeneralizedAlphaObservation(
            generalizedAlphaStage(state.get(), rate.get()));
    const auto stage = ts::resolveFreeSurfaceContactStageTemporalCoordinates(
        ts::SchemeKind::GeneralizedAlpha, history, 1, observation);
    EXPECT_DOUBLE_EQ(stage.stage_time, observation.state_time);
    EXPECT_DOUBLE_EQ(stage.stage_alpha_f, observation.provenance.alpha_f);
    ASSERT_TRUE(stage.first_order_generalized_alpha.has_value());
    EXPECT_DOUBLE_EQ(stage.first_order_generalized_alpha->alpha_m,
                     observation.provenance.alpha_m);

    // Another attempt, a missing observation, or an endpoint scheme with
    // generalized-alpha provenance are rejected.
    EXPECT_THROW(
        (void)ts::resolveFreeSurfaceContactStageTemporalCoordinates(
            ts::SchemeKind::GeneralizedAlpha, history, 2, observation),
        std::logic_error);
    EXPECT_THROW(
        (void)ts::resolveFreeSurfaceContactStageTemporalCoordinates(
            ts::SchemeKind::GeneralizedAlpha, history, 1, std::nullopt),
        std::logic_error);
    EXPECT_THROW(
        (void)ts::resolveFreeSurfaceContactStageTemporalCoordinates(
            ts::SchemeKind::BackwardEuler, history, 1, observation),
        std::logic_error);
}
