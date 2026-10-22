/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/LevelSetConservativePhaseWorkflow.h"
#include "LevelSet/LevelSetMaintenanceSchedule.h"
#include "LevelSet/LevelSetPhaseGeometryReconciliation.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ls = svmp::FE::level_set;
using svmp::FE::Real;

/// A transaction that only counts refreshes.
class CountingTransaction final : public ls::ConservativePhaseGeometryTransaction {
public:
    ls::ConservativePhaseGeometryRefresh refresh(std::span<const Real>) override
    {
        ++refreshes;
        return {};
    }
    bool active() const noexcept override { return true; }
    bool publicationStarted() const noexcept override { return false; }
    void commit() override {}
    void rollback() override {}

    int refreshes{0};
};

TEST(LevelSetConservativePhaseWorkflow, MomentMismatchIsTargetMinusCurrent)
{
    const std::vector<Real> current{1.0, 2.0, 3.0};
    const std::vector<Real> target{1.5, 1.0, 3.0};
    const auto mismatch = ls::conservativePhaseMomentMismatch(current, target);
    EXPECT_EQ(mismatch.maximum_nodal_residual, 1.0);
    EXPECT_EQ(mismatch.residual_norm, std::sqrt(Real{1.25}));
    EXPECT_EQ(mismatch.total_residual, -0.5);

    EXPECT_THROW((void)ls::conservativePhaseMomentMismatch(
                     current, std::vector<Real>(2, 0.0)),
                 std::runtime_error);
    const std::vector<Real> infinite{
        std::numeric_limits<Real>::infinity(), 2.0, 3.0};
    EXPECT_THROW((void)ls::conservativePhaseMomentMismatch(current, infinite),
                 std::runtime_error);
}

TEST(LevelSetConservativePhaseWorkflow, StageRealsCompareByBitPattern)
{
    EXPECT_TRUE(ls::sameConservativePhaseStageRealBits(0.25, 0.25));
    EXPECT_FALSE(ls::sameConservativePhaseStageRealBits(0.0, -0.0));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(ls::sameConservativePhaseStageRealBits(nan, nan));
}

TEST(LevelSetConservativePhaseWorkflow, ArtifactFingerprintMixesStringsByContent)
{
    std::uint64_t from_string = 14695981039346656037ull;
    std::uint64_t from_view = 14695981039346656037ull;
    std::uint64_t other = 14695981039346656037ull;
    ls::mixConservativePhaseArtifactFingerprint(from_string, std::string("phase"));
    ls::mixConservativePhaseArtifactFingerprint(from_view, std::string_view("phase"));
    ls::mixConservativePhaseArtifactFingerprint(other, std::string_view("phasf"));
    EXPECT_EQ(from_string, from_view);
    EXPECT_NE(from_view, other);
}

TEST(LevelSetConservativePhaseWorkflow, MaintenanceScheduleWordsAreCanonical)
{
    std::vector<std::uint64_t> words;
    ls::appendMaintenanceScheduleBool(words, true);
    ls::appendMaintenanceScheduleSigned(words, -1);
    ls::appendMaintenanceScheduleReal(words, 1.0);
    ls::appendMaintenanceScheduleString(words, "ab");
    ls::appendMaintenanceScheduleEnum(words, ls::MaintenanceSubstage::Limiting);
    const std::vector<std::uint64_t> expected{
        1u, ~std::uint64_t{0}, 0x3ff0000000000000ull, 2u, 97u, 98u, 1u};
    EXPECT_EQ(words, expected);
    EXPECT_TRUE(ls::collectiveLevelSetMaintenanceCanonicalWordsAgree(
        words, svmp::FE::systems::CollectiveCommunicator{}));
}

TEST(LevelSetConservativePhaseWorkflow, WithoutAnFESystemTheWorkflowDoesNotTouchItsRequests)
{
    const ls::LevelSetTransportOptions transport{};
    const std::optional<ls::ActiveCutVolumeRequest> volume_cut_request{};
    bool initialized = false;
    std::optional<ls::LevelSetP1PhaseTransportGraph> graph{};
    const ls::ConservativePhaseRequest::Configuration configuration{
        transport, volume_cut_request, 0.0};
    std::vector<ls::ConservativePhaseRequest> requests{
        ls::ConservativePhaseRequest{&configuration, {initialized, graph}}};
    const ls::ConservativePhaseWorkflowHost host{};

    ls::initializeConservativePhaseStates(host, requests);
    EXPECT_FALSE(initialized);
    EXPECT_FALSE(graph.has_value());

    std::vector<Real> candidate{0.0, 1.0};
    CountingTransaction transaction;
    const auto reconciliation = ls::reconcileConservativePhaseGeometry(
        host, requests.front(), {}, candidate, transaction);
    EXPECT_FALSE(reconciliation.success);
    EXPECT_EQ(reconciliation.diagnostic,
              "Conservative phase geometry reconciliation requires an FE system");
    EXPECT_EQ(transaction.refreshes, 0);
    EXPECT_EQ(candidate, (std::vector<Real>{0.0, 1.0}));
}

} // namespace
