#include "LevelSet/LevelSetConservativePhaseTransport.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <string>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

TEST(LevelSetConservativePhaseTransport,
     PreservesAConstantThroughClosedInteriorCirculations)
{
    const std::array<FE::Real, 3> volumes{1.0, 1.0, 1.0};
    const std::array<FE::Real, 3> previous{0.4, 0.4, 0.4};
    const std::array<FE::Real, 3> lower{0.0, 0.0, 0.0};
    const std::array<FE::Real, 3> upper{1.0, 1.0, 1.0};
    const std::array<level_set::LevelSetPhaseFluxEdge, 3> edges{
        level_set::LevelSetPhaseFluxEdge{0, 1, 0.125, -0.075},
        level_set::LevelSetPhaseFluxEdge{0, 2, -0.125, 0.075},
        level_set::LevelSetPhaseFluxEdge{1, 2, 0.125, -0.075},
    };

    const auto result =
        level_set::applyLevelSetConservativePhaseFluxCorrection(
            level_set::LevelSetPhaseFluxStageView{
                .lumped_control_volume = volumes,
                .previous_liquid_indicator = previous,
                .lower_liquid_indicator = lower,
                .upper_liquid_indicator = upper,
                .interior_edges = edges,
            });

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_FALSE(result.applied);
    EXPECT_TRUE(result.constant_state_input);
    EXPECT_TRUE(result.constant_preservation_satisfied);
    EXPECT_EQ(result.maximum_constant_preservation_error, 0.0);
    EXPECT_TRUE(result.interior_cancellation_satisfied);
    EXPECT_EQ(result.maximum_edge_pair_cancellation_residual, 0.0);
    EXPECT_EQ(result.low_order_nodal_cancellation_residual, 0.0);
    EXPECT_EQ(result.raw_nodal_cancellation_residual, 0.0);
    EXPECT_EQ(result.limited_nodal_cancellation_residual, 0.0);
    EXPECT_EQ(result.global_mass_balance_residual, 0.0);
    ASSERT_EQ(result.nodes.size(), previous.size());
    for (const auto& node : result.nodes) {
        EXPECT_EQ(node.low_order_liquid_indicator, 0.4);
        EXPECT_EQ(node.limited_liquid_indicator, 0.4);
        EXPECT_EQ(node.low_order_local_mass_balance_residual, 0.0);
        EXPECT_EQ(node.raw_target_local_mass_balance_residual, 0.0);
        EXPECT_EQ(node.local_mass_balance_residual, 0.0);
    }
}

TEST(LevelSetConservativePhaseTransport,
     LimitsAnOvershootWithOneSymmetricConservativeEdgeFactor)
{
    const std::array<FE::Real, 2> volumes{1.0, 1.0};
    const std::array<FE::Real, 2> previous{0.2, 0.8};
    const std::array<FE::Real, 2> lower{0.0, 0.0};
    const std::array<FE::Real, 2> upper{1.0, 1.0};
    const std::array<level_set::LevelSetPhaseFluxEdge, 1> edges{
        level_set::LevelSetPhaseFluxEdge{0, 1, 0.0, 1.0},
    };

    const auto result =
        level_set::applyLevelSetConservativePhaseFluxCorrection(
            level_set::LevelSetPhaseFluxStageView{
                .lumped_control_volume = volumes,
                .previous_liquid_indicator = previous,
                .lower_liquid_indicator = lower,
                .upper_liquid_indicator = upper,
                .interior_edges = edges,
            });

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.applied);
    EXPECT_EQ(result.limited_edges, 1u);
    ASSERT_EQ(result.edges.size(), 1u);
    EXPECT_NEAR(result.edges[0].correction_factor, 0.8, 1.0e-14);
    EXPECT_NEAR(result.edges[0].limited_antidiffusive_mass_transfer,
                0.8, 1.0e-14);
    EXPECT_NEAR(result.minimum_raw_target_liquid_indicator, -0.2, 1.0e-14);
    EXPECT_NEAR(result.maximum_raw_target_liquid_indicator, 1.2, 1.0e-14);
    ASSERT_EQ(result.nodes.size(), 2u);
    EXPECT_NEAR(result.nodes[0].limited_liquid_indicator, 1.0, 1.0e-14);
    EXPECT_NEAR(result.nodes[1].limited_liquid_indicator, 0.0, 1.0e-14);
    EXPECT_NEAR(result.total_previous_liquid_measure, 1.0, 1.0e-14);
    EXPECT_NEAR(result.total_limited_liquid_measure, 1.0, 1.0e-14);
    EXPECT_EQ(result.maximum_edge_pair_cancellation_residual, 0.0);
    EXPECT_NEAR(result.low_order_global_mass_balance_residual, 0.0, 1.0e-14);
    EXPECT_NEAR(result.raw_target_global_mass_balance_residual, 0.0,
                1.0e-14);
    EXPECT_NEAR(result.global_mass_balance_residual, 0.0, 1.0e-14);
}

TEST(LevelSetConservativePhaseTransport,
     LimitsNegativeTransfersAndSharesAReceivingNodeBudget)
{
    const std::array<FE::Real, 3> volumes{1.0, 1.0, 1.0};
    const std::array<FE::Real, 3> previous{0.8, 0.2, 0.8};
    const std::array<FE::Real, 3> lower{0.0, 0.0, 0.0};
    const std::array<FE::Real, 3> upper{1.0, 1.0, 1.0};
    const std::array<level_set::LevelSetPhaseFluxEdge, 2> edges{
        level_set::LevelSetPhaseFluxEdge{0, 1, 0.0, -0.6},
        level_set::LevelSetPhaseFluxEdge{1, 2, 0.0, 0.6},
    };

    const auto result =
        level_set::applyLevelSetConservativePhaseFluxCorrection(
            level_set::LevelSetPhaseFluxStageView{
                .lumped_control_volume = volumes,
                .previous_liquid_indicator = previous,
                .lower_liquid_indicator = lower,
                .upper_liquid_indicator = upper,
                .interior_edges = edges,
            });

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.applied);
    EXPECT_EQ(result.limited_edges, 2u);
    ASSERT_EQ(result.edges.size(), 2u);
    EXPECT_NEAR(result.edges[0].correction_factor, 2.0 / 3.0, 1.0e-14);
    EXPECT_NEAR(result.edges[1].correction_factor, 2.0 / 3.0, 1.0e-14);
    ASSERT_EQ(result.nodes.size(), 3u);
    EXPECT_NEAR(result.nodes[0].limited_liquid_indicator, 0.4, 1.0e-14);
    EXPECT_NEAR(result.nodes[1].limited_liquid_indicator, 1.0, 1.0e-14);
    EXPECT_NEAR(result.nodes[2].limited_liquid_indicator, 0.4, 1.0e-14);
    EXPECT_NEAR(result.total_previous_liquid_measure, 1.8, 1.0e-14);
    EXPECT_NEAR(result.total_limited_liquid_measure, 1.8, 1.0e-14);
    EXPECT_NEAR(result.global_mass_balance_residual, 0.0, 1.0e-14);
}

TEST(LevelSetConservativePhaseTransport,
     EnforcesNodalBoundsTighterThanTheUnitInterval)
{
    const std::array<FE::Real, 2> volumes{1.0, 1.0};
    const std::array<FE::Real, 2> previous{0.3, 0.7};
    const std::array<FE::Real, 2> lower{0.2, 0.6};
    const std::array<FE::Real, 2> upper{0.4, 0.8};
    const std::array<level_set::LevelSetPhaseFluxEdge, 1> edges{
        level_set::LevelSetPhaseFluxEdge{0, 1, 0.0, 1.0},
    };

    const auto result =
        level_set::applyLevelSetConservativePhaseFluxCorrection(
            level_set::LevelSetPhaseFluxStageView{
                .lumped_control_volume = volumes,
                .previous_liquid_indicator = previous,
                .lower_liquid_indicator = lower,
                .upper_liquid_indicator = upper,
                .interior_edges = edges,
            });

    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_EQ(result.edges.size(), 1u);
    EXPECT_NEAR(result.edges[0].correction_factor, 0.1, 1.0e-14);
    ASSERT_EQ(result.nodes.size(), 2u);
    EXPECT_NEAR(result.nodes[0].limited_liquid_indicator, 0.4, 1.0e-14);
    EXPECT_NEAR(result.nodes[1].limited_liquid_indicator, 0.6, 1.0e-14);
    EXPECT_NEAR(result.total_previous_liquid_measure, 1.0, 1.0e-14);
    EXPECT_NEAR(result.total_limited_liquid_measure, 1.0, 1.0e-14);
}

TEST(LevelSetConservativePhaseTransport,
     ClosesLocalBalancesWithBoundaryAndDivergenceContributions)
{
    const std::array<FE::Real, 3> volumes{2.0, 1.0, 3.0};
    const std::array<FE::Real, 3> previous{0.25, 0.5, 0.75};
    const std::array<FE::Real, 3> lower{0.0, 0.0, 0.0};
    const std::array<FE::Real, 3> upper{1.0, 1.0, 1.0};
    const std::array<FE::Real, 3> boundary{0.1, 0.0, -0.05};
    const std::array<FE::Real, 3> divergence{0.0, 0.02, -0.02};
    const std::array<level_set::LevelSetPhaseFluxEdge, 3> edges{
        level_set::LevelSetPhaseFluxEdge{0, 1, 0.1, 0.0},
        level_set::LevelSetPhaseFluxEdge{0, 2, 0.0, 2.0},
        level_set::LevelSetPhaseFluxEdge{1, 2, -0.03, 0.0},
    };

    const auto result =
        level_set::applyLevelSetConservativePhaseFluxCorrection(
            level_set::LevelSetPhaseFluxStageView{
                .lumped_control_volume = volumes,
                .previous_liquid_indicator = previous,
                .lower_liquid_indicator = lower,
                .upper_liquid_indicator = upper,
                .interior_edges = edges,
                .physical_boundary_mass_transfer = boundary,
                .discrete_divergence_mass_source = divergence,
            });

    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_TRUE(result.applied);
    EXPECT_TRUE(result.local_balance_satisfied);
    EXPECT_TRUE(result.global_balance_satisfied);
    EXPECT_TRUE(result.interior_cancellation_satisfied);
    EXPECT_NEAR(result.total_previous_liquid_measure, 3.25, 1.0e-14);
    EXPECT_NEAR(result.total_physical_boundary_mass_transfer, 0.05, 1.0e-14);
    EXPECT_NEAR(result.total_discrete_divergence_mass_source, 0.0, 1.0e-14);
    EXPECT_NEAR(result.total_limited_liquid_measure, 3.30, 1.0e-14);
    EXPECT_NEAR(result.low_order_global_mass_balance_residual, 0.0, 1.0e-14);
    EXPECT_NEAR(result.raw_target_global_mass_balance_residual, 0.0,
                1.0e-14);
    EXPECT_NEAR(result.global_mass_balance_residual, 0.0, 1.0e-14);
    EXPECT_LE(result.maximum_low_order_local_mass_balance_residual, 4.0e-16);
    EXPECT_LE(result.maximum_raw_target_local_mass_balance_residual, 4.0e-16);
    EXPECT_LE(result.maximum_local_mass_balance_residual, 4.0e-16);
    ASSERT_EQ(result.nodes.size(), 3u);
    EXPECT_NEAR(result.nodes[0].limited_liquid_indicator, 1.0, 1.0e-14);
    for (const auto& node : result.nodes) {
        EXPECT_GE(node.limited_liquid_indicator, 0.0);
        EXPECT_LE(node.limited_liquid_indicator, 1.0);
    }
}

TEST(LevelSetConservativePhaseTransport,
     RejectsMalformedGraphsAndAnUnboundedLowOrderPredictor)
{
    const std::array<FE::Real, 2> volumes{1.0, 1.0};
    const std::array<FE::Real, 2> previous{0.25, 0.75};
    const std::array<FE::Real, 2> lower{0.0, 0.0};
    const std::array<FE::Real, 2> upper{1.0, 1.0};
    const std::array<level_set::LevelSetPhaseFluxEdge, 2> duplicates{
        level_set::LevelSetPhaseFluxEdge{0, 1, 0.0, 0.1},
        level_set::LevelSetPhaseFluxEdge{0, 1, 0.0, -0.1},
    };
    auto result = level_set::applyLevelSetConservativePhaseFluxCorrection(
        level_set::LevelSetPhaseFluxStageView{
            .lumped_control_volume = volumes,
            .previous_liquid_indicator = previous,
            .lower_liquid_indicator = lower,
            .upper_liquid_indicator = upper,
            .interior_edges = duplicates,
        });
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("duplicate"), std::string::npos);

    const std::array<level_set::LevelSetPhaseFluxEdge, 1> reversed{
        level_set::LevelSetPhaseFluxEdge{1, 0, 0.0, 0.1},
    };
    result = level_set::applyLevelSetConservativePhaseFluxCorrection(
        level_set::LevelSetPhaseFluxStageView{
            .lumped_control_volume = volumes,
            .previous_liquid_indicator = previous,
            .lower_liquid_indicator = lower,
            .upper_liquid_indicator = upper,
            .interior_edges = reversed,
        });
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("canonical"), std::string::npos);

    const std::array<level_set::LevelSetPhaseFluxEdge, 1> unbounded_low_order{
        level_set::LevelSetPhaseFluxEdge{0, 1, 1.0, 0.0},
    };
    result = level_set::applyLevelSetConservativePhaseFluxCorrection(
        level_set::LevelSetPhaseFluxStageView{
            .lumped_control_volume = volumes,
            .previous_liquid_indicator = previous,
            .lower_liquid_indicator = lower,
            .upper_liquid_indicator = upper,
            .interior_edges = unbounded_low_order,
        });
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.low_order_bounds_satisfied);
    EXPECT_NE(result.diagnostic.find("low-order"), std::string::npos);
}

TEST(LevelSetConservativePhaseTransport,
     RequiresAnExplicitWaiverForBoundaryDrivenConstantChange)
{
    const std::array<FE::Real, 2> volumes{1.0, 1.0};
    const std::array<FE::Real, 2> previous{0.0, 0.0};
    const std::array<FE::Real, 2> lower{0.0, 0.0};
    const std::array<FE::Real, 2> upper{1.0, 1.0};
    const std::array<FE::Real, 2> boundary{0.1, 0.0};

    auto result = level_set::applyLevelSetConservativePhaseFluxCorrection(
        level_set::LevelSetPhaseFluxStageView{
            .lumped_control_volume = volumes,
            .previous_liquid_indicator = previous,
            .lower_liquid_indicator = lower,
            .upper_liquid_indicator = upper,
            .physical_boundary_mass_transfer = boundary,
        });
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.constant_state_input);
    EXPECT_TRUE(result.constant_preservation_required);
    EXPECT_FALSE(result.constant_preservation_satisfied);
    EXPECT_NE(result.diagnostic.find("constant-state"), std::string::npos);

    result = level_set::applyLevelSetConservativePhaseFluxCorrection(
        level_set::LevelSetPhaseFluxStageView{
            .lumped_control_volume = volumes,
            .previous_liquid_indicator = previous,
            .lower_liquid_indicator = lower,
            .upper_liquid_indicator = upper,
            .physical_boundary_mass_transfer = boundary,
            .require_constant_preservation = false,
        });
    ASSERT_TRUE(result.success) << result.diagnostic;
    EXPECT_FALSE(result.constant_preservation_required);
    EXPECT_FALSE(result.constant_preservation_satisfied);
    EXPECT_TRUE(result.global_balance_satisfied);
    EXPECT_NEAR(result.total_limited_liquid_measure, 0.1, 1.0e-14);
    EXPECT_NEAR(result.global_mass_balance_residual, 0.0, 1.0e-14);
}

TEST(LevelSetConservativePhaseTransport, RejectsInvalidNodalStageData)
{
    const std::array<FE::Real, 2> volumes{1.0, 0.0};
    const std::array<FE::Real, 2> previous{0.25, 0.75};
    const std::array<FE::Real, 2> lower{0.0, 0.0};
    const std::array<FE::Real, 2> upper{1.0, 1.0};

    auto result = level_set::applyLevelSetConservativePhaseFluxCorrection(
        level_set::LevelSetPhaseFluxStageView{
            .lumped_control_volume = volumes,
            .previous_liquid_indicator = previous,
            .lower_liquid_indicator = lower,
            .upper_liquid_indicator = upper,
        });
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("positive"), std::string::npos);

    const std::array<FE::Real, 2> positive_volumes{1.0, 1.0};
    const std::array<FE::Real, 2> invalid_upper{1.0, 1.1};
    result = level_set::applyLevelSetConservativePhaseFluxCorrection(
        level_set::LevelSetPhaseFluxStageView{
            .lumped_control_volume = positive_volumes,
            .previous_liquid_indicator = previous,
            .lower_liquid_indicator = lower,
            .upper_liquid_indicator = invalid_upper,
        });
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("unit-interval"), std::string::npos);

    const std::array<FE::Real, 1> incomplete_boundary{0.0};
    result = level_set::applyLevelSetConservativePhaseFluxCorrection(
        level_set::LevelSetPhaseFluxStageView{
            .lumped_control_volume = positive_volumes,
            .previous_liquid_indicator = previous,
            .lower_liquid_indicator = lower,
            .upper_liquid_indicator = upper,
            .physical_boundary_mass_transfer = incomplete_boundary,
        });
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("incomplete"), std::string::npos);

    const std::array<FE::Real, 2> nonfinite_previous{
        0.25, std::numeric_limits<FE::Real>::quiet_NaN()};
    result = level_set::applyLevelSetConservativePhaseFluxCorrection(
        level_set::LevelSetPhaseFluxStageView{
            .lumped_control_volume = positive_volumes,
            .previous_liquid_indicator = nonfinite_previous,
            .lower_liquid_indicator = lower,
            .upper_liquid_indicator = upper,
        });
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("non-finite"), std::string::npos);
}

} // namespace
