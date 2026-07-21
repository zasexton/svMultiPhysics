#include "LevelSet/LevelSetConservativePhaseRegions.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

[[nodiscard]] level_set::LevelSetPhaseFluxCorrectionResult
fourNodeCorrection()
{
    const std::array<FE::Real, 4> volumes{1.0, 1.0, 1.0, 1.0};
    const std::array<FE::Real, 4> previous{0.8, 0.6, 0.4, 0.2};
    const std::array<FE::Real, 4> lower{0.0, 0.0, 0.0, 0.0};
    const std::array<FE::Real, 4> upper{1.0, 1.0, 1.0, 1.0};
    const std::array<level_set::LevelSetPhaseFluxEdge, 3> edges{
        level_set::LevelSetPhaseFluxEdge{0, 1, 0.05, 0.30},
        level_set::LevelSetPhaseFluxEdge{1, 2, -0.02, -0.10},
        level_set::LevelSetPhaseFluxEdge{2, 3, 0.01, 0.05},
    };
    return level_set::applyLevelSetConservativePhaseFluxCorrection(
        level_set::LevelSetPhaseFluxStageView{
            .lumped_control_volume = volumes,
            .previous_liquid_indicator = previous,
            .lower_liquid_indicator = lower,
            .upper_liquid_indicator = upper,
            .interior_edges = edges,
        });
}

TEST(LevelSetConservativePhaseRegions,
     ReconstructsFilmSheetAndRimCrossingFluxes)
{
    const auto correction = fourNodeCorrection();
    ASSERT_TRUE(correction.success) << correction.diagnostic;
    const std::vector<level_set::LevelSetPhaseRegionDefinition> definitions{
        level_set::LevelSetPhaseRegionDefinition{
            .name = "bottom_film",
            .kind = level_set::LevelSetPhaseRegionKind::WallFilm,
            .node_membership = {1u, 1u, 0u, 0u},
        },
        level_set::LevelSetPhaseRegionDefinition{
            .name = "crown_sheet",
            .kind = level_set::LevelSetPhaseRegionKind::Sheet,
            .node_membership = {0u, 1u, 1u, 0u},
        },
        level_set::LevelSetPhaseRegionDefinition{
            .name = "crown_rim",
            .kind = level_set::LevelSetPhaseRegionKind::Rim,
            .node_membership = {0u, 0u, 1u, 0u},
        },
    };
    const auto result = level_set::buildLevelSetPhaseRegionLedgers(
        correction, definitions);
    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_TRUE(result.all_balances_satisfied);
    ASSERT_EQ(result.regions.size(), 3u);
    EXPECT_LE(result.maximum_balance_residual, 1.0e-14);
    EXPECT_LE(result.maximum_flux_reconstruction_residual, 1.0e-14);

    const auto& film = result.regions[0];
    EXPECT_STREQ(level_set::levelSetPhaseRegionKindName(film.kind),
                 "wall_film");
    EXPECT_EQ(film.member_nodes,
              (std::vector<FE::GlobalIndex>{0, 1}));
    EXPECT_EQ(film.internal_edges, 1u);
    ASSERT_EQ(film.crossing_edges.size(), 1u);
    EXPECT_EQ(film.crossing_edges.front().first_node, 1);
    EXPECT_EQ(film.crossing_edges.front().second_node, 2);
    EXPECT_DOUBLE_EQ(
        film.low_order_crossing_mass_transfer,
        correction.edges[1].low_order_mass_transfer);
    EXPECT_DOUBLE_EQ(
        film.raw_crossing_antidiffusive_mass_transfer,
        correction.edges[1].raw_antidiffusive_mass_transfer);
    EXPECT_DOUBLE_EQ(
        film.limited_crossing_antidiffusive_mass_transfer,
        correction.edges[1].limited_antidiffusive_mass_transfer);
    EXPECT_NEAR(film.low_order_balance_residual, 0.0, 1.0e-14);
    EXPECT_NEAR(film.raw_target_balance_residual, 0.0, 1.0e-14);
    EXPECT_NEAR(film.limited_balance_residual, 0.0, 1.0e-14);

    const auto& sheet = result.regions[1];
    EXPECT_EQ(sheet.member_nodes,
              (std::vector<FE::GlobalIndex>{1, 2}));
    EXPECT_EQ(sheet.internal_edges, 1u);
    EXPECT_EQ(sheet.crossing_edges.size(), 2u);
    EXPECT_TRUE(sheet.balance_satisfied);

    const auto& rim = result.regions[2];
    EXPECT_EQ(rim.member_nodes,
              (std::vector<FE::GlobalIndex>{2}));
    EXPECT_EQ(rim.internal_edges, 0u);
    EXPECT_EQ(rim.crossing_edges.size(), 2u);
    EXPECT_TRUE(rim.balance_satisfied);
}

TEST(LevelSetConservativePhaseRegions,
     RejectsAmbiguousOrMalformedMembership)
{
    const auto correction = fourNodeCorrection();
    ASSERT_TRUE(correction.success) << correction.diagnostic;

    std::vector<level_set::LevelSetPhaseRegionDefinition> malformed{
        level_set::LevelSetPhaseRegionDefinition{
            .name = "film",
            .kind = level_set::LevelSetPhaseRegionKind::WallFilm,
            .node_membership = {1u, 2u, 0u, 0u},
        },
    };
    auto result = level_set::buildLevelSetPhaseRegionLedgers(
        correction, malformed);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("binary membership"),
              std::string::npos);

    malformed.front().node_membership = {1u, 1u, 0u, 0u};
    malformed.push_back(malformed.front());
    result = level_set::buildLevelSetPhaseRegionLedgers(
        correction, malformed);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("unique valid names"),
              std::string::npos);

    malformed.resize(1u);
    malformed.front().kind =
        static_cast<level_set::LevelSetPhaseRegionKind>(1000);
    result = level_set::buildLevelSetPhaseRegionLedgers(
        correction, malformed);
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("valid names and kinds"),
              std::string::npos);

    auto incomplete = correction;
    incomplete.node_component_ids.pop_back();
    result = level_set::buildLevelSetPhaseRegionLedgers(
        incomplete,
        std::span<const level_set::LevelSetPhaseRegionDefinition>{});
    EXPECT_FALSE(result.success);
    EXPECT_NE(result.diagnostic.find("complete correction ledger"),
              std::string::npos);
}

TEST(LevelSetConservativePhaseRegions,
     EveryControlVolumeSubsetClosesAgainstItsCrossingEdges)
{
    const auto correction = fourNodeCorrection();
    ASSERT_TRUE(correction.success) << correction.diagnostic;
    for (unsigned int mask = 0u; mask < 16u; ++mask) {
        level_set::LevelSetPhaseRegionDefinition definition{
            .name = "subset",
            .kind = level_set::LevelSetPhaseRegionKind::Observer,
            .node_membership = std::vector<std::uint8_t>(4u, 0u),
        };
        for (std::size_t node = 0u; node < 4u; ++node) {
            definition.node_membership[node] =
                (mask & (1u << node)) != 0u ? 1u : 0u;
        }
        const std::array definitions{definition};
        const auto result = level_set::buildLevelSetPhaseRegionLedgers(
            correction, definitions);
        ASSERT_TRUE(result.success)
            << "mask=" << mask << " diagnostic=" << result.diagnostic;
        ASSERT_EQ(result.regions.size(), 1u);
        EXPECT_TRUE(result.regions.front().balance_satisfied)
            << "mask=" << mask;
        EXPECT_LE(result.maximum_balance_residual, 1.0e-14)
            << "mask=" << mask;
        EXPECT_LE(result.maximum_flux_reconstruction_residual, 1.0e-14)
            << "mask=" << mask;
    }
}

TEST(LevelSetConservativePhaseRegions,
     ParsesAndEvaluatesPredeclaredEulerianBoxes)
{
    const auto boxes = level_set::parseLevelSetPhaseRegionBoxes(
        "film|wall_film|0|1|0|0.1|*|*;"
        "rim.tip|rim|0.8|1|0.5|1|*|*");
    ASSERT_EQ(boxes.size(), 2u);
    EXPECT_EQ(boxes[0].kind,
              level_set::LevelSetPhaseRegionKind::WallFilm);
    EXPECT_EQ(boxes[1].kind,
              level_set::LevelSetPhaseRegionKind::Rim);
    const std::array<std::array<FE::Real, 3>, 4> coordinates{
        std::array<FE::Real, 3>{0.2, 0.05, 0.0},
        std::array<FE::Real, 3>{0.9, 0.05, 0.0},
        std::array<FE::Real, 3>{0.9, 0.7, 0.0},
        std::array<FE::Real, 3>{0.3, 0.7, 0.0},
    };
    const auto definitions =
        level_set::makeAxisAlignedBoxPhaseRegions(boxes, coordinates);
    ASSERT_EQ(definitions.size(), 2u);
    EXPECT_EQ(definitions[0].node_membership,
              (std::vector<std::uint8_t>{1u, 1u, 0u, 0u}));
    EXPECT_EQ(definitions[1].node_membership,
              (std::vector<std::uint8_t>{0u, 0u, 1u, 0u}));

    EXPECT_THROW(
        static_cast<void>(level_set::parseLevelSetPhaseRegionBoxes(
            "film|wall_film|1|0|0|1|*|*")),
        std::invalid_argument);
    EXPECT_THROW(
        static_cast<void>(level_set::parseLevelSetPhaseRegionBoxes(
            "film|unknown|0|1|0|1|*|*")),
        std::invalid_argument);
    EXPECT_THROW(
        static_cast<void>(level_set::parseLevelSetPhaseRegionBoxes(
            "film|sheet|0|1|0|1|*|*;"
            "film|rim|0|1|0|1|*|*")),
        std::invalid_argument);

    auto invalid_kind = boxes.front();
    invalid_kind.kind =
        static_cast<level_set::LevelSetPhaseRegionKind>(1000);
    const std::array invalid_boxes{invalid_kind};
    EXPECT_THROW(
        static_cast<void>(level_set::makeAxisAlignedBoxPhaseRegions(
            invalid_boxes, coordinates)),
        std::invalid_argument);
}

TEST(LevelSetConservativePhaseRegions,
     ExplicitlyClassifiesOnlyNonprimaryResolvedComponentsAsSatellites)
{
    const std::array<FE::Real, 5> volumes{1.0, 1.0, 1.0, 1.0, 1.0};
    const std::array<FE::Real, 5> previous{0.2, 0.2, 0.0, 0.2, 0.2};
    const std::array<FE::Real, 5> lower{0.0, 0.0, 0.0, 0.0, 0.0};
    const std::array<FE::Real, 5> upper{1.0, 1.0, 1.0, 1.0, 1.0};
    const std::array<level_set::LevelSetPhaseFluxEdge, 2> edges{
        level_set::LevelSetPhaseFluxEdge{0, 1, 0.01, 0.02},
        level_set::LevelSetPhaseFluxEdge{3, 4, 0.01, 0.02},
    };
    const auto correction =
        level_set::applyLevelSetConservativePhaseFluxCorrection(
            level_set::LevelSetPhaseFluxStageView{
                .lumped_control_volume = volumes,
                .previous_liquid_indicator = previous,
                .lower_liquid_indicator = lower,
                .upper_liquid_indicator = upper,
                .interior_edges = edges,
            });
    ASSERT_TRUE(correction.success) << correction.diagnostic;
    ASSERT_EQ(correction.components.size(), 2u);

    const auto satellites =
        level_set::makeNonprimaryComponentSatelliteRegions(correction);
    ASSERT_EQ(satellites.size(), 1u);
    EXPECT_EQ(satellites.front().name, "resolved_satellite_3");
    EXPECT_EQ(satellites.front().kind,
              level_set::LevelSetPhaseRegionKind::ResolvedSatellite);
    EXPECT_EQ(satellites.front().node_membership,
              (std::vector<std::uint8_t>{0u, 0u, 0u, 1u, 1u}));

    const auto result = level_set::buildLevelSetPhaseRegionLedgers(
        correction, satellites);
    ASSERT_TRUE(result.success) << result.diagnostic;
    ASSERT_EQ(result.regions.size(), 1u);
    EXPECT_NEAR(result.regions.front().limited_liquid_measure,
                0.4,
                1.0e-14);
    EXPECT_TRUE(result.regions.front().crossing_edges.empty());
    EXPECT_TRUE(result.regions.front().balance_satisfied);
}

} // namespace
