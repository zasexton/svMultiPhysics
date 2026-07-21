#include "LevelSet/LevelSetConservativePhaseArtifact.h"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>

namespace {

namespace FE = svmp::FE;
namespace level_set = svmp::FE::level_set;

TEST(LevelSetConservativePhaseArtifact,
     AtomicallyWritesTheCompleteAcceptedStageAndRefusesReplacement)
{
    const std::array<FE::Real, 2> volumes{1.0, 1.0};
    const std::array<FE::Real, 2> previous{0.8, 0.2};
    const std::array<FE::Real, 2> lower{0.0, 0.0};
    const std::array<FE::Real, 2> upper{1.0, 1.0};
    const std::array<level_set::LevelSetPhaseFluxEdge, 1> edges{
        level_set::LevelSetPhaseFluxEdge{0, 1, 0.05, 0.30},
    };
    auto correction =
        level_set::applyLevelSetConservativePhaseFluxCorrection(
            level_set::LevelSetPhaseFluxStageView{
                .lumped_control_volume = volumes,
                .previous_liquid_indicator = previous,
                .lower_liquid_indicator = lower,
                .upper_liquid_indicator = upper,
                .interior_edges = edges,
            });
    ASSERT_TRUE(correction.success) << correction.diagnostic;

    level_set::LevelSetP1PhaseTransportStageResult stage;
    stage.success = true;
    stage.courant_satisfied = true;
    stage.low_order_coefficients_nonnegative = true;
    stage.strong_form_decomposition_satisfied = true;
    stage.maximum_courant = 0.25;
    stage.nodal_courant = {0.25, 0.25};
    stage.physical_boundary_mass_transfer = {0.0, 0.0};
    stage.discrete_divergence_mass_source = {0.0, 0.0};
    stage.flux_edges.assign(edges.begin(), edges.end());
    stage.correction = std::move(correction);

    const auto unique = std::chrono::steady_clock::now()
                            .time_since_epoch()
                            .count();
    const auto output_directory =
        std::filesystem::temp_directory_path() /
        ("svmp-conservative-phase-artifact-" +
         std::to_string(unique));
    level_set::LevelSetConservativePhaseArtifactContext context{
        .phase_field_name = "phase_fraction",
        .level_set_field_name = "phi",
        .geometry_domain_id = "liquid_interface",
        .accepted_step = 7u,
        .accepted_time = 0.35,
        .time_step = 0.05,
        .state_revision = 41u,
        .graph_geometry_revision = 2u,
        .graph_topology_revision = 3u,
        .graph_ownership_revision = 4u,
        .graph_numbering_revision = 5u,
        .graph_dof_layout_revision = 6u,
        .geometry_validated_before_commit = true,
        .raw_post_transport_phase_measure =
            stage.correction.total_raw_target_liquid_measure,
        .post_limit_phase_measure =
            stage.correction.total_limited_liquid_measure,
        .raw_post_transport_geometry_measure = 1.0,
        .post_reinitialization_phase_measure =
            stage.correction.total_limited_liquid_measure,
        .post_reinitialization_geometry_measure = 1.0,
        .post_correction_phase_measure =
            stage.correction.total_limited_liquid_measure,
        .post_correction_geometry_measure = 1.0,
        .retained_assembly_geometry_measure = 1.0,
    };
    context.reinitialization.diagnostic = "not due \"quoted\"";
    context.reconciliation.diagnostic = "no correction required";

    const auto artifact =
        level_set::writeLevelSetConservativePhaseArtifact(
            output_directory, context, stage);
    ASSERT_TRUE(artifact.success) << artifact.diagnostic;
    EXPECT_TRUE(std::filesystem::is_regular_file(artifact.path));
    EXPECT_FALSE(std::filesystem::exists(
        std::filesystem::path(artifact.path.string() + ".tmp")));
    EXPECT_GT(artifact.bytes, 0u);
    EXPECT_EQ(artifact.nodes, 2u);
    EXPECT_EQ(artifact.edges, 1u);
    EXPECT_EQ(artifact.resolved_components, 1u);

    std::ifstream input(artifact.path);
    ASSERT_TRUE(input.is_open());
    const std::string contents{
        std::istreambuf_iterator<char>{input},
        std::istreambuf_iterator<char>{}};
    EXPECT_NE(contents.find(
                  "\"artifact\":\"conservative_phase_flux_ledger\""),
              std::string::npos);
    EXPECT_NE(contents.find("\"accepted_step\":7"),
              std::string::npos);
    EXPECT_NE(contents.find("\"nodes\":[{"),
              std::string::npos);
    EXPECT_NE(contents.find("\"edges\":[{"),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"components\":[{\"classification\":\"resolved\""),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"limited_local_mass_balance_residual\":"),
              std::string::npos);
    EXPECT_NE(contents.find("\"courant\":0.25"),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"reinitialization_diagnostic\":\"not due \\\"quoted\\\"\""),
              std::string::npos);

    const auto replacement =
        level_set::writeLevelSetConservativePhaseArtifact(
            output_directory, context, stage);
    EXPECT_FALSE(replacement.success);
    EXPECT_NE(replacement.diagnostic.find("refuses to replace"),
              std::string::npos);
    std::ifstream retained_input(artifact.path);
    ASSERT_TRUE(retained_input.is_open());
    const std::string retained_contents{
        std::istreambuf_iterator<char>{retained_input},
        std::istreambuf_iterator<char>{}};
    EXPECT_EQ(retained_contents, contents);

    auto invalid_context = context;
    invalid_context.accepted_step = 8u;
    invalid_context.time_step = 0.0;
    const auto invalid =
        level_set::writeLevelSetConservativePhaseArtifact(
            output_directory, invalid_context, stage);
    EXPECT_FALSE(invalid.success);
    EXPECT_NE(invalid.diagnostic.find("finite validated accepted stage"),
              std::string::npos);

    auto incomplete_stage = stage;
    incomplete_stage.nodal_courant.pop_back();
    auto incomplete_context = context;
    incomplete_context.accepted_step = 9u;
    const auto incomplete =
        level_set::writeLevelSetConservativePhaseArtifact(
            output_directory, incomplete_context, incomplete_stage);
    EXPECT_FALSE(incomplete.success);
    EXPECT_NE(incomplete.diagnostic.find("complete transport"),
              std::string::npos);

    auto stale_context = context;
    stale_context.accepted_step = 10u;
    const auto stale_path =
        output_directory /
        "conservative_phase_flux_phase_fraction_step_00000010.json.tmp";
    {
        std::ofstream stale_output(stale_path);
        ASSERT_TRUE(stale_output.is_open());
        stale_output << "stale";
    }
    const auto stale =
        level_set::writeLevelSetConservativePhaseArtifact(
            output_directory, stale_context, stage);
    EXPECT_FALSE(stale.success);
    EXPECT_NE(stale.diagnostic.find("refuses to replace"),
              std::string::npos);
    std::ifstream stale_input(stale_path);
    ASSERT_TRUE(stale_input.is_open());
    std::string stale_contents;
    stale_input >> stale_contents;
    EXPECT_EQ(stale_contents, "stale");

    std::error_code cleanup_error;
    std::filesystem::remove_all(output_directory, cleanup_error);
    EXPECT_FALSE(cleanup_error);
}

} // namespace
