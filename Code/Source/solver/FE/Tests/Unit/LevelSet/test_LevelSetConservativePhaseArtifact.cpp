#include "LevelSet/LevelSetConservativePhaseArtifact.h"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
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
    const std::array<FE::Real, 2> lower{0.2, 0.2};
    const std::array<FE::Real, 2> upper{0.8, 0.8};
    const std::array<FE::Real, 2> boundary_transfer{0.01, 0.01};
    const std::array<level_set::LevelSetPhaseFluxEdge, 1> edges{
        level_set::LevelSetPhaseFluxEdge{0, 1, -0.05, 0.30},
    };
    auto correction =
        level_set::applyLevelSetConservativePhaseFluxCorrection(
            level_set::LevelSetPhaseFluxStageView{
                .lumped_control_volume = volumes,
                .previous_liquid_indicator = previous,
                .lower_liquid_indicator = lower,
                .upper_liquid_indicator = upper,
                .interior_edges = edges,
                .physical_boundary_mass_transfer = boundary_transfer,
            });
    ASSERT_TRUE(correction.success) << correction.diagnostic;

    level_set::LevelSetP1PhaseTransportStageResult stage;
    stage.success = true;
    stage.courant_satisfied = true;
    stage.low_order_coefficients_nonnegative = true;
    stage.strong_form_decomposition_satisfied = true;
    stage.replicated_stage_inputs_satisfied = true;
    stage.maximum_courant = 0.25;
    stage.time_step = 0.125;
    stage.sampled_nodal_velocity = {
        std::array<FE::Real, 3>{1.0, 0.0, 0.0},
        std::array<FE::Real, 3>{0.5, 0.25, 0.0},
    };
    stage.nodal_courant = {0.25, 0.25};
    stage.physical_boundary_mass_transfer.assign(
        boundary_transfer.begin(), boundary_transfer.end());
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
        .accepted_time = 0.375,
        .time_step = 0.125,
        .state_revision = 41u,
        .graph_dimension = 2,
        .graph_nodes = 2u,
        .graph_edges = 1u,
        .graph_geometry_revision = 2u,
        .graph_topology_revision = 3u,
        .graph_ownership_revision = 4u,
        .graph_numbering_revision = 5u,
        .graph_dof_layout_revision = 6u,
        .graph_content_revision = 17u,
        .maximum_nodal_boundary_mass_transfer = 0.01,
        .boundary_mass_tolerance = 0.025,
        .boundary_flux_scope =
            "discrete_q_flux_not_pointwise_velocity_normal",
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
    const level_set::LevelSetP1PhaseGraphIdentity graph_identity{
        .dimension = context.graph_dimension,
        .nodes = context.graph_nodes,
        .edges = context.graph_edges,
        .geometry_revision = context.graph_geometry_revision,
        .topology_revision = context.graph_topology_revision,
        .ownership_revision = context.graph_ownership_revision,
        .numbering_revision = context.graph_numbering_revision,
        .dof_layout_revision = context.graph_dof_layout_revision,
        .content_revision = context.graph_content_revision,
    };
    context.split_stage_provenance =
        level_set::LevelSetP1PhaseSplitStageProvenance{
            .scheme = level_set::LevelSetP1PhaseSplitScheme::
                BackwardEulerExplicitIndicatorEndpointVelocity,
            .transport_mesh_policy =
                level_set::LevelSetP1PhaseTransportMeshPolicy::
                    FixedBackground,
            .temporal_order = 1,
            .prospective_step = context.accepted_step,
            .attempt = 2u,
            .step_start_time = 0.25,
            .step_end_time = context.accepted_time,
            .q_input_time = 0.25,
            .velocity_state_time = context.accepted_time,
            .time_step = context.time_step,
            .operator_state_revision = 31u,
            .previous_q_revision =
                level_set::levelSetP1PhaseScalarContentRevision(previous),
            .nodal_velocity_revision =
                level_set::levelSetP1PhaseVelocityContentRevision(
                    stage.sampled_nodal_velocity),
            .previous_graph_identity = graph_identity,
            .operator_graph_identity = graph_identity,
            .final_flux_ledger_digest =
                level_set::levelSetP1PhaseFluxLedgerDigest(stage),
            .stage_options = stage.executed_options,
        };
    context.reinitialization.diagnostic = "not due \"quoted\"";
    context.reinitialization.max_prescribed_contact_value_residual =
        0.0123;
    context.reinitialization
        .max_prescribed_contact_angle_error_radians = 0.0456;
    context.reconciliation.diagnostic = "no correction required";
    const std::vector<level_set::LevelSetPhaseRegionDefinition> regions{
        level_set::LevelSetPhaseRegionDefinition{
            .name = "film",
            .kind = level_set::LevelSetPhaseRegionKind::WallFilm,
            .node_membership = {1u, 0u},
        },
    };
    context.region_ledger = level_set::buildLevelSetPhaseRegionLedgers(
        stage.correction, regions);
    ASSERT_TRUE(context.region_ledger->success)
        << context.region_ledger->diagnostic;

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
    EXPECT_EQ(artifact.tracked_regions, 1u);

    std::ifstream input(artifact.path);
    ASSERT_TRUE(input.is_open());
    const std::string contents{
        std::istreambuf_iterator<char>{input},
        std::istreambuf_iterator<char>{}};
    EXPECT_NE(contents.find("\"artifact_schema_version\":3"),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"artifact\":\"conservative_phase_flux_ledger\""),
              std::string::npos);
    EXPECT_NE(contents.find("\"accepted_step\":7"),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"indicator_update\":\"explicit_q_n\""),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"indicator_bounds\":\"clamped_q_n_one_ring\""),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"velocity_sample\":\"operator_endpoint_u_np1\""),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"boundary_flux_contract\":{\"scope\":\"discrete_q_flux_not_pointwise_velocity_normal\""),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"maximum_nodal_boundary_mass_transfer\":0.01"),
              std::string::npos);
    EXPECT_NE(contents.find("\"boundary_mass_tolerance\":0.025"),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"limitation\":\"blind_to_velocity_normal_where_q_is_zero\""),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"absolute_total_within_tolerance\":true"),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"replicated_stage_inputs_satisfied\":true"),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"stage_options\":{\"invariant_tolerance\":"),
              std::string::npos);
    EXPECT_NE(contents.find("\"sampled_velocity\":[1,0,0]"),
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
                  "\"regions\":[\"film\"],\"courant\":"),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"regions\":[{\"name\":\"film\",\"kind\":\"wall_film\""),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"low_order_mass_transfer_into_region\":"),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"reinitialization_diagnostic\":\"not due \\\"quoted\\\"\""),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"reinitialization_max_prescribed_contact_value_residual\":"),
              std::string::npos);
    EXPECT_NE(contents.find(
                  "\"reinitialization_max_prescribed_contact_angle_error_radians\":"),
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

    auto wrong_boundary_scope_context = context;
    wrong_boundary_scope_context.accepted_step = 13u;
    wrong_boundary_scope_context.split_stage_provenance->prospective_step =
        13u;
    wrong_boundary_scope_context.boundary_flux_scope =
        "pointwise_velocity_normal";
    const auto wrong_boundary_scope =
        level_set::writeLevelSetConservativePhaseArtifact(
            output_directory, wrong_boundary_scope_context, stage);
    EXPECT_FALSE(wrong_boundary_scope.success);

    auto understated_boundary_maximum_context = context;
    understated_boundary_maximum_context.accepted_step = 14u;
    understated_boundary_maximum_context.split_stage_provenance
        ->prospective_step = 14u;
    understated_boundary_maximum_context
        .maximum_nodal_boundary_mass_transfer = 0.005;
    const auto understated_boundary_maximum =
        level_set::writeLevelSetConservativePhaseArtifact(
            output_directory,
            understated_boundary_maximum_context,
            stage);
    EXPECT_FALSE(understated_boundary_maximum.success);

    auto non_one_ring_stage = stage;
    non_one_ring_stage.correction.nodes.front()
        .lower_liquid_indicator = 0.0;
    auto non_one_ring_context = context;
    non_one_ring_context.accepted_step = 17u;
    non_one_ring_context.split_stage_provenance->prospective_step = 17u;
    non_one_ring_context.split_stage_provenance
        ->final_flux_ledger_digest =
        level_set::levelSetP1PhaseFluxLedgerDigest(non_one_ring_stage);
    const auto non_one_ring =
        level_set::writeLevelSetConservativePhaseArtifact(
            output_directory, non_one_ring_context, non_one_ring_stage);
    EXPECT_FALSE(non_one_ring.success);

    auto excessive_total_boundary_transfer_context = context;
    excessive_total_boundary_transfer_context.accepted_step = 15u;
    excessive_total_boundary_transfer_context.split_stage_provenance
        ->prospective_step = 15u;
    excessive_total_boundary_transfer_context.boundary_mass_tolerance =
        0.015;
    const auto excessive_total_boundary_transfer =
        level_set::writeLevelSetConservativePhaseArtifact(
            output_directory,
            excessive_total_boundary_transfer_context,
            stage);
    EXPECT_FALSE(excessive_total_boundary_transfer.success);

    auto understated_total_stage = stage;
    understated_total_stage.correction
        .total_physical_boundary_mass_transfer = 0.0;
    auto understated_total_context = context;
    understated_total_context.accepted_step = 18u;
    understated_total_context.split_stage_provenance->prospective_step =
        18u;
    understated_total_context.boundary_mass_tolerance = 0.015;
    understated_total_context.split_stage_provenance
        ->final_flux_ledger_digest =
        level_set::levelSetP1PhaseFluxLedgerDigest(
            understated_total_stage);
    const auto understated_total =
        level_set::writeLevelSetConservativePhaseArtifact(
            output_directory,
            understated_total_context,
            understated_total_stage);
    EXPECT_FALSE(understated_total.success);

    auto nonfinite_contact_history_context = context;
    nonfinite_contact_history_context.accepted_step = 16u;
    nonfinite_contact_history_context.split_stage_provenance
        ->prospective_step = 16u;
    nonfinite_contact_history_context.reinitialization
        .max_prescribed_contact_angle_error_radians =
        std::numeric_limits<FE::Real>::infinity();
    const auto nonfinite_contact_history =
        level_set::writeLevelSetConservativePhaseArtifact(
            output_directory, nonfinite_contact_history_context, stage);
    EXPECT_FALSE(nonfinite_contact_history.success);

    auto mismatched_stage_context = context;
    mismatched_stage_context.accepted_step = 12u;
    mismatched_stage_context.split_stage_provenance->prospective_step = 12u;
    mismatched_stage_context.split_stage_provenance
        ->final_flux_ledger_digest ^= 1u;
    const auto mismatched_stage =
        level_set::writeLevelSetConservativePhaseArtifact(
            output_directory, mismatched_stage_context, stage);
    EXPECT_FALSE(mismatched_stage.success);
    EXPECT_NE(mismatched_stage.diagnostic.find(
                  "finite validated accepted stage"),
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

    auto malformed_region_context = context;
    malformed_region_context.accepted_step = 11u;
    malformed_region_context.region_ledger->regions.front()
        .crossing_edges.front().second_node = 0;
    const auto malformed_region =
        level_set::writeLevelSetConservativePhaseArtifact(
            output_directory, malformed_region_context, stage);
    EXPECT_FALSE(malformed_region.success);
    EXPECT_FALSE(std::filesystem::exists(malformed_region.path));

    auto stale_context = context;
    stale_context.accepted_step = 10u;
    stale_context.split_stage_provenance->prospective_step = 10u;
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
