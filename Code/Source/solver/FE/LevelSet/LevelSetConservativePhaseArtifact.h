#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Atomic machine-readable output for conservative phase-flux stages.
 */

#include "Core/Types.h"
#include "LevelSet/LevelSetConservativePhaseOperator.h"
#include "LevelSet/LevelSetConservativePhaseRegions.h"
#include "LevelSet/LevelSetReinitialization.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace svmp::FE::level_set {

struct LevelSetConservativePhaseMismatchArtifact {
    Real maximum_nodal_residual{0.0};
    Real residual_norm{0.0};
    Real total_residual{0.0};
};

struct LevelSetConservativePhaseReconciliationArtifact {
    bool success{false};
    bool target_reached{false};
    bool limited_by_displacement{false};
    bool limited_by_topology{false};
    int iterations{0};
    int line_search_evaluations{0};
    int geometry_refresh_requests{0};
    int geometry_rebuilds{0};
    int rejected_geometry_trials{0};
    std::size_t contact_protected_nodes{0u};
    Real allowed_interface_displacement{0.0};
    Real accumulated_interface_displacement_bound{0.0};
    Real initial_residual_norm{0.0};
    Real final_residual_norm{0.0};
    Real maximum_final_nodal_residual{0.0};
    Real final_total_residual{0.0};
    Real maximum_removed_contact_increment{0.0};
    std::string last_rejected_trial_diagnostic{};
    std::string diagnostic{};
};

/**
 * @brief Accepted-state provenance and maintenance measures for one flux file.
 */
struct LevelSetConservativePhaseArtifactContext {
    std::string phase_field_name{};
    std::string level_set_field_name{};
    std::string geometry_domain_id{};
    std::uint64_t accepted_step{0u};
    Real accepted_time{0.0};
    Real time_step{0.0};
    std::uint64_t state_revision{0u};
    int graph_dimension{0};
    std::size_t graph_nodes{0u};
    std::size_t graph_edges{0u};
    /// Output-rank mesh cache stamps. These four values are diagnostic local
    /// provenance, not communicator-wide graph identities.
    std::uint64_t graph_geometry_revision{0u};
    std::uint64_t graph_topology_revision{0u};
    std::uint64_t graph_ownership_revision{0u};
    std::uint64_t graph_numbering_revision{0u};
    /// Communicator-replicated FE field-layout identity.
    std::uint64_t graph_dof_layout_revision{0u};
    /// Versioned, execution-layout-sensitive fingerprint of graph
    /// coefficients, communicator layout, and logical edge ownership.
    std::uint64_t graph_content_revision{0u};
    /**
     * Operator-stage binding for the accepted phase update.  Schema-3
     * artifacts require the fixed-background, endpoint-velocity
     * Backward-Euler split and exact clamped q^n one-ring bounds implemented
     * by the production application. The lower-level advance API remains
     * generic and may execute other caller-supplied bounds.
     */
    std::optional<LevelSetP1PhaseSplitStageProvenance>
        split_stage_provenance{};
    /**
     * Accepted discrete phase-indicator boundary-flux evidence.  This is the
     * nodal q-flux ledger produced by the conservative transport stage; it is
     * not a pointwise velocity-normal boundary condition or an ALE mesh-flux
     * claim, and it is blind to velocity-normal leakage where q is zero.
     */
    Real maximum_nodal_boundary_mass_transfer{0.0};
    Real boundary_mass_tolerance{0.0};
    std::string boundary_flux_scope{};
    bool geometry_validated_before_commit{false};
    bool reinitialization_due{false};
    bool reinitialization_applied{false};
    LevelSetSignedDistanceRepairResult reinitialization{};
    LevelSetConservativePhaseReconciliationArtifact reconciliation{};
    Real raw_post_transport_phase_measure{0.0};
    Real post_limit_phase_measure{0.0};
    Real raw_post_transport_geometry_measure{0.0};
    Real post_reinitialization_phase_measure{0.0};
    Real post_reinitialization_geometry_measure{0.0};
    LevelSetConservativePhaseMismatchArtifact
        post_reinitialization_mismatch{};
    Real post_correction_phase_measure{0.0};
    Real post_correction_geometry_measure{0.0};
    LevelSetConservativePhaseMismatchArtifact post_correction_mismatch{};
    Real retained_assembly_geometry_measure{0.0};
    std::optional<LevelSetPhaseRegionLedgerResult> region_ledger{};
};

struct LevelSetConservativePhaseArtifactResult {
    bool success{false};
    std::filesystem::path path{};
    std::uintmax_t bytes{0u};
    std::size_t nodes{0u};
    std::size_t edges{0u};
    std::size_t resolved_components{0u};
    std::size_t tracked_regions{0u};
    bool subthreshold_component_present{false};
    std::string diagnostic{};
};

/**
 * @brief Atomically write the complete accepted transport and maintenance
 * ledger as one JSON artifact.
 *
 * The destination file is never silently replaced. A temporary sibling is
 * fully closed before an atomic no-replacement publication. Callers should
 * invoke this only for an accepted transaction and only on the logical output
 * rank.
 */
[[nodiscard]] LevelSetConservativePhaseArtifactResult
writeLevelSetConservativePhaseArtifact(
    const std::filesystem::path& output_directory,
    const LevelSetConservativePhaseArtifactContext& context,
    const LevelSetP1PhaseTransportStageResult& stage);

} // namespace svmp::FE::level_set
