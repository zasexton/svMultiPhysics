#include "LevelSet/LevelSetConservativePhaseArtifact.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <string_view>
#include <system_error>

namespace svmp::FE::level_set {
namespace {

[[nodiscard]] std::string safeFileToken(std::string_view value)
{
    std::string token;
    token.reserve(value.size());
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        if (std::isalnum(character) != 0 || character == '-' ||
            character == '_') {
            token.push_back(static_cast<char>(character));
        } else {
            token.push_back('_');
        }
    }
    if (token.empty()) {
        token = "phase";
    }
    return token;
}

void writeJsonString(std::ostream& output, std::string_view value)
{
    output << '"';
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        switch (character) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (character < 0x20u) {
                output << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0')
                       << static_cast<unsigned int>(character)
                       << std::dec << std::setfill(' ');
            } else {
                output << static_cast<char>(character);
            }
            break;
        }
    }
    output << '"';
}

void writeBool(std::ostream& output, bool value)
{
    output << (value ? "true" : "false");
}

template <typename... Values>
[[nodiscard]] bool finiteValues(Values... values) noexcept
{
    return (... && std::isfinite(static_cast<Real>(values)));
}

[[nodiscard]] bool finiteMismatch(
    const LevelSetConservativePhaseMismatchArtifact& mismatch) noexcept
{
    return finiteValues(mismatch.maximum_nodal_residual,
                        mismatch.residual_norm,
                        mismatch.total_residual);
}

[[nodiscard]] bool finiteReconciliation(
    const LevelSetConservativePhaseReconciliationArtifact& reconciliation)
    noexcept
{
    return finiteValues(
        reconciliation.allowed_interface_displacement,
        reconciliation.accumulated_interface_displacement_bound,
        reconciliation.initial_residual_norm,
        reconciliation.final_residual_norm,
        reconciliation.maximum_final_nodal_residual,
        reconciliation.final_total_residual,
        reconciliation.maximum_removed_contact_increment);
}

[[nodiscard]] bool finiteReinitialization(
    const LevelSetSignedDistanceRepairResult& reinitialization) noexcept
{
    return finiteValues(
        reinitialization.max_abs_update,
        reinitialization.max_distance,
        reinitialization.max_interface_displacement,
        reinitialization.l2_interface_displacement,
        reinitialization.max_iteration_residual,
        reinitialization.max_signed_distance_error,
        reinitialization.max_unconstrained_signed_distance_error,
        reinitialization.max_wall_constrained_signed_distance_error,
        reinitialization.max_wall_contact_scale_residual,
        reinitialization.max_contact_line_displacement,
        reinitialization.max_contact_angle_change_radians,
        reinitialization.preserve_band_width);
}

[[nodiscard]] bool finiteRegionLedgerResult(
    const LevelSetPhaseRegionLedgerResult& result) noexcept
{
    if (!result.success || !result.all_balances_satisfied ||
        !finiteValues(result.maximum_balance_residual,
                      result.maximum_flux_reconstruction_residual)) {
        return false;
    }
    for (const auto& region : result.regions) {
        if (region.name.empty() || !region.balance_satisfied ||
            !finiteValues(
                region.previous_liquid_measure,
                region.low_order_liquid_measure,
                region.raw_target_liquid_measure,
                region.limited_liquid_measure,
                region.physical_boundary_mass_transfer,
                region.discrete_divergence_mass_source,
                region.low_order_nodal_interior_mass_transfer,
                region.raw_nodal_antidiffusive_mass_transfer,
                region.limited_nodal_antidiffusive_mass_transfer,
                region.low_order_crossing_mass_transfer,
                region.raw_crossing_antidiffusive_mass_transfer,
                region.limited_crossing_antidiffusive_mass_transfer,
                region.low_order_flux_reconstruction_residual,
                region.raw_flux_reconstruction_residual,
                region.limited_flux_reconstruction_residual,
                region.low_order_balance_residual,
                region.raw_target_balance_residual,
                region.limited_balance_residual,
                region.maximum_internal_pair_cancellation_residual)) {
            return false;
        }
        for (const auto& edge : region.crossing_edges) {
            if (!finiteValues(
                    edge.low_order_mass_transfer_into_region,
                    edge.raw_antidiffusive_mass_transfer_into_region,
                    edge.limited_antidiffusive_mass_transfer_into_region)) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] bool finiteContext(
    const LevelSetConservativePhaseArtifactContext& context) noexcept
{
    return finiteValues(context.accepted_time, context.time_step) &&
           context.time_step > Real{0.0} &&
           finiteReinitialization(context.reinitialization) &&
           finiteReconciliation(context.reconciliation) &&
           finiteMismatch(context.post_reinitialization_mismatch) &&
           finiteMismatch(context.post_correction_mismatch) &&
           (!context.region_ledger.has_value() ||
            finiteRegionLedgerResult(*context.region_ledger)) &&
           finiteValues(context.raw_post_transport_phase_measure,
                        context.post_limit_phase_measure,
                        context.raw_post_transport_geometry_measure,
                        context.post_reinitialization_phase_measure,
                        context.post_reinitialization_geometry_measure,
                        context.post_correction_phase_measure,
                        context.post_correction_geometry_measure,
                        context.retained_assembly_geometry_measure);
}

[[nodiscard]] bool finiteComponent(
    const LevelSetPhaseFluxComponentLedger& component) noexcept
{
    return finiteValues(component.previous_liquid_measure,
                        component.low_order_liquid_measure,
                        component.raw_target_liquid_measure,
                        component.limited_liquid_measure,
                        component.physical_boundary_mass_transfer,
                        component.discrete_divergence_mass_source,
                        component.low_order_interior_mass_transfer,
                        component.raw_antidiffusive_mass_transfer,
                        component.limited_antidiffusive_mass_transfer,
                        component.low_order_balance_residual,
                        component.raw_target_balance_residual,
                        component.limited_balance_residual);
}

[[nodiscard]] bool finiteStage(
    const LevelSetP1PhaseTransportStageResult& stage) noexcept
{
    const auto& correction = stage.correction;
    if (!finiteValues(stage.maximum_courant,
                      stage.minimum_low_order_coefficient,
                      stage.maximum_strong_form_decomposition_residual,
                      correction.total_previous_liquid_measure,
                      correction.total_low_order_liquid_measure,
                      correction.total_raw_target_liquid_measure,
                      correction.total_limited_liquid_measure,
                      correction.total_physical_boundary_mass_transfer,
                      correction.total_discrete_divergence_mass_source,
                      correction.low_order_nodal_cancellation_residual,
                      correction.raw_nodal_cancellation_residual,
                      correction.limited_nodal_cancellation_residual,
                      correction.maximum_edge_pair_cancellation_residual,
                      correction.maximum_low_order_local_mass_balance_residual,
                      correction.maximum_raw_target_local_mass_balance_residual,
                      correction.maximum_local_mass_balance_residual,
                      correction.maximum_component_balance_residual,
                      correction.component_activity_tolerance,
                      correction.previous_component_measure_closure_residual,
                      correction.low_order_component_measure_closure_residual,
                      correction.raw_target_component_measure_closure_residual,
                      correction.limited_component_measure_closure_residual,
                      correction.boundary_component_transfer_closure_residual,
                      correction.divergence_component_source_closure_residual,
                      correction.low_order_component_transfer_closure_residual,
                      correction.raw_component_transfer_closure_residual,
                      correction.limited_component_transfer_closure_residual,
                      correction.low_order_global_mass_balance_residual,
                      correction.raw_target_global_mass_balance_residual,
                      correction.global_mass_balance_residual,
                      correction.maximum_constant_preservation_error,
                      correction.minimum_low_order_liquid_indicator,
                      correction.maximum_low_order_liquid_indicator,
                      correction.minimum_raw_target_liquid_indicator,
                      correction.maximum_raw_target_liquid_indicator,
                      correction.minimum_limited_liquid_indicator,
                      correction.maximum_limited_liquid_indicator)) {
        return false;
    }
    if (stage.nodal_courant.size() != correction.nodes.size() ||
        stage.physical_boundary_mass_transfer.size() !=
            correction.nodes.size() ||
        stage.discrete_divergence_mass_source.size() !=
            correction.nodes.size() ||
        stage.flux_edges.size() != correction.edges.size()) {
        return false;
    }
    for (std::size_t index = 0u; index < correction.nodes.size(); ++index) {
        const auto& node = correction.nodes[index];
        if (!finiteValues(
                stage.nodal_courant[index],
                stage.physical_boundary_mass_transfer[index],
                stage.discrete_divergence_mass_source[index],
                node.lumped_control_volume,
                node.previous_liquid_indicator,
                node.lower_liquid_indicator,
                node.upper_liquid_indicator,
                node.physical_boundary_mass_transfer,
                node.discrete_divergence_mass_source,
                node.low_order_interior_mass_transfer,
                node.raw_antidiffusive_mass_transfer,
                node.limited_antidiffusive_mass_transfer,
                node.positive_raw_antidiffusive_mass,
                node.negative_raw_antidiffusive_mass,
                node.positive_correction_factor,
                node.negative_correction_factor,
                node.low_order_liquid_indicator,
                node.raw_target_liquid_indicator,
                node.limited_liquid_indicator,
                node.low_order_local_mass_balance_residual,
                node.raw_target_local_mass_balance_residual,
                node.local_mass_balance_residual) ||
            stage.physical_boundary_mass_transfer[index] !=
                node.physical_boundary_mass_transfer ||
            stage.discrete_divergence_mass_source[index] !=
                node.discrete_divergence_mass_source) {
            return false;
        }
    }
    for (std::size_t index = 0u; index < correction.edges.size(); ++index) {
        const auto& input_edge = stage.flux_edges[index];
        const auto& edge = correction.edges[index];
        if (!finiteValues(input_edge.low_order_mass_transfer,
                          input_edge.raw_antidiffusive_mass_transfer,
                          edge.low_order_mass_transfer,
                          edge.raw_antidiffusive_mass_transfer,
                          edge.correction_factor,
                          edge.limited_antidiffusive_mass_transfer,
                          edge.low_order_pair_cancellation_residual,
                          edge.raw_pair_cancellation_residual,
                          edge.limited_pair_cancellation_residual) ||
            input_edge.first_node != edge.first_node ||
            input_edge.second_node != edge.second_node ||
            input_edge.low_order_mass_transfer !=
                edge.low_order_mass_transfer ||
            input_edge.raw_antidiffusive_mass_transfer !=
                edge.raw_antidiffusive_mass_transfer) {
            return false;
        }
    }
    if (!std::all_of(correction.components.begin(),
                     correction.components.end(), finiteComponent)) {
        return false;
    }
    return !correction.subthreshold_component_present ||
           finiteComponent(correction.subthreshold_component);
}

[[nodiscard]] bool validRegionLedgerShape(
    const std::optional<LevelSetPhaseRegionLedgerResult>& result,
    const LevelSetPhaseFluxCorrectionResult& correction) noexcept
{
    if (!result.has_value()) {
        return true;
    }
    const auto node_count = correction.nodes.size();
    std::set<std::pair<GlobalIndex, GlobalIndex>> correction_edges;
    for (const auto& edge : correction.edges) {
        correction_edges.emplace(edge.first_node, edge.second_node);
    }
    std::set<std::string> names;
    for (const auto& region : result->regions) {
        if (std::string_view(levelSetPhaseRegionKindName(region.kind)) ==
                "unknown" ||
            !names.insert(region.name).second ||
            !std::is_sorted(region.member_nodes.begin(),
                            region.member_nodes.end()) ||
            std::adjacent_find(region.member_nodes.begin(),
                               region.member_nodes.end()) !=
                region.member_nodes.end()) {
            return false;
        }
        for (const auto node : region.member_nodes) {
            if (node < 0 || static_cast<std::size_t>(node) >= node_count) {
                return false;
            }
        }
        std::size_t expected_internal_edges = 0u;
        std::size_t expected_crossing_edges = 0u;
        for (const auto& edge : correction.edges) {
            const bool first_inside = std::binary_search(
                region.member_nodes.begin(),
                region.member_nodes.end(), edge.first_node);
            const bool second_inside = std::binary_search(
                region.member_nodes.begin(),
                region.member_nodes.end(), edge.second_node);
            expected_internal_edges += first_inside && second_inside ? 1u : 0u;
            expected_crossing_edges += first_inside != second_inside ? 1u : 0u;
        }
        if (region.internal_edges != expected_internal_edges ||
            region.crossing_edges.size() != expected_crossing_edges) {
            return false;
        }
        std::set<std::pair<GlobalIndex, GlobalIndex>> seen_crossing_edges;
        for (const auto& edge : region.crossing_edges) {
            if (edge.first_node < 0 ||
                edge.second_node <= edge.first_node ||
                static_cast<std::size_t>(edge.second_node) >= node_count ||
                !correction_edges.contains(
                    {edge.first_node, edge.second_node}) ||
                !seen_crossing_edges.emplace(
                    edge.first_node, edge.second_node).second) {
                return false;
            }
            const bool first_inside = std::binary_search(
                region.member_nodes.begin(),
                region.member_nodes.end(), edge.first_node);
            const bool second_inside = std::binary_search(
                region.member_nodes.begin(),
                region.member_nodes.end(), edge.second_node);
            if (first_inside == second_inside) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] const char* reinitializationMethodName(
    LevelSetReinitializationMethod method) noexcept
{
    switch (method) {
    case LevelSetReinitializationMethod::HamiltonJacobiPDE:
        return "HamiltonJacobiPDE";
    case LevelSetReinitializationMethod::FastMarching:
        return "FastMarching";
    case LevelSetReinitializationMethod::Projection:
        return "Projection";
    }
    return "Unknown";
}

void writeComponent(std::ostream& output,
                    const LevelSetPhaseFluxComponentLedger& component,
                    std::string_view classification)
{
    output << '{' << "\"classification\":";
    writeJsonString(output, classification);
    output << ",\"component_id\":" << component.component_id
           << ",\"nodes\":" << component.nodes
           << ",\"previous_liquid_measure\":"
           << component.previous_liquid_measure
           << ",\"low_order_liquid_measure\":"
           << component.low_order_liquid_measure
           << ",\"raw_target_liquid_measure\":"
           << component.raw_target_liquid_measure
           << ",\"limited_liquid_measure\":"
           << component.limited_liquid_measure
           << ",\"physical_boundary_mass_transfer\":"
           << component.physical_boundary_mass_transfer
           << ",\"discrete_divergence_mass_source\":"
           << component.discrete_divergence_mass_source
           << ",\"low_order_interior_mass_transfer\":"
           << component.low_order_interior_mass_transfer
           << ",\"raw_antidiffusive_mass_transfer\":"
           << component.raw_antidiffusive_mass_transfer
           << ",\"limited_antidiffusive_mass_transfer\":"
           << component.limited_antidiffusive_mass_transfer
           << ",\"low_order_balance_residual\":"
           << component.low_order_balance_residual
           << ",\"raw_target_balance_residual\":"
           << component.raw_target_balance_residual
           << ",\"limited_balance_residual\":"
           << component.limited_balance_residual << '}';
}

void writeRegion(std::ostream& output,
                 const LevelSetPhaseRegionLedger& region)
{
    output << '{' << "\"name\":";
    writeJsonString(output, region.name);
    output << ",\"kind\":";
    writeJsonString(output, levelSetPhaseRegionKindName(region.kind));
    output << ",\"balance_satisfied\":";
    writeBool(output, region.balance_satisfied);
    output << ",\"member_nodes\":[";
    for (std::size_t index = 0u;
         index < region.member_nodes.size(); ++index) {
        if (index != 0u) {
            output << ',';
        }
        output << region.member_nodes[index];
    }
    output << "],\"internal_edges\":" << region.internal_edges
           << ",\"crossing_edges\":[";
    for (std::size_t index = 0u;
         index < region.crossing_edges.size(); ++index) {
        if (index != 0u) {
            output << ',';
        }
        const auto& edge = region.crossing_edges[index];
        output << "{\"first_node\":" << edge.first_node
               << ",\"second_node\":" << edge.second_node
               << ",\"low_order_mass_transfer_into_region\":"
               << edge.low_order_mass_transfer_into_region
               << ",\"raw_antidiffusive_mass_transfer_into_region\":"
               << edge.raw_antidiffusive_mass_transfer_into_region
               << ",\"limited_antidiffusive_mass_transfer_into_region\":"
               << edge.limited_antidiffusive_mass_transfer_into_region
               << '}';
    }
    output << "]"
           << ",\"previous_liquid_measure\":"
           << region.previous_liquid_measure
           << ",\"low_order_liquid_measure\":"
           << region.low_order_liquid_measure
           << ",\"raw_target_liquid_measure\":"
           << region.raw_target_liquid_measure
           << ",\"limited_liquid_measure\":"
           << region.limited_liquid_measure
           << ",\"physical_boundary_mass_transfer\":"
           << region.physical_boundary_mass_transfer
           << ",\"discrete_divergence_mass_source\":"
           << region.discrete_divergence_mass_source
           << ",\"low_order_nodal_interior_mass_transfer\":"
           << region.low_order_nodal_interior_mass_transfer
           << ",\"raw_nodal_antidiffusive_mass_transfer\":"
           << region.raw_nodal_antidiffusive_mass_transfer
           << ",\"limited_nodal_antidiffusive_mass_transfer\":"
           << region.limited_nodal_antidiffusive_mass_transfer
           << ",\"low_order_crossing_mass_transfer\":"
           << region.low_order_crossing_mass_transfer
           << ",\"raw_crossing_antidiffusive_mass_transfer\":"
           << region.raw_crossing_antidiffusive_mass_transfer
           << ",\"limited_crossing_antidiffusive_mass_transfer\":"
           << region.limited_crossing_antidiffusive_mass_transfer
           << ",\"low_order_flux_reconstruction_residual\":"
           << region.low_order_flux_reconstruction_residual
           << ",\"raw_flux_reconstruction_residual\":"
           << region.raw_flux_reconstruction_residual
           << ",\"limited_flux_reconstruction_residual\":"
           << region.limited_flux_reconstruction_residual
           << ",\"low_order_balance_residual\":"
           << region.low_order_balance_residual
           << ",\"raw_target_balance_residual\":"
           << region.raw_target_balance_residual
           << ",\"limited_balance_residual\":"
           << region.limited_balance_residual
           << ",\"maximum_internal_pair_cancellation_residual\":"
           << region.maximum_internal_pair_cancellation_residual << '}';
}

void writeArtifact(
    std::ostream& output,
    const LevelSetConservativePhaseArtifactContext& context,
    const LevelSetP1PhaseTransportStageResult& stage)
{
    const auto& correction = stage.correction;
    output << std::setprecision(std::numeric_limits<Real>::max_digits10);
    output << "{\"artifact_schema_version\":2"
           << ",\"artifact\":\"conservative_phase_flux_ledger\""
           << ",\"phase_field\":";
    writeJsonString(output, context.phase_field_name);
    output << ",\"level_set_field\":";
    writeJsonString(output, context.level_set_field_name);
    output << ",\"geometry_domain_id\":";
    writeJsonString(output, context.geometry_domain_id);
    output << ",\"accepted_step\":" << context.accepted_step
           << ",\"accepted_time\":" << context.accepted_time
           << ",\"time_step\":" << context.time_step
           << ",\"state_revision\":" << context.state_revision
           << ",\"graph_revisions\":{\"geometry\":"
           << context.graph_geometry_revision
           << ",\"topology\":" << context.graph_topology_revision
           << ",\"ownership\":" << context.graph_ownership_revision
           << ",\"numbering\":" << context.graph_numbering_revision
           << ",\"dof_layout\":"
           << context.graph_dof_layout_revision << '}'
           << ",\"maintenance_ordering\":\"conservative_phase_transport_then_raw_geometry_rebuild_then_wall_aware_reinitialization_then_local_geometry_reconciliation_then_validation_then_commit\""
           << ",\"geometry_validated_before_commit\":";
    writeBool(output, context.geometry_validated_before_commit);
    output << ",\"transport_summary\":{\"success\":";
    writeBool(output, stage.success);
    output << ",\"courant_satisfied\":";
    writeBool(output, stage.courant_satisfied);
    output << ",\"maximum_courant\":" << stage.maximum_courant
           << ",\"minimum_low_order_coefficient\":"
           << stage.minimum_low_order_coefficient
           << ",\"low_order_coefficients_nonnegative\":";
    writeBool(output, stage.low_order_coefficients_nonnegative);
    output << ",\"strong_form_decomposition_satisfied\":";
    writeBool(output, stage.strong_form_decomposition_satisfied);
    output << ",\"maximum_strong_form_decomposition_residual\":"
           << stage.maximum_strong_form_decomposition_residual
           << ",\"correction_success\":";
    writeBool(output, correction.success);
    output << ",\"correction_applied\":";
    writeBool(output, correction.applied);
    output << ",\"low_order_bounds_satisfied\":";
    writeBool(output, correction.low_order_bounds_satisfied);
    output << ",\"limited_bounds_satisfied\":";
    writeBool(output, correction.limited_bounds_satisfied);
    output << ",\"interior_cancellation_satisfied\":";
    writeBool(output, correction.interior_cancellation_satisfied);
    output << ",\"local_balance_satisfied\":";
    writeBool(output, correction.local_balance_satisfied);
    output << ",\"global_balance_satisfied\":";
    writeBool(output, correction.global_balance_satisfied);
    output << ",\"component_balance_satisfied\":";
    writeBool(output, correction.component_balance_satisfied);
    output << ",\"component_measure_closure_satisfied\":";
    writeBool(output, correction.component_measure_closure_satisfied);
    output << ",\"constant_preservation_satisfied\":";
    writeBool(output, correction.constant_preservation_satisfied);
    output << ",\"constant_state_input\":";
    writeBool(output, correction.constant_state_input);
    output << ",\"constant_preservation_required\":";
    writeBool(output, correction.constant_preservation_required);
    output << ",\"limited_edges\":" << correction.limited_edges
           << ",\"component_activity_tolerance\":"
           << correction.component_activity_tolerance
           << ",\"total_previous_liquid_measure\":"
           << correction.total_previous_liquid_measure
           << ",\"total_low_order_liquid_measure\":"
           << correction.total_low_order_liquid_measure
           << ",\"total_raw_target_liquid_measure\":"
           << correction.total_raw_target_liquid_measure
           << ",\"total_limited_liquid_measure\":"
           << correction.total_limited_liquid_measure
           << ",\"total_physical_boundary_mass_transfer\":"
           << correction.total_physical_boundary_mass_transfer
           << ",\"total_discrete_divergence_mass_source\":"
           << correction.total_discrete_divergence_mass_source
           << ",\"low_order_nodal_cancellation_residual\":"
           << correction.low_order_nodal_cancellation_residual
           << ",\"raw_nodal_cancellation_residual\":"
           << correction.raw_nodal_cancellation_residual
           << ",\"limited_nodal_cancellation_residual\":"
           << correction.limited_nodal_cancellation_residual
           << ",\"maximum_edge_pair_cancellation_residual\":"
           << correction.maximum_edge_pair_cancellation_residual
           << ",\"maximum_low_order_local_mass_balance_residual\":"
           << correction.maximum_low_order_local_mass_balance_residual
           << ",\"maximum_raw_target_local_mass_balance_residual\":"
           << correction.maximum_raw_target_local_mass_balance_residual
           << ",\"maximum_local_mass_balance_residual\":"
           << correction.maximum_local_mass_balance_residual
           << ",\"global_mass_balance_residual\":"
           << correction.global_mass_balance_residual
           << ",\"maximum_component_balance_residual\":"
           << correction.maximum_component_balance_residual
           << ",\"previous_component_measure_closure_residual\":"
           << correction.previous_component_measure_closure_residual
           << ",\"low_order_component_measure_closure_residual\":"
           << correction.low_order_component_measure_closure_residual
           << ",\"raw_target_component_measure_closure_residual\":"
           << correction.raw_target_component_measure_closure_residual
           << ",\"limited_component_measure_closure_residual\":"
           << correction.limited_component_measure_closure_residual
           << ",\"boundary_component_transfer_closure_residual\":"
           << correction.boundary_component_transfer_closure_residual
           << ",\"divergence_component_source_closure_residual\":"
           << correction.divergence_component_source_closure_residual
           << ",\"limited_component_transfer_closure_residual\":"
           << correction.limited_component_transfer_closure_residual
           << ",\"low_order_component_transfer_closure_residual\":"
           << correction.low_order_component_transfer_closure_residual
           << ",\"raw_component_transfer_closure_residual\":"
           << correction.raw_component_transfer_closure_residual
           << ",\"low_order_global_mass_balance_residual\":"
           << correction.low_order_global_mass_balance_residual
           << ",\"raw_target_global_mass_balance_residual\":"
           << correction.raw_target_global_mass_balance_residual
           << ",\"maximum_constant_preservation_error\":"
           << correction.maximum_constant_preservation_error
           << ",\"minimum_low_order_liquid_indicator\":"
           << correction.minimum_low_order_liquid_indicator
           << ",\"maximum_low_order_liquid_indicator\":"
           << correction.maximum_low_order_liquid_indicator
           << ",\"minimum_raw_target_liquid_indicator\":"
           << correction.minimum_raw_target_liquid_indicator
           << ",\"maximum_raw_target_liquid_indicator\":"
           << correction.maximum_raw_target_liquid_indicator
           << ",\"minimum_limited_liquid_indicator\":"
           << correction.minimum_limited_liquid_indicator
           << ",\"maximum_limited_liquid_indicator\":"
           << correction.maximum_limited_liquid_indicator
           << ",\"nodes\":" << correction.nodes.size()
           << ",\"edges\":" << correction.edges.size()
           << ",\"resolved_components\":"
           << correction.components.size()
           << ",\"subthreshold_component_present\":";
    writeBool(output, correction.subthreshold_component_present);
    output << ",\"region_ledger_present\":";
    writeBool(output, context.region_ledger.has_value());
    output << ",\"tracked_regions\":"
           << (context.region_ledger.has_value()
                   ? context.region_ledger->regions.size()
                   : 0u)
           << ",\"region_balances_satisfied\":";
    writeBool(output,
              !context.region_ledger.has_value() ||
                  context.region_ledger->all_balances_satisfied);
    output << ",\"maximum_region_balance_residual\":"
           << (context.region_ledger.has_value()
                   ? context.region_ledger->maximum_balance_residual
                   : Real{0.0})
           << ",\"maximum_region_flux_reconstruction_residual\":"
           << (context.region_ledger.has_value()
                   ? context.region_ledger
                         ->maximum_flux_reconstruction_residual
                   : Real{0.0});
    output << "},\"maintenance_summary\":{\"reinitialization_due\":";
    writeBool(output, context.reinitialization_due);
    output << ",\"reinitialization_applied\":";
    writeBool(output, context.reinitialization_applied);
    const auto& reinitialization = context.reinitialization;
    const auto& reconciliation = context.reconciliation;
    output << ",\"reinitialization_success\":";
    writeBool(output, reinitialization.success);
    output << ",\"reinitialization_method\":";
    writeJsonString(
        output, reinitializationMethodName(reinitialization.method));
    output << ",\"reinitialization_converged\":";
    writeBool(output, reinitialization.converged);
    output << ",\"reinitialization_iterations\":"
           << reinitialization.iterations
           << ",\"reinitialization_repaired_dofs\":"
           << reinitialization.repaired_dofs
           << ",\"reinitialization_preserved_dofs\":"
           << reinitialization.preserved_dofs
           << ",\"reinitialization_interface_fragments\":"
           << reinitialization.interface_fragments
           << ",\"reinitialization_cut_cells\":"
           << reinitialization.cut_cells
           << ",\"reinitialization_interface_displacement_samples\":"
           << reinitialization.interface_displacement_samples
           << ",\"reinitialization_max_abs_update\":"
           << reinitialization.max_abs_update
           << ",\"reinitialization_max_distance\":"
           << reinitialization.max_distance
           << ",\"reinitialization_max_interface_displacement\":"
           << reinitialization.max_interface_displacement
           << ",\"reinitialization_l2_interface_displacement\":"
           << reinitialization.l2_interface_displacement
           << ",\"reinitialization_zero_set_bound_satisfied\":";
    writeBool(output, reinitialization.zero_set_bound_satisfied);
    output << ",\"reinitialization_max_iteration_residual\":"
           << reinitialization.max_iteration_residual
           << ",\"reinitialization_max_signed_distance_error\":"
           << reinitialization.max_signed_distance_error
           << ",\"reinitialization_max_unconstrained_signed_distance_error\":"
           << reinitialization.max_unconstrained_signed_distance_error
           << ",\"reinitialization_max_wall_constrained_signed_distance_error\":"
           << reinitialization.max_wall_constrained_signed_distance_error
           << ",\"reinitialization_wall_contact_constraints\":"
           << reinitialization.wall_contact_constraints
           << ",\"reinitialization_wall_contact_cells\":"
           << reinitialization.wall_contact_cells
           << ",\"reinitialization_wall_contact_dofs\":"
           << reinitialization.wall_contact_dofs
           << ",\"reinitialization_wall_contact_constraints_satisfied\":";
    writeBool(output, reinitialization.wall_contact_constraints_satisfied);
    output << ",\"reinitialization_max_wall_contact_scale_residual\":"
           << reinitialization.max_wall_contact_scale_residual
           << ",\"reinitialization_max_contact_line_displacement\":"
           << reinitialization.max_contact_line_displacement
           << ",\"reinitialization_max_contact_angle_change_radians\":"
           << reinitialization.max_contact_angle_change_radians
           << ",\"reinitialization_preserve_band_width\":"
           << reinitialization.preserve_band_width
           << ",\"reinitialization_diagnostic\":";
    writeJsonString(output, reinitialization.diagnostic);
    output << ",\"reconciliation_success\":";
    writeBool(output, reconciliation.success);
    output << ",\"reconciliation_target_reached\":";
    writeBool(output, reconciliation.target_reached);
    output << ",\"reconciliation_limited_by_displacement\":";
    writeBool(output, reconciliation.limited_by_displacement);
    output << ",\"reconciliation_limited_by_topology\":";
    writeBool(output, reconciliation.limited_by_topology);
    output
           << ",\"reconciliation_iterations\":"
           << reconciliation.iterations
           << ",\"reconciliation_line_search_evaluations\":"
           << reconciliation.line_search_evaluations
           << ",\"reconciliation_geometry_refresh_requests\":"
           << reconciliation.geometry_refresh_requests
           << ",\"reconciliation_geometry_rebuilds\":"
           << reconciliation.geometry_rebuilds
           << ",\"reconciliation_rejected_geometry_trials\":"
           << reconciliation.rejected_geometry_trials
           << ",\"reconciliation_contact_protected_nodes\":"
           << reconciliation.contact_protected_nodes
           << ",\"reconciliation_allowed_interface_displacement\":"
           << reconciliation.allowed_interface_displacement
           << ",\"reconciliation_interface_displacement_bound\":"
           << reconciliation.accumulated_interface_displacement_bound
           << ",\"reconciliation_initial_residual_norm\":"
           << reconciliation.initial_residual_norm
           << ",\"reconciliation_final_residual_norm\":"
           << reconciliation.final_residual_norm
           << ",\"reconciliation_maximum_final_nodal_residual\":"
           << reconciliation.maximum_final_nodal_residual
           << ",\"reconciliation_final_total_residual\":"
           << reconciliation.final_total_residual
           << ",\"reconciliation_maximum_removed_contact_increment\":"
           << reconciliation.maximum_removed_contact_increment
           << ",\"reconciliation_last_rejected_trial_diagnostic\":";
    writeJsonString(output,
                    reconciliation.last_rejected_trial_diagnostic);
    output << ",\"reconciliation_diagnostic\":";
    writeJsonString(output, reconciliation.diagnostic);
    output
           << ",\"raw_post_transport_phase_measure\":"
           << context.raw_post_transport_phase_measure
           << ",\"post_limit_phase_measure\":"
           << context.post_limit_phase_measure
           << ",\"raw_post_transport_geometry_measure\":"
           << context.raw_post_transport_geometry_measure
           << ",\"post_reinitialization_phase_measure\":"
           << context.post_reinitialization_phase_measure
           << ",\"post_reinitialization_geometry_measure\":"
           << context.post_reinitialization_geometry_measure
           << ",\"post_reinitialization_max_nodal_mismatch\":"
           << context.post_reinitialization_mismatch
                  .maximum_nodal_residual
           << ",\"post_reinitialization_mismatch_residual_norm\":"
           << context.post_reinitialization_mismatch.residual_norm
           << ",\"post_reinitialization_total_mismatch\":"
           << context.post_reinitialization_mismatch.total_residual
           << ",\"post_correction_phase_measure\":"
           << context.post_correction_phase_measure
           << ",\"post_correction_geometry_measure\":"
           << context.post_correction_geometry_measure
           << ",\"post_correction_max_nodal_mismatch\":"
           << context.post_correction_mismatch.maximum_nodal_residual
           << ",\"post_correction_mismatch_residual_norm\":"
           << context.post_correction_mismatch.residual_norm
           << ",\"post_correction_total_mismatch\":"
           << context.post_correction_mismatch.total_residual
           << ",\"retained_assembly_geometry_measure\":"
           << context.retained_assembly_geometry_measure << '}';

    output << ",\"nodes\":[";
    for (std::size_t index = 0u; index < correction.nodes.size(); ++index) {
        if (index != 0u) {
            output << ',';
        }
        const auto& node = correction.nodes[index];
        output << "{\"node\":" << node.node
               << ",\"component_id\":"
               << correction.node_component_ids[index]
               << ",\"regions\":[";
        bool first_region = true;
        if (context.region_ledger.has_value()) {
            for (const auto& region : context.region_ledger->regions) {
                if (!std::binary_search(
                        region.member_nodes.begin(),
                        region.member_nodes.end(), node.node)) {
                    continue;
                }
                if (!first_region) {
                    output << ',';
                }
                writeJsonString(output, region.name);
                first_region = false;
            }
        }
        output << "]"
               << ",\"courant\":" << stage.nodal_courant[index]
               << ",\"lumped_control_volume\":"
               << node.lumped_control_volume
               << ",\"previous_liquid_indicator\":"
               << node.previous_liquid_indicator
               << ",\"lower_liquid_indicator\":"
               << node.lower_liquid_indicator
               << ",\"upper_liquid_indicator\":"
               << node.upper_liquid_indicator
               << ",\"physical_boundary_mass_transfer\":"
               << node.physical_boundary_mass_transfer
               << ",\"discrete_divergence_mass_source\":"
               << node.discrete_divergence_mass_source
               << ",\"low_order_interior_mass_transfer\":"
               << node.low_order_interior_mass_transfer
               << ",\"raw_antidiffusive_mass_transfer\":"
               << node.raw_antidiffusive_mass_transfer
               << ",\"limited_antidiffusive_mass_transfer\":"
               << node.limited_antidiffusive_mass_transfer
               << ",\"positive_raw_antidiffusive_mass\":"
               << node.positive_raw_antidiffusive_mass
               << ",\"negative_raw_antidiffusive_mass\":"
               << node.negative_raw_antidiffusive_mass
               << ",\"positive_correction_factor\":"
               << node.positive_correction_factor
               << ",\"negative_correction_factor\":"
               << node.negative_correction_factor
               << ",\"low_order_liquid_indicator\":"
               << node.low_order_liquid_indicator
               << ",\"raw_target_liquid_indicator\":"
               << node.raw_target_liquid_indicator
               << ",\"limited_liquid_indicator\":"
               << node.limited_liquid_indicator
               << ",\"low_order_local_mass_balance_residual\":"
               << node.low_order_local_mass_balance_residual
               << ",\"raw_target_local_mass_balance_residual\":"
               << node.raw_target_local_mass_balance_residual
               << ",\"limited_local_mass_balance_residual\":"
               << node.local_mass_balance_residual << '}';
    }
    output << "],\"edges\":[";
    for (std::size_t index = 0u; index < correction.edges.size(); ++index) {
        if (index != 0u) {
            output << ',';
        }
        const auto& edge = correction.edges[index];
        output << "{\"first_node\":" << edge.first_node
               << ",\"second_node\":" << edge.second_node
               << ",\"low_order_mass_transfer\":"
               << edge.low_order_mass_transfer
               << ",\"raw_antidiffusive_mass_transfer\":"
               << edge.raw_antidiffusive_mass_transfer
               << ",\"correction_factor\":" << edge.correction_factor
               << ",\"limited_antidiffusive_mass_transfer\":"
               << edge.limited_antidiffusive_mass_transfer
               << ",\"low_order_pair_cancellation_residual\":"
               << edge.low_order_pair_cancellation_residual
               << ",\"raw_pair_cancellation_residual\":"
               << edge.raw_pair_cancellation_residual
               << ",\"limited_pair_cancellation_residual\":"
               << edge.limited_pair_cancellation_residual << '}';
    }
    output << "],\"components\":[";
    bool first_component = true;
    for (const auto& component : correction.components) {
        if (!first_component) {
            output << ',';
        }
        writeComponent(output, component, "resolved");
        first_component = false;
    }
    if (correction.subthreshold_component_present) {
        if (!first_component) {
            output << ',';
        }
        writeComponent(output,
                       correction.subthreshold_component,
                       "subthreshold");
    }
    output << "],\"regions\":[";
    if (context.region_ledger.has_value()) {
        for (std::size_t index = 0u;
             index < context.region_ledger->regions.size(); ++index) {
            if (index != 0u) {
                output << ',';
            }
            writeRegion(output, context.region_ledger->regions[index]);
        }
    }
    output << "]}\n";
}

} // namespace

LevelSetConservativePhaseArtifactResult
writeLevelSetConservativePhaseArtifact(
    const std::filesystem::path& output_directory,
    const LevelSetConservativePhaseArtifactContext& context,
    const LevelSetP1PhaseTransportStageResult& stage)
{
    LevelSetConservativePhaseArtifactResult result;
    result.nodes = stage.correction.nodes.size();
    result.edges = stage.correction.edges.size();
    result.resolved_components = stage.correction.components.size();
    result.subthreshold_component_present =
        stage.correction.subthreshold_component_present;
    result.tracked_regions = context.region_ledger.has_value()
        ? context.region_ledger->regions.size()
        : 0u;
    std::filesystem::path temporary_path;
    try {
        if (output_directory.empty() || context.phase_field_name.empty() ||
            context.level_set_field_name.empty() ||
            context.accepted_step == 0u) {
            result.diagnostic =
                "conservative phase artifact requires an output directory, nonempty field names, and a positive accepted step";
            return result;
        }
        if (!finiteContext(context) || !finiteStage(stage) ||
            !validRegionLedgerShape(
                context.region_ledger, stage.correction) ||
            !context.geometry_validated_before_commit || !stage.success ||
            !stage.correction.success ||
            stage.correction.node_component_ids.size() !=
                stage.correction.nodes.size() ||
            (context.reinitialization_applied &&
             (!context.reinitialization.success ||
              !context.reinitialization.converged))) {
            result.diagnostic =
                "conservative phase artifact requires one finite validated accepted stage with complete transport and maintenance ledgers";
            return result;
        }

        std::filesystem::create_directories(output_directory);
        std::ostringstream filename;
        filename << "conservative_phase_flux_"
                 << safeFileToken(context.phase_field_name)
                 << "_step_" << std::setw(8) << std::setfill('0')
                 << context.accepted_step << ".json";
        result.path = output_directory / filename.str();
        temporary_path = result.path;
        temporary_path += ".tmp";
        if (std::filesystem::exists(result.path) ||
            std::filesystem::exists(temporary_path)) {
            result.diagnostic =
                "conservative phase artifact refuses to replace an existing final or temporary file";
            return result;
        }

        bool write_failed = false;
        {
            std::ofstream output(
                temporary_path, std::ios::out | std::ios::trunc);
            if (!output.is_open()) {
                result.diagnostic =
                    "conservative phase artifact could not open its temporary file";
                return result;
            }
            writeArtifact(output, context, stage);
            output.flush();
            output.close();
            if (!output.good()) {
                result.diagnostic =
                    "conservative phase artifact failed while closing its temporary file";
                write_failed = true;
            }
        }
        if (write_failed) {
            std::error_code cleanup_error;
            std::filesystem::remove(temporary_path, cleanup_error);
            return result;
        }

        std::error_code publication_error;
        std::filesystem::create_hard_link(
            temporary_path, result.path, publication_error);
        if (publication_error) {
            result.diagnostic =
                "conservative phase artifact could not atomically publish without replacement: " +
                publication_error.message();
            std::error_code cleanup_error;
            std::filesystem::remove(temporary_path, cleanup_error);
            return result;
        }
        std::error_code cleanup_error;
        std::filesystem::remove(temporary_path, cleanup_error);
        if (cleanup_error) {
            result.diagnostic =
                "conservative phase artifact was published but its temporary link could not be removed: " +
                cleanup_error.message();
            std::error_code rollback_error;
            std::filesystem::remove(result.path, rollback_error);
            return result;
        }
        result.bytes = std::filesystem::file_size(result.path);
        result.success = true;
        result.diagnostic = "ok";
    } catch (const std::exception& exception) {
        if (!temporary_path.empty()) {
            std::error_code cleanup_error;
            std::filesystem::remove(temporary_path, cleanup_error);
        }
        result.diagnostic = exception.what();
    } catch (...) {
        if (!temporary_path.empty()) {
            std::error_code cleanup_error;
            std::filesystem::remove(temporary_path, cleanup_error);
        }
        result.diagnostic =
            "conservative phase artifact failed with an unknown exception";
    }
    return result;
}

} // namespace svmp::FE::level_set
