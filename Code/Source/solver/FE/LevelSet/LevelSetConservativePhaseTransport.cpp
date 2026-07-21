#include "LevelSet/LevelSetConservativePhaseTransport.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace svmp::FE::level_set {
namespace {

[[nodiscard]] Real scaledTolerance(Real tolerance,
                                   std::initializer_list<Real> values)
{
    Real scale = Real{1.0};
    for (const Real value : values) {
        scale = std::max(scale, std::abs(value));
    }
    return tolerance * scale;
}

[[nodiscard]] Real optionalNodeValue(std::span<const Real> values,
                                     std::size_t node) noexcept
{
    return values.empty() ? Real{0.0} : values[node];
}

[[nodiscard]] Real longDoubleSum(std::span<const Real> values)
{
    long double sum = 0.0L;
    for (const Real value : values) {
        sum += static_cast<long double>(value);
    }
    return static_cast<Real>(sum);
}

template <typename Projection>
[[nodiscard]] Real longDoubleNodeSum(
    const std::vector<LevelSetPhaseFluxNodeLedger>& nodes,
    Projection projection)
{
    long double sum = 0.0L;
    for (const auto& node : nodes) {
        sum += static_cast<long double>(projection(node));
    }
    return static_cast<Real>(sum);
}

[[nodiscard]] bool insideBounds(Real value,
                                Real lower,
                                Real upper,
                                Real tolerance) noexcept
{
    const Real slack = scaledTolerance(tolerance, {value, lower, upper});
    return value >= lower - slack && value <= upper + slack;
}

} // namespace

LevelSetPhaseFluxCorrectionResult
applyLevelSetConservativePhaseFluxCorrection(
    const LevelSetPhaseFluxStageView& stage)
{
    LevelSetPhaseFluxCorrectionResult result;
    result.constant_preservation_required =
        stage.require_constant_preservation;
    try {
        const std::size_t node_count = stage.lumped_control_volume.size();
        if (node_count == 0u) {
            result.diagnostic =
                "conservative phase-flux correction requires at least one node";
            return result;
        }
        if (stage.previous_liquid_indicator.size() != node_count ||
            stage.lower_liquid_indicator.size() != node_count ||
            stage.upper_liquid_indicator.size() != node_count) {
            result.diagnostic =
                "conservative phase-flux correction received inconsistent nodal spans";
            return result;
        }
        if ((!stage.physical_boundary_mass_transfer.empty() &&
             stage.physical_boundary_mass_transfer.size() != node_count) ||
            (!stage.discrete_divergence_mass_source.empty() &&
             stage.discrete_divergence_mass_source.size() != node_count)) {
            result.diagnostic =
                "conservative phase-flux correction received an incomplete boundary or divergence ledger";
            return result;
        }
        if (!std::isfinite(stage.invariant_tolerance) ||
            stage.invariant_tolerance < Real{0.0}) {
            result.diagnostic =
                "conservative phase-flux correction requires a finite nonnegative invariant tolerance";
            return result;
        }

        result.nodes.resize(node_count);
        std::vector<Real> low_order_internal(node_count, Real{0.0});
        std::vector<Real> raw_internal(node_count, Real{0.0});
        std::vector<Real> limited_internal(node_count, Real{0.0});
        std::vector<Real> positive_raw(node_count, Real{0.0});
        std::vector<Real> negative_raw(node_count, Real{0.0});
        std::vector<Real> positive_factor(node_count, Real{1.0});
        std::vector<Real> negative_factor(node_count, Real{1.0});

        for (std::size_t i = 0; i < node_count; ++i) {
            const Real volume = stage.lumped_control_volume[i];
            const Real previous = stage.previous_liquid_indicator[i];
            const Real lower = stage.lower_liquid_indicator[i];
            const Real upper = stage.upper_liquid_indicator[i];
            const Real boundary =
                optionalNodeValue(stage.physical_boundary_mass_transfer, i);
            const Real divergence =
                optionalNodeValue(stage.discrete_divergence_mass_source, i);
            if (!std::isfinite(volume) || !(volume > Real{0.0})) {
                result.diagnostic =
                    "conservative phase-flux correction requires finite positive lumped control volumes";
                return result;
            }
            if (!std::isfinite(previous) || !std::isfinite(lower) ||
                !std::isfinite(upper) || !std::isfinite(boundary) ||
                !std::isfinite(divergence)) {
                result.diagnostic =
                    "conservative phase-flux correction found a non-finite nodal quantity";
                return result;
            }
            const Real unit_slack = scaledTolerance(
                stage.invariant_tolerance, {lower, upper, previous});
            if (lower < -unit_slack || upper > Real{1.0} + unit_slack ||
                lower > upper ||
                !insideBounds(previous, lower, upper,
                              stage.invariant_tolerance)) {
                result.diagnostic =
                    "conservative phase-flux correction requires ordered unit-interval bounds containing the previous indicator";
                return result;
            }

            auto& ledger = result.nodes[i];
            ledger.node = static_cast<GlobalIndex>(i);
            ledger.lumped_control_volume = volume;
            ledger.previous_liquid_indicator = previous;
            ledger.lower_liquid_indicator = lower;
            ledger.upper_liquid_indicator = upper;
            ledger.physical_boundary_mass_transfer = boundary;
            ledger.discrete_divergence_mass_source = divergence;
        }

        std::set<std::pair<GlobalIndex, GlobalIndex>> unique_edges;
        result.edges.reserve(stage.interior_edges.size());
        for (const auto& edge : stage.interior_edges) {
            if (edge.first_node < 0 || edge.second_node < 0 ||
                edge.first_node >= static_cast<GlobalIndex>(node_count) ||
                edge.second_node >= static_cast<GlobalIndex>(node_count) ||
                edge.first_node >= edge.second_node) {
                result.diagnostic =
                    "conservative phase-flux correction requires canonical distinct in-range edge endpoints";
                return result;
            }
            if (!std::isfinite(edge.low_order_mass_transfer) ||
                !std::isfinite(edge.raw_antidiffusive_mass_transfer)) {
                result.diagnostic =
                    "conservative phase-flux correction found a non-finite edge transfer";
                return result;
            }
            if (!unique_edges.emplace(edge.first_node, edge.second_node).second) {
                result.diagnostic =
                    "conservative phase-flux correction found a duplicate algebraic edge";
                return result;
            }

            const auto first = static_cast<std::size_t>(edge.first_node);
            const auto second = static_cast<std::size_t>(edge.second_node);
            low_order_internal[first] += edge.low_order_mass_transfer;
            low_order_internal[second] -= edge.low_order_mass_transfer;
            raw_internal[first] += edge.raw_antidiffusive_mass_transfer;
            raw_internal[second] -= edge.raw_antidiffusive_mass_transfer;
            positive_raw[first] +=
                std::max(Real{0.0}, edge.raw_antidiffusive_mass_transfer);
            negative_raw[first] +=
                std::min(Real{0.0}, edge.raw_antidiffusive_mass_transfer);
            positive_raw[second] +=
                std::max(Real{0.0}, -edge.raw_antidiffusive_mass_transfer);
            negative_raw[second] +=
                std::min(Real{0.0}, -edge.raw_antidiffusive_mass_transfer);

            LevelSetPhaseFluxEdgeLedger edge_ledger;
            edge_ledger.first_node = edge.first_node;
            edge_ledger.second_node = edge.second_node;
            edge_ledger.low_order_mass_transfer =
                edge.low_order_mass_transfer;
            edge_ledger.raw_antidiffusive_mass_transfer =
                edge.raw_antidiffusive_mass_transfer;
            edge_ledger.low_order_pair_cancellation_residual =
                edge.low_order_mass_transfer +
                (-edge.low_order_mass_transfer);
            edge_ledger.raw_pair_cancellation_residual =
                edge.raw_antidiffusive_mass_transfer +
                (-edge.raw_antidiffusive_mass_transfer);
            result.edges.push_back(edge_ledger);
        }

        result.low_order_bounds_satisfied = true;
        result.minimum_low_order_liquid_indicator =
            std::numeric_limits<Real>::infinity();
        result.maximum_low_order_liquid_indicator =
            -std::numeric_limits<Real>::infinity();
        result.minimum_raw_target_liquid_indicator =
            std::numeric_limits<Real>::infinity();
        result.maximum_raw_target_liquid_indicator =
            -std::numeric_limits<Real>::infinity();
        Real previous_minimum = std::numeric_limits<Real>::infinity();
        Real previous_maximum = -std::numeric_limits<Real>::infinity();
        for (std::size_t i = 0; i < node_count; ++i) {
            if (!std::isfinite(low_order_internal[i]) ||
                !std::isfinite(raw_internal[i]) ||
                !std::isfinite(positive_raw[i]) ||
                !std::isfinite(negative_raw[i])) {
                result.diagnostic =
                    "conservative phase-flux correction overflowed a nodal edge accumulator";
                return result;
            }
            auto& ledger = result.nodes[i];
            ledger.low_order_interior_mass_transfer = low_order_internal[i];
            ledger.raw_antidiffusive_mass_transfer = raw_internal[i];
            ledger.positive_raw_antidiffusive_mass = positive_raw[i];
            ledger.negative_raw_antidiffusive_mass = negative_raw[i];
            const Real low_order_mass =
                ledger.lumped_control_volume *
                    ledger.previous_liquid_indicator +
                ledger.physical_boundary_mass_transfer +
                ledger.discrete_divergence_mass_source +
                ledger.low_order_interior_mass_transfer;
            ledger.low_order_liquid_indicator =
                low_order_mass / ledger.lumped_control_volume;
            ledger.raw_target_liquid_indicator =
                ledger.low_order_liquid_indicator +
                ledger.raw_antidiffusive_mass_transfer /
                    ledger.lumped_control_volume;
            if (!std::isfinite(ledger.low_order_liquid_indicator) ||
                !std::isfinite(ledger.raw_target_liquid_indicator)) {
                result.diagnostic =
                    "conservative phase-flux correction produced a non-finite predictor";
                return result;
            }
            result.low_order_bounds_satisfied =
                result.low_order_bounds_satisfied &&
                insideBounds(ledger.low_order_liquid_indicator,
                             ledger.lower_liquid_indicator,
                             ledger.upper_liquid_indicator,
                             stage.invariant_tolerance);
            result.minimum_low_order_liquid_indicator = std::min(
                result.minimum_low_order_liquid_indicator,
                ledger.low_order_liquid_indicator);
            result.maximum_low_order_liquid_indicator = std::max(
                result.maximum_low_order_liquid_indicator,
                ledger.low_order_liquid_indicator);
            result.minimum_raw_target_liquid_indicator = std::min(
                result.minimum_raw_target_liquid_indicator,
                ledger.raw_target_liquid_indicator);
            result.maximum_raw_target_liquid_indicator = std::max(
                result.maximum_raw_target_liquid_indicator,
                ledger.raw_target_liquid_indicator);
            previous_minimum =
                std::min(previous_minimum,
                         ledger.previous_liquid_indicator);
            previous_maximum =
                std::max(previous_maximum,
                         ledger.previous_liquid_indicator);
        }
        if (!result.low_order_bounds_satisfied) {
            result.diagnostic =
                "conservative phase-flux correction rejected an out-of-bounds low-order predictor";
            return result;
        }

        for (std::size_t i = 0; i < node_count; ++i) {
            const auto& ledger = result.nodes[i];
            const Real positive_allowance = std::max(
                Real{0.0},
                ledger.lumped_control_volume *
                    (ledger.upper_liquid_indicator -
                     ledger.low_order_liquid_indicator));
            const Real negative_allowance = std::min(
                Real{0.0},
                ledger.lumped_control_volume *
                    (ledger.lower_liquid_indicator -
                     ledger.low_order_liquid_indicator));
            if (positive_raw[i] > Real{0.0}) {
                positive_factor[i] = std::clamp(
                    positive_allowance / positive_raw[i],
                    Real{0.0}, Real{1.0});
            }
            if (negative_raw[i] < Real{0.0}) {
                negative_factor[i] = std::clamp(
                    negative_allowance / negative_raw[i],
                    Real{0.0}, Real{1.0});
            }
            result.nodes[i].positive_correction_factor = positive_factor[i];
            result.nodes[i].negative_correction_factor = negative_factor[i];
        }

        for (std::size_t edge_index = 0;
             edge_index < stage.interior_edges.size(); ++edge_index) {
            const auto& edge = stage.interior_edges[edge_index];
            auto& ledger = result.edges[edge_index];
            const auto first = static_cast<std::size_t>(edge.first_node);
            const auto second = static_cast<std::size_t>(edge.second_node);
            Real factor = Real{1.0};
            if (edge.raw_antidiffusive_mass_transfer > Real{0.0}) {
                factor = std::min(positive_factor[first],
                                  negative_factor[second]);
            } else if (edge.raw_antidiffusive_mass_transfer < Real{0.0}) {
                factor = std::min(negative_factor[first],
                                  positive_factor[second]);
            }
            ledger.correction_factor = factor;
            ledger.limited_antidiffusive_mass_transfer =
                factor * edge.raw_antidiffusive_mass_transfer;
            ledger.limited_pair_cancellation_residual =
                ledger.limited_antidiffusive_mass_transfer +
                (-ledger.limited_antidiffusive_mass_transfer);
            limited_internal[first] +=
                ledger.limited_antidiffusive_mass_transfer;
            limited_internal[second] -=
                ledger.limited_antidiffusive_mass_transfer;
            const Real correction =
                ledger.raw_antidiffusive_mass_transfer -
                ledger.limited_antidiffusive_mass_transfer;
            const Real correction_tolerance = scaledTolerance(
                stage.invariant_tolerance,
                {ledger.raw_antidiffusive_mass_transfer,
                 ledger.limited_antidiffusive_mass_transfer});
            if (std::abs(correction) > correction_tolerance) {
                ++result.limited_edges;
            }
            result.maximum_edge_pair_cancellation_residual = std::max(
                result.maximum_edge_pair_cancellation_residual,
                std::max({std::abs(ledger.low_order_pair_cancellation_residual),
                          std::abs(ledger.raw_pair_cancellation_residual),
                          std::abs(ledger.limited_pair_cancellation_residual)}));
        }
        result.applied = result.limited_edges > 0u;

        result.limited_bounds_satisfied = true;
        result.local_balance_satisfied = true;
        result.minimum_limited_liquid_indicator =
            std::numeric_limits<Real>::infinity();
        result.maximum_limited_liquid_indicator =
            -std::numeric_limits<Real>::infinity();
        for (std::size_t i = 0; i < node_count; ++i) {
            auto& ledger = result.nodes[i];
            ledger.limited_antidiffusive_mass_transfer =
                limited_internal[i];
            ledger.limited_liquid_indicator =
                ledger.low_order_liquid_indicator +
                ledger.limited_antidiffusive_mass_transfer /
                    ledger.lumped_control_volume;
            ledger.low_order_local_mass_balance_residual =
                ledger.lumped_control_volume *
                    (ledger.low_order_liquid_indicator -
                     ledger.previous_liquid_indicator) -
                (ledger.physical_boundary_mass_transfer +
                 ledger.discrete_divergence_mass_source +
                 ledger.low_order_interior_mass_transfer);
            ledger.raw_target_local_mass_balance_residual =
                ledger.lumped_control_volume *
                    (ledger.raw_target_liquid_indicator -
                     ledger.previous_liquid_indicator) -
                (ledger.physical_boundary_mass_transfer +
                 ledger.discrete_divergence_mass_source +
                 ledger.low_order_interior_mass_transfer +
                 ledger.raw_antidiffusive_mass_transfer);
            ledger.local_mass_balance_residual =
                ledger.lumped_control_volume *
                    (ledger.limited_liquid_indicator -
                     ledger.previous_liquid_indicator) -
                (ledger.physical_boundary_mass_transfer +
                 ledger.discrete_divergence_mass_source +
                 ledger.low_order_interior_mass_transfer +
                 ledger.limited_antidiffusive_mass_transfer);
            if (!std::isfinite(ledger.limited_liquid_indicator) ||
                !std::isfinite(
                    ledger.low_order_local_mass_balance_residual) ||
                !std::isfinite(
                    ledger.raw_target_local_mass_balance_residual) ||
                !std::isfinite(ledger.local_mass_balance_residual)) {
                result.diagnostic =
                    "conservative phase-flux correction produced a non-finite limited state or balance";
                return result;
            }
            result.limited_bounds_satisfied =
                result.limited_bounds_satisfied &&
                insideBounds(ledger.limited_liquid_indicator,
                             ledger.lower_liquid_indicator,
                             ledger.upper_liquid_indicator,
                             stage.invariant_tolerance);
            const Real local_tolerance = scaledTolerance(
                stage.invariant_tolerance,
                {ledger.lumped_control_volume *
                     ledger.previous_liquid_indicator,
                 ledger.lumped_control_volume *
                     ledger.low_order_liquid_indicator,
                 ledger.lumped_control_volume *
                     ledger.raw_target_liquid_indicator,
                 ledger.lumped_control_volume *
                     ledger.limited_liquid_indicator,
                 ledger.physical_boundary_mass_transfer,
                 ledger.discrete_divergence_mass_source,
                 ledger.low_order_interior_mass_transfer,
                 ledger.raw_antidiffusive_mass_transfer,
                 ledger.limited_antidiffusive_mass_transfer});
            result.local_balance_satisfied =
                result.local_balance_satisfied &&
                std::abs(ledger.low_order_local_mass_balance_residual) <=
                    local_tolerance &&
                std::abs(ledger.raw_target_local_mass_balance_residual) <=
                    local_tolerance &&
                std::abs(ledger.local_mass_balance_residual) <=
                    local_tolerance;
            result.maximum_low_order_local_mass_balance_residual = std::max(
                result.maximum_low_order_local_mass_balance_residual,
                std::abs(ledger.low_order_local_mass_balance_residual));
            result.maximum_raw_target_local_mass_balance_residual = std::max(
                result.maximum_raw_target_local_mass_balance_residual,
                std::abs(ledger.raw_target_local_mass_balance_residual));
            result.maximum_local_mass_balance_residual = std::max(
                result.maximum_local_mass_balance_residual,
                std::abs(ledger.local_mass_balance_residual));
            result.minimum_limited_liquid_indicator = std::min(
                result.minimum_limited_liquid_indicator,
                ledger.limited_liquid_indicator);
            result.maximum_limited_liquid_indicator = std::max(
                result.maximum_limited_liquid_indicator,
                ledger.limited_liquid_indicator);
        }

        result.low_order_nodal_cancellation_residual =
            longDoubleSum(low_order_internal);
        result.raw_nodal_cancellation_residual =
            longDoubleSum(raw_internal);
        result.limited_nodal_cancellation_residual =
            longDoubleSum(limited_internal);
        Real interior_scale = Real{1.0};
        for (const auto& edge : result.edges) {
            interior_scale += std::abs(edge.low_order_mass_transfer) +
                              std::abs(edge.raw_antidiffusive_mass_transfer) +
                              std::abs(
                                  edge.limited_antidiffusive_mass_transfer);
        }
        if (!std::isfinite(interior_scale)) {
            result.diagnostic =
                "conservative phase-flux correction overflowed its interior transfer scale";
            return result;
        }
        const Real interior_tolerance =
            stage.invariant_tolerance * interior_scale;
        result.interior_cancellation_satisfied =
            result.maximum_edge_pair_cancellation_residual == Real{0.0} &&
            std::abs(result.low_order_nodal_cancellation_residual) <=
                interior_tolerance &&
            std::abs(result.raw_nodal_cancellation_residual) <=
                interior_tolerance &&
            std::abs(result.limited_nodal_cancellation_residual) <=
                interior_tolerance;

        result.total_previous_liquid_measure = longDoubleNodeSum(
            result.nodes, [](const auto& node) {
                return node.lumped_control_volume *
                       node.previous_liquid_indicator;
            });
        result.total_low_order_liquid_measure = longDoubleNodeSum(
            result.nodes, [](const auto& node) {
                return node.lumped_control_volume *
                       node.low_order_liquid_indicator;
            });
        result.total_raw_target_liquid_measure = longDoubleNodeSum(
            result.nodes, [](const auto& node) {
                return node.lumped_control_volume *
                       node.raw_target_liquid_indicator;
            });
        result.total_limited_liquid_measure = longDoubleNodeSum(
            result.nodes, [](const auto& node) {
                return node.lumped_control_volume *
                       node.limited_liquid_indicator;
            });
        result.total_physical_boundary_mass_transfer = longDoubleNodeSum(
            result.nodes, [](const auto& node) {
                return node.physical_boundary_mass_transfer;
            });
        result.total_discrete_divergence_mass_source = longDoubleNodeSum(
            result.nodes, [](const auto& node) {
                return node.discrete_divergence_mass_source;
            });
        result.low_order_global_mass_balance_residual =
            result.total_low_order_liquid_measure -
            result.total_previous_liquid_measure -
            result.total_physical_boundary_mass_transfer -
            result.total_discrete_divergence_mass_source;
        result.raw_target_global_mass_balance_residual =
            result.total_raw_target_liquid_measure -
            result.total_previous_liquid_measure -
            result.total_physical_boundary_mass_transfer -
            result.total_discrete_divergence_mass_source;
        result.global_mass_balance_residual =
            result.total_limited_liquid_measure -
            result.total_previous_liquid_measure -
            result.total_physical_boundary_mass_transfer -
            result.total_discrete_divergence_mass_source;
        const Real global_tolerance = scaledTolerance(
            stage.invariant_tolerance,
            {result.total_previous_liquid_measure,
             result.total_low_order_liquid_measure,
             result.total_raw_target_liquid_measure,
             result.total_limited_liquid_measure,
             result.total_physical_boundary_mass_transfer,
             result.total_discrete_divergence_mass_source});
        result.global_balance_satisfied =
            std::abs(result.low_order_global_mass_balance_residual) <=
                global_tolerance &&
            std::abs(result.raw_target_global_mass_balance_residual) <=
                global_tolerance &&
            std::abs(result.global_mass_balance_residual) <= global_tolerance;

        result.constant_state_input =
            previous_maximum - previous_minimum <=
            scaledTolerance(stage.invariant_tolerance,
                            {previous_minimum, previous_maximum});
        result.constant_preservation_satisfied = true;
        if (result.constant_state_input) {
            for (const auto& node : result.nodes) {
                const Real error = std::max(
                    std::abs(node.low_order_liquid_indicator -
                             node.previous_liquid_indicator),
                    std::abs(node.limited_liquid_indicator -
                             node.previous_liquid_indicator));
                result.maximum_constant_preservation_error =
                    std::max(result.maximum_constant_preservation_error,
                             error);
                const Real constant_tolerance = scaledTolerance(
                    stage.invariant_tolerance,
                    {node.previous_liquid_indicator,
                     node.low_order_liquid_indicator,
                     node.limited_liquid_indicator});
                result.constant_preservation_satisfied =
                    result.constant_preservation_satisfied &&
                    error <= constant_tolerance;
            }
        }

        result.success = result.low_order_bounds_satisfied &&
                         result.limited_bounds_satisfied &&
                         result.interior_cancellation_satisfied &&
                         result.local_balance_satisfied &&
                         result.global_balance_satisfied &&
                         (!result.constant_preservation_required ||
                          result.constant_preservation_satisfied);
        if (!result.limited_bounds_satisfied) {
            result.diagnostic =
                "conservative phase-flux correction failed its limited invariant-domain check";
        } else if (!result.interior_cancellation_satisfied) {
            result.diagnostic =
                "conservative phase-flux correction failed interior edge cancellation";
        } else if (!result.local_balance_satisfied) {
            result.diagnostic =
                "conservative phase-flux correction failed a nodal control-volume balance";
        } else if (!result.global_balance_satisfied) {
            result.diagnostic =
                "conservative phase-flux correction failed global phase-measure balance";
        } else if (result.constant_preservation_required &&
                   !result.constant_preservation_satisfied) {
            result.diagnostic =
                "conservative phase-flux correction failed constant-state preservation";
        } else if (!result.applied) {
            result.diagnostic =
                "raw antidiffusive transfers satisfy all phase-indicator bounds";
        }
    } catch (const std::exception& error) {
        result.success = false;
        result.diagnostic = error.what();
    }
    return result;
}

} // namespace svmp::FE::level_set
