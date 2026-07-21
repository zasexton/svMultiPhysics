#include "LevelSet/LevelSetConservativePhaseRegions.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace svmp::FE::level_set {
namespace {

[[nodiscard]] Real scaledTolerance(
    Real tolerance,
    const LevelSetPhaseRegionLedger& region) noexcept
{
    const Real scale = std::max(
        {Real{1.0},
         std::abs(region.previous_liquid_measure),
         std::abs(region.low_order_liquid_measure),
         std::abs(region.raw_target_liquid_measure),
         std::abs(region.limited_liquid_measure),
         std::abs(region.physical_boundary_mass_transfer),
         std::abs(region.discrete_divergence_mass_source),
         std::abs(region.low_order_nodal_interior_mass_transfer),
         std::abs(region.raw_nodal_antidiffusive_mass_transfer),
         std::abs(region.limited_nodal_antidiffusive_mass_transfer),
         std::abs(region.low_order_crossing_mass_transfer),
         std::abs(region.raw_crossing_antidiffusive_mass_transfer),
         std::abs(region.limited_crossing_antidiffusive_mass_transfer)});
    return tolerance * scale;
}

[[nodiscard]] bool validCorrectionShape(
    const LevelSetPhaseFluxCorrectionResult& correction) noexcept
{
    if (!correction.success ||
        correction.node_component_ids.size() != correction.nodes.size()) {
        return false;
    }
    for (std::size_t node = 0u; node < correction.nodes.size(); ++node) {
        if (correction.nodes[node].node !=
            static_cast<GlobalIndex>(node)) {
            return false;
        }
    }
    for (const auto& edge : correction.edges) {
        if (edge.first_node < 0 || edge.second_node < 0 ||
            edge.first_node >= edge.second_node ||
            static_cast<std::size_t>(edge.second_node) >=
                correction.nodes.size()) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::string trimCopy(std::string_view value)
{
    std::size_t first = 0u;
    while (first < value.size() &&
           std::isspace(static_cast<unsigned char>(value[first])) != 0) {
        ++first;
    }
    std::size_t last = value.size();
    while (last > first &&
           std::isspace(static_cast<unsigned char>(value[last - 1u])) != 0) {
        --last;
    }
    return std::string(value.substr(first, last - first));
}

[[nodiscard]] std::vector<std::string> split(
    std::string_view value,
    char delimiter)
{
    std::vector<std::string> tokens;
    std::size_t first = 0u;
    while (first <= value.size()) {
        const auto delimiter_position = value.find(delimiter, first);
        const auto last = delimiter_position == std::string_view::npos
            ? value.size()
            : delimiter_position;
        tokens.push_back(trimCopy(value.substr(first, last - first)));
        if (delimiter_position == std::string_view::npos) {
            break;
        }
        first = delimiter_position + 1u;
    }
    return tokens;
}

[[nodiscard]] bool validRegionName(std::string_view name) noexcept
{
    return !name.empty() &&
           std::all_of(name.begin(), name.end(), [](char character) {
               const auto value =
                   static_cast<unsigned char>(character);
               return std::isalnum(value) != 0 || character == '_' ||
                      character == '-' || character == '.';
           });
}

[[nodiscard]] bool validRegionKind(LevelSetPhaseRegionKind kind) noexcept
{
    switch (kind) {
    case LevelSetPhaseRegionKind::WallFilm:
    case LevelSetPhaseRegionKind::Sheet:
    case LevelSetPhaseRegionKind::Rim:
    case LevelSetPhaseRegionKind::ResolvedSatellite:
    case LevelSetPhaseRegionKind::Observer:
        return true;
    }
    return false;
}

[[nodiscard]] Real parseCoordinate(
    std::string_view value,
    bool lower)
{
    if (value == "*") {
        return lower ? std::numeric_limits<Real>::lowest()
                     : std::numeric_limits<Real>::max();
    }
    try {
        const auto text = trimCopy(value);
        std::size_t parsed = 0u;
        const auto coordinate = static_cast<Real>(
            std::stod(text, &parsed));
        if (parsed != text.size() || !std::isfinite(coordinate)) {
            throw std::invalid_argument("invalid coordinate");
        }
        return coordinate;
    } catch (...) {
        throw std::invalid_argument(
            "phase region box has an invalid finite coordinate '" +
            std::string(value) + "'");
    }
}

} // namespace

const char* levelSetPhaseRegionKindName(
    LevelSetPhaseRegionKind kind) noexcept
{
    switch (kind) {
    case LevelSetPhaseRegionKind::WallFilm:
        return "wall_film";
    case LevelSetPhaseRegionKind::Sheet:
        return "sheet";
    case LevelSetPhaseRegionKind::Rim:
        return "rim";
    case LevelSetPhaseRegionKind::ResolvedSatellite:
        return "resolved_satellite";
    case LevelSetPhaseRegionKind::Observer:
        return "observer";
    }
    return "unknown";
}

LevelSetPhaseRegionKind parseLevelSetPhaseRegionKind(
    std::string_view value)
{
    std::string normalized;
    for (const char character : trimCopy(value)) {
        const auto unsigned_character =
            static_cast<unsigned char>(character);
        if (character == '_' || character == '-' ||
            std::isspace(unsigned_character) != 0) {
            continue;
        }
        normalized.push_back(static_cast<char>(
            std::tolower(unsigned_character)));
    }
    if (normalized == "wallfilm") {
        return LevelSetPhaseRegionKind::WallFilm;
    }
    if (normalized == "sheet") {
        return LevelSetPhaseRegionKind::Sheet;
    }
    if (normalized == "rim") {
        return LevelSetPhaseRegionKind::Rim;
    }
    if (normalized == "resolvedsatellite" || normalized == "satellite") {
        return LevelSetPhaseRegionKind::ResolvedSatellite;
    }
    if (normalized == "observer") {
        return LevelSetPhaseRegionKind::Observer;
    }
    throw std::invalid_argument(
        "unknown conservative phase region kind '" +
        std::string(value) + "'");
}

std::vector<LevelSetPhaseRegionBox> parseLevelSetPhaseRegionBoxes(
    std::string_view value)
{
    std::vector<LevelSetPhaseRegionBox> boxes;
    std::set<std::string> names;
    if (trimCopy(value).empty()) {
        return boxes;
    }
    for (const auto& entry : split(value, ';')) {
        if (entry.empty()) {
            throw std::invalid_argument(
                "conservative phase region list contains an empty entry");
        }
        const auto fields = split(entry, '|');
        if (fields.size() != 8u || !validRegionName(fields[0]) ||
            !names.insert(fields[0]).second) {
            throw std::invalid_argument(
                "each conservative phase region requires one unique valid name, kind, and six box coordinates");
        }
        LevelSetPhaseRegionBox box;
        box.name = fields[0];
        box.kind = parseLevelSetPhaseRegionKind(fields[1]);
        for (std::size_t dimension = 0u; dimension < 3u; ++dimension) {
            box.minimum[dimension] = parseCoordinate(
                fields[2u + 2u * dimension], true);
            box.maximum[dimension] = parseCoordinate(
                fields[3u + 2u * dimension], false);
            if (box.minimum[dimension] > box.maximum[dimension]) {
                throw std::invalid_argument(
                    "conservative phase region box minimum exceeds its maximum");
            }
        }
        boxes.push_back(std::move(box));
    }
    return boxes;
}

std::vector<LevelSetPhaseRegionDefinition>
makeAxisAlignedBoxPhaseRegions(
    std::span<const LevelSetPhaseRegionBox> boxes,
    std::span<const std::array<Real, 3>> node_coordinates)
{
    std::vector<LevelSetPhaseRegionDefinition> definitions;
    definitions.reserve(boxes.size());
    std::set<std::string> names;
    for (const auto& box : boxes) {
        if (!validRegionName(box.name) || !validRegionKind(box.kind) ||
            !names.insert(box.name).second) {
            throw std::invalid_argument(
                "phase region boxes require unique valid names and kinds");
        }
        for (std::size_t dimension = 0u; dimension < 3u; ++dimension) {
            if (!std::isfinite(box.minimum[dimension]) ||
                !std::isfinite(box.maximum[dimension]) ||
                box.minimum[dimension] > box.maximum[dimension]) {
                throw std::invalid_argument(
                    "phase region box bounds must be finite and ordered");
            }
        }
        LevelSetPhaseRegionDefinition definition;
        definition.name = box.name;
        definition.kind = box.kind;
        definition.node_membership.assign(node_coordinates.size(), 0u);
        for (std::size_t node = 0u; node < node_coordinates.size(); ++node) {
            bool inside = true;
            for (std::size_t dimension = 0u; dimension < 3u; ++dimension) {
                const Real coordinate = node_coordinates[node][dimension];
                if (!std::isfinite(coordinate)) {
                    throw std::invalid_argument(
                        "phase region node coordinates must be finite");
                }
                inside = inside &&
                         coordinate >= box.minimum[dimension] &&
                         coordinate <= box.maximum[dimension];
            }
            definition.node_membership[node] = inside ? 1u : 0u;
        }
        definitions.push_back(std::move(definition));
    }
    return definitions;
}

LevelSetPhaseRegionLedgerResult buildLevelSetPhaseRegionLedgers(
    const LevelSetPhaseFluxCorrectionResult& correction,
    std::span<const LevelSetPhaseRegionDefinition> definitions,
    Real invariant_tolerance)
{
    LevelSetPhaseRegionLedgerResult result;
    try {
        if (!validCorrectionShape(correction)) {
            result.diagnostic =
                "phase region ledger requires a successful complete correction ledger";
            return result;
        }
        if (!std::isfinite(invariant_tolerance) ||
            !(invariant_tolerance > Real{0.0})) {
            result.diagnostic =
                "phase region ledger requires a positive finite invariant tolerance";
            return result;
        }

        std::set<std::string> names;
        result.regions.reserve(definitions.size());
        for (const auto& definition : definitions) {
            if (!validRegionName(definition.name) ||
                !validRegionKind(definition.kind) ||
                !names.insert(definition.name).second ||
                definition.node_membership.size() !=
                    correction.nodes.size() ||
                std::any_of(
                    definition.node_membership.begin(),
                    definition.node_membership.end(),
                    [](std::uint8_t value) { return value > 1u; })) {
                result.diagnostic =
                    "phase region definitions require unique valid names and kinds and one binary membership value per node";
                return result;
            }

            LevelSetPhaseRegionLedger region;
            region.name = definition.name;
            region.kind = definition.kind;
            for (std::size_t node = 0u;
                 node < correction.nodes.size(); ++node) {
                if (definition.node_membership[node] == 0u) {
                    continue;
                }
                const auto& ledger = correction.nodes[node];
                region.member_nodes.push_back(ledger.node);
                region.previous_liquid_measure +=
                    ledger.lumped_control_volume *
                    ledger.previous_liquid_indicator;
                region.low_order_liquid_measure +=
                    ledger.lumped_control_volume *
                    ledger.low_order_liquid_indicator;
                region.raw_target_liquid_measure +=
                    ledger.lumped_control_volume *
                    ledger.raw_target_liquid_indicator;
                region.limited_liquid_measure +=
                    ledger.lumped_control_volume *
                    ledger.limited_liquid_indicator;
                region.physical_boundary_mass_transfer +=
                    ledger.physical_boundary_mass_transfer;
                region.discrete_divergence_mass_source +=
                    ledger.discrete_divergence_mass_source;
                region.low_order_nodal_interior_mass_transfer +=
                    ledger.low_order_interior_mass_transfer;
                region.raw_nodal_antidiffusive_mass_transfer +=
                    ledger.raw_antidiffusive_mass_transfer;
                region.limited_nodal_antidiffusive_mass_transfer +=
                    ledger.limited_antidiffusive_mass_transfer;
            }

            for (const auto& edge : correction.edges) {
                const auto first =
                    static_cast<std::size_t>(edge.first_node);
                const auto second =
                    static_cast<std::size_t>(edge.second_node);
                const bool first_inside =
                    definition.node_membership[first] != 0u;
                const bool second_inside =
                    definition.node_membership[second] != 0u;
                if (first_inside && second_inside) {
                    ++region.internal_edges;
                    region.maximum_internal_pair_cancellation_residual =
                        std::max(
                            region.maximum_internal_pair_cancellation_residual,
                            std::max(
                                {std::abs(edge.low_order_pair_cancellation_residual),
                                 std::abs(edge.raw_pair_cancellation_residual),
                                 std::abs(edge.limited_pair_cancellation_residual)}));
                    continue;
                }
                if (first_inside == second_inside) {
                    continue;
                }
                const Real sign = first_inside ? Real{1.0} : Real{-1.0};
                LevelSetPhaseRegionCrossingEdgeLedger crossing{
                    .first_node = edge.first_node,
                    .second_node = edge.second_node,
                    .low_order_mass_transfer_into_region =
                        sign * edge.low_order_mass_transfer,
                    .raw_antidiffusive_mass_transfer_into_region =
                        sign * edge.raw_antidiffusive_mass_transfer,
                    .limited_antidiffusive_mass_transfer_into_region =
                        sign * edge.limited_antidiffusive_mass_transfer,
                };
                region.low_order_crossing_mass_transfer +=
                    crossing.low_order_mass_transfer_into_region;
                region.raw_crossing_antidiffusive_mass_transfer +=
                    crossing.raw_antidiffusive_mass_transfer_into_region;
                region.limited_crossing_antidiffusive_mass_transfer +=
                    crossing.limited_antidiffusive_mass_transfer_into_region;
                region.crossing_edges.push_back(std::move(crossing));
            }

            region.low_order_flux_reconstruction_residual =
                region.low_order_nodal_interior_mass_transfer -
                region.low_order_crossing_mass_transfer;
            region.raw_flux_reconstruction_residual =
                region.raw_nodal_antidiffusive_mass_transfer -
                region.raw_crossing_antidiffusive_mass_transfer;
            region.limited_flux_reconstruction_residual =
                region.limited_nodal_antidiffusive_mass_transfer -
                region.limited_crossing_antidiffusive_mass_transfer;
            region.low_order_balance_residual =
                region.low_order_liquid_measure -
                region.previous_liquid_measure -
                region.physical_boundary_mass_transfer -
                region.discrete_divergence_mass_source -
                region.low_order_crossing_mass_transfer;
            region.raw_target_balance_residual =
                region.raw_target_liquid_measure -
                region.previous_liquid_measure -
                region.physical_boundary_mass_transfer -
                region.discrete_divergence_mass_source -
                region.low_order_crossing_mass_transfer -
                region.raw_crossing_antidiffusive_mass_transfer;
            region.limited_balance_residual =
                region.limited_liquid_measure -
                region.previous_liquid_measure -
                region.physical_boundary_mass_transfer -
                region.discrete_divergence_mass_source -
                region.low_order_crossing_mass_transfer -
                region.limited_crossing_antidiffusive_mass_transfer;

            const Real tolerance =
                scaledTolerance(invariant_tolerance, region);
            const Real maximum_balance = std::max(
                {std::abs(region.low_order_balance_residual),
                 std::abs(region.raw_target_balance_residual),
                 std::abs(region.limited_balance_residual)});
            const Real maximum_reconstruction = std::max(
                {std::abs(region.low_order_flux_reconstruction_residual),
                 std::abs(region.raw_flux_reconstruction_residual),
                 std::abs(region.limited_flux_reconstruction_residual),
                 region.maximum_internal_pair_cancellation_residual});
            region.balance_satisfied =
                maximum_balance <= tolerance &&
                maximum_reconstruction <= tolerance;
            result.maximum_balance_residual = std::max(
                result.maximum_balance_residual, maximum_balance);
            result.maximum_flux_reconstruction_residual = std::max(
                result.maximum_flux_reconstruction_residual,
                maximum_reconstruction);
            result.regions.push_back(std::move(region));
        }

        result.all_balances_satisfied = std::all_of(
            result.regions.begin(), result.regions.end(),
            [](const auto& region) { return region.balance_satisfied; });
        result.success = result.all_balances_satisfied;
        result.diagnostic = result.success
            ? (result.regions.empty() ? "no phase regions requested" : "ok")
            : "a phase region flux or control-volume balance did not close";
    } catch (const std::exception& error) {
        result.diagnostic = error.what();
    }
    return result;
}

std::vector<LevelSetPhaseRegionDefinition>
makeNonprimaryComponentSatelliteRegions(
    const LevelSetPhaseFluxCorrectionResult& correction)
{
    std::vector<LevelSetPhaseRegionDefinition> definitions;
    if (!validCorrectionShape(correction) ||
        correction.components.size() < 2u) {
        return definitions;
    }
    const auto primary = std::max_element(
        correction.components.begin(), correction.components.end(),
        [](const auto& lhs, const auto& rhs) {
            if (lhs.limited_liquid_measure !=
                rhs.limited_liquid_measure) {
                return lhs.limited_liquid_measure <
                       rhs.limited_liquid_measure;
            }
            return lhs.component_id > rhs.component_id;
        });
    for (const auto& component : correction.components) {
        if (component.component_id == primary->component_id) {
            continue;
        }
        LevelSetPhaseRegionDefinition definition;
        definition.name = "resolved_satellite_" +
                          std::to_string(component.component_id);
        definition.kind = LevelSetPhaseRegionKind::ResolvedSatellite;
        definition.node_membership.assign(correction.nodes.size(), 0u);
        for (std::size_t node = 0u;
             node < correction.node_component_ids.size(); ++node) {
            if (correction.node_component_ids[node] ==
                component.component_id) {
                definition.node_membership[node] = 1u;
            }
        }
        definitions.push_back(std::move(definition));
    }
    return definitions;
}

} // namespace svmp::FE::level_set
