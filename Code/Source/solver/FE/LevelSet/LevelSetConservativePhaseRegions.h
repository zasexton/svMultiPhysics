#pragma once

/**
 * @file
 * @ingroup fe_level_set
 * @brief Conservative subregion ledgers for phase-indicator transport.
 */

#include "Core/Types.h"
#include "LevelSet/LevelSetConservativePhaseTransport.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp::FE::level_set {

enum class LevelSetPhaseRegionKind {
    WallFilm,
    Sheet,
    Rim,
    ResolvedSatellite,
    Observer
};

[[nodiscard]] const char* levelSetPhaseRegionKindName(
    LevelSetPhaseRegionKind kind) noexcept;

[[nodiscard]] LevelSetPhaseRegionKind parseLevelSetPhaseRegionKind(
    std::string_view value);

struct LevelSetPhaseRegionBox {
    std::string name{};
    LevelSetPhaseRegionKind kind{LevelSetPhaseRegionKind::Observer};
    std::array<Real, 3> minimum{};
    std::array<Real, 3> maximum{};
};

/**
 * Parse semicolon-separated fixed Eulerian box observers. Each entry is
 * `name|kind|xmin|xmax|ymin|ymax|zmin|zmax`; `*` selects an unbounded lower
 * or upper coordinate. Region names may contain letters, digits, `_`, `-`,
 * and `.`.
 */
[[nodiscard]] std::vector<LevelSetPhaseRegionBox>
parseLevelSetPhaseRegionBoxes(std::string_view value);

/**
 * @brief One fixed control-volume region for a complete transport stage.
 *
 * Membership is deliberately explicit. A benchmark or geometry classifier
 * may define a wall film, sheet, or rim, but the balance kernel never infers
 * morphology from component numbering. The same mask applies to the
 * previous, low-order, raw-target, and limited states of one stage.
 */
struct LevelSetPhaseRegionDefinition {
    std::string name{};
    LevelSetPhaseRegionKind kind{LevelSetPhaseRegionKind::Observer};
    std::vector<std::uint8_t> node_membership{};
};

struct LevelSetPhaseRegionCrossingEdgeLedger {
    GlobalIndex first_node{-1};
    GlobalIndex second_node{-1};
    Real low_order_mass_transfer_into_region{0.0};
    Real raw_antidiffusive_mass_transfer_into_region{0.0};
    Real limited_antidiffusive_mass_transfer_into_region{0.0};
};

struct LevelSetPhaseRegionLedger {
    std::string name{};
    LevelSetPhaseRegionKind kind{LevelSetPhaseRegionKind::Observer};
    bool balance_satisfied{false};
    std::vector<GlobalIndex> member_nodes{};
    std::size_t internal_edges{0u};
    std::vector<LevelSetPhaseRegionCrossingEdgeLedger> crossing_edges{};
    Real previous_liquid_measure{0.0};
    Real low_order_liquid_measure{0.0};
    Real raw_target_liquid_measure{0.0};
    Real limited_liquid_measure{0.0};
    Real physical_boundary_mass_transfer{0.0};
    Real discrete_divergence_mass_source{0.0};
    Real low_order_nodal_interior_mass_transfer{0.0};
    Real raw_nodal_antidiffusive_mass_transfer{0.0};
    Real limited_nodal_antidiffusive_mass_transfer{0.0};
    Real low_order_crossing_mass_transfer{0.0};
    Real raw_crossing_antidiffusive_mass_transfer{0.0};
    Real limited_crossing_antidiffusive_mass_transfer{0.0};
    Real low_order_flux_reconstruction_residual{0.0};
    Real raw_flux_reconstruction_residual{0.0};
    Real limited_flux_reconstruction_residual{0.0};
    Real low_order_balance_residual{0.0};
    Real raw_target_balance_residual{0.0};
    Real limited_balance_residual{0.0};
    Real maximum_internal_pair_cancellation_residual{0.0};
};

struct LevelSetPhaseRegionLedgerResult {
    bool success{false};
    bool all_balances_satisfied{false};
    Real maximum_balance_residual{0.0};
    Real maximum_flux_reconstruction_residual{0.0};
    std::vector<LevelSetPhaseRegionLedger> regions{};
    std::string diagnostic{};
};

/**
 * @brief Evaluate fixed Eulerian box membership at canonical node
 * coordinates.
 */
[[nodiscard]] std::vector<LevelSetPhaseRegionDefinition>
makeAxisAlignedBoxPhaseRegions(
    std::span<const LevelSetPhaseRegionBox> boxes,
    std::span<const std::array<Real, 3>> node_coordinates);

/**
 * @brief Build exact fixed-region balance ledgers from a successful nodal
 * control-volume correction.
 */
[[nodiscard]] LevelSetPhaseRegionLedgerResult
buildLevelSetPhaseRegionLedgers(
    const LevelSetPhaseFluxCorrectionResult& correction,
    std::span<const LevelSetPhaseRegionDefinition> definitions,
    Real invariant_tolerance = 1.0e-12);

/**
 * @brief Explicit policy helper that marks every resolved component except
 * the largest limited-measure component as a separate satellite region.
 *
 * Calling this function is the opt-in policy decision. Equal-measure ties use
 * the smallest deterministic component identifier as the primary component.
 */
[[nodiscard]] std::vector<LevelSetPhaseRegionDefinition>
makeNonprimaryComponentSatelliteRegions(
    const LevelSetPhaseFluxCorrectionResult& correction);

} // namespace svmp::FE::level_set
