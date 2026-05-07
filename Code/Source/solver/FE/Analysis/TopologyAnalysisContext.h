/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_TOPOLOGY_ANALYSIS_CONTEXT_H
#define SVMP_FE_ANALYSIS_TOPOLOGY_ANALYSIS_CONTEXT_H

/**
 * @file TopologyAnalysisContext.h
 * @brief Mesh topology metadata for problem analysis
 *
 * Provides connected-component information and boundary-marker-to-region
 * mapping built from IMeshAccess.  Independent of DOF spaces — works
 * for any element type (Tet4, Hex8, DG, higher-order, etc.).
 *
 * @see ProblemAnalysisContext for storage
 * @see TopologyScopeAnalyzer for the primary consumer (Phase 7)
 */

#include "Core/Types.h"

#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace svmp {
namespace FE {

namespace assembly {
class IMeshAccess;
} // namespace assembly

namespace analysis {

enum class TopologyConnectivityMode : std::uint8_t {
    NodeConnected,
    FacetConnected,
    EdgeConnected,
    DofGraphConnected,
    OperatorGraphConnected,
    InterfaceCoupledConnected,
};

/**
 * @brief A connected component of the mesh
 */
struct ConnectedComponent {
    int region_id{0};
    std::vector<GlobalIndex> cell_indices;
    int num_vertices{0};
    int num_cells{0};
    std::set<int> boundary_markers;
};

struct ConnectedComponentSet {
    TopologyConnectivityMode mode{TopologyConnectivityMode::NodeConnected};
    std::vector<ConnectedComponent> components;
    bool exact_for_mode{true};
};

/**
 * @brief Mapping between boundary markers and mesh regions
 */
struct BoundaryRegionMapping {
    /// boundary marker → region IDs that it touches
    std::map<int, std::vector<int>> marker_to_regions;
    /// region ID → boundary markers on that region
    std::map<int, std::set<int>> region_to_markers;
};

/**
 * @brief Mapping between interface markers and region pairs
 */
struct InterfaceRegionMapping {
    /// interface marker → pairs of (region_a, region_b) it connects
    std::map<int, std::vector<std::pair<int, int>>> interface_to_region_pairs;
};

/**
 * @brief Mesh topology metadata for problem analysis
 *
 * Built from IMeshAccess using cell-cell adjacency (via shared nodes)
 * and boundary face iteration.  Independent of FE function spaces.
 */
class TopologyAnalysisContext {
public:
    TopologyAnalysisContext() = default;

    /// Connected components of the mesh
    std::vector<ConnectedComponent> components;

    /// Mode-specific connected-component sets. `components` is the default mode.
    std::map<TopologyConnectivityMode, ConnectedComponentSet> component_sets;
    TopologyConnectivityMode default_connectivity_mode{
        TopologyConnectivityMode::NodeConnected};

    /// Boundary marker ↔ region mapping
    BoundaryRegionMapping boundary_mapping;

    /// Interface marker ↔ region pair mapping
    InterfaceRegionMapping interface_mapping;

    // ---- Queries ----

    [[nodiscard]] int numRegions() const noexcept {
        return static_cast<int>(components.size());
    }

    [[nodiscard]] bool isDisconnected() const noexcept {
        return components.size() > 1;
    }

    /// Get the region ID for a cell (-1 if not found)
    [[nodiscard]] int regionForCell(GlobalIndex cell_idx) const noexcept;

    /// Get all region IDs touched by a boundary marker
    [[nodiscard]] std::vector<int> regionsForBoundaryMarker(int marker) const;

    /// Get components for a requested topology mode, if built.
    [[nodiscard]] const ConnectedComponentSet*
    connectedComponents(TopologyConnectivityMode mode) const noexcept;

    // ---- Factory ----

    /**
     * @brief Build topology context from an IMeshAccess
     *
     * Uses cell-to-cell adjacency via shared nodes (not DOF-based) to
     * find connected components.  Maps boundary face markers to components.
     * Works for any element type.
     *
     * @param mesh  Mesh access interface
     * @return      Populated TopologyAnalysisContext
     */
    [[nodiscard]] static TopologyAnalysisContext
    build(const assembly::IMeshAccess& mesh,
          TopologyConnectivityMode mode = TopologyConnectivityMode::NodeConnected);

private:
    /// Cell index → region ID lookup (built by build())
    std::vector<int> cell_to_region_;
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_TOPOLOGY_ANALYSIS_CONTEXT_H
