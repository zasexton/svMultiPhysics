/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ANALYSIS_INTERFACE_TOPOLOGY_CONTEXT_H
#define SVMP_FE_ANALYSIS_INTERFACE_TOPOLOGY_CONTEXT_H

/**
 * @file InterfaceTopologyContext.h
 * @brief Explicit interface topology from InterfaceMesh registrations
 *
 * Built from FESystem::interfaceMesh(marker) for each registered InterfaceId.
 * Provides per-face minus/plus cell incidence, orientation, and bulk region
 * annotations. Separate from TopologyAnalysisContext which handles bulk
 * connected components.
 *
 * @see TopologyAnalysisContext for bulk region analysis
 */

#include "Core/Types.h"

#include <map>
#include <set>
#include <vector>

namespace svmp {
namespace FE {
namespace analysis {

/**
 * @brief Record for a single interface face
 */
struct InterfaceFaceRecord {
    int interface_marker{-1};
    GlobalIndex minus_cell{INVALID_GLOBAL_INDEX};
    GlobalIndex plus_cell{INVALID_GLOBAL_INDEX};
    int minus_local_face{-1};
    int plus_local_face{-1};
    bool is_two_sided{false};       ///< Both minus and plus cells exist
    bool has_orientation{false};
    int minus_region{-1};           ///< Bulk region annotation (from TopologyAnalysisContext)
    int plus_region{-1};
};

/**
 * @brief Interface topology context built from InterfaceMesh registrations
 */
class InterfaceTopologyContext {
public:
    InterfaceTopologyContext() = default;

    /// All interface face records across all registered markers
    std::vector<InterfaceFaceRecord> faces;

    /// Marker → indices into faces vector
    std::map<int, std::vector<std::size_t>> marker_to_faces;

    // ---- Queries ----

    [[nodiscard]] std::set<int> markers() const {
        std::set<int> result;
        for (const auto& [m, _] : marker_to_faces) {
            result.insert(m);
        }
        return result;
    }

    [[nodiscard]] bool hasMarker(int marker) const {
        return marker_to_faces.count(marker) > 0;
    }

    [[nodiscard]] std::size_t numFaces() const noexcept { return faces.size(); }

    [[nodiscard]] std::size_t numFacesForMarker(int marker) const {
        auto it = marker_to_faces.find(marker);
        return (it != marker_to_faces.end()) ? it->second.size() : 0u;
    }

    /// True if any interface is registered
    [[nodiscard]] bool empty() const noexcept { return faces.empty(); }
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_INTERFACE_TOPOLOGY_CONTEXT_H
