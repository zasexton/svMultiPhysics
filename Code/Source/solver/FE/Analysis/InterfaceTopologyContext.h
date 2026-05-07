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

#include <algorithm>
#include <map>
#include <set>
#include <utility>
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

    [[nodiscard]] bool markerExists(int marker) const {
        return hasMarker(marker);
    }

    [[nodiscard]] std::vector<const InterfaceFaceRecord*>
    facesForMarker(int marker) const {
        std::vector<const InterfaceFaceRecord*> result;
        auto it = marker_to_faces.find(marker);
        if (it == marker_to_faces.end()) {
            return result;
        }
        result.reserve(it->second.size());
        for (std::size_t index : it->second) {
            if (index < faces.size()) {
                result.push_back(&faces[index]);
            }
        }
        return result;
    }

    [[nodiscard]] std::size_t numFaces() const noexcept { return faces.size(); }

    [[nodiscard]] std::size_t numFacesForMarker(int marker) const {
        auto it = marker_to_faces.find(marker);
        return (it != marker_to_faces.end()) ? it->second.size() : 0u;
    }

    [[nodiscard]] bool markerHasTwoSidedFaces(int marker) const {
        const auto marker_faces = facesForMarker(marker);
        return !marker_faces.empty() &&
               std::all_of(marker_faces.begin(), marker_faces.end(),
                           [](const InterfaceFaceRecord* face) {
                               return face &&
                                      face->is_two_sided &&
                                      face->minus_cell != INVALID_GLOBAL_INDEX &&
                                      face->plus_cell != INVALID_GLOBAL_INDEX;
                           });
    }

    [[nodiscard]] bool markerHasOrientation(int marker) const {
        const auto marker_faces = facesForMarker(marker);
        return !marker_faces.empty() &&
               std::all_of(marker_faces.begin(), marker_faces.end(),
                           [](const InterfaceFaceRecord* face) {
                               return face && face->has_orientation;
                           });
    }

    [[nodiscard]] bool markerHasConsistentRegions(int marker) const {
        const auto marker_faces = facesForMarker(marker);
        return !marker_faces.empty() &&
               std::all_of(marker_faces.begin(), marker_faces.end(),
                           [](const InterfaceFaceRecord* face) {
                               return face &&
                                      face->minus_region >= 0 &&
                                      face->plus_region >= 0;
                           });
    }

    [[nodiscard]] bool markerHasValidLocalFaceIndices(int marker) const {
        const auto marker_faces = facesForMarker(marker);
        return !marker_faces.empty() &&
               std::all_of(marker_faces.begin(), marker_faces.end(),
                           [](const InterfaceFaceRecord* face) {
                               return face &&
                                      face->minus_local_face >= 0 &&
                                      (!face->is_two_sided ||
                                       face->plus_local_face >= 0);
                           });
    }

    [[nodiscard]] bool markerRecordsMatchKey(int marker) const {
        const auto marker_faces = facesForMarker(marker);
        return !marker_faces.empty() &&
               std::all_of(marker_faces.begin(), marker_faces.end(),
                           [marker](const InterfaceFaceRecord* face) {
                               return face &&
                                      face->interface_marker == marker;
                           });
    }

    [[nodiscard]] std::vector<std::pair<GlobalIndex, int>>
    duplicateInterfaceIncidence(int marker) const {
        std::set<std::pair<GlobalIndex, int>> seen;
        std::set<std::pair<GlobalIndex, int>> duplicate_set;
        for (const auto* face : facesForMarker(marker)) {
            if (!face) {
                continue;
            }
            const std::pair<GlobalIndex, int> minus{
                face->minus_cell, face->minus_local_face};
            if (minus.first != INVALID_GLOBAL_INDEX && minus.second >= 0 &&
                !seen.insert(minus).second) {
                duplicate_set.insert(minus);
            }
            const std::pair<GlobalIndex, int> plus{
                face->plus_cell, face->plus_local_face};
            if (plus.first != INVALID_GLOBAL_INDEX && plus.second >= 0 &&
                !seen.insert(plus).second) {
                duplicate_set.insert(plus);
            }
        }
        return {duplicate_set.begin(), duplicate_set.end()};
    }

    [[nodiscard]] bool markerHasNonmanifoldIncidence(int marker) const {
        return !duplicateInterfaceIncidence(marker).empty();
    }

    [[nodiscard]] std::vector<int>
    interfaceCoverageForReferencedMarkers(const std::set<int>& referenced_markers) const {
        std::vector<int> uncovered;
        for (int marker : referenced_markers) {
            if (!markerExists(marker) || numFacesForMarker(marker) == 0u) {
                uncovered.push_back(marker);
            }
        }
        return uncovered;
    }

    /// True if any interface is registered
    [[nodiscard]] bool empty() const noexcept { return faces.empty(); }
};

} // namespace analysis
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ANALYSIS_INTERFACE_TOPOLOGY_CONTEXT_H
