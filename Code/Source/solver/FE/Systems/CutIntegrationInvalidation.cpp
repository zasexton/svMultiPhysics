/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/CutIntegrationInvalidation.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace svmp {
namespace FE {
namespace systems {

namespace {

[[nodiscard]] std::uint64_t cutFacetStableId(MeshIndex facet,
                                             MeshIndex first_cell,
                                             MeshIndex second_cell) noexcept
{
    std::uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](std::uint64_t value) noexcept {
        h ^= value;
        h *= 1099511628211ull;
    };
    mix(static_cast<std::uint64_t>(facet));
    mix(static_cast<std::uint64_t>(first_cell));
    mix(static_cast<std::uint64_t>(second_cell));
    return h;
}

} // namespace

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
CutIntegrationRevisionSnapshot CutIntegrationRevisionSnapshot::capture(
    const svmp::search::CutClassificationMap& map,
    std::uint64_t fe_space_revision,
    std::uint64_t fe_dof_layout_revision,
    std::uint64_t fe_constraint_layout_revision,
    std::uint64_t fe_block_layout_revision,
    std::uint64_t restart_layout_revision) {
    CutIntegrationRevisionSnapshot snapshot;
    snapshot.valid = true;
    snapshot.cut_revision_key = map.revision_key();
    snapshot.geometry_revision = map.revision.geometry_revision;
    snapshot.topology_revision = map.revision.topology_revision;
    snapshot.ownership_revision = map.revision.ownership_revision;
    snapshot.numbering_revision = map.revision.numbering_revision;
    snapshot.label_revision = map.revision.label_revision;
    snapshot.active_configuration_epoch = map.revision.active_configuration_epoch;
    snapshot.embedded_geometry_epoch = map.revision.embedded_geometry_epoch;
    snapshot.embedded_field_layout_revision = map.revision.embedded_field_layout_revision;
    snapshot.embedded_field_value_revision = map.revision.embedded_field_value_revision;
    snapshot.embedded_source_surface_revision = map.revision.embedded_source_surface_revision;
    snapshot.embedded_provenance_revision = map.revision.embedded_provenance_revision;
    snapshot.embedded_constraint_epoch = map.revision.embedded_constraint_epoch;
    std::uint64_t topo = 1469598103934665603ull;
    for (const auto& r : map.cells) {
        topo ^= r.cut_topology_id;
        topo *= 1099511628211ull;
    }
    snapshot.cut_topology_revision = topo;
    snapshot.quadrature_policy_revision = map.options.predicate_policy.revision_key();
    snapshot.fe_space_revision = fe_space_revision;
    snapshot.fe_dof_layout_revision = fe_dof_layout_revision;
    snapshot.fe_constraint_layout_revision = fe_constraint_layout_revision;
    snapshot.fe_block_layout_revision = fe_block_layout_revision;
    snapshot.restart_layout_revision = restart_layout_revision;
    snapshot.cut_cell_count = static_cast<std::size_t>(std::count_if(
        map.cells.begin(), map.cells.end(), [](const auto& r) {
            return r.classification == svmp::search::CutClassification::Cut;
        }));
    snapshot.cut_face_count = static_cast<std::size_t>(std::count_if(
        map.faces.begin(), map.faces.end(), [](const auto& r) {
            return r.classification == svmp::search::CutClassification::Cut;
        }));
    return snapshot;
}
#endif

CutIntegrationRefreshDecision classifyCutIntegrationRefresh(
    const CutIntegrationRevisionSnapshot& cached,
    const CutIntegrationRevisionSnapshot& current) noexcept {
    CutIntegrationRefreshDecision decision;
    if (!cached.valid || !current.valid) {
        decision.rebuild_cut_classification = true;
        decision.rebuild_quadrature = true;
        decision.rebuild_sparsity_pattern = true;
        decision.refresh_full_cell_domain_caches = true;
        decision.rebuild_matrix = true;
        decision.rebuild_matrix_free_data = true;
        decision.refresh_preconditioner = true;
        decision.refresh_restart_metadata = true;
        decision.update_stabilization_hooks = true;
        decision.reason = "missing cut integration revision snapshot";
        return decision;
    }

    const bool mesh_or_embedded_changed =
        cached.cut_revision_key != current.cut_revision_key ||
        cached.geometry_revision != current.geometry_revision ||
        cached.topology_revision != current.topology_revision ||
        cached.ownership_revision != current.ownership_revision ||
        cached.numbering_revision != current.numbering_revision ||
        cached.label_revision != current.label_revision ||
        cached.active_configuration_epoch != current.active_configuration_epoch ||
        cached.embedded_geometry_epoch != current.embedded_geometry_epoch ||
        cached.embedded_field_layout_revision != current.embedded_field_layout_revision ||
        cached.embedded_field_value_revision != current.embedded_field_value_revision ||
        cached.embedded_source_surface_revision != current.embedded_source_surface_revision ||
        cached.embedded_provenance_revision != current.embedded_provenance_revision ||
        cached.embedded_constraint_epoch != current.embedded_constraint_epoch ||
        cached.cut_topology_revision != current.cut_topology_revision ||
        cached.quadrature_policy_revision != current.quadrature_policy_revision ||
        cached.conditioning_revision != current.conditioning_revision ||
        cached.cut_cell_count != current.cut_cell_count ||
        cached.cut_face_count != current.cut_face_count;

    const bool cut_coupling_topology_changed =
        cached.topology_revision != current.topology_revision ||
        cached.ownership_revision != current.ownership_revision ||
        cached.numbering_revision != current.numbering_revision ||
        cached.label_revision != current.label_revision ||
        cached.active_configuration_epoch != current.active_configuration_epoch ||
        cached.embedded_field_layout_revision != current.embedded_field_layout_revision ||
        cached.embedded_field_value_revision != current.embedded_field_value_revision ||
        cached.embedded_source_surface_revision != current.embedded_source_surface_revision ||
        cached.embedded_constraint_epoch != current.embedded_constraint_epoch ||
        cached.cut_topology_revision != current.cut_topology_revision ||
        cached.conditioning_revision != current.conditioning_revision ||
        cached.cut_cell_count != current.cut_cell_count ||
        cached.cut_face_count != current.cut_face_count;

    const bool fe_layout_changed =
        cached.fe_space_revision != current.fe_space_revision ||
        cached.fe_dof_layout_revision != current.fe_dof_layout_revision ||
        cached.fe_constraint_layout_revision != current.fe_constraint_layout_revision ||
        cached.fe_block_layout_revision != current.fe_block_layout_revision;

    const bool full_cell_domain_cache_dependencies_changed =
        cached.geometry_revision != current.geometry_revision ||
        cached.topology_revision != current.topology_revision ||
        cached.ownership_revision != current.ownership_revision ||
        cached.numbering_revision != current.numbering_revision ||
        cached.label_revision != current.label_revision ||
        fe_layout_changed;

    if (mesh_or_embedded_changed) {
        decision.rebuild_cut_classification = true;
        decision.rebuild_quadrature = true;
        decision.rebuild_sparsity_pattern = cut_coupling_topology_changed;
        decision.refresh_full_cell_domain_caches = full_cell_domain_cache_dependencies_changed;
        decision.rebuild_matrix = true;
        decision.rebuild_matrix_free_data = true;
        decision.refresh_preconditioner = true;
        decision.refresh_restart_metadata = true;
        decision.update_stabilization_hooks = true;
        decision.reason = "mesh, embedded geometry, or cut topology changed";
    } else if (fe_layout_changed) {
        decision.rebuild_sparsity_pattern = true;
        decision.refresh_full_cell_domain_caches = true;
        decision.rebuild_matrix = true;
        decision.refresh_preconditioner = true;
        decision.refresh_restart_metadata = true;
        decision.reason = "FE cut integration layout changed";
    } else if (cached.restart_layout_revision != current.restart_layout_revision) {
        decision.refresh_restart_metadata = true;
        decision.reason = "restart layout changed";
    }

    return decision;
}

CutConditioningDiagnostic diagnoseCutConditioning(
    const std::vector<Real>& volume_fractions,
    Real small_fraction_threshold,
    Real degenerate_threshold) {
    CutConditioningDiagnostic diagnostic;
    for (const Real fraction : volume_fractions) {
        if (fraction <= degenerate_threshold) {
            ++diagnostic.degenerate_cut_count;
        } else if (fraction < small_fraction_threshold) {
            ++diagnostic.small_cut_cell_count;
        }
    }
    if (diagnostic.degenerate_cut_count > 0) {
        diagnostic.ok = false;
        diagnostic.messages.push_back("degenerate cut cells require aggregation or stabilization");
    }
    if (diagnostic.small_cut_cell_count > 0) {
        diagnostic.messages.push_back("small cut cells may require conditioning stabilization");
    }
    return diagnostic;
}

std::vector<CutConditioningNeighborhood> buildCutConditioningNeighborhoods(
    const std::vector<MeshIndex>& cut_cells,
    const std::vector<Real>& volume_fractions,
    const std::vector<std::pair<MeshIndex, MeshIndex>>& adjacency,
    Real small_fraction_threshold) {
    std::vector<CutConditioningNeighborhood> neighborhoods;
    for (std::size_t i = 0; i < cut_cells.size(); ++i) {
        const Real fraction = i < volume_fractions.size() ? volume_fractions[i] : Real{0.0};
        if (fraction >= small_fraction_threshold) {
            continue;
        }
        CutConditioningNeighborhood n;
        n.cut_cell = cut_cells[i];
        n.volume_fraction = fraction;
        n.conditioning_indicator = fraction > Real{0.0}
                                       ? Real{1.0} / fraction
                                       : std::numeric_limits<Real>::infinity();
        for (const auto& edge : adjacency) {
            if (edge.first == n.cut_cell) {
                n.adjacent_cells.push_back(edge.second);
            } else if (edge.second == n.cut_cell) {
                n.adjacent_cells.push_back(edge.first);
            }
        }
        std::sort(n.adjacent_cells.begin(), n.adjacent_cells.end());
        n.adjacent_cells.erase(std::unique(n.adjacent_cells.begin(), n.adjacent_cells.end()),
                               n.adjacent_cells.end());
        n.extension_patch = n.adjacent_cells;
        n.extension_patch.insert(n.extension_patch.begin(), n.cut_cell);
        n.stable_id = static_cast<std::uint64_t>(1469598103934665603ull);
        n.stable_id ^= static_cast<std::uint64_t>(n.cut_cell);
        n.stable_id *= 1099511628211ull;
        for (const auto cell : n.adjacent_cells) {
            n.stable_id ^= static_cast<std::uint64_t>(cell);
            n.stable_id *= 1099511628211ull;
        }
        neighborhoods.push_back(std::move(n));
    }
    return neighborhoods;
}

std::vector<CutAdjacentInteriorFacet> identifyCutAdjacentInteriorFacets(
    const std::vector<MeshIndex>& cut_cells,
    const std::vector<CutInteriorFacetAdjacency>& interior_facets) {
    std::vector<MeshIndex> sorted_cut_cells = cut_cells;
    std::sort(sorted_cut_cells.begin(), sorted_cut_cells.end());
    sorted_cut_cells.erase(std::unique(sorted_cut_cells.begin(), sorted_cut_cells.end()),
                           sorted_cut_cells.end());

    const auto is_cut_cell = [&sorted_cut_cells](MeshIndex cell) {
        return std::binary_search(sorted_cut_cells.begin(), sorted_cut_cells.end(), cell);
    };

    std::vector<CutAdjacentInteriorFacet> facets;
    for (const auto& adjacency : interior_facets) {
        if (adjacency.facet < 0 || adjacency.first_cell < 0 || adjacency.second_cell < 0) {
            continue;
        }
        const bool first_cut = is_cut_cell(adjacency.first_cell);
        const bool second_cut = is_cut_cell(adjacency.second_cell);
        if (!first_cut && !second_cut) {
            continue;
        }

        CutAdjacentInteriorFacet facet;
        facet.facet = adjacency.facet;
        facet.first_cell = adjacency.first_cell;
        facet.second_cell = adjacency.second_cell;
        facet.first_cell_cut = first_cut;
        facet.second_cell_cut = second_cut;
        facet.stable_id = cutFacetStableId(facet.facet, facet.first_cell, facet.second_cell);
        facets.push_back(facet);
    }

    std::sort(facets.begin(), facets.end(), [](const auto& a, const auto& b) {
        if (a.facet != b.facet) {
            return a.facet < b.facet;
        }
        if (a.first_cell != b.first_cell) {
            return a.first_cell < b.first_cell;
        }
        return a.second_cell < b.second_cell;
    });
    facets.erase(std::unique(facets.begin(),
                             facets.end(),
                             [](const auto& a, const auto& b) {
                                 return a.facet == b.facet &&
                                        a.first_cell == b.first_cell &&
                                        a.second_cell == b.second_cell;
                             }),
                 facets.end());
    return facets;
}

CutAdjacentFacetSetHandle makeCutAdjacentFacetSetHandle(
    int marker,
    std::string name,
    const std::vector<CutAdjacentInteriorFacet>& facets) {
    CutAdjacentFacetSetHandle handle;
    handle.marker = marker;
    handle.name = std::move(name);
    handle.facets.reserve(facets.size());
    for (const auto& facet : facets) {
        if (facet.facet >= 0) {
            handle.facets.push_back(facet.facet);
        }
    }
    std::sort(handle.facets.begin(), handle.facets.end());
    handle.facets.erase(std::unique(handle.facets.begin(), handle.facets.end()),
                        handle.facets.end());

    std::uint64_t h = 1469598103934665603ull;
    const auto mix = [&h](std::uint64_t value) noexcept {
        h ^= value;
        h *= 1099511628211ull;
    };
    mix(static_cast<std::uint64_t>(marker));
    for (const auto facet : handle.facets) {
        mix(static_cast<std::uint64_t>(facet));
    }
    handle.stable_id = h;
    return handle;
}

} // namespace systems
} // namespace FE
} // namespace svmp
