/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "DofTools.h"
#include "DofHandler.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace svmp {
namespace FE {
namespace dofs {

// =============================================================================
// ComponentMask Implementation
// =============================================================================

ComponentMask ComponentMask::all() noexcept {
    ComponentMask mask;
    mask.mask_.set();
    return mask;
}

ComponentMask ComponentMask::none() noexcept {
    ComponentMask mask;
    mask.mask_.reset();
    return mask;
}

ComponentMask ComponentMask::component(std::size_t idx) {
    ComponentMask mask;
    mask.mask_.reset();
    if (idx < MAX_COMPONENTS) {
        mask.mask_.set(idx);
    }
    return mask;
}

ComponentMask ComponentMask::selected(std::initializer_list<std::size_t> indices) {
    ComponentMask mask;
    mask.mask_.reset();
    for (auto idx : indices) {
        if (idx < MAX_COMPONENTS) {
            mask.mask_.set(idx);
        }
    }
    return mask;
}

ComponentMask::ComponentMask() noexcept {
    mask_.set();  // Default: all selected
}

bool ComponentMask::isSelected(std::size_t component) const noexcept {
    if (component >= MAX_COMPONENTS) return false;
    return mask_.test(component);
}

std::size_t ComponentMask::numSelected() const noexcept {
    std::size_t count = 0;
    for (std::size_t i = 0; i < n_components_; ++i) {
        if (mask_.test(i)) ++count;
    }
    return count;
}

void ComponentMask::setSize(std::size_t n_components) {
    n_components_ = std::min(n_components, MAX_COMPONENTS);
}

void ComponentMask::select(std::size_t component) {
    if (component < MAX_COMPONENTS) {
        mask_.set(component);
    }
}

void ComponentMask::deselect(std::size_t component) {
    if (component < MAX_COMPONENTS) {
        mask_.reset(component);
    }
}

ComponentMask ComponentMask::operator&(const ComponentMask& other) const noexcept {
    ComponentMask result;
    result.mask_ = mask_ & other.mask_;
    result.n_components_ = std::min(n_components_, other.n_components_);
    return result;
}

ComponentMask ComponentMask::operator|(const ComponentMask& other) const noexcept {
    ComponentMask result;
    result.mask_ = mask_ | other.mask_;
    result.n_components_ = std::max(n_components_, other.n_components_);
    return result;
}

bool ComponentMask::selectsAll() const noexcept {
    for (std::size_t i = 0; i < n_components_; ++i) {
        if (!mask_.test(i)) return false;
    }
    return true;
}

bool ComponentMask::selectsNone() const noexcept {
    for (std::size_t i = 0; i < n_components_; ++i) {
        if (mask_.test(i)) return false;
    }
    return true;
}

// =============================================================================
// FieldMask Implementation
// =============================================================================

FieldMask FieldMask::all() noexcept {
    FieldMask mask;
    mask.mask_.set();
    return mask;
}

FieldMask FieldMask::none() noexcept {
    FieldMask mask;
    mask.mask_.reset();
    return mask;
}

FieldMask FieldMask::field(std::size_t idx) {
    FieldMask mask;
    mask.mask_.reset();
    if (idx < MAX_FIELDS) {
        mask.mask_.set(idx);
    }
    return mask;
}

FieldMask FieldMask::named(std::initializer_list<std::string> names) {
    FieldMask mask;
    mask.mask_.reset();
    // Names are stored but indices must be resolved later
    for (const auto& name : names) {
        mask.field_names_.push_back(name);
    }
    return mask;
}

FieldMask::FieldMask() noexcept {
    mask_.set();  // Default: all selected
}

bool FieldMask::isSelected(std::size_t field_idx) const noexcept {
    if (field_idx >= MAX_FIELDS) return false;
    return mask_.test(field_idx);
}

void FieldMask::select(std::size_t field_idx) {
    if (field_idx < MAX_FIELDS) {
        mask_.set(field_idx);
    }
}

void FieldMask::deselect(std::size_t field_idx) {
    if (field_idx < MAX_FIELDS) {
        mask_.reset(field_idx);
    }
}

std::size_t FieldMask::numSelected() const noexcept {
    return mask_.count();
}

void FieldMask::setFieldNames(std::span<const std::string> names) {
    field_names_.assign(names.begin(), names.end());
}

int FieldMask::getFieldIndex(const std::string& name) const {
    for (std::size_t i = 0; i < field_names_.size(); ++i) {
        if (field_names_[i] == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// =============================================================================
// DofTools Namespace Implementation
// =============================================================================

namespace DofTools {

// =========================================================================
// Boundary DOF Extraction
// =========================================================================

std::vector<GlobalIndex> extractBoundaryDofs(
    const EntityDofMap& entity_map,
    int boundary_id,
    std::span<const int> facet_boundary_labels,
    std::span<const GlobalIndex> facet2vertex_offsets,
    std::span<const GlobalIndex> facet2vertex_data,
    std::span<const GlobalIndex> edge2vertex_data,
    const DofExtractionOptions& options) {

    std::vector<GlobalIndex> result;
    std::unordered_set<GlobalIndex> seen;

    if (facet_boundary_labels.size() + 1u != facet2vertex_offsets.size()) {
        throw FEException("DofTools::extractBoundaryDofs: facet2vertex_offsets size must be facet_boundary_labels.size()+1");
    }
    if (!facet2vertex_offsets.empty() && facet2vertex_offsets.front() != 0) {
        throw FEException("DofTools::extractBoundaryDofs: facet2vertex_offsets[0] must be 0");
    }
    if (!facet2vertex_offsets.empty() &&
        static_cast<std::size_t>(facet2vertex_offsets.back()) != facet2vertex_data.size()) {
        throw FEException("DofTools::extractBoundaryDofs: facet2vertex_offsets.back() must equal facet2vertex_data.size()");
    }

    auto push_dof = [&](GlobalIndex dof) {
        if (!options.remove_duplicates) {
            result.push_back(dof);
            return;
        }
        if (seen.insert(dof).second) {
            result.push_back(dof);
        }
    };

    auto push_span = [&](std::span<const GlobalIndex> dofs) {
        for (auto d : dofs) {
            push_dof(d);
        }
    };

    auto get_facet_vertices = [&](GlobalIndex facet_id) -> std::span<const GlobalIndex> {
        if (facet_id < 0) return {};
        const auto fid = static_cast<std::size_t>(facet_id);
        const auto begin = static_cast<std::size_t>(facet2vertex_offsets[fid]);
        const auto end = static_cast<std::size_t>(facet2vertex_offsets[fid + 1]);
        if (begin > end || end > facet2vertex_data.size()) {
            throw FEException("DofTools::extractBoundaryDofs: facet2vertex offsets out of range");
        }
        return {facet2vertex_data.data() + begin, end - begin};
    };

    struct EdgeKey {
        GlobalIndex a;
        GlobalIndex b;
        bool operator==(const EdgeKey& other) const noexcept { return a == other.a && b == other.b; }
    };
    struct EdgeKeyHash {
        std::size_t operator()(const EdgeKey& k) const noexcept {
            const std::size_t h1 = std::hash<GlobalIndex>{}(k.a);
            const std::size_t h2 = std::hash<GlobalIndex>{}(k.b);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    std::unordered_map<EdgeKey, GlobalIndex, EdgeKeyHash> edge_ids;
    if (entity_map.numEdges() > 0) {
        if (edge2vertex_data.size() < static_cast<std::size_t>(2) * static_cast<std::size_t>(entity_map.numEdges())) {
            throw FEException("DofTools::extractBoundaryDofs: edge2vertex_data too small for EntityDofMap edge count");
        }
        edge_ids.reserve(static_cast<std::size_t>(entity_map.numEdges()));
        for (GlobalIndex e = 0; e < entity_map.numEdges(); ++e) {
            const auto idx = static_cast<std::size_t>(2 * e);
            const GlobalIndex v0 = edge2vertex_data[idx];
            const GlobalIndex v1 = edge2vertex_data[idx + 1];
            EdgeKey key{std::min(v0, v1), std::max(v0, v1)};
            edge_ids.emplace(key, e);
        }
    }

    // Iterate over boundary facets and collect DOFs on vertices/edges/faces.
    for (GlobalIndex f = 0; f < static_cast<GlobalIndex>(facet_boundary_labels.size()); ++f) {
        if (facet_boundary_labels[static_cast<std::size_t>(f)] != boundary_id) {
            continue;
        }

        const auto verts = get_facet_vertices(f);
        if (verts.empty()) {
            continue;
        }

        // Face-interior DOFs (3D facets only).
        if (verts.size() >= 3u) {
            if (entity_map.numFaces() <= 0 || f >= entity_map.numFaces()) {
                throw FEException("DofTools::extractBoundaryDofs: facet labels imply faces, but EntityDofMap has no matching faces");
            }
            push_span(entity_map.getFaceDofs(f));
        }

        // Vertex DOFs on the facet.
        for (auto v : verts) {
            push_span(entity_map.getVertexDofs(v));
        }

        // Edge-interior DOFs on the facet boundary.
        if (!edge_ids.empty()) {
            auto add_edge = [&](GlobalIndex v0, GlobalIndex v1) {
                EdgeKey key{std::min(v0, v1), std::max(v0, v1)};
                auto it = edge_ids.find(key);
                if (it == edge_ids.end()) {
                    throw FEException("DofTools::extractBoundaryDofs: facet edge not found in edge2vertex table");
                }
                push_span(entity_map.getEdgeDofs(it->second));
            };

            if (verts.size() == 2u) {
                add_edge(verts[0], verts[1]);
            } else {
                for (std::size_t i = 0; i < verts.size(); ++i) {
                    const auto v0 = verts[i];
                    const auto v1 = verts[(i + 1u) % verts.size()];
                    add_edge(v0, v1);
                }
            }
        }
    }

    if (options.sort_result) {
        std::sort(result.begin(), result.end());
        if (options.remove_duplicates) {
            result.erase(std::unique(result.begin(), result.end()), result.end());
        }
    }

    return result;
}

std::vector<GlobalIndex> extractBoundaryDofs(
    const EntityDofMap& entity_map,
    int boundary_id,
    std::span<const int> facet_boundary_labels,
    std::span<const GlobalIndex> facet2vertex_offsets,
    std::span<const GlobalIndex> facet2vertex_data,
    std::span<const GlobalIndex> edge2vertex_data,
    const FieldMask& field_mask,
    std::span<const GlobalIndex> dofs_per_field,
    const DofExtractionOptions& options) {

    // First get all boundary DOFs
    auto all_boundary_dofs = extractBoundaryDofs(
        entity_map, boundary_id, facet_boundary_labels,
        facet2vertex_offsets, facet2vertex_data, edge2vertex_data, options);

    // If all fields selected, return all
    if (field_mask.numSelected() == dofs_per_field.size()) {
        return all_boundary_dofs;
    }

    // Compute field ranges
    std::vector<std::pair<GlobalIndex, GlobalIndex>> field_ranges;
    GlobalIndex offset = 0;
    for (auto field_dofs : dofs_per_field) {
        field_ranges.emplace_back(offset, offset + field_dofs);
        offset += field_dofs;
    }

    // Filter by field mask
    std::vector<GlobalIndex> result;
    for (auto dof : all_boundary_dofs) {
        // Find which field this DOF belongs to
        for (std::size_t f = 0; f < field_ranges.size(); ++f) {
            if (field_mask.isSelected(f) &&
                dof >= field_ranges[f].first && dof < field_ranges[f].second) {
                result.push_back(dof);
                break;
            }
        }
    }

    if (options.sort_result) {
        std::sort(result.begin(), result.end());
    }

    return result;
}

std::vector<GlobalIndex> extractBoundaryDofs(
    const EntityDofMap& entity_map,
    int boundary_id,
    std::span<const int> facet_boundary_labels,
    std::span<const GlobalIndex> facet2vertex_offsets,
    std::span<const GlobalIndex> facet2vertex_data,
    std::span<const GlobalIndex> edge2vertex_data,
    const ComponentMask& component_mask,
    std::size_t n_components,
    const DofExtractionOptions& options) {

    // First get all boundary DOFs
    DofExtractionOptions temp_options = options;
    temp_options.sort_result = true;  // Need sorted for component filtering
    auto all_boundary_dofs = extractBoundaryDofs(
        entity_map, boundary_id, facet_boundary_labels,
        facet2vertex_offsets, facet2vertex_data, edge2vertex_data, temp_options);

    // If all components selected, return all
    if (component_mask.selectsAll()) {
        return all_boundary_dofs;
    }

    if (n_components == 0) {
        throw FEException("DofTools::extractBoundaryDofs: n_components must be > 0");
    }

    // Filter by component for both common layouts:
    // - Interleaved: component = dof % n_components
    // - Block-by-component: dof = scalar + comp * stride, component = dof / stride
    //
    // We infer stride from any entity that has component replication.
    auto infer_stride = [&]() -> std::optional<GlobalIndex> {
        auto try_span = [&](std::span<const GlobalIndex> dofs) -> std::optional<GlobalIndex> {
            if (dofs.size() < n_components) {
                return std::nullopt;
            }
            if ((dofs.size() % n_components) != 0u) {
                return std::nullopt;
            }
            const std::size_t per_comp = dofs.size() / n_components;
            if (per_comp == 0u) {
                return std::nullopt;
            }
            const GlobalIndex stride = dofs[per_comp] - dofs[0];
            if (stride <= 0) {
                return std::nullopt;
            }

            const std::size_t check = std::min<std::size_t>(per_comp, 3u);
            for (std::size_t comp = 1; comp < n_components; ++comp) {
                const GlobalIndex expected_off = static_cast<GlobalIndex>(comp) * stride;
                for (std::size_t d = 0; d < check; ++d) {
                    const GlobalIndex expected = dofs[d] + expected_off;
                    if (dofs[comp * per_comp + d] != expected) {
                        return std::nullopt;
                    }
                }
            }
            return stride;
        };

        for (GlobalIndex v = 0; v < entity_map.numVertices(); ++v) {
            if (auto s = try_span(entity_map.getVertexDofs(v)); s.has_value()) return s;
        }
        for (GlobalIndex e = 0; e < entity_map.numEdges(); ++e) {
            if (auto s = try_span(entity_map.getEdgeDofs(e)); s.has_value()) return s;
        }
        for (GlobalIndex f = 0; f < entity_map.numFaces(); ++f) {
            if (auto s = try_span(entity_map.getFaceDofs(f)); s.has_value()) return s;
        }
        for (GlobalIndex c = 0; c < entity_map.numCells(); ++c) {
            if (auto s = try_span(entity_map.getCellInteriorDofs(c)); s.has_value()) return s;
        }
        return std::nullopt;
    };

    const auto stride_opt = infer_stride();
    if (!stride_opt.has_value()) {
        throw FEException("DofTools::extractBoundaryDofs: cannot infer component DOF layout from EntityDofMap");
    }
    const GlobalIndex stride = *stride_opt;
    const bool is_interleaved = (stride == 1);

    std::vector<GlobalIndex> result;
    for (auto dof : all_boundary_dofs) {
        if (dof < 0) {
            continue;
        }
        std::size_t component = 0;
        if (is_interleaved) {
            component = static_cast<std::size_t>(dof % static_cast<GlobalIndex>(n_components));
        } else {
            component = static_cast<std::size_t>(dof / stride);
        }
        if (component_mask.isSelected(component)) {
            result.push_back(dof);
        }
    }

    if (options.sort_result) {
        std::sort(result.begin(), result.end());
    }

    return result;
}

std::vector<GlobalIndex> extractAllBoundaryDofs(
    const EntityDofMap& entity_map,
    std::span<const int> facet_boundary_labels,
    std::span<const GlobalIndex> facet2vertex_offsets,
    std::span<const GlobalIndex> facet2vertex_data,
    std::span<const GlobalIndex> edge2vertex_data) {

    std::unordered_set<GlobalIndex> seen;
    std::vector<GlobalIndex> result;

    if (facet_boundary_labels.size() + 1u != facet2vertex_offsets.size()) {
        throw FEException("DofTools::extractAllBoundaryDofs: facet2vertex_offsets size must be facet_boundary_labels.size()+1");
    }
    if (!facet2vertex_offsets.empty() && facet2vertex_offsets.front() != 0) {
        throw FEException("DofTools::extractAllBoundaryDofs: facet2vertex_offsets[0] must be 0");
    }
    if (!facet2vertex_offsets.empty() &&
        static_cast<std::size_t>(facet2vertex_offsets.back()) != facet2vertex_data.size()) {
        throw FEException("DofTools::extractAllBoundaryDofs: facet2vertex_offsets.back() must equal facet2vertex_data.size()");
    }

    auto push_dof = [&](GlobalIndex dof) {
        if (seen.insert(dof).second) {
            result.push_back(dof);
        }
    };

    auto push_span = [&](std::span<const GlobalIndex> dofs) {
        for (auto d : dofs) {
            push_dof(d);
        }
    };

    auto get_facet_vertices = [&](GlobalIndex facet_id) -> std::span<const GlobalIndex> {
        if (facet_id < 0) return {};
        const auto fid = static_cast<std::size_t>(facet_id);
        const auto begin = static_cast<std::size_t>(facet2vertex_offsets[fid]);
        const auto end = static_cast<std::size_t>(facet2vertex_offsets[fid + 1]);
        if (begin > end || end > facet2vertex_data.size()) {
            throw FEException("DofTools::extractAllBoundaryDofs: facet2vertex offsets out of range");
        }
        return {facet2vertex_data.data() + begin, end - begin};
    };

    struct EdgeKey {
        GlobalIndex a;
        GlobalIndex b;
        bool operator==(const EdgeKey& other) const noexcept { return a == other.a && b == other.b; }
    };
    struct EdgeKeyHash {
        std::size_t operator()(const EdgeKey& k) const noexcept {
            const std::size_t h1 = std::hash<GlobalIndex>{}(k.a);
            const std::size_t h2 = std::hash<GlobalIndex>{}(k.b);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    std::unordered_map<EdgeKey, GlobalIndex, EdgeKeyHash> edge_ids;
    if (entity_map.numEdges() > 0) {
        if (edge2vertex_data.size() < static_cast<std::size_t>(2) * static_cast<std::size_t>(entity_map.numEdges())) {
            throw FEException("DofTools::extractAllBoundaryDofs: edge2vertex_data too small for EntityDofMap edge count");
        }
        edge_ids.reserve(static_cast<std::size_t>(entity_map.numEdges()));
        for (GlobalIndex e = 0; e < entity_map.numEdges(); ++e) {
            const auto idx = static_cast<std::size_t>(2 * e);
            const GlobalIndex v0 = edge2vertex_data[idx];
            const GlobalIndex v1 = edge2vertex_data[idx + 1];
            EdgeKey key{std::min(v0, v1), std::max(v0, v1)};
            edge_ids.emplace(key, e);
        }
    }

    for (GlobalIndex f = 0; f < static_cast<GlobalIndex>(facet_boundary_labels.size()); ++f) {
        if (facet_boundary_labels[static_cast<std::size_t>(f)] == 0) {
            continue;
        }

        const auto verts = get_facet_vertices(f);
        if (verts.empty()) {
            continue;
        }

        if (verts.size() >= 3u && entity_map.numFaces() > 0) {
            if (f >= entity_map.numFaces()) {
                throw FEException("DofTools::extractAllBoundaryDofs: facet labels imply faces, but EntityDofMap face count is smaller");
            }
            push_span(entity_map.getFaceDofs(f));
        }

        for (auto v : verts) {
            push_span(entity_map.getVertexDofs(v));
        }

        if (!edge_ids.empty()) {
            auto add_edge = [&](GlobalIndex v0, GlobalIndex v1) {
                EdgeKey key{std::min(v0, v1), std::max(v0, v1)};
                auto it = edge_ids.find(key);
                if (it == edge_ids.end()) {
                    throw FEException("DofTools::extractAllBoundaryDofs: facet edge not found in edge2vertex table");
                }
                push_span(entity_map.getEdgeDofs(it->second));
            };

            if (verts.size() == 2u) {
                add_edge(verts[0], verts[1]);
            } else {
                for (std::size_t i = 0; i < verts.size(); ++i) {
                    add_edge(verts[i], verts[(i + 1u) % verts.size()]);
                }
            }
        }
    }

    std::sort(result.begin(), result.end());
    return result;
}

// =========================================================================
// Subdomain DOF Extraction
// =========================================================================

IndexSet extractSubdomainDofs(
    const DofMap& dof_map,
    int subdomain_id,
    std::span<const int> cell_subdomain_labels) {

    std::unordered_set<GlobalIndex> dof_set;

    auto n_cells = dof_map.getNumCells();
    for (GlobalIndex c = 0; c < n_cells && c < static_cast<GlobalIndex>(cell_subdomain_labels.size()); ++c) {
        if (cell_subdomain_labels[static_cast<std::size_t>(c)] == subdomain_id) {
            auto cell_dofs = dof_map.getCellDofs(c);
            for (auto dof : cell_dofs) {
                dof_set.insert(dof);
            }
        }
    }

    std::vector<GlobalIndex> dofs(dof_set.begin(), dof_set.end());
    return IndexSet(std::move(dofs));
}

IndexSet extractSubdomainDofs(
    const DofMap& dof_map,
    std::span<const int> subdomain_ids,
    std::span<const int> cell_subdomain_labels) {

    std::unordered_set<int> target_subdomains(subdomain_ids.begin(), subdomain_ids.end());
    std::unordered_set<GlobalIndex> dof_set;

    auto n_cells = dof_map.getNumCells();
    for (GlobalIndex c = 0; c < n_cells && c < static_cast<GlobalIndex>(cell_subdomain_labels.size()); ++c) {
        if (target_subdomains.count(cell_subdomain_labels[static_cast<std::size_t>(c)]) > 0) {
            auto cell_dofs = dof_map.getCellDofs(c);
            for (auto dof : cell_dofs) {
                dof_set.insert(dof);
            }
        }
    }

    std::vector<GlobalIndex> dofs(dof_set.begin(), dof_set.end());
    return IndexSet(std::move(dofs));
}

IndexSet extractInteriorDofs(
    const EntityDofMap& entity_map,
    std::span<const int> face_boundary_labels) {

    // Get all boundary DOFs
    std::unordered_set<GlobalIndex> boundary_dofs;

    for (GlobalIndex f = 0; f < static_cast<GlobalIndex>(face_boundary_labels.size()); ++f) {
        if (face_boundary_labels[static_cast<std::size_t>(f)] != 0) {
            auto face_dofs = entity_map.getFaceDofs(f);
            for (auto dof : face_dofs) {
                boundary_dofs.insert(dof);
            }
        }
    }

    // Get cell interior DOFs (always interior)
    std::vector<GlobalIndex> interior_dofs;
    auto n_cells = entity_map.numCells();
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto cell_dofs = entity_map.getCellInteriorDofs(c);
        for (auto dof : cell_dofs) {
            interior_dofs.push_back(dof);
        }
    }

    // Also include non-boundary vertex/edge/face DOFs
    auto n_vertices = entity_map.numVertices();
    for (GlobalIndex v = 0; v < n_vertices; ++v) {
        auto vertex_dofs = entity_map.getVertexDofs(v);
        for (auto dof : vertex_dofs) {
            if (boundary_dofs.count(dof) == 0) {
                interior_dofs.push_back(dof);
            }
        }
    }

    auto n_edges = entity_map.numEdges();
    for (GlobalIndex e = 0; e < n_edges; ++e) {
        auto edge_dofs = entity_map.getEdgeDofs(e);
        for (auto dof : edge_dofs) {
            if (boundary_dofs.count(dof) == 0) {
                interior_dofs.push_back(dof);
            }
        }
    }

    return IndexSet(std::move(interior_dofs));
}

// =========================================================================
// Entity-Based Extraction
// =========================================================================

std::vector<GlobalIndex> extractEntityDofs(
    const EntityDofMap& entity_map,
    EntityKind kind,
    std::span<const GlobalIndex> entity_ids) {

    std::vector<GlobalIndex> result;

    for (auto id : entity_ids) {
        auto dofs = entity_map.getEntityDofs(kind, id);
        result.insert(result.end(), dofs.begin(), dofs.end());
    }

    sortAndUnique(result);
    return result;
}

std::vector<GlobalIndex> extractAllVertexDofs(const EntityDofMap& entity_map) {
    std::vector<GlobalIndex> result;

    auto n_vertices = entity_map.numVertices();
    for (GlobalIndex v = 0; v < n_vertices; ++v) {
        auto dofs = entity_map.getVertexDofs(v);
        result.insert(result.end(), dofs.begin(), dofs.end());
    }

    sortAndUnique(result);
    return result;
}

std::vector<GlobalIndex> extractAllEdgeDofs(const EntityDofMap& entity_map) {
    std::vector<GlobalIndex> result;

    auto n_edges = entity_map.numEdges();
    for (GlobalIndex e = 0; e < n_edges; ++e) {
        auto dofs = entity_map.getEdgeDofs(e);
        result.insert(result.end(), dofs.begin(), dofs.end());
    }

    sortAndUnique(result);
    return result;
}

std::vector<GlobalIndex> extractAllFaceDofs(const EntityDofMap& entity_map) {
    std::vector<GlobalIndex> result;

    auto n_faces = entity_map.numFaces();
    for (GlobalIndex f = 0; f < n_faces; ++f) {
        auto dofs = entity_map.getFaceDofs(f);
        result.insert(result.end(), dofs.begin(), dofs.end());
    }

    sortAndUnique(result);
    return result;
}

std::vector<GlobalIndex> extractAllCellInteriorDofs(const EntityDofMap& entity_map) {
    std::vector<GlobalIndex> result;

    auto n_cells = entity_map.numCells();
    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto dofs = entity_map.getCellInteriorDofs(c);
        result.insert(result.end(), dofs.begin(), dofs.end());
    }

    sortAndUnique(result);
    return result;
}

IndexSet extractInterfaceDofs(const EntityDofMap& entity_map) {
    auto interface_vec = entity_map.getInterfaceDofs();
    return IndexSet(std::move(interface_vec));
}

std::vector<EntityRef> getDofSupportEntities(
    GlobalIndex dof_id,
    const EntityDofMap& entity_map,
    const MeshTopologyInfo& topology) {

    std::vector<EntityRef> result;

    const auto primary_opt = entity_map.getDofEntity(dof_id);
    if (!primary_opt) {
        return result;
    }

    const EntityRef primary = *primary_opt;
    result.push_back(primary);

    auto contains = [](std::span<const MeshIndex> ids, MeshIndex value) -> bool {
        return std::find(ids.begin(), ids.end(), value) != ids.end();
    };

    // Helper: get vertices of a face from topology.face2vertex CSR.
    auto get_face_vertices = [&](GlobalIndex face_id) -> std::span<const MeshIndex> {
        if (topology.face2vertex_offsets.empty()) return {};
        if (face_id < 0 || face_id >= topology.n_faces) return {};
        const auto fid = static_cast<std::size_t>(face_id);
        if (fid + 1 >= topology.face2vertex_offsets.size()) return {};
        const auto begin = static_cast<std::size_t>(topology.face2vertex_offsets[fid]);
        const auto end = static_cast<std::size_t>(topology.face2vertex_offsets[fid + 1]);
        if (begin > end || end > topology.face2vertex_data.size()) return {};
        return {topology.face2vertex_data.data() + begin, end - begin};
    };

    // Helper: get cell vertices from CSR.
    auto get_cell_vertices = [&](GlobalIndex cell_id) {
        return topology.getCellVertices(cell_id);
    };

    // Helper: get cell edges from CSR.
    auto get_cell_edges = [&](GlobalIndex cell_id) {
        return topology.getCellEdges(cell_id);
    };

    // Helper: get cell faces from CSR.
    auto get_cell_faces = [&](GlobalIndex cell_id) {
        return topology.getCellFaces(cell_id);
    };

    const auto add_unique = [&](EntityKind kind, GlobalIndex id) {
        if (id < 0) return;
        result.push_back(EntityRef{kind, id});
    };

    if (primary.kind == EntityKind::Vertex) {
        const MeshIndex v = static_cast<MeshIndex>(primary.id);

        // Edges containing v.
        if (!topology.edge2vertex_data.empty() && topology.n_edges > 0) {
            const auto expected = static_cast<std::size_t>(2) * static_cast<std::size_t>(topology.n_edges);
            if (topology.edge2vertex_data.size() >= expected) {
                for (GlobalIndex e = 0; e < topology.n_edges; ++e) {
                    const auto idx = static_cast<std::size_t>(2 * e);
                    const auto a = topology.edge2vertex_data[idx];
                    const auto b = topology.edge2vertex_data[idx + 1];
                    if (a == v || b == v) {
                        add_unique(EntityKind::Edge, e);
                    }
                }
            }
        }

        // Faces containing v.
        if (!topology.face2vertex_offsets.empty() && topology.n_faces > 0) {
            for (GlobalIndex f = 0; f < topology.n_faces; ++f) {
                if (contains(get_face_vertices(f), v)) {
                    add_unique(EntityKind::Face, f);
                }
            }
        }

        // Cells containing v.
        for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
            if (contains(get_cell_vertices(c), v)) {
                add_unique(EntityKind::Cell, c);
            }
        }
    } else if (primary.kind == EntityKind::Edge) {
        const GlobalIndex e = primary.id;

        // Endpoints for edge->face/cell queries.
        MeshIndex v0 = -1;
        MeshIndex v1 = -1;
        if (!topology.edge2vertex_data.empty() && e >= 0 && e < topology.n_edges) {
            const auto idx = static_cast<std::size_t>(2 * e);
            if (idx + 1 < topology.edge2vertex_data.size()) {
                v0 = topology.edge2vertex_data[idx];
                v1 = topology.edge2vertex_data[idx + 1];
            }
        }

        // Faces containing both endpoints.
        if (v0 >= 0 && v1 >= 0 && !topology.face2vertex_offsets.empty() && topology.n_faces > 0) {
            for (GlobalIndex f = 0; f < topology.n_faces; ++f) {
                const auto fv = get_face_vertices(f);
                if (contains(fv, v0) && contains(fv, v1)) {
                    add_unique(EntityKind::Face, f);
                }
            }
        }

        // Cells containing this edge.
        if (!topology.cell2edge_offsets.empty() && topology.n_cells > 0) {
            for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
                if (contains(get_cell_edges(c), static_cast<MeshIndex>(e))) {
                    add_unique(EntityKind::Cell, c);
                }
            }
        } else if (v0 >= 0 && v1 >= 0) {
            for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
                const auto cv = get_cell_vertices(c);
                if (contains(cv, v0) && contains(cv, v1)) {
                    add_unique(EntityKind::Cell, c);
                }
            }
        }
    } else if (primary.kind == EntityKind::Face) {
        const GlobalIndex f = primary.id;

        // Cells containing this face.
        if (!topology.cell2face_offsets.empty() && topology.n_cells > 0) {
            for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
                if (contains(get_cell_faces(c), static_cast<MeshIndex>(f))) {
                    add_unique(EntityKind::Cell, c);
                }
            }
        } else {
            // Fallback: match by vertex set if face2vertex and cell2vertex exist.
            const auto fv = get_face_vertices(f);
            if (!fv.empty()) {
                for (GlobalIndex c = 0; c < topology.n_cells; ++c) {
                    const auto cv = get_cell_vertices(c);
                    bool all = true;
                    for (auto v : fv) {
                        if (!contains(cv, v)) {
                            all = false;
                            break;
                        }
                    }
                    if (all) {
                        add_unique(EntityKind::Cell, c);
                    }
                }
            }
        }
    } else {
        // Cell-interior DOFs are only supported by the owning cell.
    }

    std::sort(result.begin(), result.end(), [](const EntityRef& a, const EntityRef& b) {
        if (a.kind != b.kind) {
            return static_cast<std::uint8_t>(a.kind) < static_cast<std::uint8_t>(b.kind);
        }
        return a.id < b.id;
    });
    result.erase(std::unique(result.begin(), result.end(),
                             [](const EntityRef& a, const EntityRef& b) { return a == b; }),
                 result.end());
    return result;
}

// =========================================================================
// Geometric Predicate Extraction
// =========================================================================

std::vector<GlobalIndex> extractDofsInRegion(
    const DofMap& /*dof_map*/,
    const EntityDofMap& vertex_dof_map,
    std::span<const double> vertex_coords,
    int dim,
    GeometricPredicate predicate) {

    std::vector<GlobalIndex> result;

    auto n_vertices = vertex_dof_map.numVertices();
    for (GlobalIndex v = 0; v < n_vertices; ++v) {
        auto idx = static_cast<std::size_t>(v);
        double x = (idx * dim < vertex_coords.size()) ? vertex_coords[idx * dim] : 0.0;
        double y = (dim >= 2 && idx * dim + 1 < vertex_coords.size()) ? vertex_coords[idx * dim + 1] : 0.0;
        double z = (dim >= 3 && idx * dim + 2 < vertex_coords.size()) ? vertex_coords[idx * dim + 2] : 0.0;

        if (predicate(x, y, z)) {
            auto dofs = vertex_dof_map.getVertexDofs(v);
            result.insert(result.end(), dofs.begin(), dofs.end());
        }
    }

    sortAndUnique(result);
    return result;
}

std::vector<GlobalIndex> extractDofsInBox(
    const DofMap& dof_map,
    const EntityDofMap& vertex_dof_map,
    std::span<const double> vertex_coords,
    int dim,
    std::span<const double> min_corner,
    std::span<const double> max_corner) {

    auto box_predicate = [&](double x, double y, double z) {
        if (dim >= 1) {
            if (x < min_corner[0] || x > max_corner[0]) return false;
        }
        if (dim >= 2) {
            if (y < min_corner[1] || y > max_corner[1]) return false;
        }
        if (dim >= 3) {
            if (z < min_corner[2] || z > max_corner[2]) return false;
        }
        return true;
    };

    return extractDofsInRegion(dof_map, vertex_dof_map, vertex_coords, dim, box_predicate);
}

std::vector<GlobalIndex> extractDofsInSphere(
    const DofMap& dof_map,
    const EntityDofMap& vertex_dof_map,
    std::span<const double> vertex_coords,
    int dim,
    std::span<const double> center,
    double radius) {

    double r2 = radius * radius;

    auto sphere_predicate = [&](double x, double y, double z) {
        double dx = x - center[0];
        double dy = (dim >= 2) ? (y - center[1]) : 0.0;
        double dz = (dim >= 3) ? (z - center[2]) : 0.0;
        return (dx*dx + dy*dy + dz*dz) <= r2;
    };

    return extractDofsInRegion(dof_map, vertex_dof_map, vertex_coords, dim, sphere_predicate);
}

std::vector<GlobalIndex> extractDofsOnPlane(
    const DofMap& dof_map,
    const EntityDofMap& vertex_dof_map,
    std::span<const double> vertex_coords,
    int dim,
    std::span<const double> normal,
    std::span<const double> point,
    double tolerance) {

    // Compute plane constant d from n.p = d
    double d = 0.0;
    for (int i = 0; i < dim; ++i) {
        d += normal[i] * point[i];
    }

    auto plane_predicate = [&](double x, double y, double z) {
        double dist = 0.0;
        dist += normal[0] * x;
        if (dim >= 2) dist += normal[1] * y;
        if (dim >= 3) dist += normal[2] * z;
        dist -= d;
        return std::abs(dist) <= tolerance;
    };

    return extractDofsInRegion(dof_map, vertex_dof_map, vertex_coords, dim, plane_predicate);
}

// =========================================================================
// Partition Interface DOFs (Parallel)
// =========================================================================

IndexSet extractPartitionInterfaceDofs(
    const DofMap& dof_map,
    std::span<const bool> owned_cell_mask) {

    std::unordered_set<GlobalIndex> owned_dofs;
    std::unordered_set<GlobalIndex> ghost_dofs;

    auto n_cells = dof_map.getNumCells();
    for (GlobalIndex c = 0; c < n_cells && c < static_cast<GlobalIndex>(owned_cell_mask.size()); ++c) {
        auto cell_dofs = dof_map.getCellDofs(c);

        if (owned_cell_mask[static_cast<std::size_t>(c)]) {
            for (auto dof : cell_dofs) {
                owned_dofs.insert(dof);
            }
        } else {
            for (auto dof : cell_dofs) {
                ghost_dofs.insert(dof);
            }
        }
    }

    // Interface DOFs are those that appear in both owned and ghost cells
    std::vector<GlobalIndex> interface_dofs;
    for (auto dof : owned_dofs) {
        if (ghost_dofs.count(dof) > 0) {
            interface_dofs.push_back(dof);
        }
    }

    return IndexSet(std::move(interface_dofs));
}

// =========================================================================
// DOF Statistics and Analysis
// =========================================================================

DofCountsByEntity countDofsByEntity(const EntityDofMap& entity_map) {
    auto stats = entity_map.getStatistics();

    DofCountsByEntity counts;
    counts.vertex_dofs = stats.n_vertex_dofs;
    counts.edge_dofs = stats.n_edge_dofs;
    counts.face_dofs = stats.n_face_dofs;
    counts.cell_interior_dofs = stats.n_cell_interior_dofs;
    counts.total = stats.total_dofs;

    return counts;
}

DofDistributionStats computeDistributionStats(const DofMap& dof_map) {
    DofDistributionStats stats;
    stats.total_dofs = dof_map.getNumDofs();

    auto n_cells = dof_map.getNumCells();
    if (n_cells == 0) return stats;

    // Compute min, max, sum, sum of squares
    double min_val = std::numeric_limits<double>::max();
    double max_val = 0.0;
    double sum = 0.0;
    double sum_sq = 0.0;

    for (GlobalIndex c = 0; c < n_cells; ++c) {
        auto n_dofs = static_cast<double>(dof_map.getNumCellDofs(c));
        min_val = std::min(min_val, n_dofs);
        max_val = std::max(max_val, n_dofs);
        sum += n_dofs;
        sum_sq += n_dofs * n_dofs;
    }

    auto n = static_cast<double>(n_cells);
    stats.min_dofs_per_cell = min_val;
    stats.max_dofs_per_cell = max_val;
    stats.avg_dofs_per_cell = sum / n;

    // Standard deviation
    double variance = (sum_sq - (sum * sum) / n) / n;
    stats.std_dev_dofs_per_cell = std::sqrt(std::max(0.0, variance));

    return stats;
}

// =========================================================================
// Utility Functions
// =========================================================================

void sortAndUnique(std::vector<GlobalIndex>& dofs) {
    std::sort(dofs.begin(), dofs.end());
    dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
}

IndexSet toIndexSet(std::vector<GlobalIndex> dofs) {
    return IndexSet(std::move(dofs));
}

std::vector<GlobalIndex> toVector(const IndexSet& index_set) {
    return index_set.toVector();
}

IndexSet complement(const IndexSet& dofs, GlobalIndex n_total_dofs) {
    std::vector<GlobalIndex> result;
    result.reserve(static_cast<std::size_t>(n_total_dofs - dofs.size()));

    for (GlobalIndex i = 0; i < n_total_dofs; ++i) {
        if (!dofs.contains(i)) {
            result.push_back(i);
        }
    }

    return IndexSet(std::move(result));
}

} // namespace DofTools

} // namespace dofs
} // namespace FE
} // namespace svmp
