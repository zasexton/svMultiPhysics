/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/LevelSetActiveSideVertexDirichletConstraint.h"

#include "Assembly/CutIntegrationContext.h"
#include "Dofs/EntityDofMap.h"
#include "Core/Logger.h"
#include "Spaces/FunctionSpace.h"
#include "Systems/FESystem.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

namespace svmp {
namespace FE {
namespace constraints {

namespace {

[[nodiscard]] const char* sideName(LevelSetConstraintSide side) noexcept
{
    return side == LevelSetConstraintSide::Negative ? "Negative" : "Positive";
}

[[nodiscard]] bool isActive(Real phi, Real isovalue, LevelSetConstraintSide side) noexcept
{
    return side == LevelSetConstraintSide::Negative ? phi <= isovalue : phi >= isovalue;
}

[[nodiscard]] bool hasPositiveActiveSideMeasure(
    Real phi,
    Real isovalue,
    LevelSetConstraintSide side) noexcept
{
    return side == LevelSetConstraintSide::Negative ? phi < isovalue : phi > isovalue;
}

[[nodiscard]] geometry::CutIntegrationSide toCutIntegrationSide(
    LevelSetConstraintSide side) noexcept
{
    return side == LevelSetConstraintSide::Negative
               ? geometry::CutIntegrationSide::Negative
               : geometry::CutIntegrationSide::Positive;
}

[[nodiscard]] std::string formatRuns(const std::vector<GlobalIndex>& values,
                                     std::size_t max_runs = 8u)
{
    if (values.empty()) {
        return "none";
    }

    std::ostringstream oss;
    std::size_t emitted = 0u;
    std::size_t i = 0u;
    while (i < values.size() && emitted < max_runs) {
        const auto begin = values[i];
        auto end = begin;
        ++i;
        while (i < values.size() && values[i] == end + 1) {
            end = values[i];
            ++i;
        }
        if (emitted > 0u) {
            oss << '|';
        }
        oss << begin;
        if (end != begin) {
            oss << '-' << end;
        }
        ++emitted;
    }
    if (i < values.size()) {
        oss << "|...";
    }
    return oss.str();
}

[[nodiscard]] std::string formatDofList(std::span<const GlobalIndex> values)
{
    if (values.empty()) {
        return "none";
    }
    std::ostringstream oss;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0u) {
            oss << '|';
        }
        oss << values[i];
    }
    return oss.str();
}

[[nodiscard]] const char* entityKindName(dofs::EntityKind kind) noexcept
{
    switch (kind) {
    case dofs::EntityKind::Vertex:
        return "Vertex";
    case dofs::EntityKind::Edge:
        return "Edge";
    case dofs::EntityKind::Face:
        return "Face";
    case dofs::EntityKind::Cell:
        return "Cell";
    }
    return "Unknown";
}

void appendSampleDofsFromEnv(const char* raw, std::vector<GlobalIndex>& out)
{
    if (raw == nullptr || raw[0] == '\0') {
        return;
    }

    std::string text(raw);
    for (auto& ch : text) {
        if (ch == ',' || ch == '|' || ch == ';') {
            ch = ' ';
        }
    }

    std::istringstream stream(text);
    std::string token;
    while (stream >> token) {
        const auto dash_pos = token.find('-', token[0] == '-' ? 1u : 0u);
        try {
            if (dash_pos != std::string::npos) {
                const auto first = static_cast<GlobalIndex>(
                    std::stoll(token.substr(0u, dash_pos)));
                const auto last = static_cast<GlobalIndex>(
                    std::stoll(token.substr(dash_pos + 1u)));
                const auto step = first <= last ? GlobalIndex{1} : GlobalIndex{-1};
                for (GlobalIndex value = first;; value += step) {
                    out.push_back(value);
                    if (value == last) {
                        break;
                    }
                }
            } else {
                out.push_back(static_cast<GlobalIndex>(std::stoll(token)));
            }
        } catch (const std::exception&) {
            FE_LOG_INFO(
                "LevelSetActiveSideVertexDirichletConstraint: diagnostic=level_set_active_side_vertex_constraint_sample_parse skipped_token='" +
                token + "'");
        }
    }
}

[[nodiscard]] std::vector<GlobalIndex> sampledConstraintDofs(
    const std::string& field_name)
{
    std::vector<GlobalIndex> out;
    appendSampleDofsFromEnv(
        std::getenv("SVMP_LEVEL_SET_ACTIVE_CONSTRAINT_SAMPLE_DOFS"), out);
    if (field_name == "Pressure" || field_name == "p") {
        appendSampleDofsFromEnv(
            std::getenv("SVMP_ACTIVE_PRESSURE_CONSTRAINT_SAMPLE_DOFS"), out);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

struct EntityDofSupportCounts {
    std::size_t vertex{0u};
    std::size_t edge{0u};
    std::size_t face{0u};
    std::size_t cell{0u};
    std::size_t unknown{0u};
};

void incrementEntityDofCount(EntityDofSupportCounts& counts,
                             const dofs::EntityDofMap& entity_map,
                             GlobalIndex local_dof)
{
    const auto entity = entity_map.getDofEntity(local_dof);
    if (!entity.has_value()) {
        ++counts.unknown;
        return;
    }

    switch (entity->kind) {
    case dofs::EntityKind::Vertex:
        ++counts.vertex;
        break;
    case dofs::EntityKind::Edge:
        ++counts.edge;
        break;
    case dofs::EntityKind::Face:
        ++counts.face;
        break;
    case dofs::EntityKind::Cell:
        ++counts.cell;
        break;
    }
}

struct EdgeKey {
    GlobalIndex a;
    GlobalIndex b;

    bool operator==(const EdgeKey& other) const noexcept
    {
        return a == other.a && b == other.b;
    }
};

struct EdgeKeyHash {
    std::size_t operator()(const EdgeKey& key) const noexcept
    {
        const std::size_t h1 = std::hash<GlobalIndex>{}(key.a);
        const std::size_t h2 = std::hash<GlobalIndex>{}(key.b);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

struct LevelSetVertexView {
    const Real* values{nullptr};
    std::size_t components{0};
    std::size_t entity_count{0};
};

[[nodiscard]] LevelSetVertexView levelSetVertexView(
    const systems::FESystem& system,
    const std::string& field_name,
    GlobalIndex min_vertices)
{
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        throw std::runtime_error(
            "LevelSetActiveSideVertexDirichletConstraint: native mesh support is required");
    }

    const auto& mesh = native_mesh->local_mesh();
    if (!MeshFields::has_field(mesh, EntityKind::Vertex, field_name)) {
        throw std::runtime_error(
            "LevelSetActiveSideVertexDirichletConstraint: missing vertex level-set field '" +
            field_name + "'");
    }

    const auto handle = MeshFields::get_field_handle(mesh, EntityKind::Vertex, field_name);
    if (MeshFields::field_type(mesh, handle) != FieldScalarType::Float64) {
        throw std::runtime_error(
            "LevelSetActiveSideVertexDirichletConstraint: vertex level-set field '" +
            field_name + "' must be Float64");
    }

    const auto components = MeshFields::field_components(mesh, handle);
    if (components < 1u) {
        throw std::runtime_error(
            "LevelSetActiveSideVertexDirichletConstraint: vertex level-set field '" +
            field_name + "' has no scalar component");
    }

    const auto entity_count = MeshFields::field_entity_count(mesh, handle);
    if (entity_count < static_cast<std::size_t>(min_vertices)) {
        throw std::runtime_error(
            "LevelSetActiveSideVertexDirichletConstraint: vertex level-set field '" +
            field_name + "' has fewer entries than the constrained field");
    }

    const auto* values = MeshFields::field_data_as<Real>(mesh, handle);
    if (values == nullptr) {
        throw std::runtime_error(
            "LevelSetActiveSideVertexDirichletConstraint: vertex level-set field '" +
            field_name + "' has no data");
    }
    return LevelSetVertexView{values, components, entity_count};
#else
    (void)system;
    (void)field_name;
    (void)min_vertices;
    throw std::runtime_error(
        "LevelSetActiveSideVertexDirichletConstraint: native mesh support is required");
#endif
}

} // namespace

LevelSetActiveSideVertexDirichletConstraint::
    LevelSetActiveSideVertexDirichletConstraint(FieldId field,
                                                std::string level_set_field_name,
                                                LevelSetConstraintSide active_side,
                                                Real isovalue,
                                                Real inactive_value,
                                                int interface_marker)
    : field_(field)
    , level_set_field_name_(std::move(level_set_field_name))
    , active_side_(active_side)
    , isovalue_(isovalue)
    , inactive_value_(inactive_value)
    , interface_marker_(interface_marker)
{
    if (field_ == INVALID_FIELD_ID) {
        throw std::invalid_argument(
            "LevelSetActiveSideVertexDirichletConstraint: invalid FieldId");
    }
    if (level_set_field_name_.empty()) {
        throw std::invalid_argument(
            "LevelSetActiveSideVertexDirichletConstraint: level-set field name is required");
    }
}

void LevelSetActiveSideVertexDirichletConstraint::apply(
    const systems::FESystem& system,
    AffineConstraints& constraints)
{
    const auto& rec = system.fieldRecord(field_);
    if (!rec.space) {
        throw std::invalid_argument(
            "LevelSetActiveSideVertexDirichletConstraint: field has no function space");
    }
    const bool scalar_or_product_h1 =
        rec.space->space_type() == spaces::SpaceType::H1 ||
        rec.space->space_type() == spaces::SpaceType::Product;
    if (!scalar_or_product_h1 ||
        rec.space->continuity() != Continuity::C0 ||
        rec.space->value_dimension() != rec.components ||
        rec.space->element().basis().is_vector_valued()) {
        throw std::invalid_argument(
            "LevelSetActiveSideVertexDirichletConstraint requires an H1/C0 scalar field or Product H1 vector field");
    }

    const auto& dh = system.fieldDofHandler(field_);
    const auto* entity_map = dh.getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::invalid_argument(
            "LevelSetActiveSideVertexDirichletConstraint: field DofHandler has no EntityDofMap");
    }

    const auto n_vertices = entity_map->numVertices();
    const auto level_set = levelSetVertexView(system, level_set_field_name_, n_vertices);
    const auto offset = system.fieldDofOffset(field_);
    const auto& owned = system.dofHandler().getPartition().locallyOwned();
    const auto n_field_dofs = dh.getNumDofs();

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto& mesh = system.mesh()->local_mesh();
    const bool has_subvertex_trace_dofs = n_field_dofs > n_vertices;
    std::vector<std::array<index_t, 2>> derived_edge_vertices;
    const auto* edge_vertices_by_id = &mesh.edge2vertex();
    if (has_subvertex_trace_dofs && entity_map->numEdges() > 0 &&
        edge_vertices_by_id->empty() &&
        mesh.dim() == 2) {
        const auto& face_offsets = mesh.face2vertex_offsets();
        const auto& face_vertices = mesh.face2vertex();
        derived_edge_vertices.resize(mesh.n_faces());
        for (index_t face = 0; face < static_cast<index_t>(mesh.n_faces());
             ++face) {
            const auto begin =
                static_cast<std::size_t>(face_offsets[static_cast<std::size_t>(face)]);
            const auto end = static_cast<std::size_t>(
                face_offsets[static_cast<std::size_t>(face) + 1u]);
            if (end - begin != 2u) {
                throw std::runtime_error(
                    "LevelSetActiveSideVertexDirichletConstraint: expected two vertices per 2D facet");
            }
            derived_edge_vertices[static_cast<std::size_t>(face)] = {
                face_vertices[begin], face_vertices[begin + 1u]};
        }
        edge_vertices_by_id = &derived_edge_vertices;
    }

    std::unordered_map<EdgeKey, GlobalIndex, EdgeKeyHash> edge_ids;
    if (has_subvertex_trace_dofs && entity_map->numEdges() > 0) {
        if (edge_vertices_by_id->size() <
            static_cast<std::size_t>(entity_map->numEdges())) {
            throw std::runtime_error(
                "LevelSetActiveSideVertexDirichletConstraint: edge table is too small for the constrained field");
        }
        edge_ids.reserve(static_cast<std::size_t>(entity_map->numEdges()));
        for (GlobalIndex edge = 0; edge < entity_map->numEdges(); ++edge) {
            const auto& vertices =
                (*edge_vertices_by_id)[static_cast<std::size_t>(edge)];
            const auto v0 = static_cast<GlobalIndex>(vertices[0]);
            const auto v1 = static_cast<GlobalIndex>(vertices[1]);
            edge_ids.emplace(
                EdgeKey{std::min(v0, v1), std::max(v0, v1)}, edge);
        }
    }

    std::vector<unsigned char> has_active_support(
        static_cast<std::size_t>(n_vertices), static_cast<unsigned char>(0));
    std::vector<unsigned char> has_active_dof_support(
        static_cast<std::size_t>(n_field_dofs), static_cast<unsigned char>(0));
    std::vector<Real> active_dof_retained_measure(
        static_cast<std::size_t>(n_field_dofs), Real{0.0});
    std::vector<Real> active_dof_min_volume_fraction(
        static_cast<std::size_t>(n_field_dofs),
        std::numeric_limits<Real>::infinity());
    std::vector<Real> active_dof_max_volume_fraction(
        static_cast<std::size_t>(n_field_dofs),
        -std::numeric_limits<Real>::infinity());
    std::vector<std::size_t> active_dof_retained_rule_count(
        static_cast<std::size_t>(n_field_dofs), std::size_t{0u});
    std::size_t active_support_cells = 0u;
    std::size_t active_support_cells_from_volume_support = 0u;
    std::size_t active_support_cells_from_cut_adjacent_facets = 0u;
    std::string support_mode = "cell_patch";
    std::vector<unsigned char> active_cell_seen(
        mesh.n_cells(), static_cast<unsigned char>(0));
    const auto mark_local_dof_active = [&](GlobalIndex local_dof,
                                           const char* entity_name,
                                           GlobalIndex entity_id) {
        if (local_dof < 0 || local_dof >= n_field_dofs) {
            std::ostringstream oss;
            oss << "LevelSetActiveSideVertexDirichletConstraint: "
                << entity_name << ' ' << entity_id
                << " references field DOF " << local_dof
                << " outside field '" << rec.name << "'";
            throw std::runtime_error(oss.str());
        }
        has_active_dof_support[static_cast<std::size_t>(local_dof)] =
            static_cast<unsigned char>(1);
    };
    const auto mark_vertex_support = [&](GlobalIndex vertex) {
        if (vertex < 0 || vertex >= n_vertices) {
            return;
        }
        has_active_support[static_cast<std::size_t>(vertex)] =
            static_cast<unsigned char>(1);
        for (const auto local_dof : entity_map->getVertexDofs(vertex)) {
            mark_local_dof_active(local_dof, "vertex", vertex);
        }
    };
    const auto mark_edge_support = [&](GlobalIndex edge) {
        if (edge < 0 || edge >= entity_map->numEdges()) {
            std::ostringstream oss;
            oss << "LevelSetActiveSideVertexDirichletConstraint: active support references edge "
                << edge << " outside the constrained field entity map";
            throw std::runtime_error(oss.str());
        }
        for (const auto local_dof : entity_map->getEdgeDofs(edge)) {
            mark_local_dof_active(local_dof, "edge", edge);
        }
    };
    const auto mark_face_support = [&](GlobalIndex face) {
        if (face < 0 || face >= entity_map->numFaces()) {
            std::ostringstream oss;
            oss << "LevelSetActiveSideVertexDirichletConstraint: active support references face "
                << face << " outside the constrained field entity map";
            throw std::runtime_error(oss.str());
        }
        for (const auto local_dof : entity_map->getFaceDofs(face)) {
            mark_local_dof_active(local_dof, "face", face);
        }
    };
    const auto mark_cell_active = [&](GlobalIndex cell) -> bool {
        if (cell < 0 || static_cast<std::size_t>(cell) >= mesh.n_cells()) {
            std::ostringstream oss;
            oss << "LevelSetActiveSideVertexDirichletConstraint: active support references cell "
                << cell << " outside the mesh";
            throw std::runtime_error(oss.str());
        }
        if (active_cell_seen[static_cast<std::size_t>(cell)] !=
            static_cast<unsigned char>(0)) {
            return false;
        }
        active_cell_seen[static_cast<std::size_t>(cell)] =
            static_cast<unsigned char>(1);
        ++active_support_cells;

        for (const auto local_dof : dh.getCellDofs(cell)) {
            mark_local_dof_active(local_dof, "cell", cell);
        }

        const auto [vertices, count] =
            mesh.cell_vertices_span(static_cast<index_t>(cell));
        if (vertices == nullptr || count == 0u) {
            return true;
        }
        for (std::size_t i = 0; i < count; ++i) {
            mark_vertex_support(static_cast<GlobalIndex>(vertices[i]));
        }
        return true;
    };
    const auto record_retained_rule_support =
        [&](GlobalIndex cell, const geometry::CutQuadratureRule& rule) {
        if (cell < 0 || static_cast<std::size_t>(cell) >= mesh.n_cells()) {
            return;
        }
        const Real measure =
            std::isfinite(rule.measure) && rule.measure > Real{0.0}
                ? rule.measure
                : Real{0.0};
        const bool have_volume_fraction =
            std::isfinite(rule.volume_fraction);
        for (const auto local_dof : dh.getCellDofs(cell)) {
            if (local_dof < 0 || local_dof >= n_field_dofs) {
                continue;
            }
            const auto idx = static_cast<std::size_t>(local_dof);
            active_dof_retained_measure[idx] += measure;
            ++active_dof_retained_rule_count[idx];
            if (have_volume_fraction) {
                active_dof_min_volume_fraction[idx] =
                    std::min(active_dof_min_volume_fraction[idx],
                             rule.volume_fraction);
                active_dof_max_volume_fraction[idx] =
                    std::max(active_dof_max_volume_fraction[idx],
                             rule.volume_fraction);
            }
        }
    };
    const auto mark_cut_adjacent_facet_active =
        [&](GlobalIndex cell, GlobalIndex facet) -> bool {
        if (cell < 0 || static_cast<std::size_t>(cell) >= mesh.n_cells()) {
            std::ostringstream oss;
            oss << "LevelSetActiveSideVertexDirichletConstraint: active support references cell "
                << cell << " outside the mesh";
            throw std::runtime_error(oss.str());
        }

        bool first_cell_visit = false;
        if (active_cell_seen[static_cast<std::size_t>(cell)] ==
            static_cast<unsigned char>(0)) {
            active_cell_seen[static_cast<std::size_t>(cell)] =
                static_cast<unsigned char>(1);
            ++active_support_cells;
            first_cell_visit = true;
        }

        if (facet < 0 || static_cast<std::size_t>(facet) >= mesh.n_faces()) {
            return first_cell_visit;
        }
        const auto [facet_vertices, facet_vertex_count] =
            mesh.face_vertices_span(static_cast<index_t>(facet));
        if (facet_vertices == nullptr || facet_vertex_count == 0u) {
            return first_cell_visit;
        }

        std::vector<index_t> vertices(
            facet_vertices, facet_vertices + facet_vertex_count);
        std::sort(vertices.begin(), vertices.end());
        vertices.erase(std::unique(vertices.begin(), vertices.end()),
                       vertices.end());
        for (const auto vertex : vertices) {
            mark_vertex_support(static_cast<GlobalIndex>(vertex));
        }

        bool high_order_line_facet = false;
        std::size_t n_corner_vertices = facet_vertex_count;
        const auto& face_shapes = mesh.face_shapes();
        if (static_cast<std::size_t>(facet) < face_shapes.size()) {
            const auto& shape = face_shapes[static_cast<std::size_t>(facet)];
            high_order_line_facet = shape.family == CellFamily::Line &&
                                    shape.num_corners == 2 &&
                                    facet_vertex_count >= 2u;
            if (shape.num_corners > 0) {
                n_corner_vertices = std::min(
                    facet_vertex_count,
                    static_cast<std::size_t>(shape.num_corners));
            }
        }

        const auto mark_facet_edge = [&](GlobalIndex v0, GlobalIndex v1) {
            if (edge_ids.empty()) {
                return;
            }
            const auto key = EdgeKey{std::min(v0, v1), std::max(v0, v1)};
            const auto it = edge_ids.find(key);
            if (it == edge_ids.end()) {
                std::ostringstream oss;
                oss << "LevelSetActiveSideVertexDirichletConstraint: facet "
                    << facet << " references edge (" << v0 << ',' << v1
                    << ") missing from the constrained field entity map";
                throw std::runtime_error(oss.str());
            }
            mark_edge_support(it->second);
        };

        if (high_order_line_facet) {
            mark_facet_edge(
                static_cast<GlobalIndex>(facet_vertices[0]),
                static_cast<GlobalIndex>(
                    facet_vertices[facet_vertex_count - 1u]));
        } else if (n_corner_vertices == 2u) {
            mark_facet_edge(
                static_cast<GlobalIndex>(facet_vertices[0]),
                static_cast<GlobalIndex>(facet_vertices[1]));
        } else if (n_corner_vertices >= 3u) {
            if (entity_map->numFaces() > 0) {
                mark_face_support(facet);
            }
            for (std::size_t i = 0; i < n_corner_vertices; ++i) {
                mark_facet_edge(
                    static_cast<GlobalIndex>(facet_vertices[i]),
                    static_cast<GlobalIndex>(
                        facet_vertices[(i + 1u) % n_corner_vertices]));
            }
        }
        return first_cell_visit;
    };

    const auto* cut_context = system.cutIntegrationContext();
    if (cut_context != nullptr) {
        cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
            system.meshAccess());
    }
    const bool has_retained_generated_volume_support =
        interface_marker_ >= 0 && cut_context != nullptr &&
        cut_context->hasGeneratedVolumeMarker(interface_marker_);
    if (has_retained_generated_volume_support) {
        support_mode = "retained_cut_volume";
        const auto active_rule_indices =
            cut_context->generatedVolumeRuleIndexSpanForMarkerAndSide(
                interface_marker_,
                toCutIntegrationSide(active_side_));
        const auto& metadata = cut_context->metadata();
        const auto& volume_rules = cut_context->volumeRules();
        for (const auto index : active_rule_indices) {
            if (index >= metadata.size()) {
                continue;
            }
            const auto& rule_metadata = metadata[index];
            const auto cell = rule_metadata.parent_entity >= 0
                                  ? rule_metadata.parent_entity
                                  : rule_metadata.cell;
            if (index < volume_rules.size()) {
                record_retained_rule_support(
                    static_cast<GlobalIndex>(cell), volume_rules[index]);
            }
            if (mark_cell_active(static_cast<GlobalIndex>(cell))) {
                ++active_support_cells_from_volume_support;
            }
        }
    } else {
        for (GlobalIndex cell = 0;
             cell < static_cast<GlobalIndex>(mesh.n_cells());
             ++cell) {
            const auto [vertices, count] =
                mesh.cell_vertices_span(static_cast<index_t>(cell));
            if (vertices == nullptr || count == 0u) {
                continue;
            }

            bool cell_has_active_measure = false;
            for (std::size_t i = 0; i < count; ++i) {
                const auto vertex = static_cast<GlobalIndex>(vertices[i]);
                if (vertex < 0 ||
                    static_cast<std::size_t>(vertex) >= level_set.entity_count) {
                    std::ostringstream oss;
                    oss << "LevelSetActiveSideVertexDirichletConstraint: cell "
                        << cell << " references vertex " << vertex
                        << " outside level-set field '" << level_set_field_name_
                        << "'";
                    throw std::runtime_error(oss.str());
                }
                const auto phi = level_set.values[
                    static_cast<std::size_t>(vertex) * level_set.components];
                if (hasPositiveActiveSideMeasure(phi, isovalue_, active_side_)) {
                    cell_has_active_measure = true;
                    break;
                }
            }
            if (cell_has_active_measure) {
                if (mark_cell_active(cell)) {
                    ++active_support_cells_from_volume_support;
                }
            }
        }
    }

    if (has_retained_generated_volume_support) {
        const auto* facet_set =
            cut_context->facetSetHandleForMarker(interface_marker_);
        if (facet_set != nullptr && facet_set->hasFacetMetadata()) {
            support_mode += "+cut_adjacent_facets";
            for (const auto& facet : facet_set->facet_metadata) {
                if (facet.first_cell >= static_cast<MeshIndex>(0) &&
                    mark_cut_adjacent_facet_active(
                        static_cast<GlobalIndex>(facet.first_cell),
                        static_cast<GlobalIndex>(facet.facet))) {
                    ++active_support_cells_from_cut_adjacent_facets;
                }
                if (facet.second_cell >= static_cast<MeshIndex>(0) &&
                    mark_cut_adjacent_facet_active(
                        static_cast<GlobalIndex>(facet.second_cell),
                        static_cast<GlobalIndex>(facet.facet))) {
                    ++active_support_cells_from_cut_adjacent_facets;
                }
            }
        }
    } else if (interface_marker_ >= 0 && cut_context != nullptr &&
               cut_context->facetSetHandleForMarker(interface_marker_) != nullptr) {
        support_mode += "+cut_adjacent_facets_skipped_no_retained_volume";
    }
#else
    std::vector<unsigned char> has_active_support(
        static_cast<std::size_t>(n_vertices), static_cast<unsigned char>(0));
    std::vector<unsigned char> has_active_dof_support(
        static_cast<std::size_t>(n_field_dofs), static_cast<unsigned char>(0));
    std::vector<Real> active_dof_retained_measure(
        static_cast<std::size_t>(n_field_dofs), Real{0.0});
    std::vector<Real> active_dof_min_volume_fraction(
        static_cast<std::size_t>(n_field_dofs),
        std::numeric_limits<Real>::infinity());
    std::vector<Real> active_dof_max_volume_fraction(
        static_cast<std::size_t>(n_field_dofs),
        -std::numeric_limits<Real>::infinity());
    std::vector<std::size_t> active_dof_retained_rule_count(
        static_cast<std::size_t>(n_field_dofs), std::size_t{0u});
    std::size_t active_support_cells = 0u;
    std::size_t active_support_cells_from_volume_support = 0u;
    std::size_t active_support_cells_from_cut_adjacent_facets = 0u;
    std::string support_mode = "cell_patch";
#endif

    std::vector<GlobalIndex> inactive_vertices;
    inactive_vertices.reserve(static_cast<std::size_t>(n_vertices));
    std::vector<GlobalIndex> inactive_dofs;
    inactive_dofs.reserve(static_cast<std::size_t>(n_field_dofs));
    std::size_t active_sign_vertices = 0u;
    std::size_t active_support_vertices = 0u;
    std::size_t active_support_dofs = 0u;
    std::size_t inactive_sign_vertices_with_support = 0u;
    std::size_t active_sign_vertices_without_support = 0u;
    std::size_t constrained_dofs = 0u;
    EntityDofSupportCounts active_support_by_entity;
    EntityDofSupportCounts inactive_by_entity;
    EntityDofSupportCounts constrained_owned_by_entity;

    for (GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
        const auto phi = level_set.values[
            static_cast<std::size_t>(vertex) * level_set.components];
        const bool active_sign = isActive(phi, isovalue_, active_side_);
        const bool active_support =
            has_active_support[static_cast<std::size_t>(vertex)] !=
            static_cast<unsigned char>(0);
        if (active_sign) {
            ++active_sign_vertices;
        }
        if (active_support) {
            ++active_support_vertices;
            if (!active_sign) {
                ++inactive_sign_vertices_with_support;
            }
            continue;
        }
        if (isActive(phi, isovalue_, active_side_)) {
            ++active_sign_vertices_without_support;
        }

        inactive_vertices.push_back(vertex);
    }

    for (GlobalIndex local_dof = 0; local_dof < n_field_dofs; ++local_dof) {
        if (has_active_dof_support[static_cast<std::size_t>(local_dof)] !=
            static_cast<unsigned char>(0)) {
            ++active_support_dofs;
            incrementEntityDofCount(
                active_support_by_entity, *entity_map, local_dof);
            continue;
        }

        inactive_dofs.push_back(local_dof);
        incrementEntityDofCount(inactive_by_entity, *entity_map, local_dof);
        const GlobalIndex dof = offset + local_dof;
        if (owned.contains(dof)) {
            // Precedence: a DOF already slaved by another constraint (e.g.
            // small-cut aggregation to a wet-extension polynomial) is fully
            // determined by its masters, so this inactive-side pin has no
            // singular mode left to remove. Later strong Dirichlet callers
            // may still replace master-bearing lines explicitly.
            if (constraints.isConstrained(dof)) {
                continue;
            }
            constraints.addDirichlet(dof, inactive_value_);
            ++constrained_dofs;
            incrementEntityDofCount(
                constrained_owned_by_entity, *entity_map, local_dof);
        }
    }

    for (const auto local_dof : sampledConstraintDofs(rec.name)) {
        std::ostringstream sample;
        sample << "LevelSetActiveSideVertexDirichletConstraint: diagnostic=level_set_active_side_vertex_constraint_sample"
               << " field='" << rec.name << "'"
               << " level_set_field='" << level_set_field_name_ << "'"
               << " local_dof=" << local_dof;
        if (local_dof < 0 || local_dof >= n_field_dofs) {
            sample << " status=out_of_range"
                   << " total_dofs=" << n_field_dofs;
            FE_LOG_INFO(sample.str());
            continue;
        }

        const GlobalIndex global_dof = offset + local_dof;
        const bool owned_dof = owned.contains(global_dof);
        const bool active_dof_support =
            has_active_dof_support[static_cast<std::size_t>(local_dof)] !=
            static_cast<unsigned char>(0);
        sample << " status=ok"
               << " global_dof=" << global_dof
               << " owned=" << (owned_dof ? 1 : 0)
               << " active_dof_support=" << (active_dof_support ? 1 : 0)
               << " inactive_constraint=" << (active_dof_support ? 0 : 1)
               << " constrained_owned="
               << ((!active_dof_support && owned_dof) ? 1 : 0)
               << " retained_rule_count="
               << active_dof_retained_rule_count[static_cast<std::size_t>(local_dof)]
               << " retained_measure="
               << active_dof_retained_measure[static_cast<std::size_t>(local_dof)];
        if (active_dof_retained_rule_count[
                static_cast<std::size_t>(local_dof)] > 0u &&
            std::isfinite(active_dof_min_volume_fraction[
                static_cast<std::size_t>(local_dof)])) {
            sample << " retained_min_volume_fraction="
                   << active_dof_min_volume_fraction[
                          static_cast<std::size_t>(local_dof)]
                   << " retained_max_volume_fraction="
                   << active_dof_max_volume_fraction[
                          static_cast<std::size_t>(local_dof)];
        } else {
            sample << " retained_min_volume_fraction=none"
                   << " retained_max_volume_fraction=none";
        }

        const auto entity = entity_map->getDofEntity(local_dof);
        if (!entity.has_value()) {
            sample << " entity_kind=Unknown entity_id=-1 entity_dofs=none";
            FE_LOG_INFO(sample.str());
            continue;
        }

        sample << " entity_kind=" << entityKindName(entity->kind)
               << " entity_id=" << entity->id
               << " entity_dofs="
               << formatDofList(
                      entity_map->getEntityDofs(entity->kind, entity->id));
        if (entity->kind == dofs::EntityKind::Vertex &&
            entity->id >= 0 && entity->id < n_vertices) {
            const auto vertex = entity->id;
            const auto phi = level_set.values[
                static_cast<std::size_t>(vertex) * level_set.components];
            const bool active_sign = isActive(phi, isovalue_, active_side_);
            const bool active_support =
                has_active_support[static_cast<std::size_t>(vertex)] !=
                static_cast<unsigned char>(0);
            sample << " vertex_phi=" << phi
                   << " vertex_active_sign=" << (active_sign ? 1 : 0)
                   << " vertex_active_support=" << (active_support ? 1 : 0);
        }
        FE_LOG_INFO(sample.str());
    }

    std::ostringstream oss;
    oss << "LevelSetActiveSideVertexDirichletConstraint: diagnostic=level_set_active_side_vertex_constraint"
        << " field='" << rec.name << "'"
        << " level_set_field='" << level_set_field_name_ << "'"
        << " active_side=" << sideName(active_side_)
        << " isovalue=" << isovalue_
        << " support_mode=" << support_mode
        << " interface_marker=" << interface_marker_
        << " total_vertices=" << n_vertices
        << " active_sign_vertices=" << active_sign_vertices
        << " active_support_cells=" << active_support_cells
        << " active_support_cells_from_volume_support="
        << active_support_cells_from_volume_support
        << " active_support_cells_from_cut_adjacent_facets="
        << active_support_cells_from_cut_adjacent_facets
        << " active_support_vertices=" << active_support_vertices
        << " total_dofs=" << n_field_dofs
        << " active_support_dofs=" << active_support_dofs
        << " active_support_vertex_dofs=" << active_support_by_entity.vertex
        << " active_support_edge_dofs=" << active_support_by_entity.edge
        << " active_support_face_dofs=" << active_support_by_entity.face
        << " active_support_cell_dofs=" << active_support_by_entity.cell
        << " active_support_unknown_dofs=" << active_support_by_entity.unknown
        << " active_sign_vertices_without_support=" << active_sign_vertices_without_support
        << " inactive_sign_vertices_with_support=" << inactive_sign_vertices_with_support
        << " inactive_vertices=" << inactive_vertices.size()
        << " inactive_dofs=" << inactive_dofs.size()
        << " inactive_vertex_dofs=" << inactive_by_entity.vertex
        << " inactive_edge_dofs=" << inactive_by_entity.edge
        << " inactive_face_dofs=" << inactive_by_entity.face
        << " inactive_cell_dofs=" << inactive_by_entity.cell
        << " inactive_unknown_dofs=" << inactive_by_entity.unknown
        << " constrained_owned_dofs=" << constrained_dofs
        << " constrained_owned_vertex_dofs=" << constrained_owned_by_entity.vertex
        << " constrained_owned_edge_dofs=" << constrained_owned_by_entity.edge
        << " constrained_owned_face_dofs=" << constrained_owned_by_entity.face
        << " constrained_owned_cell_dofs=" << constrained_owned_by_entity.cell
        << " constrained_owned_unknown_dofs=" << constrained_owned_by_entity.unknown
        << " inactive_vertex_runs=" << formatRuns(inactive_vertices)
        << " inactive_dof_runs=" << formatRuns(inactive_dofs);
    FE_LOG_INFO(oss.str());
}

bool LevelSetActiveSideVertexDirichletConstraint::updateValues(
    const systems::FESystem& system,
    AffineConstraints& constraints,
    double time,
    double dt)
{
    (void)system;
    (void)constraints;
    (void)time;
    (void)dt;
    // Level-set value changes can change which DOFs are constrained, so this
    // constraint must be structurally rebuilt instead of value-updated in place.
    return false;
}

ConstraintDependencyDeclaration
LevelSetActiveSideVertexDirichletConstraint::dependencyDeclaration() const
{
    ConstraintDependencyDeclaration out = ISystemConstraint::dependencyDeclaration();
    out.structural.fe_constraint_layout = true;
    out.structural.mesh_field_layout = true;
    out.structural.mesh_field_values = true;
    out.structural.active_configuration = true;
    return out;
}

systems::SetupStorageRequirements
LevelSetActiveSideVertexDirichletConstraint::storageRequirements() const noexcept
{
    systems::SetupStorageRequirements req;
    req.entity_dof_map = true;
    req.vertex_topology = true;
    req.cell_topology = true;
    return req;
}

} // namespace constraints
} // namespace FE
} // namespace svmp
