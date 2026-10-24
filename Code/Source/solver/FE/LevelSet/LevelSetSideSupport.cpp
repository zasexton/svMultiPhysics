/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/LevelSetSideSupport.h"

#include "Assembly/Assembler.h"
#include "Assembly/CutIntegrationContext.h"
#include "Dofs/EntityDofMap.h"
#include "Geometry/CutQuadrature.h"
#include "Systems/FESystem.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Mesh.h"
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace level_set {

bool vertexOnLevelSetSide(
    Real phi,
    Real isovalue,
    LevelSetSupportSide side) noexcept
{
    switch (side) {
    case LevelSetSupportSide::Everywhere:
        return true;
    case LevelSetSupportSide::Negative:
        return phi <= isovalue;
    case LevelSetSupportSide::Positive:
        return phi >= isovalue;
    }
    return true;
}

namespace {

[[nodiscard]] geometry::CutIntegrationSide integrationSide(
    LevelSetSupportSide side) noexcept
{
    switch (side) {
    case LevelSetSupportSide::Negative:
        return geometry::CutIntegrationSide::Negative;
    case LevelSetSupportSide::Positive:
        return geometry::CutIntegrationSide::Positive;
    case LevelSetSupportSide::Everywhere:
        break;
    }
    return geometry::CutIntegrationSide::Negative;
}

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
[[nodiscard]] bool cellHasMeasureOnSide(
    const LevelSetSideSupportQuery& query,
    const index_t* vertices,
    std::size_t vertex_count)
{
    if (vertices == nullptr || vertex_count == 0u) {
        return false;
    }
    for (std::size_t i = 0; i < vertex_count; ++i) {
        const auto vertex = static_cast<GlobalIndex>(vertices[i]);
        if (vertex < 0 ||
            static_cast<std::size_t>(vertex) >= query.level_set.entity_count) {
            std::ostringstream oss;
            oss << query.message_prefix << " references vertex " << vertex
                << " outside the level-set field";
            throw std::runtime_error(oss.str());
        }
        const auto phi = query.level_set.values[
            static_cast<std::size_t>(vertex) * query.level_set.components];
        const bool has_positive_measure =
            query.side == LevelSetSupportSide::Negative
                ? phi < query.isovalue
                : phi > query.isovalue;
        if (has_positive_measure) {
            return true;
        }
    }
    return false;
}
#endif

} // namespace

std::vector<unsigned char> levelSetSideSupportVertices(
    const systems::FESystem& system,
    const LevelSetSideSupportQuery& query,
    GlobalIndex n_vertices)
{
    std::vector<unsigned char> support(
        static_cast<std::size_t>(n_vertices), static_cast<unsigned char>(0));
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        for (GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
            const auto phi = query.level_set.values[
                static_cast<std::size_t>(vertex) * query.level_set.components];
            support[static_cast<std::size_t>(vertex)] =
                vertexOnLevelSetSide(phi, query.isovalue, query.side)
                    ? static_cast<unsigned char>(1)
                    : static_cast<unsigned char>(0);
        }
        return support;
    }
    const auto& mesh = native_mesh->local_mesh();
    const auto mark_cell = [&](GlobalIndex cell) {
        if (cell < 0 || static_cast<std::size_t>(cell) >= mesh.n_cells()) {
            std::ostringstream oss;
            oss << query.message_prefix << " references cell " << cell
                << " outside the mesh";
            throw std::runtime_error(oss.str());
        }
        const auto [vertices, count] =
            mesh.cell_vertices_span(static_cast<index_t>(cell));
        if (vertices == nullptr || count == 0u) {
            return;
        }
        for (std::size_t i = 0; i < count; ++i) {
            const auto vertex = static_cast<GlobalIndex>(vertices[i]);
            if (vertex >= 0 && vertex < n_vertices) {
                support[static_cast<std::size_t>(vertex)] =
                    static_cast<unsigned char>(1);
            }
        }
    };
    const auto* cut_context = system.cutIntegrationContext();
    if (cut_context != nullptr) {
        cut_context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
            system.meshAccess());
    }
    if (query.use_generated_volume_rules &&
        query.interface_marker >= 0 &&
        cut_context != nullptr &&
        cut_context->hasGeneratedVolumeMarker(query.interface_marker)) {
        const auto rule_indices =
            cut_context->generatedVolumeRuleIndexSpanForMarkerAndSide(
                query.interface_marker,
                integrationSide(query.side));
        const auto& metadata = cut_context->metadata();
        for (const auto index : rule_indices) {
            if (index >= metadata.size()) {
                continue;
            }
            const auto& rule_metadata = metadata[index];
            const auto cell = rule_metadata.parent_entity >= 0
                                  ? rule_metadata.parent_entity
                                  : rule_metadata.cell;
            mark_cell(static_cast<GlobalIndex>(cell));
        }
    } else {
        for (GlobalIndex cell = 0;
             cell < static_cast<GlobalIndex>(mesh.n_cells());
             ++cell) {
            const auto [vertices, count] =
                mesh.cell_vertices_span(static_cast<index_t>(cell));
            if (cellHasMeasureOnSide(query, vertices, count)) {
                mark_cell(cell);
            }
        }
    }
#else
    (void)system;
    for (GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
        const auto phi = query.level_set.values[
            static_cast<std::size_t>(vertex) * query.level_set.components];
        support[static_cast<std::size_t>(vertex)] =
            vertexOnLevelSetSide(phi, query.isovalue, query.side)
                ? static_cast<unsigned char>(1)
                : static_cast<unsigned char>(0);
    }
#endif
    return support;
}

LevelSetVertexSide levelSetVertexSide(
    const LevelSetVertexValues& level_set,
    GlobalIndex vertex,
    Real isovalue,
    LevelSetSupportSide side,
    Real margin) noexcept
{
    LevelSetVertexSide result{};
    result.phi = level_set.values[
        static_cast<std::size_t>(vertex) * level_set.components];
    result.signed_gap = result.phi - isovalue;
    result.on_side = vertexOnLevelSetSide(result.phi, isovalue, side);
    result.within_margin = std::abs(result.signed_gap) < margin;
    return result;
}

LevelSetSideVertexChoice levelSetSideVertexChoice(
    const LevelSetSideVertexValues& values,
    GlobalIndex vertex)
{
    LevelSetSideVertexChoice choice{};
    if (values.side.has_value()) {
        const auto& side = *values.side;
        const auto vertex_offset =
            static_cast<std::size_t>(vertex) * side.level_set.components;
        const auto phi = side.level_set.values[vertex_offset];
        choice.on_side = vertexOnLevelSetSide(phi, side.isovalue, side.side);
        choice.supported =
            side.support.has_value()
                ? ((*side.support)[static_cast<std::size_t>(vertex)] !=
                   static_cast<unsigned char>(0))
                : choice.on_side;
    }
    return choice;
}

namespace {

[[nodiscard]] Real sideVertexValue(
    const LevelSetSideVertexValues& values,
    bool supported,
    GlobalIndex vertex,
    const std::array<Real, 3>& x)
{
    return supported
        ? (values.field_values != nullptr
               ? values.field_values[
                     static_cast<std::size_t>(vertex) *
                     values.field_components]
               : values.value(x))
        : values.dry_value;
}

} // namespace

Real levelSetSideVertexValue(
    const LevelSetSideVertexValues& values,
    GlobalIndex vertex,
    const std::array<Real, 3>& x)
{
    return sideVertexValue(
        values, levelSetSideVertexChoice(values, vertex).supported, vertex, x);
}

LevelSetSideVertexEntries collectVertexFieldOnLevelSetSide(
    const systems::FESystem& system,
    FieldId field,
    const dofs::EntityDofMap& vertex_dof_map,
    const LevelSetSideVertexValues& values,
    const std::string& non_finite_prefix,
    const std::string& non_finite_suffix)
{
    const auto& mesh = system.meshAccess();
    const auto field_offset = system.fieldDofOffset(field);
    const auto n_vertices = mesh.numVertices();

    LevelSetSideVertexEntries result{};
    auto& statistics = result.statistics;
    statistics.value_min = std::numeric_limits<Real>::infinity();
    statistics.value_max = -std::numeric_limits<Real>::infinity();
    statistics.supported_value_min = std::numeric_limits<Real>::infinity();
    statistics.supported_value_max = -std::numeric_limits<Real>::infinity();
    for (GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
        const auto vertex_dofs = vertex_dof_map.getVertexDofs(vertex);
        if (vertex_dofs.empty()) {
            continue;
        }

        const auto x = mesh.getNodeCoordinates(vertex);
        const auto choice = levelSetSideVertexChoice(values, vertex);
        if (values.side.has_value()) {
            if (choice.on_side) {
                ++statistics.on_side_vertices;
            } else {
                ++statistics.off_side_vertices;
            }
            if (choice.supported) {
                ++statistics.supported_vertices;
                if (!choice.on_side) {
                    ++statistics.off_side_supported_vertices;
                }
            }
        }
        const Real value = sideVertexValue(values, choice.supported, vertex, x);
        if (!std::isfinite(value)) {
            std::ostringstream oss;
            oss << non_finite_prefix << vertex << non_finite_suffix;
            throw std::runtime_error(oss.str());
        }
        statistics.value_min = std::min(statistics.value_min, value);
        statistics.value_max = std::max(statistics.value_max, value);
        if (choice.supported) {
            statistics.supported_value_min =
                std::min(statistics.supported_value_min, value);
            statistics.supported_value_max =
                std::max(statistics.supported_value_max, value);
        }

        for (const auto local_dof : vertex_dofs) {
            result.dofs.push_back(field_offset + local_dof);
            result.values.push_back(value);
        }
    }
    return result;
}

} // namespace level_set
} // namespace FE
} // namespace svmp
