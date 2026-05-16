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
#include <sstream>
#include <stdexcept>
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
    if (rec.space->space_type() != spaces::SpaceType::H1 ||
        rec.space->continuity() != Continuity::C0 ||
        rec.space->value_dimension() != 1 ||
        rec.components != 1) {
        throw std::invalid_argument(
            "LevelSetActiveSideVertexDirichletConstraint requires a scalar H1/C0 field");
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
    std::vector<unsigned char> has_active_support(
        static_cast<std::size_t>(n_vertices), static_cast<unsigned char>(0));
    std::vector<unsigned char> has_active_dof_support(
        static_cast<std::size_t>(n_field_dofs), static_cast<unsigned char>(0));
    std::size_t active_support_cells = 0u;
    const char* support_mode = "cell_patch";
    std::vector<unsigned char> active_cell_seen(
        mesh.n_cells(), static_cast<unsigned char>(0));
    const auto mark_cell_active = [&](GlobalIndex cell) {
        if (cell < 0 || static_cast<std::size_t>(cell) >= mesh.n_cells()) {
            std::ostringstream oss;
            oss << "LevelSetActiveSideVertexDirichletConstraint: active support references cell "
                << cell << " outside the mesh";
            throw std::runtime_error(oss.str());
        }
        if (active_cell_seen[static_cast<std::size_t>(cell)] !=
            static_cast<unsigned char>(0)) {
            return;
        }
        active_cell_seen[static_cast<std::size_t>(cell)] =
            static_cast<unsigned char>(1);
        ++active_support_cells;

        for (const auto local_dof : dh.getCellDofs(cell)) {
            if (local_dof < 0 || local_dof >= n_field_dofs) {
                std::ostringstream oss;
                oss << "LevelSetActiveSideVertexDirichletConstraint: cell "
                    << cell << " references field DOF " << local_dof
                    << " outside field '" << rec.name << "'";
                throw std::runtime_error(oss.str());
            }
            has_active_dof_support[static_cast<std::size_t>(local_dof)] =
                static_cast<unsigned char>(1);
        }

        const auto [vertices, count] =
            mesh.cell_vertices_span(static_cast<index_t>(cell));
        if (vertices == nullptr || count == 0u) {
            return;
        }
        for (std::size_t i = 0; i < count; ++i) {
            const auto vertex = static_cast<GlobalIndex>(vertices[i]);
            if (vertex >= 0 && vertex < n_vertices) {
                has_active_support[static_cast<std::size_t>(vertex)] =
                    static_cast<unsigned char>(1);
            }
        }
    };

    const auto* cut_context = system.cutIntegrationContext();
    if (interface_marker_ >= 0 && cut_context != nullptr &&
        cut_context->hasGeneratedVolumeMarker(interface_marker_)) {
        support_mode = "retained_cut_volume";
        const auto active_rule_indices =
            cut_context->generatedVolumeRuleIndicesForMarkerAndSide(
                interface_marker_,
                toCutIntegrationSide(active_side_));
        const auto& metadata = cut_context->metadata();
        for (const auto index : active_rule_indices) {
            if (index >= metadata.size()) {
                continue;
            }
            const auto& rule_metadata = metadata[index];
            const auto cell = rule_metadata.parent_entity >= 0
                                  ? rule_metadata.parent_entity
                                  : rule_metadata.cell;
            mark_cell_active(static_cast<GlobalIndex>(cell));
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
                mark_cell_active(cell);
            }
        }
    }
#else
    std::vector<unsigned char> has_active_support(
        static_cast<std::size_t>(n_vertices), static_cast<unsigned char>(0));
    std::vector<unsigned char> has_active_dof_support(
        static_cast<std::size_t>(n_field_dofs), static_cast<unsigned char>(0));
    std::size_t active_support_cells = 0u;
    const char* support_mode = "cell_patch";
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
            continue;
        }

        inactive_dofs.push_back(local_dof);
        const GlobalIndex dof = offset + local_dof;
        if (owned.contains(dof)) {
            constraints.addDirichlet(dof, inactive_value_);
            ++constrained_dofs;
        }
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
        << " active_support_vertices=" << active_support_vertices
        << " total_dofs=" << n_field_dofs
        << " active_support_dofs=" << active_support_dofs
        << " active_sign_vertices_without_support=" << active_sign_vertices_without_support
        << " inactive_sign_vertices_with_support=" << inactive_sign_vertices_with_support
        << " inactive_vertices=" << inactive_vertices.size()
        << " inactive_dofs=" << inactive_dofs.size()
        << " constrained_owned_dofs=" << constrained_dofs
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
    return false;
}

ConstraintDependencyDeclaration
LevelSetActiveSideVertexDirichletConstraint::dependencyDeclaration() const
{
    ConstraintDependencyDeclaration out = ISystemConstraint::dependencyDeclaration();
    out.structural.mesh_field_layout = true;
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
