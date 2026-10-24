/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/VertexFieldInitialization.h"

#include "Backends/Interfaces/GenericVector.h"
#include "Dofs/EntityDofMap.h"
#include "Systems/FESystem.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#  include "Mesh/Fields/MeshFields.h"
#  include "Mesh/Mesh.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

std::size_t initializeFieldFromMeshVertexField(
    const FESystem& system,
    backends::GenericVector& u0,
    FieldId field_id,
    const std::string& message_prefix)
{
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    if (field_id == INVALID_FIELD_ID) {
        return 0u;
    }
    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        return 0u;
    }
    const auto& rec = system.fieldRecord(field_id);
    const auto& local_mesh = native_mesh->local_mesh();
    if (!MeshFields::has_field(local_mesh, EntityKind::Vertex, rec.name)) {
        return 0u;
    }
    const auto handle =
        MeshFields::get_field_handle(local_mesh, EntityKind::Vertex, rec.name);
    if (MeshFields::field_type(local_mesh, handle) != FieldScalarType::Float64) {
        throw std::runtime_error(
            message_prefix +
            " requires a Float64 vertex mesh field '" + rec.name + "'");
    }
    const auto components = static_cast<std::size_t>(std::max(1, rec.components));
    const auto mesh_components = MeshFields::field_components(local_mesh, handle);
    if (mesh_components < components) {
        throw std::runtime_error(
            message_prefix +
            " mesh field '" + rec.name + "' has fewer components than the FE field");
    }
    const auto n_vertices = static_cast<GlobalIndex>(native_mesh->n_vertices());
    const auto entity_count = MeshFields::field_entity_count(local_mesh, handle);
    if (entity_count < static_cast<std::size_t>(n_vertices)) {
        throw std::runtime_error(
            message_prefix +
            " mesh field '" + rec.name + "' has fewer entries than mesh vertices");
    }
    const auto* mesh_values = MeshFields::field_data_as<Real>(local_mesh, handle);
    if (mesh_values == nullptr) {
        throw std::runtime_error(
            message_prefix +
            " found an empty mesh field '" + rec.name + "'");
    }
    const auto& field_dofs = system.fieldDofHandler(field_id);
    const auto* entity_map = field_dofs.getEntityDofMap();
    if (entity_map == nullptr || entity_map->numVertices() < n_vertices) {
        throw std::runtime_error(
            message_prefix +
            " requires FE vertex DOF metadata for field '" + rec.name + "'");
    }
    const auto field_offset = system.fieldDofOffset(field_id);
    const auto n_field_dofs = static_cast<std::size_t>(field_dofs.getNumDofs());
    std::vector<Real> coefficients(n_field_dofs, Real{0.0});
    std::vector<std::uint8_t> assigned(n_field_dofs, 0u);
    const auto projection =
        system.projectMeshVertexValuesToFieldCoefficients(
            field_id,
            std::span<const Real>(
                reinterpret_cast<const Real*>(mesh_values),
                entity_count * mesh_components),
            mesh_components,
            std::span<Real>(coefficients.data(), coefficients.size()),
            std::span<std::uint8_t>(assigned.data(), assigned.size()),
            message_prefix);
    if (projection.unassigned_dofs != 0u) {
        throw std::runtime_error(
            message_prefix +
            " could not safely project " +
            std::to_string(projection.unassigned_dofs) +
            " field coefficient(s) from mesh vertices for field '" +
            rec.name + "'");
    }
    std::vector<GlobalIndex> dofs;
    std::vector<Real> values;
    dofs.reserve(n_field_dofs);
    values.reserve(n_field_dofs);
    for (std::size_t dof = 0; dof < coefficients.size(); ++dof) {
        if (assigned[dof] == 0u) {
            continue;
        }
        dofs.push_back(field_offset + static_cast<GlobalIndex>(dof));
        values.push_back(coefficients[dof]);
    }
    if (dofs.empty()) {
        return 0u;
    }
    auto view = u0.createAssemblyView();
    if (!view) {
        throw std::runtime_error(
            message_prefix +
            " could not create a vector view");
    }
    view->beginAssemblyPhase();
    view->setVectorEntries(dofs, values);
    view->endAssemblyPhase();
    view->finalizeAssembly();
    return dofs.size();
#else
    (void)system;
    (void)u0;
    (void)field_id;
    (void)message_prefix;
    return 0u;
#endif
}

MeshVertexFieldView requireMeshVertexField(
    const FESystem& system,
    const std::string& field_name,
    GlobalIndex n_vertices,
    const MeshVertexFieldMessages& messages)
{
#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        throw std::runtime_error(messages.no_native_mesh);
    }

    const auto& local_mesh = native_mesh->local_mesh();
    if (!MeshFields::has_field(local_mesh, EntityKind::Vertex, field_name)) {
        throw std::runtime_error(
            messages.missing_field_prefix + field_name + "'");
    }

    const auto handle = MeshFields::get_field_handle(
        local_mesh, EntityKind::Vertex, field_name);
    if (MeshFields::field_type(local_mesh, handle) != FieldScalarType::Float64) {
        throw std::runtime_error(messages.not_float64);
    }

    const auto components = MeshFields::field_components(local_mesh, handle);
    if (components < 1u) {
        throw std::runtime_error(messages.no_components);
    }

    const auto entity_count = MeshFields::field_entity_count(local_mesh, handle);
    if (entity_count < static_cast<std::size_t>(n_vertices)) {
        throw std::runtime_error(messages.too_few_entries);
    }

    const auto* values = MeshFields::field_data_as<Real>(local_mesh, handle);
    if (values == nullptr) {
        throw std::runtime_error(messages.empty);
    }

    if (!messages.non_finite_prefix.empty()) {
        for (GlobalIndex vertex = 0; vertex < n_vertices; ++vertex) {
            const auto value =
                values[static_cast<std::size_t>(vertex) * components];
            if (!std::isfinite(value)) {
                std::ostringstream oss;
                oss << messages.non_finite_prefix << vertex;
                throw std::runtime_error(oss.str());
            }
        }
    }

    return MeshVertexFieldView{values, components, entity_count};
#else
    (void)system;
    (void)field_name;
    (void)n_vertices;
    throw std::runtime_error(messages.no_mesh_support);
#endif
}

std::optional<GlobalIndex> localVertexOf(
    const FESystem& system,
    constraints::VertexIdMode id_mode,
    GlobalIndex node_id,
    const LocalVertexMessages& messages)
{
    if (node_id < 0) {
        throw std::invalid_argument(messages.negative_id);
    }

    switch (id_mode) {
    case constraints::VertexIdMode::LocalVertexId:
        return node_id;
    case constraints::VertexIdMode::GlobalVertexGid:
        break;
    }

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    const auto* native_mesh = system.mesh();
    if (native_mesh == nullptr) {
        throw std::invalid_argument(messages.no_native_mesh);
    }
    const auto local_vertex =
        native_mesh->local_mesh().global_to_local_vertex(static_cast<gid_t>(node_id));
    if (local_vertex == INVALID_INDEX) {
        return std::nullopt;
    }
    return static_cast<GlobalIndex>(local_vertex);
#else
    (void)system;
    throw std::invalid_argument(messages.no_mesh_support);
#endif
}

const dofs::EntityDofMap& requireVertexDofMap(
    const FESystem& system,
    FieldId field,
    const std::string& message)
{
    const auto* entity_map = system.fieldDofHandler(field).getEntityDofMap();
    if (entity_map == nullptr) {
        throw std::runtime_error(message);
    }
    return *entity_map;
}

void writeVectorEntries(
    backends::GenericVector& u0,
    const std::vector<GlobalIndex>& dofs,
    const std::vector<Real>& values,
    const std::string& no_entries_message,
    const std::string& no_view_message)
{
    if (dofs.empty()) {
        throw std::runtime_error(no_entries_message);
    }

    auto view = u0.createAssemblyView();
    if (!view) {
        throw std::runtime_error(no_view_message);
    }
    view->beginAssemblyPhase();
    view->setVectorEntries(dofs, values);
    view->endAssemblyPhase();
    view->finalizeAssembly();
}

} // namespace systems
} // namespace FE
} // namespace svmp
