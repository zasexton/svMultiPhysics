/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_VERTEX_FIELD_INITIALIZATION_H
#define SVMP_FE_SYSTEMS_VERTEX_FIELD_INITIALIZATION_H

/**
 * @file VertexFieldInitialization.h
 * @brief Initial values of an FE field from the vertex fields of the
 *        system's native mesh, the validated view of such a field, the local
 *        vertex of a vertex id, and the write of vertex coefficients.
 */

#include "Core/Types.h"
#include "Constraints/VertexDirichletConstraint.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace backends {
class GenericVector;
}
namespace dofs {
class EntityDofMap;
}
namespace systems {

class FESystem;

/**
 * @brief Set the coefficients of @p field_id in @p u0 from the vertex mesh
 *        field of the same name (FESystem::projectMeshVertexValuesToFieldCoefficients).
 *
 * Nothing happens (0 is returned) without a field id, a native mesh, or a
 * vertex mesh field of the field's name.  Otherwise the mesh field must be
 * Float64 with at least the field's components and an entry per mesh vertex,
 * the field must have vertex DOF metadata, and every coefficient must be
 * assigned; a violation throws std::runtime_error with a message that starts
 * with @p message_prefix.
 *
 * @return the number of coefficients set
 */
std::size_t initializeFieldFromMeshVertexField(
    const FESystem& system,
    backends::GenericVector& u0,
    FieldId field_id,
    const std::string& message_prefix);

/// A vertex field of the native mesh: @p components values per vertex.
struct MeshVertexFieldView {
    const Real* values{nullptr};
    std::size_t components{0};
    std::size_t entity_count{0};
};

/// The messages of requireMeshVertexField (complete texts; the field name is
/// inserted after @p missing_field_prefix, followed by "'").
struct MeshVertexFieldMessages {
    std::string no_native_mesh{};
    std::string missing_field_prefix{};
    std::string not_float64{};
    std::string no_components{};
    std::string too_few_entries{};
    std::string empty{};
    /// Without native mesh support in the build.
    std::string no_mesh_support{};
    /// When not empty, component 0 of each of the first n_vertices vertices
    /// must be finite; the vertex index is appended to this text.
    std::string non_finite_prefix{};
};

/**
 * @brief The Float64 vertex field @p field_name of the native mesh, with at
 *        least one component and an entry for each of the first
 *        @p n_vertices vertices; std::runtime_error with @p messages otherwise.
 */
[[nodiscard]] MeshVertexFieldView requireMeshVertexField(
    const FESystem& system,
    const std::string& field_name,
    GlobalIndex n_vertices,
    const MeshVertexFieldMessages& messages);

/// The messages of localVertexOf (complete texts).
struct LocalVertexMessages {
    std::string negative_id{};
    /// A global id without a native mesh.
    std::string no_native_mesh{};
    /// A global id without native mesh support in the build.
    std::string no_mesh_support{};
};

/**
 * @brief The local mesh vertex that @p node_id names: itself for local
 *        ids; for global ids the native mesh's local vertex, or nothing when
 *        the vertex is not on this rank.  std::invalid_argument with
 *        @p messages for a negative id or a global id without a native mesh.
 */
[[nodiscard]] std::optional<GlobalIndex> localVertexOf(
    const FESystem& system,
    constraints::VertexIdMode id_mode,
    GlobalIndex node_id,
    const LocalVertexMessages& messages);

/**
 * @brief The vertex DOF map of @p field; std::runtime_error with @p message
 *        when the field has none.
 */
[[nodiscard]] const dofs::EntityDofMap& requireVertexDofMap(
    const FESystem& system,
    FieldId field,
    const std::string& message);

/**
 * @brief Set @p values at @p dofs of @p u0 through one assembly view;
 *        std::runtime_error with @p no_entries_message when @p dofs is empty
 *        and with @p no_view_message without a view.
 */
void writeVectorEntries(
    backends::GenericVector& u0,
    const std::vector<GlobalIndex>& dofs,
    const std::vector<Real>& values,
    const std::string& no_entries_message,
    const std::string& no_view_message);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_VERTEX_FIELD_INITIALIZATION_H
