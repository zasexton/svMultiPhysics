/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_LEVELSET_LEVELSET_SIDE_SUPPORT_H
#define SVMP_FE_LEVELSET_LEVELSET_SIDE_SUPPORT_H

/**
 * @file LevelSetSideSupport.h
 * @brief The mesh vertices that support one side of a level set: the
 *        vertices of every cell with positive measure on that side.
 */

#include "Core/Types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace dofs {
class EntityDofMap;
}
namespace systems {
class FESystem;
}
namespace level_set {

/// The side of a level set a quantity lives on.
enum class LevelSetSupportSide : std::uint8_t {
    Everywhere,
    Negative,   ///< phi <= isovalue
    Positive    ///< phi >= isovalue
};

/// Whether a vertex value @p phi is on @p side of @p isovalue (inclusive).
[[nodiscard]] bool vertexOnLevelSetSide(
    Real phi,
    Real isovalue,
    LevelSetSupportSide side) noexcept;

/// A level-set vertex field: @p components values per vertex.
struct LevelSetVertexValues {
    const Real* values{nullptr};
    std::size_t components{0};
    std::size_t entity_count{0};
};

/// Which vertices support one side of a level set.
struct LevelSetSideSupportQuery {
    LevelSetVertexValues level_set{};
    Real isovalue{0.0};
    /// Negative or Positive (Everywhere marks every vertex).
    LevelSetSupportSide side{LevelSetSupportSide::Negative};
    /// The generated interface marker of the side's cut volume, or -1.
    int interface_marker{-1};
    /// Use the generated cut-volume rules of @p interface_marker when the
    /// system's cut-integration context has them.
    bool use_generated_volume_rules{false};
    /// Start of the messages ("<prefix> references cell 7 outside the mesh").
    std::string message_prefix{};
};

/**
 * @brief The support vertices (1) of the query's side among the first
 *        @p n_vertices mesh vertices.
 *
 * Without a native mesh: the vertices on the side.  With one: the vertices
 * of every cell that has a generated cut-volume rule of the marker and side
 * (when requested and present), or else of every cell with a vertex strictly
 * on the side.  The system's free-surface geometry snapshots must be
 * current.  Throws std::runtime_error for a cell or vertex outside the mesh
 * or the level-set field.
 */
[[nodiscard]] std::vector<unsigned char> levelSetSideSupportVertices(
    const systems::FESystem& system,
    const LevelSetSideSupportQuery& query,
    GlobalIndex n_vertices);

/// The side of one level-set vertex value, and its distance to the isovalue.
struct LevelSetVertexSide {
    Real phi{0.0};
    /// phi - isovalue
    Real signed_gap{0.0};
    bool on_side{false};
    /// |phi - isovalue| < margin
    bool within_margin{false};
};

/// The side of vertex @p vertex of @p level_set relative to @p isovalue.
[[nodiscard]] LevelSetVertexSide levelSetVertexSide(
    const LevelSetVertexValues& level_set,
    GlobalIndex vertex,
    Real isovalue,
    LevelSetSupportSide side,
    Real margin) noexcept;

/// The values of a field's vertices on one side of a level set.
struct LevelSetSideVertexValues {
    /// The level set, when the field is restricted to one of its sides.
    struct Side {
        LevelSetVertexValues level_set{};
        Real isovalue{0.0};
        LevelSetSupportSide side{LevelSetSupportSide::Negative};
        /// The support vertices (levelSetSideSupportVertices); without them
        /// a vertex is supported when it is on the side.
        std::optional<std::vector<unsigned char>> support{};
    };
    std::optional<Side> side{};
    /// Supported vertices take component 0 of this vertex field when it is
    /// given, else value(x).
    const Real* field_values{nullptr};
    std::size_t field_components{0};
    std::function<Real(const std::array<Real, 3>&)> value{};
    /// Unsupported vertices take this value.
    Real dry_value{0.0};
};

/// Whether @p vertex takes the supported value, and whether it is on the side.
struct LevelSetSideVertexChoice {
    bool supported{true};
    bool on_side{true};
};

[[nodiscard]] LevelSetSideVertexChoice levelSetSideVertexChoice(
    const LevelSetSideVertexValues& values,
    GlobalIndex vertex);

/// The value of @p vertex at coordinates @p x.
[[nodiscard]] Real levelSetSideVertexValue(
    const LevelSetSideVertexValues& values,
    GlobalIndex vertex,
    const std::array<Real, 3>& x);

/// What collectVertexFieldOnLevelSetSide collected.
struct LevelSetSideVertexStatistics {
    std::size_t on_side_vertices{0};
    std::size_t off_side_vertices{0};
    std::size_t supported_vertices{0};
    std::size_t off_side_supported_vertices{0};
    /// Minimum and maximum of the values (infinite when there is none).
    Real value_min{0.0};
    Real value_max{0.0};
    /// Minimum and maximum of the supported values (infinite when none).
    Real supported_value_min{0.0};
    Real supported_value_max{0.0};
};

/// The vertex coefficients of a field and their statistics.
struct LevelSetSideVertexEntries {
    std::vector<GlobalIndex> dofs{};
    std::vector<Real> values{};
    LevelSetSideVertexStatistics statistics{};
};

/**
 * @brief The vertex coefficients of @p field, whose vertex DOF map is
 *        @p vertex_dof_map (systems::requireVertexDofMap): every mesh vertex
 *        with DOFs takes levelSetSideVertexValue, for each of its DOFs.
 *
 * The side counts are taken when the values are restricted to a side.
 * Throws std::runtime_error "<non_finite_prefix><vertex><non_finite_suffix>"
 * for a non-finite value.  Write the result with systems::writeVectorEntries.
 */
[[nodiscard]] LevelSetSideVertexEntries collectVertexFieldOnLevelSetSide(
    const systems::FESystem& system,
    FieldId field,
    const dofs::EntityDofMap& vertex_dof_map,
    const LevelSetSideVertexValues& values,
    const std::string& non_finite_prefix,
    const std::string& non_finite_suffix);

} // namespace level_set
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_LEVELSET_LEVELSET_SIDE_SUPPORT_H
