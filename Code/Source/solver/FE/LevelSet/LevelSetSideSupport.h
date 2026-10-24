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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
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

} // namespace level_set
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_LEVELSET_LEVELSET_SIDE_SUPPORT_H
