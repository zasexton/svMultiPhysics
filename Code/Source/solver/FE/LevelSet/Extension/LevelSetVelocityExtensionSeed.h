/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file LevelSetVelocityExtensionSeed.h
 * @ingroup fe_level_set
 * @brief The known set of a velocity extension into the dry part of an
 *        unfitted level-set mesh, and the revision of its frozen rows.
 *
 * The known set of the PDE velocity extension
 * (LevelSetPdeVelocityExtension.h) is the wet vertices plus every vertex of
 * the retained interface cells, so that the extension equals the physical
 * velocity on the interface.  These functions find the interface cells from
 * nodal level-set values (when no generated cut context is authoritative),
 * mark their vertices, make the mask agree across the communicator and count
 * owned vertices once.  VelocityExtensionMapRevision keys the frozen rows of
 * a monolithic extension on the mesh, the free-surface geometry, the level
 * set and the known set.
 *
 * Moved from Application/Core/LevelSetVelocityExtensionMap.{h,cpp} with the
 * bodies and messages unchanged when the algebraic wall-compatible wet
 * extension was retired (decisions D47 and D55).
 */

#include "Core/Types.h"
#include "Mesh/Core/MeshComm.h"
#include "Mesh/Mesh.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace level_set {

/// The revision of a frozen extension map: the mesh revisions, the
/// free-surface geometry revision, and hashes of the level-set values and of
/// the known set.  key() is rank-local; communicatorCombinedRevision()
/// (LevelSetPdeVelocityExtension.h) makes it agree across ranks.
struct VelocityExtensionMapRevision {
  std::uint64_t mesh_geometry{0u};
  std::uint64_t mesh_topology{0u};
  std::uint64_t mesh_ownership{0u};
  std::uint64_t mesh_numbering{0u};
  std::uint64_t free_surface_geometry{0u};
  std::uint64_t level_set_values{0u};
  std::uint64_t active_set{0u};

  [[nodiscard]] std::uint64_t key() const noexcept;
  [[nodiscard]] bool complete() const noexcept;
  bool operator==(const VelocityExtensionMapRevision&) const = default;
};

[[nodiscard]] VelocityExtensionMapRevision velocityExtensionMapRevision(
    std::uint64_t mesh_geometry,
    std::uint64_t mesh_topology,
    std::uint64_t mesh_ownership,
    std::uint64_t mesh_numbering,
    std::uint64_t free_surface_geometry,
    std::span<const double> level_set_values,
    std::span<const std::uint8_t> active_set);

/// The number of vertices set in `mask` over the communicator, each counted
/// on its owner rank.
[[nodiscard]] std::size_t globalOwnedVelocityExtensionMaskCount(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    std::span<const std::uint8_t> mask);

/// The sum of a rank-local interface sample count over the communicator.
[[nodiscard]] std::size_t globalVelocityExtensionGeometrySampleCount(
    std::size_t local_count,
    const svmp::MeshComm& comm);

/// Marks every vertex of `cells` in `trace_support`; returns the number of
/// newly marked vertices.
[[nodiscard]] std::size_t markVelocityExtensionTraceSupportCells(
    const svmp::Mesh& mesh,
    std::span<const svmp::FE::MeshIndex> cells,
    std::vector<std::uint8_t>& trace_support);

/// The local cells whose nodal level-set values change sign across, or touch,
/// `isovalue`.
[[nodiscard]] std::vector<svmp::FE::MeshIndex>
nodalVelocityExtensionInterfaceCells(
    const svmp::Mesh& mesh,
    std::span<const double> phi,
    double isovalue);

/// Marks every local copy of a vertex that any rank marked; returns the
/// global owned count of the synchronized mask.
[[nodiscard]] std::size_t synchronizeVelocityExtensionTraceSupportMask(
    const svmp::Mesh& mesh,
    const svmp::MeshComm& comm,
    std::vector<std::uint8_t>& trace_support);

} // namespace level_set
} // namespace FE
} // namespace svmp
