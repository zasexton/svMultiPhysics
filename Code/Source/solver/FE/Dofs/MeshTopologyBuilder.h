/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_DOFS_MESHTOPOLOGYBUILDER_H
#define SVMP_FE_DOFS_MESHTOPOLOGYBUILDER_H

#include "Core/Types.h"

#include <array>
#include <span>
#include <vector>

namespace svmp {
namespace FE {
namespace dofs {

struct CellToEntityCSR {
    std::vector<MeshOffset> offsets; // size = n_cells + 1
    std::vector<MeshIndex> data;     // flat entity IDs per cell
};

CellToEntityCSR buildCellToEdgesRefOrder(
    int dim,
    std::span<const MeshOffset> cell2vertex_offsets,
    std::span<const MeshIndex> cell2vertex,
    std::span<const std::array<MeshIndex, 2>> edge2vertex);

CellToEntityCSR buildCellToFacesRefOrder(
    int dim,
    std::span<const MeshOffset> cell2vertex_offsets,
    std::span<const MeshIndex> cell2vertex,
    std::span<const MeshOffset> face2vertex_offsets,
    std::span<const MeshIndex> face2vertex);

} // namespace dofs
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_DOFS_MESHTOPOLOGYBUILDER_H

