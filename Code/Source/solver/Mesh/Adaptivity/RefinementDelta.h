/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SVMP_REFINEMENT_DELTA_H
#define SVMP_REFINEMENT_DELTA_H

/**
 * @file RefinementDelta.h
 * @brief Mesh ↔ FE adaptivity contract types (GID-based)
 *
 * This header defines small, dependency-light data structures that describe a
 * mesh refinement operation in terms of **stable global IDs (GIDs)**.
 *
 * Design goals:
 * - Mesh remains independent of FE.
 * - FE can consume these records (optionally) to build prolongation/restriction
 *   operators and hanging-node constraints at the DOF level.
 * - Records are stable across mesh rebuilds as long as Mesh preserves GIDs for
 *   existing entities and assigns unique new GIDs for created entities.
 */

#include "../Core/MeshTypes.h"
#include "RefinementRules.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace svmp {

/**
 * @brief Provenance record for a newly-created vertex
 *
 * For linear (P1/Q1) nodal fields, this is sufficient to drive prolongation and
 * vertex-level hanging constraints. Higher-order DOF provenance (edge/face/volume
 * nodes) is solver/FE-specific and is intentionally not represented here yet.
 */
struct VertexProvenanceRecord {
  gid_t new_vertex_gid = INVALID_GID;
  std::vector<std::pair<gid_t, double>> parent_vertex_weights;
};

/**
 * @brief Record describing how one parent cell was refined
 */
struct CellRefinementRecord {
  gid_t parent_cell_gid = INVALID_GID;
  CellFamily family = CellFamily::Point;
  RefinementSpec spec{};
  std::vector<gid_t> child_cell_gids;
};

/**
 * @brief Summary of a single refinement pass, expressed in stable IDs
 */
struct RefinementDelta {
  static constexpr std::uint32_t kVersion = 1;
  std::uint32_t version = kVersion;

  std::vector<CellRefinementRecord> refined_cells;
  std::vector<VertexProvenanceRecord> new_vertices;
};

} // namespace svmp

#endif // SVMP_REFINEMENT_DELTA_H

