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

#ifndef SVMP_NODE_ORDERING_H
#define SVMP_NODE_ORDERING_H

#include "../Core/MeshTypes.h"
#include "CellTopology.h"

#include <vector>

namespace svmp {

enum class NodeOrderingFormat {
  VTK,   // internal canonical ordering used by CurvilinearEval / CellTopology
  Gmsh,  // Gmsh .msh ordering (high-order nodes follow Gmsh conventions)
};

class NodeOrdering {
public:
  // Return a permutation `perm` such that:
  //   nodes_vtk[i] = nodes_in[ perm[i] ]
  // where nodes_in is in `fmt` ordering and nodes_vtk is in internal canonical (VTK-like) ordering.
  static std::vector<index_t> permutation_to_vtk(NodeOrderingFormat fmt,
                                                 CellFamily family,
                                                 int order,
                                                 size_t node_count);

  // In-place reorder from `fmt` ordering to internal canonical (VTK-like) ordering.
  static void reorder_to_vtk(NodeOrderingFormat fmt,
                             CellFamily family,
                             int order,
                             std::vector<size_t>& nodes);

  // Return a permutation `perm` such that:
  //   nodes_out[i] = nodes_vtk[ perm[i] ]
  // where nodes_vtk is in internal canonical ordering and nodes_out is in `fmt` ordering.
  static std::vector<index_t> permutation_from_vtk(NodeOrderingFormat fmt,
                                                   CellFamily family,
                                                   int order,
                                                   size_t node_count);

  // In-place reorder from internal canonical (VTK-like) ordering to `fmt` ordering.
  static void reorder_from_vtk(NodeOrderingFormat fmt,
                               CellFamily family,
                               int order,
                               std::vector<size_t>& nodes);
};

} // namespace svmp

#endif // SVMP_NODE_ORDERING_H
