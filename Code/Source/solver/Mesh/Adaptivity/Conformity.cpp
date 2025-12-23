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

#include "Conformity.h"
#include "../Core/MeshBase.h"
#include "../Fields/MeshFields.h"
#include "../Topology/CellTopology.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>

namespace svmp {

namespace {

using EdgeKey = std::pair<index_t, index_t>;
using FaceKey = std::vector<index_t>;

struct EdgeIncident {
  index_t cell = INVALID_INDEX;
  int local_edge = -1;
};

struct FaceIncident {
  index_t cell = INVALID_INDEX;
  int local_face = -1;
  std::vector<index_t> local_vertices;  // local vertex indices of this face in the cell
};

EdgeKey make_edge_key(index_t a, index_t b) {
  if (a < b) {
    return {a, b};
  }
  return {b, a};
}

FaceKey make_face_key(const std::vector<index_t>& vertices) {
  FaceKey key = vertices;
  std::sort(key.begin(), key.end());
  return key;
}

std::vector<index_t> cell_vertices(const MeshBase& mesh, index_t cell) {
  auto [ptr, count] = mesh.cell_vertices_span(cell);
  if (!ptr || count == 0) {
    return {};
  }
  return std::vector<index_t>(ptr, ptr + count);
}

std::map<EdgeKey, std::vector<EdgeIncident>> build_edge_incidence(const MeshBase& mesh) {
  std::map<EdgeKey, std::vector<EdgeIncident>> edges;

  for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
    const CellFamily family = mesh.cell_shape(c).family;
    const auto conn = cell_vertices(mesh, c);
    if (conn.empty()) continue;

    const auto ev = CellTopology::get_edges_view(family);
    for (int e = 0; e < ev.edge_count; ++e) {
      const index_t lv0 = ev.pairs_flat[2 * e + 0];
      const index_t lv1 = ev.pairs_flat[2 * e + 1];
      if (lv0 < 0 || lv1 < 0) continue;
      if (static_cast<size_t>(lv0) >= conn.size() || static_cast<size_t>(lv1) >= conn.size()) continue;
      const index_t gv0 = conn[static_cast<size_t>(lv0)];
      const index_t gv1 = conn[static_cast<size_t>(lv1)];
      edges[make_edge_key(gv0, gv1)].push_back(EdgeIncident{c, e});
    }
  }

  return edges;
}

std::map<FaceKey, std::vector<FaceIncident>> build_face_incidence(const MeshBase& mesh) {
  std::map<FaceKey, std::vector<FaceIncident>> faces;
  if (mesh.dim() != 3) {
    return faces;
  }

  for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
    const CellFamily family = mesh.cell_shape(c).family;
    const auto conn = cell_vertices(mesh, c);
    if (conn.empty()) continue;

    const auto fv = CellTopology::get_boundary_faces_canonical_view(family);
    if (!fv.indices || !fv.offsets || fv.face_count <= 0) continue;

    for (int f = 0; f < fv.face_count; ++f) {
      const int start = fv.offsets[f];
      const int end = fv.offsets[f + 1];
      if (end <= start) continue;

      std::vector<index_t> face_vertices;
      face_vertices.reserve(static_cast<size_t>(end - start));
      std::vector<index_t> face_locals;
      face_locals.reserve(static_cast<size_t>(end - start));
      for (int i = start; i < end; ++i) {
        const index_t lv = fv.indices[i];
        if (lv < 0 || static_cast<size_t>(lv) >= conn.size()) continue;
        face_vertices.push_back(conn[static_cast<size_t>(lv)]);
        face_locals.push_back(lv);
      }
      if (face_vertices.size() < 3) continue;

      faces[make_face_key(face_vertices)].push_back(FaceIncident{c, f, std::move(face_locals)});
    }
  }

  return faces;
}

bool triangle_splits_edge(const RefinementSpec& spec, int edge_id) {
  const int e = edge_id % 3;
  switch (spec.pattern) {
    case RefinementPattern::RED:
    case RefinementPattern::ISOTROPIC:
      return true;
    case RefinementPattern::GREEN:
      return e == static_cast<int>(spec.selector % 3u);
    case RefinementPattern::BLUE: {
      const int s = static_cast<int>(spec.selector % 3u);
      if (s == 0) return e == 0 || e == 2;
      if (s == 1) return e == 0 || e == 1;
      return e == 1 || e == 2;
    }
    case RefinementPattern::BISECTION:
      return e == static_cast<int>(spec.selector % 3u);
    default:
      return true;
  }
}

RefinementSpec triangle_spec_from_required_edges(std::uint8_t mask, bool allow_green_blue) {
  if (!allow_green_blue) {
    return RefinementSpec{RefinementPattern::RED, 0u};
  }

  const std::uint8_t m = static_cast<std::uint8_t>(mask & 0x7u);
  const auto popcnt = [](std::uint8_t x) {
    int c = 0;
    while (x) {
      c += (x & 1u) ? 1 : 0;
      x = static_cast<std::uint8_t>(x >> 1);
    }
    return c;
  };

  const int bits = popcnt(m);
  if (bits >= 3) {
    return RefinementSpec{RefinementPattern::RED, 0u};
  }
  if (bits == 1) {
    for (std::uint32_t e = 0; e < 3u; ++e) {
      if (m & (1u << e)) {
        return RefinementSpec{RefinementPattern::GREEN, e};
      }
    }
    return RefinementSpec{RefinementPattern::RED, 0u};
  }
  if (bits == 2) {
    // Determine shared vertex of the two required edges.
    // Edge ids: 0:(0,1), 1:(1,2), 2:(2,0)
    if (m == 0x3u) {  // edges 0 and 1 share vertex 1
      return RefinementSpec{RefinementPattern::BLUE, 1u};
    }
    if (m == 0x6u) {  // edges 1 and 2 share vertex 2
      return RefinementSpec{RefinementPattern::BLUE, 2u};
    }
    // edges 0 and 2 share vertex 0
    return RefinementSpec{RefinementPattern::BLUE, 0u};
  }
  return RefinementSpec{RefinementPattern::RED, 0u};
}

bool quad_splits_edge(const RefinementSpec& spec, int edge_id) {
  const int e = edge_id % 4;
  switch (spec.pattern) {
    case RefinementPattern::RED:
    case RefinementPattern::ISOTROPIC:
      return true;
    case RefinementPattern::ANISOTROPIC: {
      const int dir = static_cast<int>(spec.selector % 2u);
      if (dir == 0) {
        return e == 0 || e == 2;
      }
      return e == 1 || e == 3;
    }
    default:
      return true;
  }
}

RefinementSpec quad_spec_from_required_edges(std::uint8_t mask, bool allow_anisotropic) {
  const std::uint8_t m = static_cast<std::uint8_t>(mask & 0xFu);
  if (!allow_anisotropic) {
    return RefinementSpec{RefinementPattern::RED, 0u};
  }

  const std::uint8_t dir0 = static_cast<std::uint8_t>((1u << 0) | (1u << 2));
  const std::uint8_t dir1 = static_cast<std::uint8_t>((1u << 1) | (1u << 3));

  if ((m & ~dir0) == 0u) {
    return RefinementSpec{RefinementPattern::ANISOTROPIC, 0u};
  }
  if ((m & ~dir1) == 0u) {
    return RefinementSpec{RefinementPattern::ANISOTROPIC, 1u};
  }
  return RefinementSpec{RefinementPattern::RED, 0u};
}

bool tetra_splits_face(const RefinementSpec& spec, const std::vector<index_t>& face_locals) {
  switch (spec.pattern) {
    case RefinementPattern::RED:
    case RefinementPattern::ISOTROPIC:
      return true;
    case RefinementPattern::GREEN: {
      const index_t ov = static_cast<index_t>(spec.selector % 4u);
      for (index_t lv : face_locals) {
        if (lv == ov) {
          return false;
        }
      }
      return true;
    }
    default:
      return true;
  }
}

index_t tetra_opposite_vertex_for_face(const std::vector<index_t>& face_locals) {
  bool used[4] = {false, false, false, false};
  for (index_t lv : face_locals) {
    if (lv >= 0 && lv < 4) {
      used[lv] = true;
    }
  }
  for (index_t i = 0; i < 4; ++i) {
    if (!used[i]) return i;
  }
  return 3;
}

bool pyramid_splits_face(const RefinementSpec& spec, const std::vector<index_t>& face_locals) {
  switch (spec.pattern) {
    case RefinementPattern::RED:
    case RefinementPattern::ISOTROPIC:
      return true;
    case RefinementPattern::ANISOTROPIC:
      // Base split: only the quad base face (four corners) is refined.
      return face_locals.size() == 4;
    default:
      return true;
  }
}

bool wedge_splits_face(const RefinementSpec& spec, const std::vector<index_t>& face_locals) {
  switch (spec.pattern) {
    case RefinementPattern::RED:
    case RefinementPattern::ISOTROPIC:
      return true;
    case RefinementPattern::GREEN: {
      if (face_locals.size() != 4) {
        return false;
      }
      std::vector<int> base;
      base.reserve(2);
      for (index_t lv : face_locals) {
        if (lv >= 0 && lv <= 2) {
          base.push_back(static_cast<int>(lv));
        }
      }
      if (base.size() != 2) {
        return false;
      }
      std::sort(base.begin(), base.end());
      const int a = base[0];
      const int b = base[1];
      int base_edge = -1;
      if (a == 0 && b == 1) base_edge = 0;
      if (a == 1 && b == 2) base_edge = 1;
      if (a == 0 && b == 2) base_edge = 2;
      if (base_edge < 0) return false;
      return static_cast<std::uint32_t>(base_edge) == (spec.selector % 3u);
    }
    default:
      return true;
  }
}

std::uint32_t hex_axis_mask_for_face(const std::vector<index_t>& face_locals) {
  // Determine which coordinate bit is constant for this face in the standard Hex local ordering.
  // Vertex bits: index -> (x,y,z) in {0,1}^3.
  static constexpr int bits[8][3] = {
      {0, 0, 0},  // 0
      {1, 0, 0},  // 1
      {1, 1, 0},  // 2
      {0, 1, 0},  // 3
      {0, 0, 1},  // 4
      {1, 0, 1},  // 5
      {1, 1, 1},  // 6
      {0, 1, 1}   // 7
  };

  if (face_locals.empty()) {
    return 7u;
  }

  int x0 = bits[face_locals[0]][0];
  int y0 = bits[face_locals[0]][1];
  int z0 = bits[face_locals[0]][2];
  bool x_const = true;
  bool y_const = true;
  bool z_const = true;
  for (index_t lv : face_locals) {
    if (lv < 0 || lv >= 8) continue;
    x_const = x_const && (bits[lv][0] == x0);
    y_const = y_const && (bits[lv][1] == y0);
    z_const = z_const && (bits[lv][2] == z0);
  }

  // Split the two tangential axes for that face.
  if (x_const) return 2u | 4u;  // split Y+Z
  if (y_const) return 1u | 4u;  // split X+Z
  if (z_const) return 1u | 2u;  // split X+Y
  return 7u;
}

bool hex_splits_face(const RefinementSpec& spec, const std::vector<index_t>& face_locals) {
  switch (spec.pattern) {
    case RefinementPattern::RED:
    case RefinementPattern::ISOTROPIC:
      return true;
    case RefinementPattern::ANISOTROPIC: {
      const std::uint32_t axis_mask = spec.selector;
      const std::uint32_t need = hex_axis_mask_for_face(face_locals);
      // For face conformity with a regular refinement neighbor, the face must be
      // split in both tangential directions (need is a 2-bit mask).
      return (axis_mask & need) == need;
    }
    default:
      return true;
  }
}

} // namespace

ClosureConformityEnforcer::ClosureConformityEnforcer(const Config& config)
    : config_(config) {}

NonConformity ClosureConformityEnforcer::check_conformity(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks) const {
  NonConformity non_conformity;
  non_conformity.max_level_difference = 0;

  const size_t n_cells = mesh.n_cells();
  if (marks.size() != n_cells) {
    return non_conformity;
  }

  bool has_surface_cells = false;
  bool has_volume_cells = false;
  for (index_t c = 0; c < static_cast<index_t>(n_cells); ++c) {
    const CellFamily family = mesh.cell_shape(c).family;
    has_surface_cells = has_surface_cells || (family == CellFamily::Triangle || family == CellFamily::Quad || family == CellFamily::Polygon);
    has_volume_cells = has_volume_cells || (family == CellFamily::Tetra || family == CellFamily::Hex || family == CellFamily::Wedge ||
                                            family == CellFamily::Pyramid || family == CellFamily::Polyhedron);
  }

  const auto edges = build_edge_incidence(mesh);
  const auto faces = build_face_incidence(mesh);

  const auto target_level = [&](index_t c) -> size_t {
    const size_t base = mesh.refinement_level(c);
    if (marks[static_cast<size_t>(c)] == MarkType::REFINE) {
      return base + 1;
    }
    if (marks[static_cast<size_t>(c)] == MarkType::COARSEN && base > 0) {
      return base - 1;
    }
    return base;
  };

  if (config_.check_edge_conformity && has_surface_cells) {
    for (const auto& [edge, inc] : edges) {
      if (inc.size() != 2) continue;
      const index_t c0 = inc[0].cell;
      const index_t c1 = inc[1].cell;
      if (c0 == INVALID_INDEX || c1 == INVALID_INDEX) continue;
      const size_t l0 = target_level(c0);
      const size_t l1 = target_level(c1);
      const size_t diff = (l0 > l1) ? (l0 - l1) : (l1 - l0);
      non_conformity.max_level_difference = std::max(non_conformity.max_level_difference, diff);

      const bool r0 = marks[static_cast<size_t>(c0)] == MarkType::REFINE;
      const bool r1 = marks[static_cast<size_t>(c1)] == MarkType::REFINE;
      if (r0 != r1) {
        non_conformity.non_conforming_edges.insert(edge);
        non_conformity.cells_needing_closure.insert(static_cast<size_t>(r0 ? c1 : c0));
      }
    }
  }

  if (config_.check_face_conformity && has_volume_cells) {
    for (const auto& [face, inc] : faces) {
      if (inc.size() != 2) continue;
      const index_t c0 = inc[0].cell;
      const index_t c1 = inc[1].cell;
      if (c0 == INVALID_INDEX || c1 == INVALID_INDEX) continue;
      const size_t l0 = target_level(c0);
      const size_t l1 = target_level(c1);
      const size_t diff = (l0 > l1) ? (l0 - l1) : (l1 - l0);
      non_conformity.max_level_difference = std::max(non_conformity.max_level_difference, diff);

      const bool r0 = marks[static_cast<size_t>(c0)] == MarkType::REFINE;
      const bool r1 = marks[static_cast<size_t>(c1)] == MarkType::REFINE;
      if (r0 != r1) {
        std::vector<size_t> face_sz;
        face_sz.reserve(face.size());
        for (index_t v : face) face_sz.push_back(static_cast<size_t>(v));
        non_conformity.non_conforming_faces.insert(std::move(face_sz));
        non_conformity.cells_needing_closure.insert(static_cast<size_t>(r0 ? c1 : c0));
      }
    }
  }

  return non_conformity;
}

size_t ClosureConformityEnforcer::enforce_conformity(
    const MeshBase& mesh,
    std::vector<MarkType>& marks,
    const AdaptivityOptions& options) const {
  cell_refinement_specs_.clear();

  const size_t n_cells = mesh.n_cells();
  if (marks.size() != n_cells) {
    marks.resize(n_cells, MarkType::NONE);
  }

  const std::vector<MarkType> initial_marks = marks;

  bool has_surface_cells = false;
  bool has_volume_cells = false;
  for (index_t c = 0; c < static_cast<index_t>(n_cells); ++c) {
    const CellFamily family = mesh.cell_shape(c).family;
    has_surface_cells = has_surface_cells || (family == CellFamily::Triangle || family == CellFamily::Quad || family == CellFamily::Polygon);
    has_volume_cells = has_volume_cells || (family == CellFamily::Tetra || family == CellFamily::Hex || family == CellFamily::Wedge ||
                                            family == CellFamily::Pyramid || family == CellFamily::Polyhedron);
  }

  const auto edges = build_edge_incidence(mesh);
  const auto faces = build_face_incidence(mesh);

  std::vector<std::uint8_t> tri_required_edges(n_cells, 0);
  std::vector<std::uint8_t> quad_required_edges(n_cells, 0);
  std::vector<std::uint8_t> tet_required_faces(n_cells, 0);
  std::vector<std::uint32_t> hex_required_axis_mask(n_cells, 0u);
  std::vector<int> wedge_required_base_edge(n_cells, -1);
  std::vector<bool> pyramid_base_required(n_cells, false);

  // Initialize specs for user-marked refinement.
  for (size_t c = 0; c < n_cells; ++c) {
    if (marks[c] != MarkType::REFINE) continue;

    const CellFamily family = mesh.cell_shape(static_cast<index_t>(c)).family;
    cell_refinement_specs_[c] = RefinementSpec{RefinementPattern::RED, 0u};

    if (family == CellFamily::Triangle) tri_required_edges[c] = 0x7u;
    if (family == CellFamily::Quad) quad_required_edges[c] = 0xFu;
    if (family == CellFamily::Tetra) tet_required_faces[c] = 0xFu;
    if (family == CellFamily::Hex) hex_required_axis_mask[c] = 0x7u;
    if (family == CellFamily::Wedge) wedge_required_base_edge[c] = -2; // sentinel meaning RED
    if (family == CellFamily::Pyramid) pyramid_base_required[c] = false; // RED refines everything
  }

  const bool allow_minimal = options.use_green_closure && config_.use_green_closure;
  const size_t max_iters = std::min(options.max_closure_iterations, config_.max_iterations);
  size_t iterations = 0;

  auto ensure_refined = [&](size_t c) {
    if (marks[c] == MarkType::NONE) {
      marks[c] = MarkType::REFINE;
    }
  };

  auto set_spec = [&](size_t c, const RefinementSpec& spec) {
    const auto it = cell_refinement_specs_.find(c);
    if (it == cell_refinement_specs_.end() || it->second.pattern != spec.pattern || it->second.selector != spec.selector) {
      cell_refinement_specs_[c] = spec;
      return true;
    }
    return false;
  };

  const auto cell_splits_edge = [&](size_t c, int local_edge) {
    const auto it = cell_refinement_specs_.find(c);
    if (it == cell_refinement_specs_.end()) return false;
    const CellFamily family = mesh.cell_shape(static_cast<index_t>(c)).family;
    if (family == CellFamily::Triangle) return triangle_splits_edge(it->second, local_edge);
    if (family == CellFamily::Quad) return quad_splits_edge(it->second, local_edge);
    // For other families, treat as splitting all edges when refined.
    return true;
  };

  const auto cell_splits_face = [&](size_t c, const std::vector<index_t>& face_locals) {
    const auto it = cell_refinement_specs_.find(c);
    if (it == cell_refinement_specs_.end()) return false;
    const CellFamily family = mesh.cell_shape(static_cast<index_t>(c)).family;
    if (family == CellFamily::Tetra) return tetra_splits_face(it->second, face_locals);
    if (family == CellFamily::Hex) return hex_splits_face(it->second, face_locals);
    if (family == CellFamily::Wedge) return wedge_splits_face(it->second, face_locals);
    if (family == CellFamily::Pyramid) return pyramid_splits_face(it->second, face_locals);
    return true;
  };

  for (; iterations < max_iters; ++iterations) {
    bool changed = false;

    if (config_.check_edge_conformity && has_surface_cells) {
      for (const auto& [edge, inc] : edges) {
        if (inc.size() != 2) continue;
        const size_t c0 = static_cast<size_t>(inc[0].cell);
        const size_t c1 = static_cast<size_t>(inc[1].cell);
        if (c0 >= n_cells || c1 >= n_cells) continue;

        const bool s0 = cell_splits_edge(c0, inc[0].local_edge);
        const bool s1 = cell_splits_edge(c1, inc[1].local_edge);
        if (s0 == s1) continue;

        const size_t coarse = s0 ? c1 : c0;
        const int local_edge = s0 ? inc[1].local_edge : inc[0].local_edge;
        const CellFamily family = mesh.cell_shape(static_cast<index_t>(coarse)).family;

        ensure_refined(coarse);

        if (initial_marks[coarse] == MarkType::REFINE) {
          // User-requested refinement remains RED.
          changed = set_spec(coarse, RefinementSpec{RefinementPattern::RED, 0u}) || changed;
          continue;
        }

        if (family == CellFamily::Triangle) {
          tri_required_edges[coarse] |= static_cast<std::uint8_t>(1u << (local_edge % 3));
          const auto spec = triangle_spec_from_required_edges(tri_required_edges[coarse], allow_minimal);
          changed = set_spec(coarse, spec) || changed;
        } else if (family == CellFamily::Quad) {
          quad_required_edges[coarse] |= static_cast<std::uint8_t>(1u << (local_edge % 4));
          const auto spec = quad_spec_from_required_edges(quad_required_edges[coarse], allow_minimal && options.allow_anisotropic);
          changed = set_spec(coarse, spec) || changed;
        } else {
          changed = set_spec(coarse, RefinementSpec{RefinementPattern::RED, 0u}) || changed;
        }
      }
    }

    if (config_.check_face_conformity && has_volume_cells) {
      for (const auto& [face, inc] : faces) {
        if (inc.size() != 2) continue;
        const size_t c0 = static_cast<size_t>(inc[0].cell);
        const size_t c1 = static_cast<size_t>(inc[1].cell);
        if (c0 >= n_cells || c1 >= n_cells) continue;

        const bool s0 = cell_splits_face(c0, inc[0].local_vertices);
        const bool s1 = cell_splits_face(c1, inc[1].local_vertices);
        if (s0 == s1) continue;

        const size_t coarse = s0 ? c1 : c0;
        const auto& face_locals = s0 ? inc[1].local_vertices : inc[0].local_vertices;
        const CellFamily family = mesh.cell_shape(static_cast<index_t>(coarse)).family;

        ensure_refined(coarse);

        if (initial_marks[coarse] == MarkType::REFINE) {
          changed = set_spec(coarse, RefinementSpec{RefinementPattern::RED, 0u}) || changed;
          continue;
        }

        if (!allow_minimal) {
          changed = set_spec(coarse, RefinementSpec{RefinementPattern::RED, 0u}) || changed;
          continue;
        }

        if (family == CellFamily::Tetra) {
          tet_required_faces[coarse] |= static_cast<std::uint8_t>(1u << (inc[0].cell == static_cast<index_t>(coarse) ? inc[0].local_face : inc[1].local_face));
          // If only one face is required, choose face-GREEN with selector = opposite vertex.
          const std::uint8_t mask = tet_required_faces[coarse] & 0xFu;
          if ((mask & (mask - 1u)) == 0u) {
            const index_t ov = tetra_opposite_vertex_for_face(face_locals);
            changed = set_spec(coarse, RefinementSpec{RefinementPattern::GREEN, static_cast<std::uint32_t>(ov)}) || changed;
          } else {
            changed = set_spec(coarse, RefinementSpec{RefinementPattern::RED, 0u}) || changed;
          }
        } else if (family == CellFamily::Hex) {
          hex_required_axis_mask[coarse] |= hex_axis_mask_for_face(face_locals);
          changed =
              set_spec(coarse, RefinementSpec{RefinementPattern::ANISOTROPIC, hex_required_axis_mask[coarse]}) ||
              changed;
        } else if (family == CellFamily::Wedge) {
          // Determine required base edge for this quad face.
          if (face_locals.size() == 4) {
            std::vector<int> base;
            base.reserve(2);
            for (index_t lv : face_locals) {
              if (lv >= 0 && lv <= 2) base.push_back(static_cast<int>(lv));
            }
            if (base.size() == 2) {
              std::sort(base.begin(), base.end());
              const int a = base[0];
              const int b = base[1];
              int base_edge = -1;
              if (a == 0 && b == 1) base_edge = 0;
              if (a == 1 && b == 2) base_edge = 1;
              if (a == 0 && b == 2) base_edge = 2;
              if (wedge_required_base_edge[coarse] >= 0 && wedge_required_base_edge[coarse] != base_edge) {
                changed = set_spec(coarse, RefinementSpec{RefinementPattern::RED, 0u}) || changed;
              } else if (base_edge >= 0) {
                wedge_required_base_edge[coarse] = base_edge;
                changed =
                    set_spec(coarse, RefinementSpec{RefinementPattern::GREEN, static_cast<std::uint32_t>(base_edge)}) ||
                    changed;
              } else {
                changed = set_spec(coarse, RefinementSpec{RefinementPattern::RED, 0u}) || changed;
              }
            } else {
              changed = set_spec(coarse, RefinementSpec{RefinementPattern::RED, 0u}) || changed;
            }
          } else {
            changed = set_spec(coarse, RefinementSpec{RefinementPattern::RED, 0u}) || changed;
          }
        } else if (family == CellFamily::Pyramid) {
          pyramid_base_required[coarse] = true;
          changed = set_spec(coarse, RefinementSpec{RefinementPattern::ANISOTROPIC, 0u}) || changed;
        } else {
          changed = set_spec(coarse, RefinementSpec{RefinementPattern::RED, 0u}) || changed;
        }
      }
    }

    if (!changed) {
      break;
    }
  }

  // Ensure all refined cells have a spec entry.
  for (size_t c = 0; c < n_cells; ++c) {
    if (marks[c] != MarkType::REFINE) continue;
    if (cell_refinement_specs_.count(c) > 0) continue;
    cell_refinement_specs_[c] = RefinementSpec{RefinementPattern::RED, 0u};
  }

  return iterations;
}

std::map<size_t, std::map<size_t, double>> ClosureConformityEnforcer::generate_constraints(
    const MeshBase& mesh,
    const NonConformity& non_conformity) const {
  (void)mesh;
  (void)non_conformity;
  return {};
}

bool ClosureConformityEnforcer::is_edge_conforming(
    const MeshBase& mesh,
    size_t v1, size_t v2,
    const std::vector<MarkType>& marks) const {
  if (marks.size() != mesh.n_cells()) {
    return true;
  }
  const auto cells = find_edge_elements(mesh, v1, v2);
  if (cells.size() != 2) {
    return true;
  }
  const auto lvl0 = mesh.refinement_level(static_cast<index_t>(cells[0])) + (marks[cells[0]] == MarkType::REFINE ? 1 : 0);
  const auto lvl1 = mesh.refinement_level(static_cast<index_t>(cells[1])) + (marks[cells[1]] == MarkType::REFINE ? 1 : 0);
  return lvl0 == lvl1;
}

bool ClosureConformityEnforcer::is_face_conforming(
    const MeshBase& mesh,
    const std::vector<size_t>& face_vertices,
    const std::vector<MarkType>& marks) const {
  if (marks.size() != mesh.n_cells()) {
    return true;
  }
  const auto cells = find_face_elements(mesh, face_vertices);
  if (cells.size() != 2) {
    return true;
  }
  const auto lvl0 = mesh.refinement_level(static_cast<index_t>(cells[0])) + (marks[cells[0]] == MarkType::REFINE ? 1 : 0);
  const auto lvl1 = mesh.refinement_level(static_cast<index_t>(cells[1])) + (marks[cells[1]] == MarkType::REFINE ? 1 : 0);
  return lvl0 == lvl1;
}

std::vector<size_t> ClosureConformityEnforcer::find_edge_elements(
    const MeshBase& mesh,
    size_t v1, size_t v2) const {
  std::vector<size_t> cells;
  cells.reserve(4);
  const index_t vv1 = static_cast<index_t>(v1);
  const index_t vv2 = static_cast<index_t>(v2);
  for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
    auto [ptr, count] = mesh.cell_vertices_span(c);
    bool has1 = false;
    bool has2 = false;
    for (size_t i = 0; i < count; ++i) {
      has1 = has1 || (ptr[i] == vv1);
      has2 = has2 || (ptr[i] == vv2);
    }
    if (has1 && has2) {
      cells.push_back(static_cast<size_t>(c));
    }
  }
  return cells;
}

std::vector<size_t> ClosureConformityEnforcer::find_face_elements(
    const MeshBase& mesh,
    const std::vector<size_t>& face_vertices) const {
  std::vector<size_t> cells;
  cells.reserve(4);
  if (face_vertices.empty()) {
    return cells;
  }
  std::vector<index_t> face;
  face.reserve(face_vertices.size());
  for (size_t v : face_vertices) face.push_back(static_cast<index_t>(v));
  std::sort(face.begin(), face.end());

  for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
    const auto conn = cell_vertices(mesh, c);
    if (conn.size() < face.size()) continue;
    std::vector<index_t> sorted_conn = conn;
    std::sort(sorted_conn.begin(), sorted_conn.end());
    if (std::includes(sorted_conn.begin(), sorted_conn.end(), face.begin(), face.end())) {
      cells.push_back(static_cast<size_t>(c));
    }
  }
  return cells;
}

void ClosureConformityEnforcer::mark_for_closure(
    std::vector<MarkType>& marks,
    size_t elem_id) const {
  if (elem_id < marks.size() && marks[elem_id] == MarkType::NONE) {
    marks[elem_id] = MarkType::REFINE;
  }
}

HangingNodeConformityEnforcer::HangingNodeConformityEnforcer(const Config& config)
    : config_(config) {}

NonConformity HangingNodeConformityEnforcer::check_conformity(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks) const {
  NonConformity non_conformity;
  non_conformity.max_level_difference = 0;

  if (marks.size() != mesh.n_cells()) {
    return non_conformity;
  }

  const auto edges = build_edge_incidence(mesh);
  const auto faces = build_face_incidence(mesh);

  const auto target_level = [&](index_t c) -> size_t {
    const size_t base = mesh.refinement_level(c);
    if (marks[static_cast<size_t>(c)] == MarkType::REFINE) {
      return base + 1;
    }
    if (marks[static_cast<size_t>(c)] == MarkType::COARSEN && base > 0) {
      return base - 1;
    }
    return base;
  };

  // Edge-based hanging nodes (midpoints).
  for (const auto& [edge, inc] : edges) {
    if (inc.size() != 2) continue;
    const index_t c0 = inc[0].cell;
    const index_t c1 = inc[1].cell;
    const size_t l0 = target_level(c0);
    const size_t l1 = target_level(c1);
    const size_t diff = (l0 > l1) ? (l0 - l1) : (l1 - l0);
    non_conformity.max_level_difference = std::max(non_conformity.max_level_difference, diff);
    if (diff == 0) continue;

    HangingNode hn;
    hn.node_id = static_cast<size_t>(std::numeric_limits<size_t>::max());  // symbolic (not a real vertex yet)
    hn.parent_entity = {static_cast<size_t>(edge.first), static_cast<size_t>(edge.second)};
    hn.on_edge = true;
    hn.level_difference = diff;
    hn.constraints[static_cast<size_t>(edge.first)] = 0.5;
    hn.constraints[static_cast<size_t>(edge.second)] = 0.5;
    non_conformity.hanging_nodes.push_back(std::move(hn));
    non_conformity.non_conforming_edges.insert(edge);
  }

  // Face-based hanging nodes (centroids) for 3D.
  for (const auto& [face, inc] : faces) {
    if (inc.size() != 2) continue;
    const index_t c0 = inc[0].cell;
    const index_t c1 = inc[1].cell;
    const size_t l0 = target_level(c0);
    const size_t l1 = target_level(c1);
    const size_t diff = (l0 > l1) ? (l0 - l1) : (l1 - l0);
    non_conformity.max_level_difference = std::max(non_conformity.max_level_difference, diff);
    if (diff == 0) continue;

    HangingNode hn;
    hn.node_id = static_cast<size_t>(std::numeric_limits<size_t>::max());
    hn.on_edge = false;
    hn.level_difference = diff;
    if (!face.empty()) {
      hn.parent_entity = {static_cast<size_t>(face.front()), static_cast<size_t>(face.back())};
    }
    const double w = 1.0 / static_cast<double>(face.size());
    for (index_t v : face) {
      hn.constraints[static_cast<size_t>(v)] = w;
    }
    non_conformity.hanging_nodes.push_back(std::move(hn));

    std::vector<size_t> face_sz;
    face_sz.reserve(face.size());
    for (index_t v : face) face_sz.push_back(static_cast<size_t>(v));
    non_conformity.non_conforming_faces.insert(std::move(face_sz));
  }

  return non_conformity;
}

size_t HangingNodeConformityEnforcer::enforce_conformity(
    const MeshBase& mesh,
    std::vector<MarkType>& marks,
    const AdaptivityOptions& options) const {
  (void)mesh;
  (void)marks;
  (void)options;
  // Hanging nodes are allowed; no closure refinement is applied.
  return 0;
}

std::map<size_t, std::map<size_t, double>> HangingNodeConformityEnforcer::generate_constraints(
    const MeshBase& mesh,
    const NonConformity& non_conformity) const {
  (void)mesh;
  std::map<size_t, std::map<size_t, double>> constraints;
  for (const auto& hn : non_conformity.hanging_nodes) {
    std::map<size_t, double> eq;
    double sum = 0.0;
    for (const auto& [master, w] : hn.constraints) {
      if (std::abs(w) < config_.constraint_tolerance) continue;
      eq[master] = w;
      sum += w;
    }
    if (eq.empty()) continue;
    // Normalize weights to sum to 1 when possible.
    if (std::abs(sum) > config_.constraint_tolerance) {
      for (auto& [master, w] : eq) {
        w /= sum;
      }
    }
    constraints[hn.node_id] = std::move(eq);
  }
  return constraints;
}

std::vector<HangingNode> HangingNodeConformityEnforcer::find_edge_hanging_nodes(
    const MeshBase& mesh,
    size_t v1, size_t v2,
    const std::vector<MarkType>& marks) const {
  (void)mesh;
  (void)v1;
  (void)v2;
  (void)marks;
  return {};
}

std::vector<HangingNode> HangingNodeConformityEnforcer::find_face_hanging_nodes(
    const MeshBase& mesh,
    const std::vector<size_t>& face_vertices,
    const std::vector<MarkType>& marks) const {
  (void)mesh;
  (void)face_vertices;
  (void)marks;
  return {};
}

std::map<size_t, double> HangingNodeConformityEnforcer::generate_node_constraint(
    const MeshBase& mesh,
    const HangingNode& node) const {
  (void)mesh;
  return node.constraints;
}

MinimalClosureEnforcer::MinimalClosureEnforcer(const Config& config)
    : config_(config) {}

NonConformity MinimalClosureEnforcer::check_conformity(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks) const {
  // Minimal closure shares the same detection as closure refinement.
  ClosureConformityEnforcer closure;
  return closure.check_conformity(mesh, marks);
}

size_t MinimalClosureEnforcer::enforce_conformity(
    const MeshBase& mesh,
    std::vector<MarkType>& marks,
    const AdaptivityOptions& options) const {
  ClosureConformityEnforcer::Config cfg;
  cfg.max_iterations = options.max_closure_iterations;
  cfg.max_level_difference = options.max_level_difference;
  cfg.use_green_closure = options.use_green_closure;
  cfg.propagate_closure = true;
  cfg.check_face_conformity = true;
  cfg.check_edge_conformity = true;

  ClosureConformityEnforcer closure(cfg);
  return closure.enforce_conformity(mesh, marks, options);
}

std::map<size_t, std::map<size_t, double>> MinimalClosureEnforcer::generate_constraints(
    const MeshBase& mesh,
    const NonConformity& non_conformity) const {
  (void)mesh;
  (void)non_conformity;
  return {};
}

std::vector<std::pair<size_t, RefinementPattern>> MinimalClosureEnforcer::compute_minimal_closure(
    const MeshBase& mesh,
    const std::vector<MarkType>& marks,
    const NonConformity& non_conformity) const {
  (void)mesh;
  (void)marks;
  (void)non_conformity;
  return {};
}

double MinimalClosureEnforcer::compute_closure_cost(
    const std::vector<std::pair<size_t, RefinementPattern>>& closure) const {
  double cost = 0.0;
  for (const auto& [cell, pattern] : closure) {
    (void)cell;
    cost += config_.refinement_cost;
    if (pattern != RefinementPattern::RED && pattern != RefinementPattern::ISOTROPIC) {
      cost += config_.pattern_cost;
    }
  }
  return cost;
}

std::unique_ptr<ConformityEnforcer> ConformityEnforcerFactory::create(
    const AdaptivityOptions& options) {
  switch (options.conformity_mode) {
    case AdaptivityOptions::ConformityMode::ENFORCE_CONFORMING:
      return create_closure();
    case AdaptivityOptions::ConformityMode::ALLOW_HANGING_NODES:
      return create_hanging_node();
    case AdaptivityOptions::ConformityMode::MINIMAL_CLOSURE:
      return create_minimal_closure();
  }
  return create_closure();
}

std::unique_ptr<ConformityEnforcer> ConformityEnforcerFactory::create_closure(
    const ClosureConformityEnforcer::Config& config) {
  return std::make_unique<ClosureConformityEnforcer>(config);
}

std::unique_ptr<ConformityEnforcer> ConformityEnforcerFactory::create_hanging_node(
    const HangingNodeConformityEnforcer::Config& config) {
  return std::make_unique<HangingNodeConformityEnforcer>(config);
}

std::unique_ptr<ConformityEnforcer> ConformityEnforcerFactory::create_minimal_closure(
    const MinimalClosureEnforcer::Config& config) {
  return std::make_unique<MinimalClosureEnforcer>(config);
}

bool ConformityUtils::is_mesh_conforming(const MeshBase& mesh) {
  const size_t n_cells = mesh.n_cells();
  if (n_cells == 0) return true;

  // Build per-cell boundary entities.
  struct CellBoundary {
    CellFamily family;
    std::set<EdgeKey> edges;
    std::set<FaceKey> faces;
    std::vector<index_t> vertices;
  };

  std::vector<CellBoundary> bnd(n_cells);
  for (size_t c = 0; c < n_cells; ++c) {
    const index_t cell = static_cast<index_t>(c);
    bnd[c].family = mesh.cell_shape(cell).family;
    bnd[c].vertices = cell_vertices(mesh, cell);

    const auto conn = bnd[c].vertices;
    if (conn.empty()) continue;

    const auto ev = CellTopology::get_edges_view(bnd[c].family);
    for (int e = 0; e < ev.edge_count; ++e) {
      const index_t lv0 = ev.pairs_flat[2 * e + 0];
      const index_t lv1 = ev.pairs_flat[2 * e + 1];
      if (lv0 < 0 || lv1 < 0) continue;
      if (static_cast<size_t>(lv0) >= conn.size() || static_cast<size_t>(lv1) >= conn.size()) continue;
      bnd[c].edges.insert(make_edge_key(conn[static_cast<size_t>(lv0)], conn[static_cast<size_t>(lv1)]));
    }

    if (mesh.dim() == 3) {
      const auto fv = CellTopology::get_boundary_faces_canonical_view(bnd[c].family);
      if (fv.indices && fv.offsets && fv.face_count > 0) {
        for (int f = 0; f < fv.face_count; ++f) {
          const int start = fv.offsets[f];
          const int end = fv.offsets[f + 1];
          if (end <= start) continue;
          std::vector<index_t> face;
          face.reserve(static_cast<size_t>(end - start));
          for (int i = start; i < end; ++i) {
            const index_t lv = fv.indices[i];
            if (lv < 0 || static_cast<size_t>(lv) >= conn.size()) continue;
            face.push_back(conn[static_cast<size_t>(lv)]);
          }
          if (face.size() < 3) continue;
          bnd[c].faces.insert(make_face_key(face));
        }
      }
    }
  }

  // Vertex-to-cell adjacency for candidate neighbors.
  std::vector<std::vector<index_t>> v2c(mesh.n_vertices());
  for (size_t c = 0; c < n_cells; ++c) {
    for (index_t v : bnd[c].vertices) {
      if (v >= 0 && static_cast<size_t>(v) < v2c.size()) {
        v2c[static_cast<size_t>(v)].push_back(static_cast<index_t>(c));
      }
    }
  }

  std::set<std::pair<index_t, index_t>> candidate_pairs;
  for (const auto& cells : v2c) {
    for (size_t i = 0; i < cells.size(); ++i) {
      for (size_t j = i + 1; j < cells.size(); ++j) {
        const index_t a = std::min(cells[i], cells[j]);
        const index_t b = std::max(cells[i], cells[j]);
        candidate_pairs.insert({a, b});
      }
    }
  }

  for (const auto& [ca, cb] : candidate_pairs) {
    const auto& va = bnd[static_cast<size_t>(ca)].vertices;
    const auto& vb = bnd[static_cast<size_t>(cb)].vertices;
    if (va.empty() || vb.empty()) continue;

    std::vector<index_t> common;
    for (index_t a : va) {
      for (index_t b : vb) {
        if (a == b) {
          common.push_back(a);
          break;
        }
      }
    }

    if (mesh.dim() == 2) {
      if (common.size() < 2) continue;

      bool has_common_edge = false;
      for (const auto& e : bnd[static_cast<size_t>(ca)].edges) {
        if (bnd[static_cast<size_t>(cb)].edges.count(e) > 0) {
          has_common_edge = true;
          break;
        }
      }

      if (!has_common_edge) {
        return false;
      }
    } else if (mesh.dim() == 3) {
      if (common.size() < 3) continue;

      std::vector<index_t> unique_common = common;
      std::sort(unique_common.begin(), unique_common.end());
      unique_common.erase(std::unique(unique_common.begin(), unique_common.end()), unique_common.end());
      if (unique_common.size() < 3) continue;

      const auto& faces_a = bnd[static_cast<size_t>(ca)].faces;
      const auto& faces_b = bnd[static_cast<size_t>(cb)].faces;
      bool shares_face = false;
      if (faces_a.size() <= faces_b.size()) {
        for (const auto& f : faces_a) {
          if (faces_b.count(f) > 0) {
            shares_face = true;
            break;
          }
        }
      } else {
        for (const auto& f : faces_b) {
          if (faces_a.count(f) > 0) {
            shares_face = true;
            break;
          }
        }
      }

      if (!shares_face) {
        return false;
      }
    }
  }

  return true;
}

std::vector<HangingNode> ConformityUtils::find_hanging_nodes(const MeshBase& mesh) {
  if (is_mesh_conforming(mesh)) {
    return {};
  }

  // Best-effort: return symbolic hanging nodes by detecting pairs of cells that share vertices but not an edge/face.
  std::vector<HangingNode> nodes;
  const auto edges = build_edge_incidence(mesh);
  std::set<EdgeKey> edge_set;
  for (const auto& [k, inc] : edges) {
    (void)inc;
    edge_set.insert(k);
  }

  // A very lightweight heuristic for 2D hanging nodes: find an edge (a,b) that exists,
  // and a vertex m that lies on segment (a,b) such that edges (a,m) and (m,b) also exist.
  if (mesh.dim() == 2) {
    const size_t n_vertices = mesh.n_vertices();
    for (const auto& [ab, inc] : edges) {
      (void)inc;
      const index_t a = ab.first;
      const index_t b = ab.second;
      const auto pa = mesh.get_vertex_coords(a);
      const auto pb = mesh.get_vertex_coords(b);
      for (index_t m = 0; m < static_cast<index_t>(n_vertices); ++m) {
        if (m == a || m == b) continue;
        if (edge_set.count(make_edge_key(a, m)) == 0 || edge_set.count(make_edge_key(m, b)) == 0) continue;
        const auto pm = mesh.get_vertex_coords(m);
        const std::array<double, 3> A = {static_cast<double>(pa[0]), static_cast<double>(pa[1]), static_cast<double>(pa[2])};
        const std::array<double, 3> B = {static_cast<double>(pb[0]), static_cast<double>(pb[1]), static_cast<double>(pb[2])};
        const std::array<double, 3> M = {static_cast<double>(pm[0]), static_cast<double>(pm[1]), static_cast<double>(pm[2])};
        const std::array<double, 3> AB = {B[0] - A[0], B[1] - A[1], B[2] - A[2]};
        const std::array<double, 3> AM = {M[0] - A[0], M[1] - A[1], M[2] - A[2]};
        const double cross = AB[0] * AM[1] - AB[1] * AM[0];
        if (std::abs(cross) > 1e-12) continue;
        HangingNode hn;
        hn.node_id = static_cast<size_t>(m);
        hn.parent_entity = {static_cast<size_t>(a), static_cast<size_t>(b)};
        hn.on_edge = true;
        hn.level_difference = 1;
        hn.constraints[static_cast<size_t>(a)] = 0.5;
        hn.constraints[static_cast<size_t>(b)] = 0.5;
        nodes.push_back(std::move(hn));
        break;
      }
    }
  }
  return nodes;
}

size_t ConformityUtils::check_level_difference(
    const MeshBase& mesh,
    size_t elem1,
    size_t elem2) {
  if (elem1 >= mesh.n_cells() || elem2 >= mesh.n_cells()) {
    return 0;
  }
  const size_t l1 = mesh.refinement_level(static_cast<index_t>(elem1));
  const size_t l2 = mesh.refinement_level(static_cast<index_t>(elem2));
  return (l1 > l2) ? (l1 - l2) : (l2 - l1);
}

void ConformityUtils::apply_constraints(
    std::vector<double>& solution,
    const std::map<size_t, std::map<size_t, double>>& constraints) {
  if (constraints.empty()) return;
  const std::vector<double> base = solution;
  for (const auto& [slave, eq] : constraints) {
    if (slave >= solution.size()) continue;
    double val = 0.0;
    for (const auto& [master, w] : eq) {
      if (master >= base.size()) continue;
      val += w * base[master];
    }
    solution[slave] = val;
  }
}

void ConformityUtils::eliminate_constraints(
    std::vector<std::vector<double>>& matrix,
    std::vector<double>& rhs,
    const std::map<size_t, std::map<size_t, double>>& constraints) {
  if (constraints.empty()) return;
  const size_t n = matrix.size();
  if (rhs.size() != n) return;
  for (const auto& [slave, eq] : constraints) {
    if (slave >= n) continue;

    // Substitute x_slave = sum w_j x_j into all other equations.
    for (size_t r = 0; r < n; ++r) {
      if (r == slave) continue;
      const double a_rs = matrix[r][slave];
      if (a_rs == 0.0) continue;
      for (const auto& [master, w] : eq) {
        if (master >= n) continue;
        matrix[r][master] += a_rs * w;
      }
      matrix[r][slave] = 0.0;
    }

    // Replace the slave row with the constraint equation: x_slave - sum w_j x_j = 0.
    std::fill(matrix[slave].begin(), matrix[slave].end(), 0.0);
    matrix[slave][slave] = 1.0;
    for (const auto& [master, w] : eq) {
      if (master >= n) continue;
      matrix[slave][master] = -w;
    }
    rhs[slave] = 0.0;
  }
}

void ConformityUtils::write_nonconformity_to_field(
    MeshFields& fields,
    const MeshBase& mesh,
    const NonConformity& non_conformity) {
  (void)fields;
  auto& writable_mesh = const_cast<MeshBase&>(mesh);

  const FieldHandle hanging_h =
      MeshFields::attach_field(writable_mesh, EntityKind::Vertex, "conformity_hanging_level",
                               FieldScalarType::Int32, 1);
  const FieldHandle closure_h =
      MeshFields::attach_field(writable_mesh, EntityKind::Volume, "conformity_needs_closure",
                               FieldScalarType::UInt8, 1);
  const FieldHandle maxdiff_h =
      MeshFields::attach_field(writable_mesh, EntityKind::Volume, "conformity_max_level_difference",
                               FieldScalarType::Int32, 1);

  auto* hanging = static_cast<int32_t*>(MeshFields::field_data(writable_mesh, hanging_h));
  auto* closure = static_cast<std::uint8_t*>(MeshFields::field_data(writable_mesh, closure_h));
  auto* maxdiff = static_cast<int32_t*>(MeshFields::field_data(writable_mesh, maxdiff_h));

  if (hanging) {
    std::fill(hanging, hanging + writable_mesh.n_vertices(), 0);
    for (const auto& hn : non_conformity.hanging_nodes) {
      if (hn.node_id < writable_mesh.n_vertices()) {
        hanging[hn.node_id] = static_cast<int32_t>(hn.level_difference);
      }
    }
  }

  if (closure) {
    std::fill(closure, closure + writable_mesh.n_cells(), static_cast<std::uint8_t>(0));
    for (size_t c : non_conformity.cells_needing_closure) {
      if (c < writable_mesh.n_cells()) {
        closure[c] = static_cast<std::uint8_t>(1);
      }
    }
  }

  if (maxdiff) {
    for (size_t c = 0; c < writable_mesh.n_cells(); ++c) {
      maxdiff[c] = static_cast<int32_t>(non_conformity.max_level_difference);
    }
  }
}

std::map<size_t, std::map<size_t, double>> ConformityUtils::build_hanging_vertex_constraints(
    const MeshBase& mesh) {
  (void)mesh;
  return {};
}

std::map<gid_t, std::map<gid_t, double>> ConformityUtils::build_hanging_vertex_constraints_gid(
    const MeshBase& mesh) {
  (void)mesh;
  return {};
}

} // namespace svmp
