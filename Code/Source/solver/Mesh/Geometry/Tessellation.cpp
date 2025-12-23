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

#include "Tessellation.h"

#include "../Core/MeshBase.h"
#include "../Topology/CellTopology.h"
#include "../Topology/FaceEmbedding.h"
#include "CurvilinearEval.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace svmp {

namespace {

inline int pow2(int level) {
    if (level <= 0) return 1;
    return 1 << level;
}

inline TessParamPoint corner_param(CellFamily family, int corner_id) {
    switch (family) {
        case CellFamily::Line:
            return (corner_id == 0) ? TessParamPoint{-1, 0, 0} : TessParamPoint{1, 0, 0};
        case CellFamily::Triangle: {
            switch (corner_id) {
                case 0: return {0, 0, 0};
                case 1: return {1, 0, 0};
                case 2: return {0, 1, 0};
                default: return {0, 0, 0};
            }
        }
        case CellFamily::Quad: {
            switch (corner_id) {
                case 0: return {-1, -1, 0};
                case 1: return {1, -1, 0};
                case 2: return {1, 1, 0};
                case 3: return {-1, 1, 0};
                default: return {0, 0, 0};
            }
        }
        case CellFamily::Tetra: {
            switch (corner_id) {
                case 0: return {0, 0, 0};
                case 1: return {1, 0, 0};
                case 2: return {0, 1, 0};
                case 3: return {0, 0, 1};
                default: return {0, 0, 0};
            }
        }
        case CellFamily::Hex: {
            switch (corner_id) {
                case 0: return {-1, -1, -1};
                case 1: return {1, -1, -1};
                case 2: return {1, 1, -1};
                case 3: return {-1, 1, -1};
                case 4: return {-1, -1, 1};
                case 5: return {1, -1, 1};
                case 6: return {1, 1, 1};
                case 7: return {-1, 1, 1};
                default: return {0, 0, 0};
            }
        }
        case CellFamily::Wedge: {
            switch (corner_id) {
                case 0: return {0, 0, -1};
                case 1: return {1, 0, -1};
                case 2: return {0, 1, -1};
                case 3: return {0, 0, 1};
                case 4: return {1, 0, 1};
                case 5: return {0, 1, 1};
                default: return {0, 0, 0};
            }
        }
        case CellFamily::Pyramid: {
            switch (corner_id) {
                case 0: return {-1, -1, 0};
                case 1: return {1, -1, 0};
                case 2: return {1, 1, 0};
                case 3: return {-1, 1, 0};
                case 4: return {0, 0, 1};
                default: return {0, 0, 0};
            }
        }
        default:
            return {0, 0, 0};
    }
}

inline TessParamPoint lerp(const TessParamPoint& a, const TessParamPoint& b, real_t t) {
    return {(1 - t) * a[0] + t * b[0], (1 - t) * a[1] + t * b[1], (1 - t) * a[2] + t * b[2]};
}

inline std::array<real_t,3> sub3(const std::array<real_t,3>& a, const std::array<real_t,3>& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline std::array<real_t,3> add3(const std::array<real_t,3>& a, const std::array<real_t,3>& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline std::array<real_t,3> scale3(const std::array<real_t,3>& a, real_t s) {
    return {a[0] * s, a[1] * s, a[2] * s};
}

inline real_t norm3(const std::array<real_t,3>& a) {
    return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

inline const std::vector<real_t>& coords_for(const MeshBase& mesh, Configuration cfg) {
    return ((cfg == Configuration::Current || cfg == Configuration::Deformed) && mesh.has_current_coords())
               ? mesh.X_cur()
               : mesh.X_ref();
}

inline std::array<real_t,3> map_face_point(
    const MeshBase& mesh,
    index_t face,
    const TessParamPoint& xi,
    Configuration cfg) {

    const CellShape& shape = mesh.face_shapes().at(static_cast<size_t>(face));
    auto [verts, n_nodes] = mesh.face_vertices_span(face);
    const auto sf = CurvilinearEvaluator::evaluate_shape_functions(shape, n_nodes, xi);

    const auto& X = coords_for(mesh, cfg);
    const int dim = mesh.dim();

    std::array<real_t,3> x{0.0, 0.0, 0.0};
    for (size_t i = 0; i < n_nodes; ++i) {
        const index_t v = verts[i];
        if (dim >= 1) x[0] += sf.N[i] * X[static_cast<size_t>(v * dim + 0)];
        if (dim >= 2) x[1] += sf.N[i] * X[static_cast<size_t>(v * dim + 1)];
        if (dim >= 3) x[2] += sf.N[i] * X[static_cast<size_t>(v * dim + 2)];
    }
    return x;
}

template <typename MapFn>
real_t estimate_edge_chord_error(CellFamily family, int level, const MapFn& map) {
    const int n_div = pow2(level);
    if (n_div <= 0) return 0.0;

    std::vector<std::pair<int,int>> edges;
    if (family == CellFamily::Line) {
        edges.push_back({0, 1});
    } else {
        const auto eview = CellTopology::get_edges_view(family);
        edges.reserve(static_cast<size_t>(eview.edge_count));
        for (int ei = 0; ei < eview.edge_count; ++ei) {
            edges.push_back({eview.pairs_flat[2 * ei + 0], eview.pairs_flat[2 * ei + 1]});
        }
    }

    const real_t eps = static_cast<real_t>(1e-12);
    real_t max_err = 0.0;

    for (const auto& [a, b] : edges) {
        const auto A = corner_param(family, a);
        const auto B = corner_param(family, b);

        std::vector<std::array<real_t,3>> x_end(static_cast<size_t>(n_div + 1));
        for (int i = 0; i <= n_div; ++i) {
            const real_t t = static_cast<real_t>(i) / static_cast<real_t>(n_div);
            x_end[static_cast<size_t>(i)] = map(lerp(A, B, t));
        }

        for (int i = 0; i < n_div; ++i) {
            const real_t tmid = (static_cast<real_t>(i) + static_cast<real_t>(0.5)) / static_cast<real_t>(n_div);
            const auto xm = map(lerp(A, B, tmid));

            const auto x0 = x_end[static_cast<size_t>(i)];
            const auto x1 = x_end[static_cast<size_t>(i + 1)];

            const auto xlin = scale3(add3(x0, x1), static_cast<real_t>(0.5));
            const real_t denom = std::max(norm3(sub3(x1, x0)), eps);
            const real_t err = norm3(sub3(xm, xlin)) / denom;

            max_err = std::max(max_err, err);
        }
    }

    return max_err;
}

template <typename MapFn>
real_t segment_chord_error(const MapFn& map, const TessParamPoint& A, const TessParamPoint& B) {
    const real_t eps = static_cast<real_t>(1e-12);
    const auto x0 = map(A);
    const auto x1 = map(B);
    const auto xm = map(lerp(A, B, static_cast<real_t>(0.5)));
    const auto xlin = scale3(add3(x0, x1), static_cast<real_t>(0.5));
    const real_t denom = std::max(norm3(sub3(x1, x0)), eps);
    return norm3(sub3(xm, xlin)) / denom;
}

int adaptive_refinement_level_for_cell(const MeshBase& mesh, index_t cell, const TessellationConfig& cfg) {
    int level = std::max(0, cfg.refinement_level);
    if (!(cfg.adaptive || cfg.local_adaptive) || cfg.curvature_threshold <= 0) {
        return level;
    }

    const CellShape shape = mesh.cell_shape(cell);
    switch (shape.family) {
        case CellFamily::Point:
        case CellFamily::Line:
        case CellFamily::Triangle:
        case CellFamily::Quad:
        case CellFamily::Tetra:
        case CellFamily::Hex:
        case CellFamily::Wedge:
        case CellFamily::Pyramid:
            break;
        default:
            return level;
    }
    auto [_, n_nodes] = mesh.cell_vertices_span(cell);
    const int order = CurvilinearEvaluator::deduce_order(shape, n_nodes);
    level = std::max(level, std::max(0, cfg.min_refinement_level));
    level = std::max(level, Tessellator::suggest_refinement_level(order));

    const int max_level = std::max(level, std::max(0, cfg.max_refinement_level));
    auto map = [&](const TessParamPoint& xi) -> std::array<real_t,3> {
        return CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg.configuration).coordinates;
    };

    for (; level < max_level; ++level) {
        const real_t err = estimate_edge_chord_error(shape.family, level, map);
        if (err <= cfg.curvature_threshold) break;
    }
    return level;
}

int adaptive_refinement_level_for_face(const MeshBase& mesh, index_t face, const TessellationConfig& cfg) {
    int level = std::max(0, cfg.refinement_level);
    if (!(cfg.adaptive || cfg.local_adaptive) || cfg.curvature_threshold <= 0) {
        return level;
    }

    const CellShape shape = mesh.face_shapes().at(static_cast<size_t>(face));
    switch (shape.family) {
        case CellFamily::Line:
        case CellFamily::Triangle:
        case CellFamily::Quad:
            break;
        default:
            return level;
    }
    auto [_, n_nodes] = mesh.face_vertices_span(face);
    const int order = CurvilinearEvaluator::deduce_order(shape, n_nodes);
    level = std::max(level, std::max(0, cfg.min_refinement_level));
    level = std::max(level, Tessellator::suggest_refinement_level(order));

    const int max_level = std::max(level, std::max(0, cfg.max_refinement_level));
    auto map = [&](const TessParamPoint& xi) -> std::array<real_t,3> {
        return map_face_point(mesh, face, xi, cfg.configuration);
    };

    for (; level < max_level; ++level) {
        const real_t err = estimate_edge_chord_error(shape.family, level, map);
        if (err <= cfg.curvature_threshold) break;
    }
    return level;
}

index_t boundary_incident_cell(const MeshBase& mesh, index_t face) {
    const auto& fc = mesh.face_cells(face);
    if (fc[0] != INVALID_INDEX) return fc[0];
    if (fc[1] != INVALID_INDEX) return fc[1];
    return INVALID_INDEX;
}

int find_local_face_id_in_cell(const MeshBase& mesh, index_t cell, index_t face) {
    const CellShape cshape = mesh.cell_shape(cell);
    const auto view = CellTopology::get_oriented_boundary_faces_view(cshape.family);

    const auto& fshape = mesh.face_shapes().at(static_cast<size_t>(face));
    const int fc = fshape.num_corners;
    if (fc < 2) return -1;

    auto [cell_verts, _n_cell_nodes] = mesh.cell_vertices_span(cell);
    auto [face_verts, n_face_nodes] = mesh.face_vertices_span(face);
    if (!cell_verts || !face_verts || n_face_nodes < static_cast<size_t>(fc)) return -1;

    std::vector<index_t> fcorner(face_verts, face_verts + fc);
    std::sort(fcorner.begin(), fcorner.end());

    for (int fi = 0; fi < view.face_count; ++fi) {
        const int b = view.offsets[fi];
        const int e = view.offsets[fi + 1];
        const int fv = e - b;
        if (fv != fc) continue;
        std::vector<index_t> ccorner;
        ccorner.reserve(static_cast<size_t>(fv));
        for (int j = b; j < e; ++j) {
            const int loc = view.indices[j];
            ccorner.push_back(cell_verts[static_cast<size_t>(loc)]);
        }
        std::sort(ccorner.begin(), ccorner.end());
        if (ccorner == fcorner) return fi;
    }
    return -1;
}

std::vector<int> face_corner_perm_to_cell(const std::vector<index_t>& face_corners,
                                          const std::vector<index_t>& cell_face_corners) {
    if (face_corners.size() != cell_face_corners.size()) {
        throw std::runtime_error("Tessellation: face corner size mismatch");
    }
    std::vector<int> perm(face_corners.size(), -1);
    for (size_t k = 0; k < face_corners.size(); ++k) {
        const index_t v = face_corners[k];
        int found = -1;
        for (size_t j = 0; j < cell_face_corners.size(); ++j) {
            if (cell_face_corners[j] == v) { found = static_cast<int>(j); break; }
        }
        if (found < 0) {
            throw std::runtime_error("Tessellation: failed to match face corner to cell face corner");
        }
        perm[k] = found;
    }
    return perm;
}

TessParamPoint map_face_param_to_cell_face_param(CellFamily face_family,
                                                 const TessParamPoint& xi_face,
                                                 const std::vector<int>& perm) {
    if (face_family == CellFamily::Triangle) {
        const real_t r = xi_face[0];
        const real_t s = xi_face[1];
        std::array<real_t, 3> w_face = {1.0 - r - s, r, s};
        std::array<real_t, 3> w_cell = {0.0, 0.0, 0.0};
        for (int k = 0; k < 3; ++k) w_cell[static_cast<size_t>(perm[static_cast<size_t>(k)])] = w_face[static_cast<size_t>(k)];
        return {w_cell[1], w_cell[2], 0.0};
    }
    if (face_family == CellFamily::Quad) {
        const real_t u = xi_face[0];
        const real_t v = xi_face[1];
        std::array<real_t, 4> w_face = {
            static_cast<real_t>(0.25) * (1 - u) * (1 - v),
            static_cast<real_t>(0.25) * (1 + u) * (1 - v),
            static_cast<real_t>(0.25) * (1 + u) * (1 + v),
            static_cast<real_t>(0.25) * (1 - u) * (1 + v),
        };
        std::array<real_t, 4> w_cell = {0.0, 0.0, 0.0, 0.0};
        for (int k = 0; k < 4; ++k) w_cell[static_cast<size_t>(perm[static_cast<size_t>(k)])] = w_face[static_cast<size_t>(k)];
        const real_t u_cell = (w_cell[1] + w_cell[2]) - (w_cell[0] + w_cell[3]);
        const real_t v_cell = (w_cell[2] + w_cell[3]) - (w_cell[0] + w_cell[1]);
        return {u_cell, v_cell, 0.0};
    }
    if (face_family == CellFamily::Line) {
        const real_t x = xi_face[0];
        std::array<real_t, 2> w_face = {static_cast<real_t>(0.5) * (1 - x), static_cast<real_t>(0.5) * (1 + x)};
        std::array<real_t, 2> w_cell = {0.0, 0.0};
        for (int k = 0; k < 2; ++k) w_cell[static_cast<size_t>(perm[static_cast<size_t>(k)])] = w_face[static_cast<size_t>(k)];
        const real_t x_cell = w_cell[1] - w_cell[0];
        return {x_cell, 0.0, 0.0};
    }
    return xi_face;
}

int adaptive_refinement_level_for_boundary_face_from_cell(const MeshBase& mesh,
                                                          index_t face,
                                                          index_t cell,
                                                          int local_face_id,
                                                          const std::vector<int>& perm,
                                                          const TessellationConfig& cfg) {
    int level = std::max(0, cfg.refinement_level);
    if (!(cfg.adaptive || cfg.local_adaptive) || cfg.curvature_threshold <= 0) {
        return level;
    }

    const auto fshape = mesh.face_shapes().at(static_cast<size_t>(face));
    if (fshape.family != CellFamily::Line && fshape.family != CellFamily::Triangle && fshape.family != CellFamily::Quad) {
        return level;
    }

    const auto cshape = mesh.cell_shape(cell);
    auto [_, n_nodes] = mesh.cell_vertices_span(cell);
    const int order = CurvilinearEvaluator::deduce_order(cshape, n_nodes);
    level = std::max(level, std::max(0, cfg.min_refinement_level));
    level = std::max(level, Tessellator::suggest_refinement_level(order));

    const int max_level = std::max(level, std::max(0, cfg.max_refinement_level));
    auto map = [&](const TessParamPoint& xi_face) -> std::array<real_t, 3> {
        const auto xi_cf = map_face_param_to_cell_face_param(fshape.family, xi_face, perm);
        const auto xi_cell = FaceEmbedding::embed_face_point(cshape.family, local_face_id, xi_cf);
        return CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi_cell, cfg.configuration).coordinates;
    };

    for (; level < max_level; ++level) {
        const real_t err = estimate_edge_chord_error(fshape.family, level, map);
        if (err <= cfg.curvature_threshold) break;
    }
    return level;
}

inline CellShape linear_shape(CellFamily family) {
    CellShape s;
    s.family = family;
    s.order = 1;
    switch (family) {
        case CellFamily::Line:    s.num_corners = 2; break;
        case CellFamily::Triangle:s.num_corners = 3; break;
        case CellFamily::Quad:    s.num_corners = 4; break;
        case CellFamily::Tetra:   s.num_corners = 4; break;
        case CellFamily::Hex:     s.num_corners = 8; break;
        case CellFamily::Wedge:   s.num_corners = 6; break;
        case CellFamily::Pyramid: s.num_corners = 5; break;
        case CellFamily::Point:   s.num_corners = 1; break;
        default:                  s.num_corners = 0; break;
    }
    return s;
}

} // namespace

//=============================================================================
// Public API
//=============================================================================

TessellatedCell Tessellator::tessellate_cell(
    const MeshBase& mesh,
    index_t cell,
    const TessellationConfig& config) {

    TessellatedCell out;
    out.cell_id = cell;
    out.cell_shape = mesh.cell_shape(cell);

    SubdivisionGrid grid;
    switch (out.cell_shape.family) {
        case CellFamily::Point:
            out.sub_element_shape = linear_shape(CellFamily::Point);
            out.offsets = {0};
            return out;
        case CellFamily::Line:
            if (config.local_adaptive) {
                auto [_, n_nodes] = mesh.cell_vertices_span(cell);
                const int order = CurvilinearEvaluator::deduce_order(out.cell_shape, n_nodes);
                const int base_level = std::max({0, config.refinement_level, config.min_refinement_level,
                                                 Tessellator::suggest_refinement_level(order)});
                const int max_level = std::max(base_level, std::max(0, config.max_refinement_level));
                std::function<std::array<real_t, 3>(const TessParamPoint&)> map =
                    [&](const TessParamPoint& xi) -> std::array<real_t, 3> {
                    return CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, config.configuration).coordinates;
                };
                grid = subdivide_line_local_adaptive(base_level, max_level, config.curvature_threshold, map);
            } else {
                const int level = adaptive_refinement_level_for_cell(mesh, cell, config);
                grid = subdivide_line(level);
            }
            out.sub_element_shape = linear_shape(CellFamily::Line);
            break;
        case CellFamily::Triangle:
        {
            const int level = adaptive_refinement_level_for_cell(mesh, cell, config);
            grid = subdivide_triangle(level);
            out.sub_element_shape = linear_shape(CellFamily::Triangle);
            break;
        }
        case CellFamily::Quad:
            if (config.local_adaptive) {
                auto [_, n_nodes] = mesh.cell_vertices_span(cell);
                const int order = CurvilinearEvaluator::deduce_order(out.cell_shape, n_nodes);
                const int base_level = std::max({0, config.refinement_level, config.min_refinement_level,
                                                 Tessellator::suggest_refinement_level(order)});
                const int max_level = std::max(base_level, std::max(0, config.max_refinement_level));
                std::function<std::array<real_t, 3>(const TessParamPoint&)> map =
                    [&](const TessParamPoint& xi) -> std::array<real_t, 3> {
                    return CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, config.configuration).coordinates;
                };
                grid = subdivide_quad_local_adaptive(base_level, max_level, config.curvature_threshold, map);
            } else {
                const int level = adaptive_refinement_level_for_cell(mesh, cell, config);
                grid = subdivide_quad(level);
            }
            out.sub_element_shape = linear_shape(CellFamily::Quad);
            break;
        case CellFamily::Tetra:
        {
            const int level = adaptive_refinement_level_for_cell(mesh, cell, config);
            grid = subdivide_tet(level);
            out.sub_element_shape = linear_shape(CellFamily::Tetra);
            break;
        }
        case CellFamily::Hex:
        {
            const int level = adaptive_refinement_level_for_cell(mesh, cell, config);
            grid = subdivide_hex(level);
            out.sub_element_shape = linear_shape(CellFamily::Hex);
            break;
        }
        case CellFamily::Wedge:
        {
            const int level = adaptive_refinement_level_for_cell(mesh, cell, config);
            grid = subdivide_wedge(level);
            out.sub_element_shape = linear_shape(CellFamily::Wedge);
            break;
        }
        case CellFamily::Pyramid:
        {
            const int level = adaptive_refinement_level_for_cell(mesh, cell, config);
            grid = subdivide_pyramid(level);
            // Pyramid tessellation outputs tetrahedra (robust, widely supported).
            out.sub_element_shape = linear_shape(CellFamily::Tetra);
            break;
        }
        default:
            throw std::runtime_error("Tessellator::tessellate_cell: unsupported cell family");
    }

    map_subdivision_to_physical(mesh, cell, grid, config.configuration, out);
    out.field_values.clear();
    if (config.field_evaluator) {
        out.field_values.resize(grid.points.size());
        for (size_t i = 0; i < grid.points.size(); ++i) {
            config.field_evaluator(cell, grid.points[i], out.field_values[i]);
        }
    }
    out.connectivity = std::move(grid.connectivity);
    out.offsets = std::move(grid.offsets);
    return out;
}

TessellatedFace Tessellator::tessellate_face(
    const MeshBase& mesh,
    index_t face,
    const TessellationConfig& config) {

    TessellatedFace out;
    out.face_id = face;
    out.face_shape = mesh.face_shapes().at(static_cast<size_t>(face));

    auto [verts, n_nodes] = mesh.face_vertices_span(face);
    if (n_nodes == 0) return out;

    // Polygon faces: keep a simple triangle-fan fallback for now.
    if (out.face_shape.family == CellFamily::Polygon) {
        const auto& X = coords_for(mesh, config.configuration);
        const int dim = mesh.dim();

        out.vertices.reserve(n_nodes);
        for (size_t i = 0; i < n_nodes; ++i) {
            const index_t v = verts[i];
            std::array<real_t,3> p{0,0,0};
            for (int d = 0; d < dim; ++d) p[static_cast<size_t>(d)] = X[static_cast<size_t>(v * dim + d)];
            out.vertices.push_back(p);
        }

        out.sub_element_shape = linear_shape(CellFamily::Triangle);
        out.offsets.push_back(0);
        for (size_t i = 1; i + 1 < n_nodes; ++i) {
            out.connectivity.push_back(0);
            out.connectivity.push_back(static_cast<index_t>(i));
            out.connectivity.push_back(static_cast<index_t>(i + 1));
            out.offsets.push_back(static_cast<int>(out.connectivity.size()));
        }
        return out;
    }

    SubdivisionGrid grid;
    switch (out.face_shape.family) {
        case CellFamily::Line:
            if (config.local_adaptive) {
                const int order = CurvilinearEvaluator::deduce_order(out.face_shape, n_nodes);
                const int base_level = std::max({0, config.refinement_level, config.min_refinement_level,
                                                 Tessellator::suggest_refinement_level(order)});
                const int max_level = std::max(base_level, std::max(0, config.max_refinement_level));
                std::function<std::array<real_t, 3>(const TessParamPoint&)> map =
                    [&](const TessParamPoint& xi) -> std::array<real_t, 3> {
                    return map_face_point(mesh, face, xi, config.configuration);
                };
                grid = subdivide_line_local_adaptive(base_level, max_level, config.curvature_threshold, map);
            } else {
                const int level = adaptive_refinement_level_for_face(mesh, face, config);
                grid = subdivide_line(level);
            }
            out.sub_element_shape = linear_shape(CellFamily::Line);
            break;
        case CellFamily::Triangle:
        {
            const int level = adaptive_refinement_level_for_face(mesh, face, config);
            grid = subdivide_triangle(level);
            out.sub_element_shape = linear_shape(CellFamily::Triangle);
            break;
        }
        case CellFamily::Quad:
            if (config.local_adaptive) {
                const int order = CurvilinearEvaluator::deduce_order(out.face_shape, n_nodes);
                const int base_level = std::max({0, config.refinement_level, config.min_refinement_level,
                                                 Tessellator::suggest_refinement_level(order)});
                const int max_level = std::max(base_level, std::max(0, config.max_refinement_level));
                std::function<std::array<real_t, 3>(const TessParamPoint&)> map =
                    [&](const TessParamPoint& xi) -> std::array<real_t, 3> {
                    return map_face_point(mesh, face, xi, config.configuration);
                };
                grid = subdivide_quad_local_adaptive(base_level, max_level, config.curvature_threshold, map);
            } else {
                const int level = adaptive_refinement_level_for_face(mesh, face, config);
                grid = subdivide_quad(level);
            }
            out.sub_element_shape = linear_shape(CellFamily::Quad);
            break;
        default:
            throw std::runtime_error("Tessellator::tessellate_face: unsupported face family");
    }

    map_face_subdivision_to_physical(mesh, face, grid, config.configuration, out);
    out.field_values.clear();
    if (config.field_evaluator) {
        const index_t cell = boundary_incident_cell(mesh, face);
        if (cell != INVALID_INDEX) {
            const CellShape cshape = mesh.cell_shape(cell);
            int local_face_id = -1;
            try {
                local_face_id = find_local_face_id_in_cell(mesh, cell, face);
            } catch (...) {
                local_face_id = -1;
            }
            if (local_face_id >= 0) {
                auto [cell_verts, _n] = mesh.cell_vertices_span(cell);
                const int fc = out.face_shape.num_corners;
                if (cell_verts && verts && fc >= 2 && n_nodes >= static_cast<size_t>(fc)) {
                    try {
                        const auto view = CellTopology::get_oriented_boundary_faces_view(cshape.family);
                        const int b = view.offsets[local_face_id];
                        const int e = view.offsets[local_face_id + 1];
                        const int fv = e - b;
                        if (fv == fc) {
                            std::vector<index_t> face_corners(verts, verts + fc);
                            std::vector<index_t> cell_face_corners;
                            cell_face_corners.reserve(static_cast<size_t>(fv));
                            for (int j = b; j < e; ++j) {
                                cell_face_corners.push_back(cell_verts[static_cast<size_t>(view.indices[j])]);
                            }
                            const auto perm = face_corner_perm_to_cell(face_corners, cell_face_corners);
                            out.field_values.resize(grid.points.size());
                            for (size_t i = 0; i < grid.points.size(); ++i) {
                                const auto xi_cf = map_face_param_to_cell_face_param(out.face_shape.family, grid.points[i], perm);
                                const auto xi_cell = FaceEmbedding::embed_face_point(cshape.family, local_face_id, xi_cf);
                                config.field_evaluator(cell, xi_cell, out.field_values[i]);
                            }
                        }
                    } catch (...) {
                        out.field_values.clear();
                    }
                }
            }
        }
    }
    out.connectivity = std::move(grid.connectivity);
    out.offsets = std::move(grid.offsets);
    return out;
}

std::vector<TessellatedCell> Tessellator::tessellate_mesh(
    const MeshBase& mesh,
    const TessellationConfig& config) {

    std::vector<TessellatedCell> out;
    out.reserve(mesh.n_cells());
    for (index_t c = 0; c < static_cast<index_t>(mesh.n_cells()); ++c) {
        out.push_back(tessellate_cell(mesh, c, config));
    }
    return out;
}

std::vector<TessellatedFace> Tessellator::tessellate_boundary(
    const MeshBase& mesh,
    const TessellationConfig& config) {

    std::vector<TessellatedFace> out;
    auto faces = mesh.boundary_faces();
    out.reserve(faces.size());
    for (index_t f : faces) {
        const auto fshape = mesh.face_shapes().at(static_cast<size_t>(f));
        if (fshape.family != CellFamily::Line && fshape.family != CellFamily::Triangle && fshape.family != CellFamily::Quad) {
            out.push_back(tessellate_face(mesh, f, config));
            continue;
        }

        const index_t cell = boundary_incident_cell(mesh, f);
        if (cell == INVALID_INDEX) {
            out.push_back(tessellate_face(mesh, f, config));
            continue;
        }

        const CellShape cshape = mesh.cell_shape(cell);
        auto [_, n_cell_nodes] = mesh.cell_vertices_span(cell);
        const int cell_order = CurvilinearEvaluator::deduce_order(cshape, n_cell_nodes);
        if (cell_order <= 1) {
            out.push_back(tessellate_face(mesh, f, config));
            continue;
        }

        const int local_face_id = find_local_face_id_in_cell(mesh, cell, f);
        if (local_face_id < 0) {
            out.push_back(tessellate_face(mesh, f, config));
            continue;
        }

        // Corner ordering map: face corners -> cell oriented face corners.
        auto [cell_verts, _n] = mesh.cell_vertices_span(cell);
        auto [face_verts, n_face_nodes] = mesh.face_vertices_span(f);
        const int fc = fshape.num_corners;
        if (!cell_verts || !face_verts || n_face_nodes < static_cast<size_t>(fc)) {
            out.push_back(tessellate_face(mesh, f, config));
            continue;
        }

        const auto view = CellTopology::get_oriented_boundary_faces_view(cshape.family);
        const int b = view.offsets[local_face_id];
        const int e = view.offsets[local_face_id + 1];
        const int fv = e - b;
        if (fv != fc) {
            out.push_back(tessellate_face(mesh, f, config));
            continue;
        }

        std::vector<index_t> face_corners(face_verts, face_verts + fc);
        std::vector<index_t> cell_face_corners;
        cell_face_corners.reserve(static_cast<size_t>(fv));
        for (int j = b; j < e; ++j) {
            cell_face_corners.push_back(cell_verts[static_cast<size_t>(view.indices[j])]);
        }

        const auto perm = face_corner_perm_to_cell(face_corners, cell_face_corners);

        SubdivisionGrid grid;
        TessellatedFace tess;
        tess.face_id = f;
        tess.face_shape = fshape;
        switch (fshape.family) {
            case CellFamily::Line:
            case CellFamily::Triangle:
            case CellFamily::Quad:
                break;
            default:
                out.push_back(tessellate_face(mesh, f, config));
                continue;
        }

        const bool use_local = config.local_adaptive && (fshape.family == CellFamily::Line || fshape.family == CellFamily::Quad);
        if (use_local) {
            const int base_level = std::max({0, config.refinement_level, config.min_refinement_level,
                                             Tessellator::suggest_refinement_level(cell_order)});
            const int max_level = std::max(base_level, std::max(0, config.max_refinement_level));
            std::function<std::array<real_t, 3>(const TessParamPoint&)> map =
                [&](const TessParamPoint& xi_face) -> std::array<real_t, 3> {
                const auto xi_cf = map_face_param_to_cell_face_param(fshape.family, xi_face, perm);
                const auto xi_cell = FaceEmbedding::embed_face_point(cshape.family, local_face_id, xi_cf);
                return CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi_cell, config.configuration).coordinates;
            };
            if (fshape.family == CellFamily::Line) {
                grid = subdivide_line_local_adaptive(base_level, max_level, config.curvature_threshold, map);
                tess.sub_element_shape = linear_shape(CellFamily::Line);
            } else {
                grid = subdivide_quad_local_adaptive(base_level, max_level, config.curvature_threshold, map);
                tess.sub_element_shape = linear_shape(CellFamily::Quad);
            }
        } else {
            const int level = adaptive_refinement_level_for_boundary_face_from_cell(mesh, f, cell, local_face_id, perm, config);
            switch (fshape.family) {
                case CellFamily::Line:
                    grid = subdivide_line(level);
                    tess.sub_element_shape = linear_shape(CellFamily::Line);
                    break;
                case CellFamily::Triangle:
                    grid = subdivide_triangle(level);
                    tess.sub_element_shape = linear_shape(CellFamily::Triangle);
                    break;
                case CellFamily::Quad:
                    grid = subdivide_quad(level);
                    tess.sub_element_shape = linear_shape(CellFamily::Quad);
                    break;
                default:
                    break;
            }
        }

        tess.vertices.reserve(grid.points.size());
        tess.field_values.clear();
        if (config.field_evaluator) tess.field_values.resize(grid.points.size());
        for (size_t i = 0; i < grid.points.size(); ++i) {
            const auto& xi_face = grid.points[i];
            const auto xi_cf = map_face_param_to_cell_face_param(fshape.family, xi_face, perm);
            const auto xi_cell = FaceEmbedding::embed_face_point(cshape.family, local_face_id, xi_cf);
            const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi_cell, config.configuration);
            tess.vertices.push_back(eval.coordinates);
            if (config.field_evaluator) {
                config.field_evaluator(cell, xi_cell, tess.field_values[i]);
            }
        }
        tess.connectivity = std::move(grid.connectivity);
        tess.offsets = std::move(grid.offsets);
        out.push_back(std::move(tess));
    }
    return out;
}

int Tessellator::suggest_refinement_level(int order) {
    return std::max(0, order - 1);
}

//=============================================================================
// Subdivision generators
//=============================================================================

Tessellator::SubdivisionGrid Tessellator::subdivide_line(int level) {
    SubdivisionGrid grid;
    const int n_div = pow2(level);
    grid.points.reserve(static_cast<size_t>(n_div + 1));
    for (int i = 0; i <= n_div; ++i) {
        const real_t x = -1.0 + 2.0 * static_cast<real_t>(i) / static_cast<real_t>(n_div);
        grid.points.push_back({x, 0, 0});
    }
    grid.offsets.push_back(0);
    for (int i = 0; i < n_div; ++i) {
        grid.connectivity.push_back(i);
        grid.connectivity.push_back(i + 1);
        grid.offsets.push_back(static_cast<int>(grid.connectivity.size()));
    }
    return grid;
}

Tessellator::SubdivisionGrid Tessellator::subdivide_triangle(int level) {
    SubdivisionGrid grid;
    const int n_div = pow2(level);
    const int n_points = (n_div + 1) * (n_div + 2) / 2;
    grid.points.reserve(static_cast<size_t>(n_points));

    for (int j = 0; j <= n_div; ++j) {
        for (int i = 0; i <= n_div - j; ++i) {
            const real_t xi = static_cast<real_t>(i) / static_cast<real_t>(n_div);
            const real_t eta = static_cast<real_t>(j) / static_cast<real_t>(n_div);
            grid.points.push_back({xi, eta, 0});
        }
    }

    auto index = [n_div](int i, int j) -> int {
        return j * (2 * n_div + 3 - j) / 2 + i;
    };

    grid.offsets.push_back(0);
    for (int j = 0; j < n_div; ++j) {
        for (int i = 0; i < n_div - j; ++i) {
            // lower-left
            grid.connectivity.push_back(index(i, j));
            grid.connectivity.push_back(index(i + 1, j));
            grid.connectivity.push_back(index(i, j + 1));
            grid.offsets.push_back(static_cast<int>(grid.connectivity.size()));

            // upper-right
            if (i + j + 1 < n_div) {
                grid.connectivity.push_back(index(i + 1, j));
                grid.connectivity.push_back(index(i + 1, j + 1));
                grid.connectivity.push_back(index(i, j + 1));
                grid.offsets.push_back(static_cast<int>(grid.connectivity.size()));
            }
        }
    }

    return grid;
}

Tessellator::SubdivisionGrid Tessellator::subdivide_quad(int level) {
    SubdivisionGrid grid;
    const int n_div = pow2(level);
    const int n1 = n_div + 1;
    grid.points.reserve(static_cast<size_t>(n1 * n1));

    for (int j = 0; j < n1; ++j) {
        for (int i = 0; i < n1; ++i) {
            const real_t xi = -1.0 + 2.0 * static_cast<real_t>(i) / static_cast<real_t>(n_div);
            const real_t eta = -1.0 + 2.0 * static_cast<real_t>(j) / static_cast<real_t>(n_div);
            grid.points.push_back({xi, eta, 0});
        }
    }

    auto index = [n1](int i, int j) -> int { return j * n1 + i; };

    grid.offsets.push_back(0);
    for (int j = 0; j < n_div; ++j) {
        for (int i = 0; i < n_div; ++i) {
            grid.connectivity.push_back(index(i, j));
            grid.connectivity.push_back(index(i + 1, j));
            grid.connectivity.push_back(index(i + 1, j + 1));
            grid.connectivity.push_back(index(i, j + 1));
            grid.offsets.push_back(static_cast<int>(grid.connectivity.size()));
        }
    }
    return grid;
}

Tessellator::SubdivisionGrid Tessellator::subdivide_line_local_adaptive(
    int base_level,
    int max_level,
    real_t curvature_threshold,
    const std::function<std::array<real_t, 3>(const TessParamPoint&)>& map) {

    base_level = std::max(0, base_level);
    max_level = std::max(base_level, max_level);
    if (max_level == base_level || curvature_threshold <= 0 || !map) {
        return subdivide_line(base_level);
    }

    const int N = pow2(max_level);

    struct Segment {
        int i0 = 0;
        int i1 = 0;
    };

    std::vector<Segment> segments;
    segments.reserve(static_cast<size_t>(N));

    auto refine = [&](auto&& self, int i0, int i1, int level) -> void {
        const real_t x0 = -1.0 + 2.0 * static_cast<real_t>(i0) / static_cast<real_t>(N);
        const real_t x1 = -1.0 + 2.0 * static_cast<real_t>(i1) / static_cast<real_t>(N);
        const TessParamPoint A{x0, 0.0, 0.0};
        const TessParamPoint B{x1, 0.0, 0.0};
        const real_t err = segment_chord_error(map, A, B);
        if (err > curvature_threshold && level < max_level && (i1 - i0) > 1) {
            const int im = (i0 + i1) / 2;
            self(self, i0, im, level + 1);
            self(self, im, i1, level + 1);
        } else {
            segments.push_back({i0, i1});
        }
    };

    const int n_base = pow2(base_level);
    const int step = N / n_base;
    for (int k = 0; k < n_base; ++k) {
        refine(refine, k * step, (k + 1) * step, base_level);
    }

    std::sort(segments.begin(), segments.end(), [](const Segment& a, const Segment& b) { return a.i0 < b.i0; });

    std::vector<int> breakpoints;
    breakpoints.reserve(segments.size() + 1);
    breakpoints.push_back(segments.empty() ? 0 : segments[0].i0);
    int expected_start = breakpoints.back();
    for (const auto& seg : segments) {
        if (seg.i0 != expected_start) {
            throw std::runtime_error("subdivide_line_local_adaptive: non-contiguous segment list");
        }
        breakpoints.push_back(seg.i1);
        expected_start = seg.i1;
    }

    SubdivisionGrid grid;
    grid.points.reserve(breakpoints.size());
    grid.connectivity.reserve(segments.size() * 2);
    grid.offsets.reserve(segments.size() + 1);

    for (int i : breakpoints) {
        const real_t x = -1.0 + 2.0 * static_cast<real_t>(i) / static_cast<real_t>(N);
        grid.points.push_back({x, 0.0, 0.0});
    }

    grid.offsets.push_back(0);
    for (index_t i = 0; i + 1 < static_cast<index_t>(breakpoints.size()); ++i) {
        grid.connectivity.push_back(i);
        grid.connectivity.push_back(i + 1);
        grid.offsets.push_back(static_cast<int>(grid.connectivity.size()));
    }

    return grid;
}

Tessellator::SubdivisionGrid Tessellator::subdivide_quad_local_adaptive(
    int base_level,
    int max_level,
    real_t curvature_threshold,
    const std::function<std::array<real_t, 3>(const TessParamPoint&)>& map) {

    base_level = std::max(0, base_level);
    max_level = std::max(base_level, max_level);
    if (max_level == base_level || curvature_threshold <= 0 || !map) {
        return subdivide_quad(base_level);
    }

    const int N = pow2(max_level);

    struct Quad {
        int i0 = 0;
        int i1 = 0;
        int j0 = 0;
        int j1 = 0;
        int level = 0;
    };

    std::vector<Quad> leaves;

    auto coord = [N](int i) -> real_t {
        return -1.0 + 2.0 * static_cast<real_t>(i) / static_cast<real_t>(N);
    };

    auto quad_error = [&](int i0, int i1, int j0, int j1) -> real_t {
        const real_t u0 = coord(i0);
        const real_t u1 = coord(i1);
        const real_t v0 = coord(j0);
        const real_t v1 = coord(j1);

        const TessParamPoint p00{u0, v0, 0.0};
        const TessParamPoint p10{u1, v0, 0.0};
        const TessParamPoint p11{u1, v1, 0.0};
        const TessParamPoint p01{u0, v1, 0.0};

        real_t err = 0.0;
        err = std::max(err, segment_chord_error(map, p00, p10));
        err = std::max(err, segment_chord_error(map, p10, p11));
        err = std::max(err, segment_chord_error(map, p11, p01));
        err = std::max(err, segment_chord_error(map, p01, p00));
        return err;
    };

    auto refine = [&](auto&& self, int i0, int i1, int j0, int j1, int level) -> void {
        const real_t err = quad_error(i0, i1, j0, j1);
        if (err > curvature_threshold && level < max_level && (i1 - i0) > 1 && (j1 - j0) > 1) {
            const int im = (i0 + i1) / 2;
            const int jm = (j0 + j1) / 2;
            self(self, i0, im, j0, jm, level + 1);
            self(self, im, i1, j0, jm, level + 1);
            self(self, im, i1, jm, j1, level + 1);
            self(self, i0, im, jm, j1, level + 1);
        } else {
            leaves.push_back({i0, i1, j0, j1, level});
        }
    };

    const int n_base = pow2(base_level);
    const int step = N / n_base;
    leaves.reserve(static_cast<size_t>(n_base * n_base));
    for (int j = 0; j < n_base; ++j) {
        for (int i = 0; i < n_base; ++i) {
            const int i0 = i * step;
            const int i1 = (i + 1) * step;
            const int j0 = j * step;
            const int j1 = (j + 1) * step;
            refine(refine, i0, i1, j0, j1, base_level);
        }
    }

    SubdivisionGrid grid;
    grid.connectivity.reserve(leaves.size() * 4);
    grid.offsets.reserve(leaves.size() + 1);

    auto pack = [](int i, int j) -> long long {
        return (static_cast<long long>(i) << 32) | static_cast<unsigned int>(j);
    };

    std::unordered_map<long long, index_t> point_ids;
    point_ids.reserve(leaves.size() * 2);

    auto point = [&](int i, int j) -> index_t {
        const long long key = pack(i, j);
        auto it = point_ids.find(key);
        if (it != point_ids.end()) return it->second;
        const index_t id = static_cast<index_t>(grid.points.size());
        grid.points.push_back({coord(i), coord(j), 0.0});
        point_ids.emplace(key, id);
        return id;
    };

    grid.offsets.push_back(0);
    for (const auto& q : leaves) {
        const index_t a = point(q.i0, q.j0);
        const index_t b = point(q.i1, q.j0);
        const index_t c = point(q.i1, q.j1);
        const index_t d = point(q.i0, q.j1);
        grid.connectivity.push_back(a);
        grid.connectivity.push_back(b);
        grid.connectivity.push_back(c);
        grid.connectivity.push_back(d);
        grid.offsets.push_back(static_cast<int>(grid.connectivity.size()));
    }

    return grid;
}

Tessellator::SubdivisionGrid Tessellator::subdivide_hex(int level) {
    SubdivisionGrid grid;
    const int n_div = pow2(level);
    const int n1 = n_div + 1;
    grid.points.reserve(static_cast<size_t>(n1 * n1 * n1));

    for (int k = 0; k < n1; ++k) {
        for (int j = 0; j < n1; ++j) {
            for (int i = 0; i < n1; ++i) {
                const real_t xi = -1.0 + 2.0 * static_cast<real_t>(i) / static_cast<real_t>(n_div);
                const real_t eta = -1.0 + 2.0 * static_cast<real_t>(j) / static_cast<real_t>(n_div);
                const real_t zeta = -1.0 + 2.0 * static_cast<real_t>(k) / static_cast<real_t>(n_div);
                grid.points.push_back({xi, eta, zeta});
            }
        }
    }

    auto index = [n1](int i, int j, int k) -> int { return k * n1 * n1 + j * n1 + i; };

    grid.offsets.push_back(0);
    for (int k = 0; k < n_div; ++k) {
        for (int j = 0; j < n_div; ++j) {
            for (int i = 0; i < n_div; ++i) {
                grid.connectivity.push_back(index(i, j, k));
                grid.connectivity.push_back(index(i + 1, j, k));
                grid.connectivity.push_back(index(i + 1, j + 1, k));
                grid.connectivity.push_back(index(i, j + 1, k));
                grid.connectivity.push_back(index(i, j, k + 1));
                grid.connectivity.push_back(index(i + 1, j, k + 1));
                grid.connectivity.push_back(index(i + 1, j + 1, k + 1));
                grid.connectivity.push_back(index(i, j + 1, k + 1));
                grid.offsets.push_back(static_cast<int>(grid.connectivity.size()));
            }
        }
    }

    return grid;
}

Tessellator::SubdivisionGrid Tessellator::subdivide_tet(int level) {
    SubdivisionGrid grid;
    const int N = pow2(level);

    struct Int3 {
        int i, j, k;
    };

    auto pack = [](int i, int j, int k) -> long long {
        // 21 bits each (N is small in practice)
        return (static_cast<long long>(i) << 42) |
               (static_cast<long long>(j) << 21) |
               static_cast<long long>(k);
    };

    std::unordered_map<long long, index_t> map;
    std::vector<Int3> coords;
    coords.reserve(static_cast<size_t>((N + 1) * (N + 2) * (N + 3) / 6));

    auto add_point = [&](int i, int j, int k) -> index_t {
        long long key = pack(i, j, k);
        auto it = map.find(key);
        if (it != map.end()) return it->second;
        index_t idx = static_cast<index_t>(coords.size());
        coords.push_back({i, j, k});
        map.emplace(key, idx);
        return idx;
    };

    auto midpoint = [&](index_t a, index_t b) -> index_t {
        const auto pa = coords[static_cast<size_t>(a)];
        const auto pb = coords[static_cast<size_t>(b)];
        return add_point((pa.i + pb.i) / 2, (pa.j + pb.j) / 2, (pa.k + pb.k) / 2);
    };

    // Start with reference tet corners in integer coordinates.
    const index_t v0 = add_point(0, 0, 0);
    const index_t v1 = add_point(N, 0, 0);
    const index_t v2 = add_point(0, N, 0);
    const index_t v3 = add_point(0, 0, N);

    std::vector<std::array<index_t, 4>> tets;
    tets.push_back({v0, v1, v2, v3});

    for (int l = 0; l < level; ++l) {
        std::vector<std::array<index_t, 4>> next;
        next.reserve(tets.size() * 8);
        for (const auto& T : tets) {
            const index_t a = T[0], b = T[1], c = T[2], d = T[3];
            const index_t m01 = midpoint(a, b);
            const index_t m02 = midpoint(a, c);
            const index_t m03 = midpoint(a, d);
            const index_t m12 = midpoint(b, c);
            const index_t m13 = midpoint(b, d);
            const index_t m23 = midpoint(c, d);

            next.push_back({a,  m01, m02, m03});
            next.push_back({b,  m01, m12, m13});
            next.push_back({c,  m02, m12, m23});
            next.push_back({d,  m03, m13, m23});

            // central octahedron split along diagonal m01-m23
            next.push_back({m01, m02, m03, m23});
            next.push_back({m01, m03, m13, m23});
            next.push_back({m01, m13, m12, m23});
            next.push_back({m01, m12, m02, m23});
        }
        tets.swap(next);
    }

    // Convert unique integer vertices to parametric points (xi,eta,zeta) in [0,1]
    grid.points.reserve(coords.size());
    for (const auto& p : coords) {
        grid.points.push_back({
            static_cast<real_t>(p.i) / static_cast<real_t>(N),
            static_cast<real_t>(p.j) / static_cast<real_t>(N),
            static_cast<real_t>(p.k) / static_cast<real_t>(N)
        });
    }

    grid.offsets.push_back(0);
    for (const auto& T : tets) {
        grid.connectivity.insert(grid.connectivity.end(), T.begin(), T.end());
        grid.offsets.push_back(static_cast<int>(grid.connectivity.size()));
    }

    return grid;
}

Tessellator::SubdivisionGrid Tessellator::subdivide_wedge(int level) {
    SubdivisionGrid grid;
    const int n_div = pow2(level);

    // Base triangle subdivision
    const SubdivisionGrid tri = subdivide_triangle(level);
    const int n_tri_pts = static_cast<int>(tri.points.size());
    const int n_layers = n_div + 1;

    grid.points.reserve(static_cast<size_t>(n_layers * n_tri_pts));
    for (int k = 0; k < n_layers; ++k) {
        const real_t z = -1.0 + 2.0 * static_cast<real_t>(k) / static_cast<real_t>(n_div);
        for (const auto& p : tri.points) {
            grid.points.push_back({p[0], p[1], z});
        }
    }

    grid.offsets.push_back(0);
    const int n_tris = tri.n_sub_elements();
    for (int k = 0; k < n_div; ++k) {
        for (int ti = 0; ti < n_tris; ++ti) {
            const int b = tri.offsets[ti];
            const int e = tri.offsets[ti + 1];
            if (e - b != 3) continue;
            const int a0 = tri.connectivity[static_cast<size_t>(b + 0)];
            const int a1 = tri.connectivity[static_cast<size_t>(b + 1)];
            const int a2 = tri.connectivity[static_cast<size_t>(b + 2)];

            const int lo = k * n_tri_pts;
            const int hi = (k + 1) * n_tri_pts;

            grid.connectivity.push_back(lo + a0);
            grid.connectivity.push_back(lo + a1);
            grid.connectivity.push_back(lo + a2);
            grid.connectivity.push_back(hi + a0);
            grid.connectivity.push_back(hi + a1);
            grid.connectivity.push_back(hi + a2);
            grid.offsets.push_back(static_cast<int>(grid.connectivity.size()));
        }
    }

    return grid;
}

Tessellator::SubdivisionGrid Tessellator::subdivide_pyramid(int level) {
    SubdivisionGrid grid;
    // Pyramid is tessellated as 2 tetrahedra (base diagonal split), with optional
    // red refinement applied level times (2 * 8^level tetrahedra).
    grid.points = {
        {-1, -1, 0},
        { 1, -1, 0},
        { 1,  1, 0},
        {-1,  1, 0},
        { 0,  0, 1}
    };

    std::vector<std::array<index_t, 4>> tets;
    tets.push_back({0, 1, 2, 4});
    tets.push_back({0, 2, 3, 4});

    struct EdgeKey {
        index_t a;
        index_t b;
        bool operator==(const EdgeKey& o) const noexcept { return a == o.a && b == o.b; }
    };
    struct EdgeKeyHash {
        size_t operator()(const EdgeKey& k) const noexcept {
            const size_t h1 = std::hash<index_t>{}(k.a);
            const size_t h2 = std::hash<index_t>{}(k.b);
            // hash_combine
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    for (int l = 0; l < level; ++l) {
        std::unordered_map<EdgeKey, index_t, EdgeKeyHash> mid_cache;
        mid_cache.reserve(tets.size() * 6);

        auto midpoint = [&](index_t a, index_t b) -> index_t {
            const EdgeKey key{std::min(a, b), std::max(a, b)};
            auto it = mid_cache.find(key);
            if (it != mid_cache.end()) return it->second;
            const auto& pa = grid.points[static_cast<size_t>(a)];
            const auto& pb = grid.points[static_cast<size_t>(b)];
            const index_t idx = static_cast<index_t>(grid.points.size());
            grid.points.push_back({0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]), 0.5 * (pa[2] + pb[2])});
            mid_cache.emplace(key, idx);
            return idx;
        };

        std::vector<std::array<index_t, 4>> next;
        next.reserve(tets.size() * 8);
        for (const auto& T : tets) {
            const index_t a = T[0], b = T[1], c = T[2], d = T[3];
            const index_t m01 = midpoint(a, b);
            const index_t m02 = midpoint(a, c);
            const index_t m03 = midpoint(a, d);
            const index_t m12 = midpoint(b, c);
            const index_t m13 = midpoint(b, d);
            const index_t m23 = midpoint(c, d);

            next.push_back({a,  m01, m02, m03});
            next.push_back({b,  m01, m12, m13});
            next.push_back({c,  m02, m12, m23});
            next.push_back({d,  m03, m13, m23});

            // central octahedron split along diagonal m01-m23 (matches subdivide_tet)
            next.push_back({m01, m02, m03, m23});
            next.push_back({m01, m03, m13, m23});
            next.push_back({m01, m13, m12, m23});
            next.push_back({m01, m12, m02, m23});
        }
        tets.swap(next);
    }

    grid.offsets.push_back(0);
    for (const auto& T : tets) {
        grid.connectivity.insert(grid.connectivity.end(), T.begin(), T.end());
        grid.offsets.push_back(static_cast<int>(grid.connectivity.size()));
    }

    return grid;
}

//=============================================================================
// Mapping to physical coordinates
//=============================================================================

void Tessellator::map_subdivision_to_physical(
    const MeshBase& mesh,
    index_t cell,
    const SubdivisionGrid& grid,
    Configuration cfg,
    TessellatedCell& result) {

    result.vertices.clear();
    result.vertices.reserve(grid.points.size());

    for (const auto& xi : grid.points) {
        const auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
        result.vertices.push_back(eval.coordinates);
    }
}

void Tessellator::map_face_subdivision_to_physical(
    const MeshBase& mesh,
    index_t face,
    const SubdivisionGrid& grid,
    Configuration cfg,
    TessellatedFace& result) {

    result.vertices.clear();
    result.vertices.reserve(grid.points.size());

    for (const auto& xi : grid.points) {
        result.vertices.push_back(map_face_point(mesh, face, xi, cfg));
    }
}

} // namespace svmp
