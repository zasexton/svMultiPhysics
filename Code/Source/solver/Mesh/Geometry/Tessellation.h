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

#ifndef SVMP_TESSELLATION_H
#define SVMP_TESSELLATION_H

#include "../Core/MeshTypes.h"
#include "../Topology/CellShape.h"

#include <array>
#include <functional>
#include <string>
#include <vector>

namespace svmp {

class MeshBase;

using TessParamPoint = std::array<real_t, 3>;

struct TessellatedCell {
    index_t cell_id{INVALID_INDEX};
    CellShape cell_shape{};
    CellShape sub_element_shape{};

    std::vector<std::array<real_t, 3>> vertices;
    std::vector<index_t> connectivity;
    std::vector<int> offsets; // offsets.size() == n_sub_elements + 1

    std::vector<std::vector<real_t>> field_values;

    int n_sub_elements() const { return std::max(0, static_cast<int>(offsets.size()) - 1); }
};

struct TessellatedFace {
    index_t face_id{INVALID_INDEX};
    CellShape face_shape{};
    CellShape sub_element_shape{};

    std::vector<std::array<real_t, 3>> vertices;
    std::vector<index_t> connectivity;
    std::vector<int> offsets;
    std::vector<std::vector<real_t>> field_values;

    int n_sub_elements() const { return std::max(0, static_cast<int>(offsets.size()) - 1); }
};

struct TessellationConfig {
    int refinement_level{0};                 // 0 = minimal
    bool adaptive{false};                    // if true, increase refinement_level until chord error <= curvature_threshold
    real_t curvature_threshold{0.1};         // relative chord-error threshold used when adaptive=true
    bool local_adaptive{false};              // if true, perform per-subcell adaptive refinement (line/quad supported; quads emit triangles)
    int min_refinement_level{0};             // minimum refinement level when adaptive/local_adaptive is enabled
    int max_refinement_level{8};             // maximum refinement level when adaptive/local_adaptive is enabled

    // Optional field evaluator. If provided, tessellation will evaluate this callback
    // at each tessellated vertex and store values in `field_values` (per-vertex vectors).
    std::function<void(index_t cell, const TessParamPoint& xi, std::vector<real_t>& out)> field_evaluator;

    bool interpolate_fields{false};          // legacy/reserved
    std::vector<std::string> field_names;    // legacy/reserved
    Configuration configuration{Configuration::Reference};
};

class Tessellator {
public:
    static TessellatedCell tessellate_cell(
        const MeshBase& mesh,
        index_t cell,
        const TessellationConfig& config = TessellationConfig{});
    // Note: Pyramid cells are tessellated into linear tetrahedra for robustness.

    static TessellatedFace tessellate_face(
        const MeshBase& mesh,
        index_t face,
        const TessellationConfig& config = TessellationConfig{});

    static std::vector<TessellatedCell> tessellate_mesh(
        const MeshBase& mesh,
        const TessellationConfig& config = TessellationConfig{});

    static std::vector<TessellatedFace> tessellate_boundary(
        const MeshBase& mesh,
        const TessellationConfig& config = TessellationConfig{});

    static int suggest_refinement_level(int order);

private:
    struct SubdivisionGrid {
        std::vector<TessParamPoint> points;  // parametric points
        std::vector<index_t> connectivity;   // flat connectivity
        std::vector<int> offsets;            // per-sub-element offsets

        int n_sub_elements() const { return std::max(0, static_cast<int>(offsets.size()) - 1); }
    };

    static SubdivisionGrid subdivide_line(int level);
    static SubdivisionGrid subdivide_triangle(int level);
    static SubdivisionGrid subdivide_quad(int level);
    static SubdivisionGrid subdivide_tet(int level);
    static SubdivisionGrid subdivide_hex(int level);
    static SubdivisionGrid subdivide_wedge(int level);
    static SubdivisionGrid subdivide_pyramid(int level);

    static SubdivisionGrid subdivide_line_local_adaptive(
        int base_level,
        int max_level,
        real_t curvature_threshold,
        const std::function<std::array<real_t, 3>(const TessParamPoint&)>& map);

    static SubdivisionGrid subdivide_quad_local_adaptive(
        int base_level,
        int max_level,
        real_t curvature_threshold,
        const std::function<std::array<real_t, 3>(const TessParamPoint&)>& map);

    static void map_subdivision_to_physical(
        const MeshBase& mesh,
        index_t cell,
        const SubdivisionGrid& grid,
        Configuration cfg,
        TessellatedCell& result);

    static void map_face_subdivision_to_physical(
        const MeshBase& mesh,
        index_t face,
        const SubdivisionGrid& grid,
        Configuration cfg,
        TessellatedFace& result);
};

} // namespace svmp

#endif // SVMP_TESSELLATION_H
