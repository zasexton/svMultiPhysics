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
#include <vector>
#include <array>

namespace svmp {

// Forward declarations
class MeshBase;

/**
 * @brief Linearized (tessellated) representation of a high-order element
 *
 * Breaks a curved/high-order element into linear sub-elements for
 * visualization, I/O export, or conservative integration.
 *
 * Example: A quadratic triangle (6 nodes) → 4 linear triangles
 */
struct TessellatedCell {
    /// Original cell index
    index_t cell_id{INVALID_INDEX};

    /// Original cell shape
    CellShape cell_shape{CellShape::Vertex};

    /// Sub-element shape (linear version of cell_shape)
    CellShape sub_element_shape{CellShape::Vertex};

    /// Tessellated vertex coordinates (physical space)
    std::vector<std::array<real_t, 3>> vertices;

    /// Sub-element connectivity (indices into vertices array)
    /// Flattened: [sub_elem_0_nodes..., sub_elem_1_nodes..., ...]
    std::vector<index_t> connectivity;

    /// Offsets into connectivity array for each sub-element
    std::vector<int> offsets;

    /// Field values at tessellation vertices (optional)
    std::vector<std::vector<real_t>> field_values;

    /// Get number of sub-elements
    int n_sub_elements() const {
        return static_cast<int>(offsets.size()) - 1;
    }

    /// Get connectivity for a specific sub-element
    std::vector<index_t> get_sub_element(int sub_elem_id) const {
        int start = offsets[sub_elem_id];
        int end = offsets[sub_elem_id + 1];
        return std::vector<index_t>(connectivity.begin() + start,
                                    connectivity.begin() + end);
    }
};

/**
 * @brief Tessellated representation of a high-order face
 */
struct TessellatedFace {
    index_t face_id{INVALID_INDEX};
    CellShape face_shape{CellShape::Vertex};
    CellShape sub_element_shape{CellShape::Vertex};

    std::vector<std::array<real_t, 3>> vertices;
    std::vector<index_t> connectivity;
    std::vector<int> offsets;
    std::vector<std::vector<real_t>> field_values;

    int n_sub_elements() const {
        return static_cast<int>(offsets.size()) - 1;
    }
};

/**
 * @brief Tessellation configuration
 */
struct TessellationConfig {
    /// Refinement level (0 = minimal, higher = finer)
    /// Each level doubles the number of sub-elements along each edge
    int refinement_level{0};

    /// Whether to use uniform or adaptive subdivision
    bool adaptive{false};

    /// Curvature threshold for adaptive refinement
    real_t curvature_threshold{0.1};

    /// Whether to include field interpolation
    bool interpolate_fields{false};

    /// Field names to interpolate
    std::vector<std::string> field_names;

    /// Configuration to tessellate (Reference or Current)
    Configuration configuration{Configuration::Reference};
};

/**
 * @brief Tessellation engine for high-order elements
 *
 * Converts curvilinear/high-order elements into piecewise-linear representations
 * by subdividing in parametric space and evaluating geometry at subdivision points.
 *
 * **Use Cases:**
 * - VTK/ParaView visualization (requires linear cells)
 * - Conservative data transfer between meshes
 * - Accurate surface/volume integration
 * - Ray-tracing and collision detection
 *
 * **Supported Elements (all polynomial orders):**
 * - Line → linear sub-edges
 * - Triangle → linear sub-triangles
 * - Quad → linear sub-quads (or triangulated)
 * - Tet → linear sub-tets
 * - Hex → linear sub-hexes (or tetrahedral decomposition)
 * - Wedge → linear sub-wedges
 * - Pyramid → linear sub-pyramids
 *
 * **Tessellation Strategy:**
 * - Uniform: Regular parametric subdivision
 * - Adaptive: Refine based on local curvature (future)
 */
class Tessellator {
public:
    /**
     * @brief Tessellate a single cell
     * @param mesh The mesh
     * @param cell Cell index
     * @param config Tessellation configuration
     * @return Tessellated representation
     */
    static TessellatedCell tessellate_cell(
        const MeshBase& mesh,
        index_t cell,
        const TessellationConfig& config = TessellationConfig{});

    /**
     * @brief Tessellate a single face
     * @param mesh The mesh
     * @param face Face index
     * @param config Tessellation configuration
     * @return Tessellated representation
     */
    static TessellatedFace tessellate_face(
        const MeshBase& mesh,
        index_t face,
        const TessellationConfig& config = TessellationConfig{});

    /**
     * @brief Tessellate all cells in mesh
     * @param mesh The mesh
     * @param config Tessellation configuration
     * @return Vector of tessellated cells
     */
    static std::vector<TessellatedCell> tessellate_mesh(
        const MeshBase& mesh,
        const TessellationConfig& config = TessellationConfig{});

    /**
     * @brief Tessellate all boundary faces
     * @param mesh The mesh
     * @param config Tessellation configuration
     * @return Vector of tessellated faces
     */
    static std::vector<TessellatedFace> tessellate_boundary(
        const MeshBase& mesh,
        const TessellationConfig& config = TessellationConfig{});

    /**
     * @brief Determine minimal refinement level for polynomial order
     * @param order Polynomial order
     * @return Suggested refinement level
     *
     * Returns refinement level needed to resolve geometry curvature.
     * For order p, suggests level = p - 1 (quadratic → 1 subdivision, etc.)
     */
    static int suggest_refinement_level(int order);

    /**
     * @brief Export tessellated mesh to VTK format
     * @param tessellation Tessellated cells
     * @param filename Output VTK file path
     * @param include_fields Whether to export field data
     */
    static void export_to_vtk(
        const std::vector<TessellatedCell>& tessellation,
        const std::string& filename,
        bool include_fields = false);

private:
    // ---- Parametric subdivision generators ----

    /// Generate uniform subdivision points in parametric space
    struct SubdivisionGrid {
        std::vector<std::array<real_t, 3>> points;  // Parametric coords
        std::vector<index_t> connectivity;           // Sub-element connectivity
        std::vector<int> offsets;                    // Sub-element offsets
    };

    static SubdivisionGrid subdivide_line(int level);
    static SubdivisionGrid subdivide_triangle(int level);
    static SubdivisionGrid subdivide_quad(int level);
    static SubdivisionGrid subdivide_tet(int level);
    static SubdivisionGrid subdivide_hex(int level);
    static SubdivisionGrid subdivide_wedge(int level);
    static SubdivisionGrid subdivide_pyramid(int level);

    /// Map parametric subdivision to physical coordinates
    static void map_subdivision_to_physical(
        const MeshBase& mesh,
        index_t cell,
        const SubdivisionGrid& grid,
        Configuration cfg,
        TessellatedCell& result);

    /// Interpolate field values at subdivision points
    static void interpolate_fields(
        const MeshBase& mesh,
        index_t cell,
        const SubdivisionGrid& grid,
        const TessellationConfig& config,
        TessellatedCell& result);

    /// Estimate local curvature for adaptive refinement
    static real_t estimate_curvature(
        const MeshBase& mesh,
        index_t cell,
        const std::array<real_t, 3>& xi,
        Configuration cfg);
};

/**
 * @brief Triangle subdivision patterns
 *
 * Provides standard subdivision patterns for triangles (uniform refinement).
 */
class TriangleSubdivision {
public:
    /**
     * @brief Subdivide triangle into 4^level sub-triangles
     * @param level Refinement level (0 = no subdivision, 1 = 4 tris, 2 = 16 tris, ...)
     * @return Subdivision connectivity in barycentric coordinates
     */
    static Tessellator::SubdivisionGrid uniform_subdivision(int level);

    /**
     * @brief Get barycentric coordinates for vertices in uniform subdivision
     * @param level Refinement level
     * @param n_divisions Number of divisions per edge (2^level)
     * @return Grid of (λ₀, λ₁, λ₂) points
     */
    static std::vector<std::array<real_t, 3>> barycentric_grid(int level);
};

/**
 * @brief Quad/Hex tensor-product subdivision
 */
class TensorProductSubdivision {
public:
    /**
     * @brief Subdivide quad into (2^level)² sub-quads
     * @param level Refinement level
     * @return Subdivision in (ξ, η) ∈ [-1,1]²
     */
    static Tessellator::SubdivisionGrid subdivide_quad(int level);

    /**
     * @brief Subdivide hex into (2^level)³ sub-hexes
     * @param level Refinement level
     * @return Subdivision in (ξ, η, ζ) ∈ [-1,1]³
     */
    static Tessellator::SubdivisionGrid subdivide_hex(int level);

    /**
     * @brief Alternative: subdivide quad into triangles
     * @param level Refinement level
     * @return Triangulated subdivision
     */
    static Tessellator::SubdivisionGrid triangulate_quad(int level);

    /**
     * @brief Alternative: subdivide hex into tets
     * @param level Refinement level
     * @return Tetrahedral decomposition
     */
    static Tessellator::SubdivisionGrid tetrahedralize_hex(int level);
};

/**
 * @brief Curved surface extractor
 *
 * Extracts surface tessellation from volume elements for rendering.
 */
class SurfaceExtractor {
public:
    /**
     * @brief Extract tessellated surface from mesh
     * @param mesh The mesh
     * @param config Tessellation configuration
     * @return Tessellated boundary faces
     */
    static std::vector<TessellatedFace> extract_surface(
        const MeshBase& mesh,
        const TessellationConfig& config = TessellationConfig{});

    /**
     * @brief Extract isosurface of a field
     * @param mesh The mesh
     * @param field_name Field name
     * @param isovalue Isovalue
     * @param config Tessellation configuration
     * @return Triangulated isosurface
     */
    static std::vector<TessellatedFace> extract_isosurface(
        const MeshBase& mesh,
        const std::string& field_name,
        real_t isovalue,
        const TessellationConfig& config = TessellationConfig{});
};

} // namespace svmp

#endif // SVMP_TESSELLATION_H
