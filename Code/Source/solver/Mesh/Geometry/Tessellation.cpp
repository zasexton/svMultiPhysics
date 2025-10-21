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
#include "CurvilinearEval.h"
#include <fstream>
#include <stdexcept>
#include <cmath>

namespace svmp {

//=============================================================================
// Tessellator Implementation
//=============================================================================

TessellatedCell Tessellator::tessellate_cell(
    const MeshBase& mesh,
    index_t cell,
    const TessellationConfig& config) {

    TessellatedCell result;
    result.cell_id = cell;
    result.cell_shape = mesh.cell_shape(cell);

    // Generate parametric subdivision
    SubdivisionGrid grid;

    switch (result.cell_shape) {
        case CellShape::Vertex:
            result.sub_element_shape = CellShape::Vertex;
            return result;  // No subdivision needed

        case CellShape::Line:
            grid = subdivide_line(config.refinement_level);
            result.sub_element_shape = CellShape::Line;
            break;

        case CellShape::Triangle:
            grid = subdivide_triangle(config.refinement_level);
            result.sub_element_shape = CellShape::Triangle;
            break;

        case CellShape::Quad:
            grid = subdivide_quad(config.refinement_level);
            result.sub_element_shape = CellShape::Quad;
            break;

        case CellShape::Tetrahedron:
            grid = subdivide_tet(config.refinement_level);
            result.sub_element_shape = CellShape::Tetrahedron;
            break;

        case CellShape::Hexahedron:
            grid = subdivide_hex(config.refinement_level);
            result.sub_element_shape = CellShape::Hexahedron;
            break;

        case CellShape::Wedge:
            grid = subdivide_wedge(config.refinement_level);
            result.sub_element_shape = CellShape::Wedge;
            break;

        case CellShape::Pyramid:
            grid = subdivide_pyramid(config.refinement_level);
            result.sub_element_shape = CellShape::Pyramid;
            break;

        default:
            throw std::runtime_error("Unsupported cell shape for tessellation");
    }

    // Map to physical coordinates
    map_subdivision_to_physical(mesh, cell, grid, config.configuration, result);

    // Interpolate fields if requested
    if (config.interpolate_fields) {
        interpolate_fields(mesh, cell, grid, config, result);
    }

    // Copy connectivity
    result.connectivity = grid.connectivity;
    result.offsets = grid.offsets;

    return result;
}

TessellatedFace Tessellator::tessellate_face(
    const MeshBase& mesh,
    index_t face,
    const TessellationConfig& config) {

    TessellatedFace result;
    result.face_id = face;

    // Get face vertices and determine shape
    auto vertices = mesh.face_vertices(face);

    if (vertices.size() == 2) {
        result.face_shape = CellShape::Line;
    } else if (vertices.size() == 3) {
        result.face_shape = CellShape::Triangle;
    } else if (vertices.size() == 4) {
        result.face_shape = CellShape::Quad;
    } else {
        // Polygon: triangulate
        result.face_shape = CellShape::Triangle;
    }

    // For now, simplified implementation: just copy vertices
    auto get_coords = [&](index_t vid) -> std::array<real_t, 3> {
        if (config.configuration == Configuration::Current ||
            config.configuration == Configuration::Deformed) {
            return mesh.vertex_current_coordinates(vid);
        } else {
            return mesh.vertex_reference_coordinates(vid);
        }
    };

    for (auto vid : vertices) {
        result.vertices.push_back(get_coords(vid));
    }

    // Simple linear connectivity
    result.connectivity.resize(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        result.connectivity[i] = static_cast<index_t>(i);
    }

    result.offsets = {0, static_cast<int>(vertices.size())};
    result.sub_element_shape = result.face_shape;

    return result;
}

std::vector<TessellatedCell> Tessellator::tessellate_mesh(
    const MeshBase& mesh,
    const TessellationConfig& config) {

    std::vector<TessellatedCell> tessellation;
    tessellation.reserve(mesh.n_cells());

    for (index_t cell = 0; cell < mesh.n_cells(); ++cell) {
        tessellation.push_back(tessellate_cell(mesh, cell, config));
    }

    return tessellation;
}

std::vector<TessellatedFace> Tessellator::tessellate_boundary(
    const MeshBase& mesh,
    const TessellationConfig& config) {

    std::vector<TessellatedFace> tessellation;

    for (index_t face = 0; face < mesh.n_faces(); ++face) {
        if (mesh.is_boundary_face(face)) {
            tessellation.push_back(tessellate_face(mesh, face, config));
        }
    }

    return tessellation;
}

int Tessellator::suggest_refinement_level(int order) {
    // For polynomial order p, use p-1 subdivisions to resolve curvature
    return std::max(0, order - 1);
}

void Tessellator::export_to_vtk(
    const std::vector<TessellatedCell>& tessellation,
    const std::string& filename,
    bool include_fields) {

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open VTK file: " + filename);
    }

    // Count total vertices and cells
    size_t total_vertices = 0;
    size_t total_cells = 0;

    for (const auto& tess : tessellation) {
        total_vertices += tess.vertices.size();
        total_cells += tess.n_sub_elements();
    }

    // Write VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "Tessellated Mesh\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";

    // Write points
    file << "POINTS " << total_vertices << " double\n";
    for (const auto& tess : tessellation) {
        for (const auto& v : tess.vertices) {
            file << v[0] << " " << v[1] << " " << v[2] << "\n";
        }
    }

    // Write cells
    size_t connectivity_size = 0;
    for (const auto& tess : tessellation) {
        connectivity_size += tess.n_sub_elements() + tess.connectivity.size();
    }

    file << "CELLS " << total_cells << " " << connectivity_size << "\n";

    size_t vertex_offset = 0;
    for (const auto& tess : tessellation) {
        for (int i = 0; i < tess.n_sub_elements(); ++i) {
            auto sub_conn = tess.get_sub_element(i);
            file << sub_conn.size();
            for (auto idx : sub_conn) {
                file << " " << (vertex_offset + idx);
            }
            file << "\n";
        }
        vertex_offset += tess.vertices.size();
    }

    // Write cell types
    file << "CELL_TYPES " << total_cells << "\n";
    for (const auto& tess : tessellation) {
        int vtk_type = 1;  // VTK_VERTEX
        switch (tess.sub_element_shape) {
            case CellShape::Line: vtk_type = 3; break;        // VTK_LINE
            case CellShape::Triangle: vtk_type = 5; break;    // VTK_TRIANGLE
            case CellShape::Quad: vtk_type = 9; break;        // VTK_QUAD
            case CellShape::Tetrahedron: vtk_type = 10; break;// VTK_TETRA
            case CellShape::Hexahedron: vtk_type = 12; break; // VTK_HEXAHEDRON
            case CellShape::Wedge: vtk_type = 13; break;      // VTK_WEDGE
            case CellShape::Pyramid: vtk_type = 14; break;    // VTK_PYRAMID
            default: vtk_type = 1;
        }

        for (int i = 0; i < tess.n_sub_elements(); ++i) {
            file << vtk_type << "\n";
        }
    }

    // Write field data if requested
    if (include_fields && !tessellation.empty() &&
        !tessellation[0].field_values.empty()) {
        file << "POINT_DATA " << total_vertices << "\n";

        // Assume all fields have same structure
        for (size_t field_idx = 0; field_idx < tessellation[0].field_values.size(); ++field_idx) {
            file << "SCALARS field_" << field_idx << " double 1\n";
            file << "LOOKUP_TABLE default\n";

            for (const auto& tess : tessellation) {
                for (auto val : tess.field_values[field_idx]) {
                    file << val << "\n";
                }
            }
        }
    }

    file.close();
}

//=============================================================================
// Parametric Subdivision Generators
//=============================================================================

Tessellator::SubdivisionGrid Tessellator::subdivide_line(int level) {
    int n_div = (1 << level);  // 2^level divisions
    int n_points = n_div + 1;

    SubdivisionGrid grid;
    grid.points.resize(n_points);

    for (int i = 0; i < n_points; ++i) {
        real_t xi = -1.0 + 2.0 * i / n_div;
        grid.points[i] = {xi, 0, 0};
    }

    // Connectivity: n_div line segments
    grid.connectivity.reserve(2 * n_div);
    grid.offsets.reserve(n_div + 1);
    grid.offsets.push_back(0);

    for (int i = 0; i < n_div; ++i) {
        grid.connectivity.push_back(i);
        grid.connectivity.push_back(i + 1);
        grid.offsets.push_back(grid.connectivity.size());
    }

    return grid;
}

Tessellator::SubdivisionGrid Tessellator::subdivide_triangle(int level) {
    return TriangleSubdivision::uniform_subdivision(level);
}

Tessellator::SubdivisionGrid Tessellator::subdivide_quad(int level) {
    return TensorProductSubdivision::subdivide_quad(level);
}

Tessellator::SubdivisionGrid Tessellator::subdivide_tet(int level) {
    // Simplified: single tet for level 0
    SubdivisionGrid grid;

    if (level == 0) {
        grid.points = {
            {0, 0, 0},
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1}
        };
        grid.connectivity = {0, 1, 2, 3};
        grid.offsets = {0, 4};
    } else {
        // TODO: Implement recursive tet subdivision
        grid = subdivide_tet(0);
    }

    return grid;
}

Tessellator::SubdivisionGrid Tessellator::subdivide_hex(int level) {
    return TensorProductSubdivision::subdivide_hex(level);
}

Tessellator::SubdivisionGrid Tessellator::subdivide_wedge(int level) {
    // Wedge = triangle × line
    SubdivisionGrid grid;

    if (level == 0) {
        grid.points = {
            {0, 0, -1}, {1, 0, -1}, {0, 1, -1},  // Bottom triangle
            {0, 0,  1}, {1, 0,  1}, {0, 1,  1}   // Top triangle
        };
        grid.connectivity = {0, 1, 2, 3, 4, 5};
        grid.offsets = {0, 6};
    } else {
        // TODO: Implement wedge subdivision
        grid = subdivide_wedge(0);
    }

    return grid;
}

Tessellator::SubdivisionGrid Tessellator::subdivide_pyramid(int level) {
    SubdivisionGrid grid;

    if (level == 0) {
        grid.points = {
            {-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0},  // Base
            {0, 0, 1}  // Apex
        };
        grid.connectivity = {0, 1, 2, 3, 4};
        grid.offsets = {0, 5};
    } else {
        // TODO: Implement pyramid subdivision
        grid = subdivide_pyramid(0);
    }

    return grid;
}

void Tessellator::map_subdivision_to_physical(
    const MeshBase& mesh,
    index_t cell,
    const SubdivisionGrid& grid,
    Configuration cfg,
    TessellatedCell& result) {

    result.vertices.reserve(grid.points.size());

    for (const auto& xi : grid.points) {
        auto eval = CurvilinearEvaluator::evaluate_geometry(mesh, cell, xi, cfg);
        result.vertices.push_back(eval.coordinates);
    }
}

void Tessellator::interpolate_fields(
    const MeshBase& mesh,
    index_t cell,
    const SubdivisionGrid& grid,
    const TessellationConfig& config,
    TessellatedCell& result) {

    // Placeholder: would interpolate field values from cell nodes to subdivision points
    // using shape functions
    result.field_values.resize(config.field_names.size());

    for (size_t field_idx = 0; field_idx < config.field_names.size(); ++field_idx) {
        result.field_values[field_idx].resize(grid.points.size(), 0.0);
        // TODO: Actual field interpolation using shape functions
    }
}

real_t Tessellator::estimate_curvature(
    const MeshBase& mesh,
    index_t cell,
    const std::array<real_t, 3>& xi,
    Configuration cfg) {

    // Simplified curvature estimate: ||∂²x/∂ξ²||
    // Would require second derivatives of shape functions
    return 0.0;  // Placeholder
}

//=============================================================================
// TriangleSubdivision Implementation
//=============================================================================

Tessellator::SubdivisionGrid TriangleSubdivision::uniform_subdivision(int level) {
    Tessellator::SubdivisionGrid grid;

    int n_div = (1 << level);  // 2^level divisions per edge
    int n_points = (n_div + 1) * (n_div + 2) / 2;

    grid.points.reserve(n_points);

    // Generate barycentric grid
    for (int j = 0; j <= n_div; ++j) {
        for (int i = 0; i <= n_div - j; ++i) {
            real_t lambda0 = static_cast<real_t>(n_div - i - j) / n_div;
            real_t lambda1 = static_cast<real_t>(i) / n_div;
            real_t lambda2 = static_cast<real_t>(j) / n_div;

            grid.points.push_back({lambda1, lambda2, 0});
        }
    }

    // Generate connectivity for sub-triangles
    auto index = [n_div](int i, int j) -> int {
        return j * (2*n_div + 3 - j) / 2 + i;
    };

    grid.offsets.push_back(0);

    for (int j = 0; j < n_div; ++j) {
        for (int i = 0; i < n_div - j; ++i) {
            // Lower-left triangle
            grid.connectivity.push_back(index(i, j));
            grid.connectivity.push_back(index(i+1, j));
            grid.connectivity.push_back(index(i, j+1));
            grid.offsets.push_back(grid.connectivity.size());

            // Upper-right triangle (if not on hypotenuse)
            if (i + j + 1 < n_div) {
                grid.connectivity.push_back(index(i+1, j));
                grid.connectivity.push_back(index(i+1, j+1));
                grid.connectivity.push_back(index(i, j+1));
                grid.offsets.push_back(grid.connectivity.size());
            }
        }
    }

    return grid;
}

std::vector<std::array<real_t, 3>> TriangleSubdivision::barycentric_grid(int level) {
    auto grid = uniform_subdivision(level);
    return grid.points;
}

//=============================================================================
// TensorProductSubdivision Implementation
//=============================================================================

Tessellator::SubdivisionGrid TensorProductSubdivision::subdivide_quad(int level) {
    int n_div = (1 << level);
    int n_points_1d = n_div + 1;
    int n_points = n_points_1d * n_points_1d;

    Tessellator::SubdivisionGrid grid;
    grid.points.reserve(n_points);

    // Generate grid points
    for (int j = 0; j < n_points_1d; ++j) {
        for (int i = 0; i < n_points_1d; ++i) {
            real_t xi = -1.0 + 2.0 * i / n_div;
            real_t eta = -1.0 + 2.0 * j / n_div;
            grid.points.push_back({xi, eta, 0});
        }
    }

    // Generate connectivity
    grid.offsets.push_back(0);

    for (int j = 0; j < n_div; ++j) {
        for (int i = 0; i < n_div; ++i) {
            int i0 = j * n_points_1d + i;
            int i1 = j * n_points_1d + i + 1;
            int i2 = (j+1) * n_points_1d + i + 1;
            int i3 = (j+1) * n_points_1d + i;

            grid.connectivity.push_back(i0);
            grid.connectivity.push_back(i1);
            grid.connectivity.push_back(i2);
            grid.connectivity.push_back(i3);
            grid.offsets.push_back(grid.connectivity.size());
        }
    }

    return grid;
}

Tessellator::SubdivisionGrid TensorProductSubdivision::subdivide_hex(int level) {
    int n_div = (1 << level);
    int n_points_1d = n_div + 1;
    int n_points = n_points_1d * n_points_1d * n_points_1d;

    Tessellator::SubdivisionGrid grid;
    grid.points.reserve(n_points);

    // Generate grid points
    for (int k = 0; k < n_points_1d; ++k) {
        for (int j = 0; j < n_points_1d; ++j) {
            for (int i = 0; i < n_points_1d; ++i) {
                real_t xi = -1.0 + 2.0 * i / n_div;
                real_t eta = -1.0 + 2.0 * j / n_div;
                real_t zeta = -1.0 + 2.0 * k / n_div;
                grid.points.push_back({xi, eta, zeta});
            }
        }
    }

    // Generate connectivity
    grid.offsets.push_back(0);

    auto index = [n_points_1d](int i, int j, int k) -> int {
        return k * n_points_1d * n_points_1d + j * n_points_1d + i;
    };

    for (int k = 0; k < n_div; ++k) {
        for (int j = 0; j < n_div; ++j) {
            for (int i = 0; i < n_div; ++i) {
                // Hex connectivity (8 vertices)
                grid.connectivity.push_back(index(i,   j,   k));
                grid.connectivity.push_back(index(i+1, j,   k));
                grid.connectivity.push_back(index(i+1, j+1, k));
                grid.connectivity.push_back(index(i,   j+1, k));
                grid.connectivity.push_back(index(i,   j,   k+1));
                grid.connectivity.push_back(index(i+1, j,   k+1));
                grid.connectivity.push_back(index(i+1, j+1, k+1));
                grid.connectivity.push_back(index(i,   j+1, k+1));
                grid.offsets.push_back(grid.connectivity.size());
            }
        }
    }

    return grid;
}

Tessellator::SubdivisionGrid TensorProductSubdivision::triangulate_quad(int level) {
    auto quad_grid = subdivide_quad(level);

    Tessellator::SubdivisionGrid tri_grid;
    tri_grid.points = quad_grid.points;
    tri_grid.offsets.push_back(0);

    // Each quad → 2 triangles
    for (int i = 0; i < quad_grid.n_sub_elements(); ++i) {
        auto quad = quad_grid.get_sub_element(i);

        // Triangle 1: v0, v1, v2
        tri_grid.connectivity.push_back(quad[0]);
        tri_grid.connectivity.push_back(quad[1]);
        tri_grid.connectivity.push_back(quad[2]);
        tri_grid.offsets.push_back(tri_grid.connectivity.size());

        // Triangle 2: v0, v2, v3
        tri_grid.connectivity.push_back(quad[0]);
        tri_grid.connectivity.push_back(quad[2]);
        tri_grid.connectivity.push_back(quad[3]);
        tri_grid.offsets.push_back(tri_grid.connectivity.size());
    }

    return tri_grid;
}

Tessellator::SubdivisionGrid TensorProductSubdivision::tetrahedralize_hex(int level) {
    auto hex_grid = subdivide_hex(level);

    Tessellator::SubdivisionGrid tet_grid;
    tet_grid.points = hex_grid.points;
    tet_grid.offsets.push_back(0);

    // Each hex → 5 or 6 tets (standard decomposition)
    // Simplified: use 6-tet decomposition
    for (int i = 0; i < hex_grid.n_sub_elements(); ++i) {
        auto hex = hex_grid.get_sub_element(i);

        // 6-tet decomposition (placeholder)
        // Would need proper hex-to-tet templates
        tet_grid.connectivity.insert(tet_grid.connectivity.end(), hex.begin(), hex.begin() + 4);
        tet_grid.offsets.push_back(tet_grid.connectivity.size());
    }

    return tet_grid;
}

//=============================================================================
// SurfaceExtractor Implementation
//=============================================================================

std::vector<TessellatedFace> SurfaceExtractor::extract_surface(
    const MeshBase& mesh,
    const TessellationConfig& config) {

    return Tessellator::tessellate_boundary(mesh, config);
}

std::vector<TessellatedFace> SurfaceExtractor::extract_isosurface(
    const MeshBase& mesh,
    const std::string& field_name,
    real_t isovalue,
    const TessellationConfig& config) {

    // Placeholder for marching cubes/tetrahedra implementation
    std::vector<TessellatedFace> isosurface;

    // Would implement marching cubes/tetrahedra algorithm here
    // For each cell:
    //   1. Sample field at grid points
    //   2. Find edge crossings where field = isovalue
    //   3. Triangulate isosurface within cell

    return isosurface;
}

} // namespace svmp
