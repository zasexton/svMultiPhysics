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

#include "MFEMReader.h"
#include "MeshIO.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <set>
#include <map>

namespace svmp {

MFEMReader::MFEMReader() = default;
MFEMReader::~MFEMReader() = default;

MeshBase MFEMReader::read(const MeshIOOptions& options) {
    std::string filename = options.path.empty() ? options.filename : options.path;
    return read(filename);
}

MeshBase MFEMReader::read(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("MFEMReader: Cannot open file: " + filename);
    }

    // Read and verify header
    std::string line;

    // Skip empty lines and comments at the beginning
    while (std::getline(file, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;  // Empty line

        // Check for comment
        if (line[start] == '#') continue;

        // Should be "MFEM mesh" header
        if (line.find("MFEM mesh") != std::string::npos ||
            line.find("MFEM INLINE mesh") != std::string::npos) {
            break;
        }

        throw std::runtime_error("MFEMReader: Invalid MFEM file header: " + filename);
    }

    return parse_mesh(file, filename);
}

void MFEMReader::register_with_mesh_io() {
    // Register MFEM reader
    MeshIO::register_reader("mfem", [](const MeshIOOptions& opts) -> std::unique_ptr<MeshBase> {
        MeshBase mesh = MFEMReader::read(opts);
        return std::make_unique<MeshBase>(std::move(mesh));
    });

    // Register capabilities
    MeshIO::FormatCapabilities caps;
    caps.supports_3d = true;
    caps.supports_2d = true;
    caps.supports_1d = true;
    caps.supports_mixed_cells = true;
    caps.supports_high_order = false;  // Basic reader doesn't handle high-order
    caps.supports_fields = false;
    caps.supports_labels = true;  // Attributes become labels
    caps.supports_binary = false;
    caps.supports_ascii = true;
    caps.supports_compression = false;

    MeshIO::register_capabilities("mfem", caps);
}

bool MFEMReader::is_mfem_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        if (line[start] == '#') continue;

        return line.find("MFEM mesh") != std::string::npos ||
               line.find("MFEM INLINE mesh") != std::string::npos;
    }

    return false;
}

std::string MFEMReader::get_format_version(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return "";

    std::string line;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        if (line[start] == '#') continue;

        // Look for version string like "MFEM mesh v1.0"
        size_t pos = line.find("MFEM mesh");
        if (pos != std::string::npos) {
            size_t ver_pos = line.find("v", pos + 9);
            if (ver_pos != std::string::npos) {
                // Extract version
                size_t end = line.find_first_of(" \t\r\n", ver_pos);
                if (end == std::string::npos) end = line.length();
                return line.substr(ver_pos, end - ver_pos);
            }
            return "v1.0";  // Default version
        }
        break;
    }

    return "";
}

// ==========================================
// Parsing functions
// ==========================================

MeshBase MFEMReader::parse_mesh(std::ifstream& file, const std::string& filename) {
    std::vector<MFEMElement> elements;
    std::vector<MFEMElement> boundary;
    std::vector<real_t> coords;
    int dimension = 0;
    int space_dim = 0;

    std::string line;

    // Parse sections
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        if (line[start] == '#') continue;

        // Check for section headers
        if (line.find("dimension") != std::string::npos) {
            dimension = parse_dimension(file);
        }
        else if (line.find("elements") != std::string::npos) {
            parse_elements(file, elements);
        }
        else if (line.find("boundary") != std::string::npos) {
            parse_boundary(file, boundary);
        }
        else if (line.find("vertices") != std::string::npos) {
            parse_vertices(file, coords, space_dim);
        }
    }

    // Validate parsed data
    if (elements.empty()) {
        throw std::runtime_error("MFEMReader: No elements found in file: " + filename);
    }
    if (coords.empty()) {
        throw std::runtime_error("MFEMReader: No vertices found in file: " + filename);
    }
    if (dimension == 0) {
        // Infer dimension from elements
        for (const auto& elem : elements) {
            dimension = std::max(dimension, mfem_element_dimension(elem.geometry_type));
        }
    }
    if (space_dim == 0) {
        space_dim = dimension;
    }

    return build_mesh(dimension, elements, boundary, coords, space_dim);
}

int MFEMReader::parse_dimension(std::ifstream& file) {
    std::string line;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        if (line[start] == '#') continue;

        int dim;
        std::istringstream iss(line);
        if (iss >> dim) {
            return dim;
        }
        break;
    }
    return 0;
}

void MFEMReader::parse_elements(std::ifstream& file, std::vector<MFEMElement>& elements) {
    std::string line;

    // Read number of elements
    int num_elements = 0;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        if (line[start] == '#') continue;

        std::istringstream iss(line);
        if (iss >> num_elements) {
            break;
        }
    }

    elements.reserve(num_elements);

    // Read elements
    for (int i = 0; i < num_elements; ++i) {
        if (!std::getline(file, line)) break;

        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) { --i; continue; }
        if (line[start] == '#') { --i; continue; }

        std::istringstream iss(line);

        MFEMElement elem;
        iss >> elem.attribute >> elem.geometry_type;

        // Read vertices
        int num_verts = mfem_element_num_vertices(elem.geometry_type);
        elem.vertices.resize(num_verts);
        for (int v = 0; v < num_verts; ++v) {
            iss >> elem.vertices[v];
        }

        elements.push_back(std::move(elem));
    }
}

void MFEMReader::parse_boundary(std::ifstream& file, std::vector<MFEMElement>& boundary) {
    std::string line;

    // Read number of boundary elements
    int num_boundary = 0;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        if (line[start] == '#') continue;

        std::istringstream iss(line);
        if (iss >> num_boundary) {
            break;
        }
    }

    boundary.reserve(num_boundary);

    // Read boundary elements
    for (int i = 0; i < num_boundary; ++i) {
        if (!std::getline(file, line)) break;

        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) { --i; continue; }
        if (line[start] == '#') { --i; continue; }

        std::istringstream iss(line);

        MFEMElement elem;
        iss >> elem.attribute >> elem.geometry_type;

        int num_verts = mfem_element_num_vertices(elem.geometry_type);
        elem.vertices.resize(num_verts);
        for (int v = 0; v < num_verts; ++v) {
            iss >> elem.vertices[v];
        }

        boundary.push_back(std::move(elem));
    }
}

void MFEMReader::parse_vertices(std::ifstream& file, std::vector<real_t>& coords, int& space_dim) {
    std::string line;

    // Read number of vertices
    int num_vertices = 0;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        if (line[start] == '#') continue;

        std::istringstream iss(line);
        if (iss >> num_vertices) {
            break;
        }
    }

    // Read space dimension
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        if (line[start] == '#') continue;

        std::istringstream iss(line);
        if (iss >> space_dim) {
            break;
        }
    }

    // Allocate coordinates (always 3D internally)
    coords.resize(num_vertices * 3, 0.0);

    // Read vertices
    for (int i = 0; i < num_vertices; ++i) {
        if (!std::getline(file, line)) break;

        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) { --i; continue; }
        if (line[start] == '#') { --i; continue; }

        std::istringstream iss(line);

        for (int d = 0; d < space_dim && d < 3; ++d) {
            iss >> coords[i * 3 + d];
        }
    }
}

// ==========================================
// Conversion functions
// ==========================================

CellShape MFEMReader::mfem_to_cellshape(int mfem_type) {
    CellShape shape;
    shape.order = 1;  // MFEM basic reader only handles linear elements

    switch (mfem_type) {
        case MFEM_POINT:
            shape.family = CellFamily::Point;
            shape.num_corners = 1;
            break;
        case MFEM_SEGMENT:
            shape.family = CellFamily::Line;
            shape.num_corners = 2;
            break;
        case MFEM_TRIANGLE:
            shape.family = CellFamily::Triangle;
            shape.num_corners = 3;
            break;
        case MFEM_SQUARE:
            shape.family = CellFamily::Quad;
            shape.num_corners = 4;
            break;
        case MFEM_TETRAHEDRON:
            shape.family = CellFamily::Tetra;
            shape.num_corners = 4;
            break;
        case MFEM_CUBE:
            shape.family = CellFamily::Hex;
            shape.num_corners = 8;
            break;
        case MFEM_PRISM:
            shape.family = CellFamily::Wedge;
            shape.num_corners = 6;
            break;
        case MFEM_PYRAMID:
            shape.family = CellFamily::Pyramid;
            shape.num_corners = 5;
            break;
        default:
            throw std::runtime_error("MFEMReader: Unknown MFEM geometry type: " +
                                   std::to_string(mfem_type));
    }

    return shape;
}

int MFEMReader::mfem_element_num_vertices(int mfem_type) {
    switch (mfem_type) {
        case MFEM_POINT:       return 1;
        case MFEM_SEGMENT:     return 2;
        case MFEM_TRIANGLE:    return 3;
        case MFEM_SQUARE:      return 4;
        case MFEM_TETRAHEDRON: return 4;
        case MFEM_CUBE:        return 8;
        case MFEM_PRISM:       return 6;
        case MFEM_PYRAMID:     return 5;
        default:
            return 0;
    }
}

int MFEMReader::mfem_element_dimension(int mfem_type) {
    switch (mfem_type) {
        case MFEM_POINT:       return 0;
        case MFEM_SEGMENT:     return 1;
        case MFEM_TRIANGLE:    return 2;
        case MFEM_SQUARE:      return 2;
        case MFEM_TETRAHEDRON: return 3;
        case MFEM_CUBE:        return 3;
        case MFEM_PRISM:       return 3;
        case MFEM_PYRAMID:     return 3;
        default:
            return 0;
    }
}

// ==========================================
// Mesh building
// ==========================================

MeshBase MFEMReader::build_mesh(int dimension,
                                 const std::vector<MFEMElement>& elements,
                                 const std::vector<MFEMElement>& boundary,
                                 const std::vector<real_t>& coords,
                                 int space_dim) {
    // Separate elements by dimension
    std::vector<MFEMElement> volume_elements;
    std::vector<MFEMElement> surface_elements;

    for (const auto& elem : elements) {
        int elem_dim = mfem_element_dimension(elem.geometry_type);
        if (elem_dim == dimension) {
            volume_elements.push_back(elem);
        } else if (elem_dim == dimension - 1) {
            surface_elements.push_back(elem);
        }
    }

    // Build connectivity arrays using correct types
    std::vector<offset_t> cell2vertex_offsets;
    std::vector<index_t> cell2vertex;
    std::vector<CellShape> cell_shapes;
    std::vector<label_t> cell_regions;

    cell2vertex_offsets.reserve(volume_elements.size() + 1);
    cell2vertex_offsets.push_back(0);

    for (const auto& elem : volume_elements) {
        cell_shapes.push_back(mfem_to_cellshape(elem.geometry_type));
        cell_regions.push_back(static_cast<label_t>(elem.attribute));

        for (size_t v : elem.vertices) {
            cell2vertex.push_back(static_cast<index_t>(v));
        }
        cell2vertex_offsets.push_back(static_cast<offset_t>(cell2vertex.size()));
    }

    // Build mesh using correct API
    // Always use 3D coordinates (z=0 for 2D meshes) like GmshReader
    MeshBase mesh;
    mesh.build_from_arrays(3, coords, cell2vertex_offsets,
                          cell2vertex, cell_shapes);

    // Set region labels for each cell
    for (size_t c = 0; c < cell_regions.size(); ++c) {
        mesh.set_region_label(static_cast<index_t>(c), cell_regions[c]);
    }

    // Set global IDs (sequential numbering)
    std::vector<gid_t> vertex_gids(coords.size() / 3);
    for (size_t i = 0; i < vertex_gids.size(); ++i) {
        vertex_gids[i] = static_cast<gid_t>(i);
    }
    mesh.set_vertex_gids(vertex_gids);

    std::vector<gid_t> cell_gids(volume_elements.size());
    for (size_t i = 0; i < cell_gids.size(); ++i) {
        cell_gids[i] = static_cast<gid_t>(i);
    }
    mesh.set_cell_gids(cell_gids);

    // Store boundary info for later face labeling
    // Full integration requires matching face vertices after finalize
    (void)boundary;  // Suppress unused warning

    // Finalize the mesh
    mesh.finalize();

    return mesh;
}

void MFEMReader::skip_comments(std::ifstream& file) {
    std::string line;
    std::streampos pos = file.tellg();

    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) {
            pos = file.tellg();
            continue;
        }
        if (line[start] == '#') {
            pos = file.tellg();
            continue;
        }
        // Found non-comment, seek back
        file.seekg(pos);
        return;
    }
}

bool MFEMReader::skip_to_section(std::ifstream& file, const std::string& section_name) {
    std::string line;
    while (std::getline(file, line)) {
        if (line.find(section_name) != std::string::npos) {
            return true;
        }
    }
    return false;
}

} // namespace svmp
