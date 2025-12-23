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

#ifndef SVMP_MFEM_READER_H
#define SVMP_MFEM_READER_H

#include "../Core/MeshBase.h"
#include "../Core/MeshTypes.h"
#include <string>
#include <memory>
#include <vector>
#include <fstream>

namespace svmp {

/**
 * @brief MFEM native mesh format reader
 *
 * This class reads mesh data from MFEM's native mesh format (v1.0, v1.1, v1.2).
 * The format is a simple ASCII text format with sections for:
 * - dimension
 * - elements (with material/attribute IDs)
 * - boundary (boundary elements with attribute IDs)
 * - vertices (coordinates)
 *
 * MFEM geometry types:
 * - POINT       = 0
 * - SEGMENT     = 1
 * - TRIANGLE    = 2
 * - SQUARE      = 3 (Quadrilateral)
 * - TETRAHEDRON = 4
 * - CUBE        = 5 (Hexahedron)
 * - PRISM       = 6 (Wedge)
 * - PYRAMID     = 7
 *
 * @see https://mfem.org/mesh-formats/
 */
class MFEMReader {
public:
    MFEMReader();
    ~MFEMReader();

    /**
     * @brief Read a mesh from an MFEM format file
     * @param options IO options including file path
     * @return Loaded mesh
     * @throws std::runtime_error if file cannot be read or format is invalid
     */
    static MeshBase read(const MeshIOOptions& options);

    /**
     * @brief Read a mesh from an MFEM format file
     * @param filename Path to the .mesh file
     * @return Loaded mesh
     */
    static MeshBase read(const std::string& filename);

    /**
     * @brief Register MFEM reader with MeshIO registry
     *
     * Registers reader for "mfem" and "mesh" formats
     */
    static void register_with_mesh_io();

    /**
     * @brief Check if a file appears to be an MFEM mesh file
     * @param filename Path to check
     * @return True if file starts with "MFEM mesh"
     */
    static bool is_mfem_file(const std::string& filename);

    /**
     * @brief Get the MFEM mesh format version from a file
     * @param filename Path to the .mesh file
     * @return Version string (e.g., "v1.0") or empty if not detected
     */
    static std::string get_format_version(const std::string& filename);

private:
    /**
     * @brief Internal element data from parsing
     */
    struct MFEMElement {
        int attribute;           // Material/region attribute
        int geometry_type;       // MFEM geometry type ID
        std::vector<size_t> vertices;  // Vertex indices (0-based)
    };

    /**
     * @brief Parse mesh file
     */
    static MeshBase parse_mesh(std::ifstream& file, const std::string& filename);

    /**
     * @brief Parse dimension section
     */
    static int parse_dimension(std::ifstream& file);

    /**
     * @brief Parse elements section
     */
    static void parse_elements(std::ifstream& file, std::vector<MFEMElement>& elements);

    /**
     * @brief Parse boundary section
     */
    static void parse_boundary(std::ifstream& file, std::vector<MFEMElement>& boundary);

    /**
     * @brief Parse vertices section
     */
    static void parse_vertices(std::ifstream& file, std::vector<real_t>& coords, int& space_dim);

    /**
     * @brief Convert MFEM geometry type to CellShape
     * @param mfem_type MFEM geometry type ID
     * @return Corresponding CellShape
     */
    static CellShape mfem_to_cellshape(int mfem_type);

    /**
     * @brief Get number of vertices for an MFEM geometry type
     * @param mfem_type MFEM geometry type ID
     * @return Number of vertices
     */
    static int mfem_element_num_vertices(int mfem_type);

    /**
     * @brief Get topological dimension of MFEM geometry type
     * @param mfem_type MFEM geometry type ID
     * @return Dimension (0=point, 1=segment, 2=surface, 3=volume)
     */
    static int mfem_element_dimension(int mfem_type);

    /**
     * @brief Build mesh from parsed data
     */
    static MeshBase build_mesh(int dimension,
                               const std::vector<MFEMElement>& elements,
                               const std::vector<MFEMElement>& boundary,
                               const std::vector<real_t>& coords,
                               int space_dim);

    /**
     * @brief Skip to a specific section in the file
     */
    static bool skip_to_section(std::ifstream& file, const std::string& section_name);

    /**
     * @brief Skip comment lines (lines starting with #)
     */
    static void skip_comments(std::ifstream& file);

    /**
     * @brief MFEM geometry type constants
     */
    enum MFEMGeometryType {
        MFEM_POINT       = 0,
        MFEM_SEGMENT     = 1,
        MFEM_TRIANGLE    = 2,
        MFEM_SQUARE      = 3,  // Quadrilateral
        MFEM_TETRAHEDRON = 4,
        MFEM_CUBE        = 5,  // Hexahedron
        MFEM_PRISM       = 6,  // Wedge
        MFEM_PYRAMID     = 7
    };
};

} // namespace svmp

#endif // SVMP_MFEM_READER_H
