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

#ifndef SVMP_GMSH_READER_H
#define SVMP_GMSH_READER_H

#include "../Core/MeshBase.h"
#include "../Core/MeshTypes.h"
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <fstream>

namespace svmp {

/**
 * @brief Gmsh mesh reader supporting MSH 2.x and 4.x formats
 *
 * This class reads mesh data from Gmsh MSH format files and converts them
 * to the svmp::MeshBase format. It supports:
 * - MSH 2.2 ASCII format (legacy, widely used)
 * - MSH 4.1 ASCII format (current)
 * - Physical groups for region/boundary labels
 * - Linear and high-order elements
 * - Mixed element meshes (tets, hexes, pyramids, wedges, etc.)
 *
 * Gmsh element types supported:
 * - 1D: Line (2-node), Line3 (3-node)
 * - 2D: Triangle (3-node), Quad (4-node), Triangle6, Quad8, Quad9
 * - 3D: Tetrahedron (4-node), Hexahedron (8-node), Prism/Wedge (6-node),
 *       Pyramid (5-node), and high-order variants
 *
 * @see https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
 */
class GmshReader {
public:
  GmshReader();
  ~GmshReader();

  /**
   * @brief Read a mesh from a Gmsh MSH file
   * @param options IO options including file path
   * @return Loaded mesh
   * @throws std::runtime_error if file cannot be read or format is unsupported
   */
  static MeshBase read(const MeshIOOptions& options);

  /**
   * @brief Read a mesh from a Gmsh MSH file
   * @param filename Path to the .msh file
   * @return Loaded mesh
   */
  static MeshBase read(const std::string& filename);

  /**
   * @brief Register Gmsh reader with MeshBase I/O registry
   *
   * Registers reader for "gmsh" and "msh" formats
   */
  static void register_with_mesh_io();

  /**
   * @brief Check if a file appears to be a Gmsh MSH file
   * @param filename Path to check
   * @return True if file starts with $MeshFormat
   */
  static bool is_gmsh_file(const std::string& filename);

  /**
   * @brief Get the MSH format version from a file
   * @param filename Path to the .msh file
   * @return Version number (e.g., 2.2, 4.1) or 0.0 if not detected
   */
  static double get_format_version(const std::string& filename);

private:
  /**
   * @brief Internal element data from parsing
   */
  struct GmshElement {
    int type;                    // Gmsh element type ID
    int physical_tag;            // Physical group tag (for labels)
    int entity_tag;              // Geometric entity tag
    std::vector<size_t> nodes;   // Node indices (1-based in file, converted to 0-based)
  };

  /**
   * @brief Physical group information
   */
  struct PhysicalGroup {
    int dimension;
    int tag;
    std::string name;
  };

  /**
   * @brief Read MSH 2.x format file
   */
  static MeshBase read_msh2(std::ifstream& file, const std::string& filename);

  /**
   * @brief Read MSH 2.x binary format file
   */
  static MeshBase read_msh2_binary(std::ifstream& file,
                                   const std::string& filename,
                                   int data_size,
                                   bool swap_endian);

  /**
   * @brief Read MSH 4.x format file
   */
  static MeshBase read_msh4(std::ifstream& file, const std::string& filename);

  /**
   * @brief Read MSH 4.x binary format file
   */
  static MeshBase read_msh4_binary(std::ifstream& file,
                                   const std::string& filename,
                                   int data_size,
                                   bool swap_endian);

  /**
   * @brief Parse $Nodes section (MSH 2.x)
   */
  static void parse_nodes_v2(std::ifstream& file,
                             std::vector<real_t>& coords,
                             std::unordered_map<size_t, size_t>& node_id_map);

  /**
   * @brief Parse $Nodes section (MSH 2.x binary)
   */
  static void parse_nodes_v2_binary(std::ifstream& file,
                                    std::vector<real_t>& coords,
                                    std::unordered_map<size_t, size_t>& node_id_map,
                                    int data_size,
                                    bool swap_endian);

  /**
   * @brief Parse $Nodes section (MSH 4.x)
   */
  static void parse_nodes_v4(std::ifstream& file,
                             std::vector<real_t>& coords,
                             std::unordered_map<size_t, size_t>& node_id_map);

  /**
   * @brief Parse $Nodes section (MSH 4.x binary)
   */
  static void parse_nodes_v4_binary(std::ifstream& file,
                                    std::vector<real_t>& coords,
                                    std::unordered_map<size_t, size_t>& node_id_map,
                                    int data_size,
                                    bool swap_endian);

  /**
   * @brief Parse $Elements section (MSH 2.x)
   */
  static void parse_elements_v2(std::ifstream& file,
                                std::vector<GmshElement>& elements);

  /**
   * @brief Parse $Elements section (MSH 2.x binary)
   */
  static void parse_elements_v2_binary(std::ifstream& file,
                                       std::vector<GmshElement>& elements,
                                       bool swap_endian);

  /**
   * @brief Parse $Elements section (MSH 4.x)
   */
  static void parse_elements_v4(std::ifstream& file,
                                std::vector<GmshElement>& elements);

  /**
   * @brief Parse $Elements section (MSH 4.x binary)
   */
  static void parse_elements_v4_binary(std::ifstream& file,
                                       std::vector<GmshElement>& elements,
                                       int data_size,
                                       bool swap_endian);

  /**
   * @brief Parse $PhysicalNames section
   */
  static void parse_physical_names(std::ifstream& file,
                                   std::vector<PhysicalGroup>& groups);

  /**
   * @brief Convert Gmsh element type to CellShape
   * @param gmsh_type Gmsh element type ID
   * @return Corresponding CellShape
   */
  static CellShape gmsh_to_cellshape(int gmsh_type);

  /**
   * @brief Get number of nodes for a Gmsh element type
   * @param gmsh_type Gmsh element type ID
   * @return Number of nodes
   */
  static int gmsh_element_num_nodes(int gmsh_type);

  /**
   * @brief Get topological dimension of Gmsh element type
   * @param gmsh_type Gmsh element type ID
   * @return Dimension (0=point, 1=line, 2=surface, 3=volume)
   */
  static int gmsh_element_dimension(int gmsh_type);

  /**
   * @brief Reorder Gmsh node numbering to svmp convention
   *
   * Gmsh uses different node orderings for some elements.
   * This function reorders nodes to match our convention.
   *
   * @param gmsh_type Gmsh element type ID
   * @param nodes Node indices to reorder (modified in place)
   */
  static void reorder_nodes_to_svmp(int gmsh_type, std::vector<size_t>& nodes);

  /**
   * @brief Build mesh from parsed data
   */
  static MeshBase build_mesh(const std::vector<real_t>& coords,
                             const std::vector<GmshElement>& elements,
                             const std::vector<PhysicalGroup>& physical_groups,
                             const std::unordered_map<size_t, size_t>& node_id_map);

  /**
   * @brief Skip to end of current section
   */
  static void skip_section(std::ifstream& file, const std::string& end_tag);

  /**
   * @brief Gmsh element type constants
   */
  enum GmshElementType {
    GMSH_LINE = 1,
    GMSH_TRIANGLE = 2,
    GMSH_QUAD = 3,
    GMSH_TETRAHEDRON = 4,
    GMSH_HEXAHEDRON = 5,
    GMSH_PRISM = 6,        // Wedge
    GMSH_PYRAMID = 7,
    GMSH_LINE3 = 8,        // 3-node line (quadratic)
    GMSH_TRIANGLE6 = 9,    // 6-node triangle (quadratic)
    GMSH_QUAD9 = 10,       // 9-node quad (quadratic)
    GMSH_TETRAHEDRON10 = 11,
    GMSH_HEXAHEDRON27 = 12,
    GMSH_PRISM18 = 13,
    GMSH_PYRAMID14 = 14,
    GMSH_POINT = 15,
    GMSH_QUAD8 = 16,       // 8-node serendipity quad
    GMSH_HEXAHEDRON20 = 17,
    GMSH_PRISM15 = 18,
    GMSH_PYRAMID13 = 19,
    GMSH_TRIANGLE9 = 20,   // 9-node cubic triangle
    GMSH_TRIANGLE10 = 21,  // 10-node cubic triangle
    GMSH_TRIANGLE12 = 22,
    GMSH_TRIANGLE15 = 23,
    GMSH_TRIANGLE15_IC = 24,
    GMSH_TRIANGLE21 = 25,
    GMSH_EDGE4 = 26,
    GMSH_EDGE5 = 27,
    GMSH_EDGE6 = 28,
    GMSH_TETRAHEDRON20 = 29,
    GMSH_TETRAHEDRON35 = 30,
    GMSH_TETRAHEDRON56 = 31,
    GMSH_HEXAHEDRON64 = 92,
    GMSH_HEXAHEDRON125 = 93
  };
};

} // namespace svmp

#endif // SVMP_GMSH_READER_H
